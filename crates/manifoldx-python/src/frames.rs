//! Fused frame ownership. The GIL is acquired only by explicit participants.
use super::*;
use manifoldx_core::clock::{conflicts, Cadence, FrameClock};
use manifoldx_core::ecs::{Compare, ComponentReadout, ComponentSchema, Scalar, Selection};
use manifoldx_core::runtime::{Admission, Command, Control, Staged, MAX_COMMAND_BYTES};
use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const MAX_SYSTEMS: usize = 32;
const MAX_QUEUE: usize = 64;
const MAX_PREDICATES: usize = 16;
const STRUCTURAL: usize = usize::MAX;
const MAX_SNAPSHOT_BYTES: usize = 16 * 1024 * 1024;
const MAX_FRAMES: u64 = 1_000_000;

type Predicate = (usize, usize, String, (Option<f32>, Option<u32>));
type Registration = (String, Option<f64>, Vec<usize>, Vec<usize>, Py<PyAny>);

fn scalar(value: (Option<f32>, Option<u32>)) -> PyResult<Scalar> {
    match value {
        (Some(value), None) if value.is_finite() => Ok(Scalar::F32(value)),
        (None, Some(value)) => Ok(Scalar::U32(value)),
        _ => Err(PyValueError::new_err("expected a finite f32 or a u32")),
    }
}

fn selection(component: usize, predicates: Vec<Predicate>) -> PyResult<Selection> {
    if predicates.len() > MAX_PREDICATES {
        return Err(PyValueError::new_err("too many predicates"));
    }
    Ok(Selection {
        component,
        predicates: predicates
            .into_iter()
            .map(|(index, field, op, value)| {
                let op = match op.as_str() {
                    "eq" => Compare::Eq,
                    "ne" => Compare::Ne,
                    "lt" => Compare::Lt,
                    "le" => Compare::Le,
                    "gt" => Compare::Gt,
                    "ge" => Compare::Ge,
                    _ => return Err(PyValueError::new_err("unknown comparison")),
                };
                Ok((index, field, op, scalar(value)?))
            })
            .collect::<PyResult<_>>()?,
    })
}

fn set_command(
    root: usize,
    predicates: Vec<Predicate>,
    component: usize,
    field: usize,
    value: (Option<f32>, Option<u32>),
) -> PyResult<Command> {
    Ok(Command::Set {
        selection: selection(root, predicates)?,
        component,
        field,
        value: scalar(value)?,
    })
}

#[derive(Default)]
struct ReadCounts {
    rows: u64,
    bytes: u64,
    calls: u64,
}

struct Snapshot {
    clock: (u64, f64, f64),
    overruns: Vec<(String, u64)>,
    columns: Vec<(usize, Arc<ComponentReadout>)>,
}

#[pyclass(frozen)]
struct FrameSnapshot {
    live: Mutex<Option<Snapshot>>,
    counts: Arc<Mutex<ReadCounts>>,
}

#[pymethods]
impl FrameSnapshot {
    #[getter]
    fn clock(&self) -> PyResult<(u64, f64, f64)> {
        self.live
            .lock()
            .unwrap()
            .as_ref()
            .map(|s| s.clock)
            .ok_or_else(|| PyRuntimeError::new_err("snapshot lease expired"))
    }

    #[getter]
    fn overruns(&self) -> PyResult<Vec<(String, u64)>> {
        let overruns = self
            .live
            .lock()
            .unwrap()
            .as_ref()
            .map(|s| s.overruns.clone());
        overruns.ok_or_else(|| PyRuntimeError::new_err("snapshot lease expired"))
    }

    fn read<'py>(&self, py: Python<'py>, column: usize) -> PyResult<Bound<'py, PyDict>> {
        let readout = {
            let snapshot = self.live.lock().unwrap();
            snapshot.as_ref().map(|s| {
                s.columns
                    .iter()
                    .find(|(id, _)| *id == column)
                    .map(|(_, data)| Arc::clone(data))
            })
        };
        let readout = readout
            .ok_or_else(|| PyRuntimeError::new_err("snapshot lease expired"))?
            .ok_or_else(|| PyValueError::new_err("column was not declared for reading"))?;
        let result = readout_dict(py, &readout)?;
        count_read(&self.counts, &readout);
        Ok(result)
    }
}

fn readout_dict<'py>(py: Python<'py>, readout: &ComponentReadout) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    let membership: Vec<_> = readout.membership.iter().copied().map(u8::from).collect();
    dict.set_item("kind", readout.schema.element.as_str())?;
    dict.set_item("width", readout.schema.width)?;
    dict.set_item("version", readout.version)?;
    dict.set_item("count", readout.ids.len())?;
    dict.set_item("handles", PyBytes::new(py, &handles_bytes(&readout.ids)))?;
    dict.set_item("membership", PyBytes::new(py, &membership))?;
    dict.set_item("data", PyBytes::new(py, &column_bytes(&readout.data)))?;
    Ok(dict)
}

fn readout_bytes(readout: &ComponentReadout) -> u64 {
    (readout.ids.len() * 17 + readout.data.byte_len()) as u64
}

fn count_read(counts: &Mutex<ReadCounts>, readout: &ComponentReadout) {
    let mut counts = counts.lock().unwrap();
    counts.rows += readout.ids.len() as u64;
    counts.bytes += readout_bytes(readout);
    counts.calls += 1;
}

#[derive(Default)]
struct Batch {
    commands: Vec<Staged>,
    events: Vec<(String, String)>,
    bytes: usize,
}

#[pyclass(frozen)]
struct FrameCommands {
    live: Mutex<Option<Batch>>,
    writes: Arc<[usize]>,
    reads: Arc<[usize]>,
    state: CallState,
}

#[pymethods]
impl FrameCommands {
    fn set(
        &self,
        root: usize,
        predicates: Vec<Predicate>,
        component: usize,
        field: usize,
        value: (Option<f32>, Option<u32>),
    ) -> PyResult<u64> {
        if !self.writes.contains(&component) {
            return Err(PyValueError::new_err("column was not declared for writing"));
        }
        if predicates
            .iter()
            .any(|(column, ..)| !self.reads.contains(column))
        {
            return Err(PyValueError::new_err(
                "predicate column was not declared for reading",
            ));
        }
        self.stage(set_command(root, predicates, component, field, value)?)
    }

    fn spawn(&self, py: Python<'_>, count: u32, payload: &Bound<'_, PyAny>) -> PyResult<u64> {
        self.check_structural()?;
        let credit = self.state.control.reserve().map_err(admission_error)?;
        let payload = payload.call0()?;
        let command = spawn_command(
            py,
            &self.state.schemas,
            self.state.max_rows,
            count,
            payload.cast::<PyDict>()?,
        )?;
        self.stage_with(|| credit.stage(command))
    }

    fn destroy(&self, py: Python<'_>, handles: &Bound<'_, PyAny>) -> PyResult<u64> {
        self.check_structural()?;
        let credit = self.state.control.reserve().map_err(admission_error)?;
        let handles = handles.call0()?;
        let ids = decode_handles_limited(py, &handles, Some((self.state.max_rows, 16)))?;
        self.stage_with(|| credit.stage(Command::Destroy(ids)))
    }

    fn emit(&self, name: String, value: String) -> PyResult<()> {
        self.push(name.len() + value.len(), |batch| {
            batch.events.push((name, value))
        })
    }
}

impl FrameCommands {
    fn check_structural(&self) -> PyResult<()> {
        let active = self.live.lock().unwrap().is_some();
        if !active {
            return Err(PyRuntimeError::new_err("command lease expired"));
        }
        if self.writes.contains(&STRUCTURAL) {
            Ok(())
        } else {
            Err(PyValueError::new_err("structural access was not declared"))
        }
    }

    fn stage(&self, command: Command) -> PyResult<u64> {
        self.stage_with(|| self.state.control.stage(command))
    }

    fn stage_with(&self, stage: impl FnOnce() -> Result<Staged, Admission>) -> PyResult<u64> {
        let mut guard = self.live.lock().unwrap();
        let Some(batch) = guard.as_mut() else {
            drop(guard);
            return Err(PyRuntimeError::new_err("command lease expired"));
        };
        let result = stage().map(|staged| {
            let id = staged.id();
            batch.commands.push(staged);
            id
        });
        drop(guard);
        result.map_err(admission_error)
    }

    fn push(&self, bytes: usize, push: impl FnOnce(&mut Batch)) -> PyResult<()> {
        let mut guard = self.live.lock().unwrap();
        let Some(batch) = guard.as_mut() else {
            drop(guard);
            return Err(PyRuntimeError::new_err("command lease expired"));
        };
        if batch.commands.len() + batch.events.len() >= MAX_QUEUE
            || bytes > MAX_COMMAND_BYTES - batch.bytes
        {
            drop(guard);
            return Err(PyRuntimeError::new_err("system command queue is full"));
        }
        push(batch);
        batch.bytes += bytes;
        Ok(())
    }
}

struct Job {
    snapshot: Snapshot,
}

struct Completed {
    batch: Result<Batch, String>,
    ns: u64,
}

fn nanos(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

fn call(callback: &Py<PyAny>, snapshot: Snapshot, access: &Access, state: &CallState) -> Completed {
    let start = Instant::now();
    let batch = Python::attach(|py| {
        let snapshot = Bound::new(
            py,
            FrameSnapshot {
                live: Mutex::new(Some(snapshot)),
                counts: Arc::clone(&state.counts),
            },
        )?;
        let commands = Bound::new(
            py,
            FrameCommands {
                live: Mutex::new(Some(Batch::default())),
                writes: Arc::clone(&access.writes),
                reads: Arc::clone(&access.reads),
                state: state.clone(),
            },
        )?;
        let result = callback.call1(py, (&snapshot, &commands));
        snapshot.get().live.lock().unwrap().take();
        let batch = commands.get().live.lock().unwrap().take().unwrap();
        result.map(|_| batch)
    });
    let batch = batch.map_err(|error| Python::attach(|_| error.to_string()));
    Completed {
        batch,
        ns: nanos(start.elapsed()),
    }
}

#[derive(Clone)]
struct Access {
    reads: Arc<[usize]>,
    writes: Arc<[usize]>,
}

// Compare declared inputs and write targets at the same barrier that will
// publish the batch. A writer that ran while Python was detached wins over a
// result computed from that older version. Structural work conflicts globally.
enum Revision {
    World(u64),
    Columns {
        entities: Option<u64>,
        versions: Vec<(usize, u64)>,
    },
}

impl Revision {
    fn capture(world: &World, access: &Access) -> Result<Self, String> {
        if access.writes.contains(&STRUCTURAL) {
            return Ok(Self::World(world.version()));
        }
        let versions: Vec<_> = access
            .reads
            .iter()
            .chain(
                access
                    .writes
                    .iter()
                    .filter(|column| !access.reads.contains(column)),
            )
            .map(|&column| {
                world
                    .column_version(column)
                    .map(|version| (column, version))
            })
            .collect::<Result<_, _>>()?;
        let entities = (!versions.is_empty()).then(|| world.entity_version());
        Ok(Self::Columns { entities, versions })
    }

    fn matches(&self, world: &World) -> bool {
        match self {
            Self::World(version) => *version == world.version(),
            Self::Columns { entities, versions } => {
                entities.is_none_or(|version| version == world.entity_version())
                    && versions
                        .iter()
                        .all(|&(column, version)| world.column_version(column) == Ok(version))
            }
        }
    }
}

#[derive(Clone)]
struct CallState {
    control: Control,
    counts: Arc<Mutex<ReadCounts>>,
    schemas: Arc<[ComponentSchema]>,
    max_rows: usize,
}

struct Worker {
    sender: Option<SyncSender<Job>>,
    receiver: Receiver<Completed>,
    thread: Option<JoinHandle<()>>,
    pending: Option<Revision>,
}

impl Worker {
    fn new(callback: Py<PyAny>, access: Access, state: CallState) -> std::io::Result<Self> {
        let (sender, jobs) = mpsc::sync_channel::<Job>(1);
        let (results, receiver) = mpsc::sync_channel(1);
        let thread = thread::Builder::new().spawn(move || {
            while let Ok(job) = jobs.recv() {
                let result = call(&callback, job.snapshot, &access, &state);
                if results.send(result).is_err() {
                    break;
                }
            }
            // Drop Python references while attached on this worker, never on the frame lane.
            Python::attach(|py| callback.drop_ref(py));
        })?;
        Ok(Self {
            sender: Some(sender),
            receiver,
            thread: Some(thread),
            pending: None,
        })
    }

    fn take(&mut self) -> Result<Option<(Revision, Completed)>, String> {
        if self.pending.is_none() {
            return Ok(None);
        }
        match self.receiver.try_recv() {
            Ok(completed) => Ok(Some((self.pending.take().unwrap(), completed))),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err("async worker exited".into()),
        }
    }

    fn launch(&mut self, snapshot: Snapshot, revision: Revision) -> Result<(), String> {
        self.sender
            .as_ref()
            .unwrap()
            .try_send(Job { snapshot })
            .map_err(|_| "async worker unavailable".to_string())?;
        self.pending = Some(revision);
        Ok(())
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

struct System {
    name: String,
    access: Access,
    work: Work,
    ns: u64,
    calls: u64,
    over_budget: u64,
    last_ns: u64,
    skipped: u64,
    last_start: Option<u64>,
}

enum Work {
    Sync(Py<PyAny>),
    Async(Cadence, Worker),
}

impl System {
    fn finish(
        &mut self,
        world: &mut World,
        completed: Completed,
        report: &mut Report,
        tick: u64,
        current: bool,
    ) -> Result<(), String> {
        self.ns = self.ns.max(completed.ns);
        self.last_ns = completed.ns;
        self.calls += 1;
        self.over_budget += u64::from(completed.ns > 1_000_000);
        let batch = completed
            .batch
            .map_err(|error| format!("{}: {error}", self.name))?;
        if !current {
            report.stale_system_results += 1;
            for command in batch.commands {
                command.reject(tick, "stale async result: declared World versions changed");
            }
            return Ok(());
        }
        for command in batch.commands {
            report.system_commands += u64::from(command.commit(world, tick));
        }
        for (event, value) in batch.events {
            if report.events.len() == MAX_QUEUE {
                return Err("event queue is full".into());
            }
            report.events.push((tick, event, value));
        }
        Ok(())
    }
}

#[derive(Default)]
struct Report {
    frame_intervals_ns: Vec<u64>,
    events: Vec<(u64, String, String)>,
    system_commands: u64,
    stale_system_results: u64,
    snapshot_copies: u64,
    snapshot_bytes: u64,
    bridge_calls: u64,
    bridge_rows: u64,
    bridge_bytes: u64,
    bridge_ns: u64,
}

type Cache = HashMap<usize, ((u64, u64), Arc<ComponentReadout>)>;

fn capture(
    world: &World,
    clock: &FrameClock,
    reads: &[usize],
    cache: &mut Cache,
    report: &mut Report,
) -> Result<Snapshot, String> {
    let mut columns = Vec::with_capacity(reads.len());
    for &column in reads {
        // Include the entity table version because readouts carry live identities.
        let version = (world.entity_version(), world.column_version(column)?);
        let entry = cache.entry(column);
        if let std::collections::hash_map::Entry::Occupied(entry) = &entry {
            let (current, readout) = entry.get();
            if *current == version {
                columns.push((column, Arc::clone(readout)));
                continue;
            }
        }
        let readout = Arc::new(
            world
                .read_component(column, None)
                .map_err(|e| e.to_string())?,
        );
        report.snapshot_copies += 1;
        report.snapshot_bytes += readout_bytes(&readout);
        cache.insert(column, (version, Arc::clone(&readout)));
        columns.push((column, readout));
    }
    Ok(Snapshot {
        clock: (clock.tick(), clock.time(), clock.dt()),
        overruns: Vec::new(),
        columns,
    })
}

struct Bridge {
    reads: Vec<usize>,
    callback: Py<PyAny>,
}

impl Bridge {
    fn present(
        &self,
        world: &World,
        clock: &FrameClock,
        cache: &mut Cache,
        report: &mut Report,
        counts: &Arc<Mutex<ReadCounts>>,
        systems: &[System],
    ) -> Result<(), String> {
        let mut snapshot = capture(world, clock, &self.reads, cache, report)?;
        snapshot.overruns = systems
            .iter()
            .filter(|s| matches!(s.work, Work::Sync(_)) && s.last_ns > 1_000_000)
            .map(|s| (s.name.clone(), s.last_ns))
            .collect();
        let rows: usize = snapshot
            .columns
            .iter()
            .map(|(_, readout)| readout.ids.len())
            .sum();
        let bytes: u64 = snapshot
            .columns
            .iter()
            .map(|(_, readout)| readout_bytes(readout))
            .sum();
        let start = Instant::now();
        Python::attach(|py| {
            let result = (|| -> PyResult<_> {
                let snapshot = Bound::new(
                    py,
                    FrameSnapshot {
                        live: Mutex::new(Some(snapshot)),
                        counts: Arc::clone(counts),
                    },
                )?;
                let result = self.callback.call1(py, (&snapshot,));
                snapshot.get().live.lock().unwrap().take();
                result
            })();
            result.map(|_| ()).map_err(|error| error.to_string())
        })?;
        report.bridge_calls += 1;
        report.bridge_rows += rows as u64;
        report.bridge_bytes += bytes;
        report.bridge_ns += nanos(start.elapsed());
        Ok(())
    }
}

struct Loop {
    systems: Vec<System>,
    bridge: Option<Bridge>,
    state: CallState,
}

impl Loop {
    fn new(
        registrations: Vec<Registration>,
        bridge: Option<Bridge>,
        state: CallState,
    ) -> std::io::Result<Self> {
        let mut systems = Vec::with_capacity(registrations.len());
        for (name, hz, reads, writes, callback) in registrations {
            let access = Access {
                reads: reads.into(),
                writes: writes.into(),
            };
            let work = match hz {
                Some(hz) => Work::Async(
                    Cadence::new(hz),
                    Worker::new(callback, access.clone(), state.clone())?,
                ),
                None => Work::Sync(callback),
            };
            systems.push(System {
                name,
                access,
                work,
                ns: 0,
                calls: 0,
                over_budget: 0,
                last_ns: 0,
                skipped: 0,
                last_start: None,
            });
        }
        Ok(Self {
            systems,
            bridge,
            state,
        })
    }

    fn execute(
        &mut self,
        world: &mut World,
        control: &Control,
        frames: u64,
        step: Duration,
        paced: bool,
        batches_per_frame: usize,
    ) -> Result<Report, String> {
        let mut report = Report::default();
        report.frame_intervals_ns.reserve(frames as usize);
        let mut previous = None;
        let mut clock = FrameClock::new(step);
        let mut cache = Cache::new();
        let mut due = Vec::with_capacity(self.systems.len());
        for _ in 0..frames {
            let start = Instant::now();
            if !control.frame_boundary(world, clock.tick(), batches_per_frame) {
                break;
            }
            if let Some(previous) = previous.replace(start) {
                report
                    .frame_intervals_ns
                    .push(nanos(start.duration_since(previous)));
            }
            // Settle ready work in registration order. A sync callback sees
            // the commits from all preceding registrations at this barrier.
            for system in &mut self.systems {
                let (ready, current) = match &mut system.work {
                    Work::Async(_, worker) => {
                        let Some((revision, completed)) = worker.take()? else {
                            continue;
                        };
                        (completed, revision.matches(world))
                    }
                    Work::Sync(callback) => {
                        let snap =
                            capture(world, &clock, &system.access.reads, &mut cache, &mut report)?;
                        (call(callback, snap, &system.access, &self.state), true)
                    }
                };
                system.finish(world, ready, &mut report, clock.tick(), current)?;
            }
            // Prefer the least recently dispatched registration when starts
            // conflict. Ties use registration order. No missed start is queued.
            due.clear();
            for (index, system) in self.systems.iter_mut().enumerate() {
                if let Work::Async(cadence, _) = &mut system.work {
                    if cadence.due(clock.time()) {
                        due.push((system.last_start, index));
                    }
                }
            }
            due.sort_unstable();
            for &(_, index) in &due {
                let (before, current) = self.systems.split_at_mut(index);
                let (system, after) = current.split_first_mut().unwrap();
                let Work::Async(_, worker) = &mut system.work else {
                    unreachable!()
                };
                let blocked = before.iter().chain(after.iter()).any(|other| {
                    matches!(&other.work, Work::Async(_, worker) if worker.pending.is_some())
                        && (system.access.writes.contains(&STRUCTURAL)
                            || other.access.writes.contains(&STRUCTURAL)
                            || conflicts(
                                &system.access.reads,
                                &system.access.writes,
                                &other.access.reads,
                                &other.access.writes,
                            ))
                });
                if blocked || worker.pending.is_some() {
                    system.skipped += 1;
                } else {
                    let snapshot =
                        capture(world, &clock, &system.access.reads, &mut cache, &mut report)?;
                    let revision = Revision::capture(world, &system.access)?;
                    worker.launch(snapshot, revision)?;
                    system.last_start = Some(clock.tick());
                }
            }
            if let Some(bridge) = &self.bridge {
                bridge.present(
                    world,
                    &clock,
                    &mut cache,
                    &mut report,
                    &self.state.counts,
                    &self.systems,
                )?;
            }
            control.record_frame();
            let deadline = clock.advance();
            if paced {
                control.wait_until(deadline);
            }
        }
        Ok(report)
    }

    fn join(&mut self) {
        // Close admission before joining. Uncommitted worker results are discarded.
        for system in &mut self.systems {
            if let Work::Async(_, worker) = &mut system.work {
                worker.sender.take();
            }
        }
        for system in &mut self.systems {
            if let Work::Async(_, worker) = &mut system.work {
                if let Some(thread) = worker.thread.take() {
                    let _ = thread.join();
                }
                drop(worker.receiver.try_recv());
            }
        }
    }
}

#[pyclass(frozen)]
struct FrameLoop {
    owner: Arc<Mutex<Option<World>>>,
    control: Control,
    rows: Arc<Mutex<ReadCounts>>,
    batches_per_frame: usize,
    max_rows: usize,
    owner_thread: thread::ThreadId,
}

#[pymethods]
impl FrameLoop {
    #[new]
    fn new(
        world: &PyNativeWorld,
        capacity: usize,
        batches_per_frame: usize,
        max_rows: usize,
    ) -> PyResult<Self> {
        if !(1..=MAX_QUEUE).contains(&capacity)
            || !(1..=capacity).contains(&batches_per_frame)
            || max_rows == 0
        {
            return Err(PyValueError::new_err("invalid frame queue limits"));
        }
        let control = {
            let world = world.lock()?;
            Control::new(&world, capacity, max_rows)
        };
        Ok(Self {
            owner: Arc::clone(&world.inner),
            control,
            batches_per_frame,
            max_rows,
            rows: Arc::new(Mutex::new(ReadCounts::default())),
            owner_thread: thread::current().id(),
        })
    }

    fn set(
        &self,
        root: usize,
        predicates: Vec<Predicate>,
        component: usize,
        field: usize,
        value: (Option<f32>, Option<u32>),
    ) -> PyResult<u64> {
        self.admit(set_command(root, predicates, component, field, value)?)
    }

    fn destroy(&self, py: Python<'_>, handles: &Bound<'_, PyAny>) -> PyResult<u64> {
        let credit = self.control.reserve().map_err(admission_error)?;
        let handles = handles.call0()?;
        let ids = decode_handles_limited(py, &handles, Some((self.max_rows, 16)))?;
        credit
            .submit(Command::Destroy(ids))
            .map_err(admission_error)
    }

    fn spawn(&self, py: Python<'_>, count: u32, payload: &Bound<'_, PyAny>) -> PyResult<u64> {
        let credit = self.control.reserve().map_err(admission_error)?;
        let schemas = {
            let guard = self.owner.lock().unwrap();
            guard
                .as_ref()
                .map(|world| world.schemas().cloned().collect::<Vec<_>>())
        };
        let schemas = schemas.ok_or_else(|| {
            PyRuntimeError::new_err("use a structural command lease during a frame")
        })?;
        let payload = payload.call0()?;
        let command = spawn_command(
            py,
            &schemas,
            self.max_rows,
            count,
            payload.cast::<PyDict>()?,
        )?;
        credit.submit(command).map_err(admission_error)
    }

    fn stop(&self) {
        self.control.stop();
    }

    fn outcome(&self) -> Option<(u64, u64, Option<String>)> {
        let outcome = self.control.take()?;
        Some((outcome.sequence, outcome.frame, outcome.result.err()))
    }

    #[getter]
    fn stopped(&self) -> bool {
        self.control.phase() == manifoldx_core::runtime::Phase::Stopped
    }

    #[getter]
    fn full(&self) -> u64 {
        self.control.stats().full
    }

    #[getter]
    fn reads(&self) -> (u64, u64, u64) {
        let counts = self.rows.lock().unwrap();
        (counts.rows, counts.bytes, counts.calls)
    }

    fn read<'py>(
        &self,
        py: Python<'py>,
        root: usize,
        predicates: Vec<Predicate>,
        column: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let selection = selection(root, predicates)?;
        let result = {
            let guard = self.owner.lock().unwrap();
            guard.as_ref().map(|world| {
                let ids = world.select(&selection)?;
                world
                    .read_component(column, Some(&ids))
                    .map_err(|e| e.to_string())
            })
        };
        let readout = result
            .ok_or_else(|| {
                PyRuntimeError::new_err("readback requires an idle World or a snapshot lease")
            })?
            .map_err(PyValueError::new_err)?;
        let result = readout_dict(py, &readout)?;
        count_read(&self.rows, &readout);
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn run<'py>(
        &self,
        py: Python<'py>,
        frames: u64,
        hz: f64,
        paced: bool,
        registrations: Vec<Registration>,
        bridge: Option<(Vec<usize>, Py<PyAny>)>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if thread::current().id() != self.owner_thread {
            return Err(PyRuntimeError::new_err("run on the World owner thread"));
        }
        if !(1..=MAX_FRAMES).contains(&frames) || !hz.is_finite() || !(1.0..=1000.0).contains(&hz) {
            return Err(PyValueError::new_err("invalid frame count or rate"));
        }
        if registrations.len() > MAX_SYSTEMS {
            return Err(PyValueError::new_err("too many systems"));
        }
        let mut guard = self.owner.lock().unwrap();
        let world = guard.as_ref();
        let preflight = world.ok_or("World already running").and_then(|world| {
            for (_, hz, reads, writes, _) in &registrations {
                if hz.is_some_and(|hz| !hz.is_finite() || !(1.0..=30.0).contains(&hz)) {
                    return Err("async rate must be between 1 and 30 Hz");
                }
                validate_columns(world, reads, writes)?;
            }
            if let Some((reads, _)) = &bridge {
                validate_columns(world, reads, &[])?;
            }
            self.control.start()
        });
        if let Err(error) = preflight {
            drop(guard);
            return Err(PyValueError::new_err(error));
        }
        let mut world = guard.take().unwrap();
        drop(guard);

        let step = Duration::from_secs_f64(1.0 / hz);
        let bridge = bridge.map(|(reads, callback)| Bridge { reads, callback });
        let state = CallState {
            control: self.control.clone(),
            counts: Arc::clone(&self.rows),
            schemas: world.schemas().cloned().collect(),
            max_rows: self.max_rows,
        };
        let execution = py.detach(|| {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut run = Loop::new(registrations, bridge, state).map_err(|e| e.to_string())?;
                let report = run.execute(
                    &mut world,
                    &self.control,
                    frames,
                    step,
                    paced,
                    self.batches_per_frame,
                );
                self.control.complete();
                run.join();
                Ok::<_, String>((report?, run))
            }));
            self.control.complete();
            result.map_err(|_| "native frame loop panicked".to_string())?
        });
        *self.owner.lock().unwrap() = Some(world);
        let (report, run) = execution.map_err(PyRuntimeError::new_err)?;
        let result = PyDict::new(py);
        result.set_item("frames", self.control.stats().frames)?;
        result.set_item("frame_intervals_ns", report.frame_intervals_ns)?;
        result.set_item("events", report.events)?;
        result.set_item("system_commands", report.system_commands)?;
        result.set_item("stale_system_results", report.stale_system_results)?;
        result.set_item("snapshot_copies", report.snapshot_copies)?;
        result.set_item("snapshot_bytes", report.snapshot_bytes)?;
        result.set_item(
            "bridge",
            (
                report.bridge_calls,
                report.bridge_rows,
                report.bridge_bytes,
                report.bridge_ns,
            ),
        )?;
        let mut systems = Vec::new();
        for system in run.systems {
            systems.push((
                system.name,
                matches!(system.work, Work::Async(..)),
                system.calls,
                system.skipped,
                system.ns,
                system.over_budget,
            ));
        }
        result.set_item("systems", systems)?;
        Ok(result)
    }
}

impl FrameLoop {
    fn admit(&self, command: Command) -> PyResult<u64> {
        match self.control.submit(command) {
            Admission::Admitted(id) => Ok(id),
            other => Err(admission_error(other)),
        }
    }
}

fn admission_error(admission: Admission) -> PyErr {
    match admission {
        Admission::Full => PyRuntimeError::new_err("frame command queue is full"),
        Admission::Stopped => PyRuntimeError::new_err("frame command queue is stopped"),
        Admission::Error(error) => PyValueError::new_err(error),
        Admission::Admitted(_) => unreachable!(),
    }
}

fn validate_columns(world: &World, reads: &[usize], writes: &[usize]) -> Result<(), &'static str> {
    let mut bytes = 0;
    for &index in reads {
        let schema = world.schema(index).ok_or("unknown read column")?;
        bytes += world.capacity() as usize * (17 + schema.width as usize * 4);
        if bytes > MAX_SNAPSHOT_BYTES {
            return Err("declared snapshot exceeds byte limit");
        }
    }
    for &index in writes {
        if index != STRUCTURAL {
            world.schema(index).ok_or("unknown write column")?;
        }
    }
    Ok(())
}

fn spawn_command(
    py: Python<'_>,
    schemas: &[ComponentSchema],
    max_rows: usize,
    count: u32,
    args: &Bound<'_, PyDict>,
) -> PyResult<Command> {
    if count as usize > max_rows || args.len() > schemas.len() {
        return Err(PyValueError::new_err(
            "spawn command exceeds row or column limit",
        ));
    }
    let mut bytes = u64::from(count) * 16;
    if bytes > MAX_COMMAND_BYTES as u64 {
        return Err(PyValueError::new_err("spawn command exceeds byte limit"));
    }
    let mut payload = Vec::with_capacity(args.len());
    for (component, value) in args {
        let component = component.extract::<usize>()?;
        let schema = schemas
            .get(component)
            .ok_or_else(|| PyValueError::new_err("unknown spawn component"))?;
        let width = schema.width as usize;
        bytes += 8 + u64::from(count) * u64::from(schema.width) * 4;
        if bytes > MAX_COMMAND_BYTES as u64 {
            return Err(PyValueError::new_err("spawn command exceeds byte limit"));
        }
        let rows = count as usize;
        let data = match schema.element {
            ElementType::F32 if width == 0 && value.is_none() => ColumnData::F32(Vec::new()),
            ElementType::U32 if width == 0 && value.is_none() => ColumnData::U32(Vec::new()),
            ElementType::F32 => ColumnData::F32(read_matrix::<f32>(
                py,
                &value,
                rows,
                width,
                schema.element,
                "spawn rows",
            )?),
            ElementType::U32 => ColumnData::U32(read_matrix::<u32>(
                py,
                &value,
                rows,
                width,
                schema.element,
                "spawn rows",
            )?),
        };
        payload.push(ComponentPayload { component, data });
    }
    Ok(Command::Spawn { count, payload })
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("_STRUCTURAL_ACCESS", STRUCTURAL)?;
    module.add_class::<FrameLoop>()?;
    module.add_class::<FrameSnapshot>()?;
    module.add_class::<FrameCommands>()?;
    Ok(())
}
