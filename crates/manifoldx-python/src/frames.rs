//! Fused frame ownership. The GIL is acquired only by explicit participants.
use super::*;
use manifoldx_core::clock::{conflicts, Cadence, FrameClock};
use manifoldx_core::ecs::{Compare, ComponentReadout, ComponentSchema, Scalar, Selection};
use manifoldx_core::kernel::Program;
use manifoldx_core::runtime::{apply, Admission, Command, Control, Staged, MAX_COMMAND_BYTES};
use manifoldx_render::{
    desktop::{self, DesktopApplication, DesktopInfo, DesktopOptions, PresentedFrame},
    resident::{PositionSpec, ResidentKernel},
    wgpu,
    winit::{
        dpi::PhysicalSize,
        event::{MouseButton, MouseScrollDelta, WindowEvent},
        keyboard::{Key, ModifiersState, NamedKey},
        window::Window,
    },
    FrameTarget, NativeRenderer, RenderError, RenderStats, RendererOptions,
};
use pyo3::types::PyTuple;
use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};
use std::sync::OnceLock;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

const MAX_SYSTEMS: usize = 32;
const MAX_QUEUE: usize = 64;
const MAX_PREDICATES: usize = 16;
const STRUCTURAL: usize = usize::MAX;
const MAX_SNAPSHOT_BYTES: usize = 16 * 1024 * 1024;
const MAX_FRAMES: u64 = 1_000_000;
// Leave a short tail for the clock instead of letting a timed wait overshoot
// the frame deadline. The stop request can still wake the blocking portion.
const DEADLINE_GUARD: Duration = Duration::from_micros(1200);

type Predicate = (usize, usize, String, (Option<f32>, Option<u32>));
type Registration = (String, Option<f64>, Vec<usize>, Vec<usize>, Py<PyAny>);
type Prepared = (String, Option<f64>, Vec<usize>, Vec<usize>, Participant);

enum Participant {
    Callback(Py<PyAny>),
    Kernel(usize, Arc<Program>),
}

fn prepare(py: Python<'_>, registration: Registration) -> PyResult<Prepared> {
    let (name, hz, reads, writes, callback) = registration;
    let participant = if let Ok(spec) = callback.bind(py).cast::<PyTuple>() {
        let (column, program): (usize, PyRef<'_, kernel::KernelProgram>) = spec.extract()?;
        if hz.is_some() {
            return Err(PyValueError::new_err("kernel must run on the frame lane"));
        }
        Participant::Kernel(column, Arc::clone(&program.0))
    } else {
        Participant::Callback(callback)
    };
    Ok((name, hz, reads, writes, participant))
}

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

    fn replace(
        &self,
        py: Python<'_>,
        component: usize,
        handles: &Bound<'_, PyAny>,
        rows: &Bound<'_, PyAny>,
    ) -> PyResult<u64> {
        if self.live.lock().unwrap().is_none() {
            return Err(PyRuntimeError::new_err("command lease expired"));
        }
        if !self.writes.contains(&component) {
            return Err(PyValueError::new_err("column was not declared for writing"));
        }
        let schema = self
            .state
            .schemas
            .get(component)
            .ok_or_else(|| PyValueError::new_err("unknown replacement component"))?;
        let credit = self.state.control.reserve().map_err(admission_error)?;
        let width = schema.width as usize;
        let handles = handles.call0()?;
        let ids =
            decode_handles_limited(py, &handles, Some((self.state.max_rows, 16 + width * 4)))?;
        let data = match schema.element {
            ElementType::F32 => ColumnData::F32(read_matrix::<f32>(
                py,
                rows,
                ids.len(),
                width,
                schema.element,
                "replacement rows",
            )?),
            ElementType::U32 => ColumnData::U32(read_matrix::<u32>(
                py,
                rows,
                ids.len(),
                width,
                schema.element,
                "replacement rows",
            )?),
        };
        self.stage_with(|| {
            credit.stage(Command::Update {
                component,
                ids,
                data,
            })
        })
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
    ran: bool,
}

/// One presented frame's live samples for `manifoldx run --debug`.
///
/// These are the same measurements the desktop loop takes to draw and present
/// every frame, collected only when the dev channel asks for them. GPU timing
/// is absent because the timestamp queries stay reserved for bounded
/// diagnostic runs; no counter here is estimated.
#[derive(Clone, Default)]
struct FrameMetrics {
    interval_ns: u64,
    acquire_ns: u64,
    submit_ns: u64,
    present_ns: u64,
    update_ns: u64,
    render_ns: u64,
    systems: Vec<(String, u64)>,
    over_budget: Vec<(String, u64)>,
}

enum Work {
    Kernel(usize, Arc<Program>),
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
        apply: &mut impl FnMut(&mut World, Command) -> Result<Option<ComponentReadout>, String>,
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
            report.system_commands +=
                u64::from(command.commit_with(tick, |command| apply(world, command)));
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

// Start lateness, command barrier, kernel work, active frame work, return
// from the pacing wait relative to the deadline, and the slowest kernel.
type FrameTiming = (u64, u64, u64, u64, Option<i64>, Option<(usize, u64)>);

#[derive(Default)]
struct Report {
    frame_timing_ns: Vec<FrameTiming>,
    frame_intervals_ns: Vec<u64>,
    events: Vec<(u64, String, String)>,
    system_commands: u64,
    stale_system_results: u64,
    snapshot_copies: u64,
    kernel_dispatches: u64,
    kernel_rows: u64,
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
    profile: bool,
    state: CallState,
}

impl Loop {
    fn new(
        registrations: Vec<Prepared>,
        bridge: Option<Bridge>,
        state: CallState,
        profile: bool,
    ) -> std::io::Result<Self> {
        let mut systems = Vec::with_capacity(registrations.len());
        for (name, hz, reads, writes, participant) in registrations {
            let access = Access {
                reads: reads.into(),
                writes: writes.into(),
            };
            let work = match (hz, participant) {
                (_, Participant::Kernel(column, program)) => Work::Kernel(column, program),
                (Some(hz), Participant::Callback(callback)) => Work::Async(
                    Cadence::new(hz),
                    Worker::new(callback, access.clone(), state.clone())?,
                ),
                (None, Participant::Callback(callback)) => Work::Sync(callback),
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
                ran: false,
            });
        }
        Ok(Self {
            systems,
            bridge,
            state,
            profile,
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
        let profile = self.profile;
        let mut report = Report::default();
        report.frame_intervals_ns.reserve(frames as usize);
        if profile {
            report.frame_timing_ns.reserve(frames as usize);
        }
        let mut previous = None;
        let mut scheduled = None;
        let mut clock = FrameClock::new(step);
        let mut cache = Cache::new();
        let mut due = Vec::with_capacity(self.systems.len());
        for _ in 0..frames {
            let start = Instant::now();
            if !control.frame_boundary(world, clock.tick(), batches_per_frame) {
                break;
            }
            let barrier = if profile { nanos(start.elapsed()) } else { 0 };
            if let Some(previous) = previous.replace(start) {
                report
                    .frame_intervals_ns
                    .push(nanos(start.duration_since(previous)));
            }
            let (kernels, slowest) = self.systems_at_barrier(
                world,
                &clock,
                &mut cache,
                &mut due,
                &mut report,
                &mut apply,
            )?;
            control.record_frame();
            let deadline = clock.advance();
            let active = if profile { nanos(start.elapsed()) } else { 0 };
            let mut wake_offset = None;
            if paced {
                control.wait_until(deadline - DEADLINE_GUARD.min(step / 10));
                if profile {
                    let wake = Instant::now();
                    wake_offset = Some(if wake >= deadline {
                        nanos(wake.duration_since(deadline)) as i64
                    } else {
                        -(nanos(deadline.duration_since(wake)) as i64)
                    });
                }
                while Instant::now() < deadline {
                    std::hint::spin_loop();
                }
            }
            if profile {
                let late = scheduled.map_or(0, |deadline| {
                    nanos(start.saturating_duration_since(deadline))
                });
                report
                    .frame_timing_ns
                    .push((late, barrier, kernels, active, wake_offset, slowest));
                scheduled = paced.then_some(deadline);
            }
        }
        Ok(report)
    }

    fn systems_at_barrier(
        &mut self,
        world: &mut World,
        clock: &FrameClock,
        cache: &mut Cache,
        due: &mut Vec<(Option<u64>, usize)>,
        report: &mut Report,
        apply: &mut impl FnMut(&mut World, Command) -> Result<Option<ComponentReadout>, String>,
    ) -> Result<(u64, Option<(usize, u64)>), String> {
        let profile = self.profile;
        let mut kernels = 0;
        let mut slowest = None;
        for system in &mut self.systems {
            system.ran = false;
        }
        // Settle ready work in registration order. A sync callback sees
        // the commits from all preceding registrations at this barrier.
        for (index, system) in self.systems.iter_mut().enumerate() {
            let (ready, current) = match &mut system.work {
                Work::Kernel(column, program) => {
                    let start = Instant::now();
                    let rows = program
                        .dispatch(world, *column, clock.dt() as f32)
                        .map_err(|err| format!("Rust CPU kernel {}: {err}", program.name()))?;
                    report.kernel_dispatches += 1;
                    report.kernel_rows += rows as u64;
                    system.calls += 1;
                    system.last_ns = nanos(start.elapsed());
                    system.ns = system.ns.max(system.last_ns);
                    system.ran = true;
                    if profile {
                        kernels += system.last_ns;
                        if slowest.is_none_or(|(_, ns)| system.last_ns > ns) {
                            slowest = Some((index, system.last_ns));
                        }
                    }
                    continue;
                }
                Work::Async(_, worker) => {
                    let Some((revision, completed)) = worker.take()? else {
                        continue;
                    };
                    system.ran = true;
                    (completed, revision.matches(world))
                }
                Work::Sync(callback) => {
                    let snap = capture(world, clock, &system.access.reads, cache, report)?;
                    system.ran = true;
                    (call(callback, snap, &system.access, &self.state), true)
                }
            };
            system.finish(world, ready, report, clock.tick(), current, apply)?;
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
        for &(_, index) in due.iter() {
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
                let snapshot = capture(world, clock, &system.access.reads, cache, report)?;
                let revision = Revision::capture(world, &system.access)?;
                worker.launch(snapshot, revision)?;
                system.last_start = Some(clock.tick());
            }
        }
        if let Some(bridge) = &self.bridge {
            bridge.present(
                world,
                clock,
                cache,
                report,
                &self.state.counts,
                &self.systems,
            )?;
        }
        Ok((kernels, slowest))
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
    schemas: OnceLock<Arc<[ComponentSchema]>>,
    control: Control,
    rows: Arc<Mutex<ReadCounts>>,
    metrics: Arc<Mutex<Option<FrameMetrics>>>,
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
            schemas: OnceLock::new(),
            control,
            batches_per_frame,
            max_rows,
            rows: Arc::new(Mutex::new(ReadCounts::default())),
            metrics: Arc::new(Mutex::new(None)),
            owner_thread: thread::current().id(),
        })
    }

    /// The last presented frame's measured samples, or None before one has
    /// completed. Only `run_desktop(debug=true)` fills it.
    fn last_metrics<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let guard = self.metrics.lock().unwrap();
        let Some(metrics) = guard.as_ref() else {
            return Ok(None);
        };
        let out = PyDict::new(py);
        out.set_item("interval_ns", metrics.interval_ns)?;
        out.set_item("acquire_ns", metrics.acquire_ns)?;
        out.set_item("submit_ns", metrics.submit_ns)?;
        out.set_item("present_ns", metrics.present_ns)?;
        out.set_item("update_ns", metrics.update_ns)?;
        out.set_item("render_ns", metrics.render_ns)?;
        out.set_item("systems", metrics.systems.clone())?;
        out.set_item("over_budget", metrics.over_budget.clone())?;
        Ok(Some(out))
    }

    fn dispatch_kernel(
        &self,
        py: Python<'_>,
        column: usize,
        program: &kernel::KernelProgram,
        args: (u32, f32),
    ) -> PyResult<usize> {
        if thread::current().id() != self.owner_thread {
            return Err(PyRuntimeError::new_err(
                "dispatch on the World owner thread",
            ));
        }
        let (version, dt) = args;
        if version != manifoldx_core::kernel::VERSION {
            return Err(PyValueError::new_err("unsupported kernel argument version"));
        }
        py.detach(|| {
            let mut guard = self.owner.lock().unwrap();
            let world = guard.as_mut().ok_or("World already running")?;
            program.0.dispatch(world, column, dt)
        })
        .map_err(|error| PyValueError::new_err(format!("Rust CPU kernel: {error}")))
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
            self.schemas.get().cloned().or_else(|| {
                guard
                    .as_ref()
                    .map(|world| world.schemas().cloned().collect())
            })
        };
        let schemas = schemas.ok_or_else(|| PyRuntimeError::new_err("World is unavailable"))?;
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

    #[pyo3(signature = (graphics, options, registrations, startup, keyboard, position, diagnostics, application=None, debug=false))]
    #[allow(clippy::too_many_arguments)]
    fn run_desktop<'py>(
        &self,
        py: Python<'py>,
        graphics: &graphics::NativeGraphics,
        options: DesktopInput,
        registrations: Vec<Registration>,
        startup: Py<PyAny>,
        keyboard: Py<PyAny>,
        position: Option<PositionInput>,
        diagnostics: DesktopDiagnostics,
        application: Option<ApplicationInput>,
        debug: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        if thread::current().id() != self.owner_thread {
            return Err(PyRuntimeError::new_err("run on the World owner thread"));
        }
        let (title, width, height, fullscreen, monitor, mode, hz, frame_limit) = options;
        if width == 0
            || height == 0
            || !hz.is_finite()
            || !(1.0..=1000.0).contains(&hz)
            || frame_limit.is_some_and(|n| !(1..=100_000).contains(&n))
            || registrations.len() > MAX_SYSTEMS
        {
            return Err(PyValueError::new_err(
                "invalid desktop dimensions, frame limits or systems",
            ));
        }
        let (skip_dispatch, suppress_present, diagnostic_readback, resize_after_three) =
            diagnostics;
        if (skip_dispatch
            || suppress_present
            || diagnostic_readback
            || resize_after_three.is_some())
            && frame_limit.is_none()
        {
            return Err(PyValueError::new_err(
                "desktop diagnostics require a frame limit",
            ));
        }
        let position =
            position.map(
                |(column, width, offset, kernel_version, writes, wgsl)| PositionSpec {
                    column,
                    width,
                    offset,
                    kernel_version,
                    writes,
                    wgsl,
                },
            );
        if (skip_dispatch || diagnostic_readback) && position.is_none() {
            return Err(PyValueError::new_err(
                "resident diagnostics require a position binding",
            ));
        }
        let present_mode = match mode.as_str() {
            "fifo" => wgpu::PresentMode::Fifo,
            "mailbox" => wgpu::PresentMode::Mailbox,
            "immediate" => wgpu::PresentMode::Immediate,
            _ => {
                return Err(PyValueError::new_err(
                    "present must be fifo, mailbox or immediate",
                ))
            }
        };
        let registrations = registrations
            .into_iter()
            .map(|item| prepare(py, item))
            .collect::<PyResult<Vec<_>>>()?;
        *self.metrics.lock().unwrap() = None;
        let mut frames = WindowFrames {
            owner: self,
            registrations: Some(registrations),
            startup,
            keyboard,
            application,
            events: Vec::new(),
            cursor: None,
            modifiers: ModifiersState::empty(),
            application_systems: 0,
            position,
            resident: None,
            skip_dispatch,
            world: None,
            runner: None,
            scene: Arc::clone(&graphics.scene),
            clock: FrameClock::new(Duration::from_secs_f64(1.0 / hz)),
            report: Report::default(),
            cache: Cache::new(),
            due: Vec::new(),
            display: None,
            frames: Vec::new(),
            frame_limit,
            debug,
            metrics: Arc::clone(&self.metrics),
            pending_metrics: None,
        };
        let options = DesktopOptions {
            title,
            physical_px: PhysicalSize::new(width, height),
            fullscreen,
            monitor,
            present_mode,
            renderer: RendererOptions::default(),
            sample_limit: frame_limit,
            suppress_present,
            resize_after_three: resize_after_three
                .map(|(width, height)| PhysicalSize::new(width, height)),
            presented: Some(graphics.presented()),
        };
        let (presented, gpu_ns, diagnostic) = py.detach(|| {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                desktop::run_desktop(options, &mut frames)
            }));
            frames.finish();
            let (presented, gpu_ns) = result
                .map_err(|_| PyRuntimeError::new_err("native desktop frame loop panicked"))?
                .map_err(graphics::render_error)?;
            let diagnostic = if diagnostic_readback {
                frames
                    .resident
                    .as_ref()
                    .map(ResidentKernel::diagnostic_readback)
                    .transpose()
                    .map_err(graphics::render_error)?
            } else {
                None
            };
            Ok::<_, PyErr>((presented, gpu_ns, diagnostic))
        })?;
        let result = PyDict::new(py);
        result.set_item("presented", presented)?;
        result.set_item("scheduler_frames", self.control.stats().frames)?;
        result.set_item("bridge_calls", frames.report.bridge_calls)?;
        result.set_item("system_commands", frames.report.system_commands)?;
        result.set_item("snapshot_bytes", frames.report.snapshot_bytes)?;
        if let Some((info, adapter)) = frames.display {
            let monitors: Vec<_> = info
                .monitors
                .iter()
                .map(|m| {
                    (
                        &m.name,
                        m.physical_px.width,
                        m.physical_px.height,
                        m.scale_factor,
                        m.refresh_millihertz,
                        m.primary,
                    )
                })
                .collect();
            result.set_item("monitors", monitors)?;
            result.set_item("monitor", info.selected_monitor)?;
            result.set_item(
                "initial_px",
                (info.physical_px.width, info.physical_px.height),
            )?;
            result.set_item("scale", info.scale_factor)?;
            result.set_item("present_mode", format!("{:?}", info.present_mode))?;
            result.set_item("format", format!("{:?}", info.format))?;
            result.set_item("adapter", adapter)?;
        }
        let samples: Vec<_> = frames
            .frames
            .iter()
            .map(|frame| {
                let sample = graphics::stats_dict(py, &frame.render)?;
                sample.set_item("sequence", frame.sequence)?;
                sample.set_item(
                    "physical_px",
                    (frame.physical_px.width, frame.physical_px.height),
                )?;
                sample.set_item("scale", frame.scale_factor)?;
                sample.set_item("wall_interval_ns", frame.wall_interval.map(nanos))?;
                sample.set_item("acquire_ns", nanos(frame.acquire))?;
                sample.set_item("submit_ns", nanos(frame.submit))?;
                sample.set_item("present_ns", nanos(frame.present))?;
                sample.set_item("gpu_render_ns", gpu_ns.get(frame.sequence as usize - 1))?;
                Ok::<_, PyErr>(sample)
            })
            .collect::<PyResult<_>>()?;
        result.set_item("samples", samples)?;
        if let Some(resident) = &frames.resident {
            let stats = &resident.stats;
            let gpu = PyDict::new(py);
            gpu.set_item("dispatches", stats.dispatches)?;
            gpu.set_item("version", stats.version)?;
            gpu.set_item("cpu_version", stats.cpu_version)?;
            gpu.set_item("row_uploads", stats.row_uploads)?;
            gpu.set_item("row_upload_bytes", stats.row_upload_bytes)?;
            gpu.set_item("field_uploads", stats.field_uploads)?;
            gpu.set_item("field_upload_bytes", stats.field_upload_bytes)?;
            gpu.set_item("mask_uploads", stats.mask_uploads)?;
            gpu.set_item("mask_upload_bytes", stats.mask_upload_bytes)?;
            gpu.set_item("uniform_uploads", stats.uniform_uploads)?;
            gpu.set_item("uniform_upload_bytes", stats.uniform_uploads * 16)?;
            gpu.set_item("generations_seeded", stats.generations_seeded)?;
            gpu.set_item("generations_retired", stats.generations_retired)?;
            if let Some((bytes, rows)) = diagnostic {
                gpu.set_item("diagnostic_readbacks", u64::from(bytes != 0))?;
                gpu.set_item("diagnostic_bytes", bytes)?;
                gpu.set_item("diagnostic_rows", rows)?;
            } else {
                gpu.set_item("diagnostic_readbacks", 0)?;
                gpu.set_item("diagnostic_bytes", 0)?;
            }
            result.set_item("resident", gpu)?;
        }
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (frames, hz, paced, registrations, bridge, profile=false, freeze_clock=false))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        frames: u64,
        hz: f64,
        paced: bool,
        registrations: Vec<Registration>,
        bridge: Option<(Vec<usize>, Py<PyAny>)>,
        profile: bool,
        freeze_clock: bool,
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
        if freeze_clock && (paced || frames != 1) {
            return Err(PyValueError::new_err(
                "a frozen clock requires one unpaced frame",
            ));
        }
        let registrations = registrations
            .into_iter()
            .map(|registration| prepare(py, registration))
            .collect::<PyResult<Vec<_>>>()?;
        let (mut world, state) = self
            .begin(
                &registrations,
                bridge.as_ref().map(|(reads, _)| reads.as_slice()),
            )
            .map_err(PyValueError::new_err)?;
        let step = if freeze_clock {
            Duration::ZERO
        } else {
            Duration::from_secs_f64(1.0 / hz)
        };
        let bridge = bridge.map(|(reads, callback)| Bridge { reads, callback });
        let execution = py.detach(|| {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut run =
                    Loop::new(registrations, bridge, state, profile).map_err(|e| e.to_string())?;
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
        if profile {
            result.set_item("frame_timing_ns", report.frame_timing_ns)?;
        }
        result.set_item("events", report.events)?;
        result.set_item("system_commands", report.system_commands)?;
        result.set_item("stale_system_results", report.stale_system_results)?;
        result.set_item("snapshot_copies", report.snapshot_copies)?;
        result.set_item("kernel_backend", "rust-cpu")?;
        result.set_item("kernel_dispatches", report.kernel_dispatches)?;
        result.set_item("kernel_rows", report.kernel_rows)?;
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
    fn begin(
        &self,
        registrations: &[Prepared],
        bridge_reads: Option<&[usize]>,
    ) -> Result<(World, CallState), String> {
        let mut guard = self.owner.lock().unwrap();
        let world = guard.as_ref();
        let preflight = world.ok_or("World already running").and_then(|world| {
            for (_, hz, reads, writes, participant) in registrations {
                if let Participant::Kernel(column, program) = participant {
                    program.validate_column(world, *column)?;
                }
                if hz.is_some_and(|hz| !hz.is_finite() || !(1.0..=30.0).contains(&hz)) {
                    return Err("async rate must be between 1 and 30 Hz");
                }
                validate_columns(world, reads, writes)?;
            }
            if let Some(reads) = bridge_reads {
                validate_columns(world, reads, &[])?;
            }
            self.control.start()
        });
        preflight.map_err(str::to_owned)?;
        // Publish the startup schemas before moving the World out of the owner slot.
        let schemas = self
            .schemas
            .get_or_init(|| guard.as_ref().unwrap().schemas().cloned().collect())
            .clone();
        let world = guard.take().unwrap();
        drop(guard);

        let state = CallState {
            control: self.control.clone(),
            counts: Arc::clone(&self.rows),
            schemas,
            max_rows: self.max_rows,
        };
        Ok((world, state))
    }

    fn admit(&self, command: Command) -> PyResult<u64> {
        match self.control.submit(command) {
            Admission::Admitted(id) => Ok(id),
            other => Err(admission_error(other)),
        }
    }
}

type DesktopInput = (
    String,
    u32,
    u32,
    bool,
    Option<String>,
    String,
    f64,
    Option<u64>,
);

type PositionInput = (usize, u32, u32, u32, Vec<(u32, u32)>, String);
type DesktopDiagnostics = (bool, bool, bool, Option<(u32, u32)>);
type ApplicationInput = (Py<PyAny>, Py<PyAny>);
type WindowInput = (&'static str, String, [f64; 4], u16, Vec<&'static str>);

struct WindowFrames<'a> {
    owner: &'a FrameLoop,
    registrations: Option<Vec<Prepared>>,
    startup: Py<PyAny>,
    keyboard: Py<PyAny>,
    application: Option<ApplicationInput>,
    events: Vec<WindowInput>,
    cursor: Option<[f64; 2]>,
    modifiers: ModifiersState,
    application_systems: usize,
    position: Option<PositionSpec>,
    resident: Option<ResidentKernel>,
    skip_dispatch: bool,
    world: Option<World>,
    runner: Option<Loop>,
    scene: Arc<Mutex<graphics::Scene>>,
    clock: FrameClock,
    report: Report,
    cache: Cache,
    due: Vec<(Option<u64>, usize)>,
    display: Option<(DesktopInfo, String)>,
    frames: Vec<PresentedFrame>,
    frame_limit: Option<u64>,
    debug: bool,
    metrics: Arc<Mutex<Option<FrameMetrics>>>,
    pending_metrics: Option<FrameMetrics>,
}

impl WindowFrames<'_> {
    fn application_frame(&mut self, size: [u32; 2], dpi: f64) -> Result<(), RenderError> {
        let Some((begin, _)) = &self.application else {
            return Ok(());
        };
        // Lifecycle callbacks run between barriers, while the owner has exclusive
        // access to the World. Systems run after it returns to the Rust scheduler.
        let owner_slot = self.resident.is_none();
        if owner_slot {
            *self.owner.owner.lock().unwrap() = self.world.take();
        }
        let events = std::mem::take(&mut self.events);
        let updates = Python::attach(|py| {
            let result = begin.call1(py, (size, dpi, self.clock.time(), events, owner_slot))?;
            result
                .extract::<Option<Vec<Registration>>>(py)
                .and_then(|rows| {
                    rows.map(|rows| rows.into_iter().map(|row| prepare(py, row)).collect())
                        .transpose()
                })
        });
        if owner_slot {
            self.world = self.owner.owner.lock().unwrap().take();
        }
        let updates: Option<Vec<Prepared>> =
            updates.map_err(|error| RenderError::Window(error.to_string()))?;
        if let Some(updates) = updates {
            let world = self
                .world
                .as_ref()
                .expect("desktop owns the World after callbacks");
            let runner = self
                .runner
                .as_mut()
                .expect("desktop application has a scheduler");
            let original = runner.systems.len() - self.application_systems;
            if original + updates.len() > MAX_SYSTEMS {
                return Err(RenderError::Window("too many desktop systems".into()));
            }
            for (_, hz, reads, writes, participant) in &updates {
                validate_columns(world, reads, writes)
                    .map_err(|error| RenderError::Window(error.into()))?;
                if hz.is_some() || !matches!(participant, Participant::Callback(_)) {
                    return Err(RenderError::Window(
                        "application systems must be synchronous callbacks".into(),
                    ));
                }
                if let Some(resident) = &self.resident {
                    if reads.contains(&resident.column()) {
                        return Err(RenderError::Window(
                            "resident column cannot use CPU reads".into(),
                        ));
                    }
                }
            }
            self.application_systems = updates.len();
            let extra = Loop::new(updates, None, runner.state.clone(), runner.profile)
                .map_err(|error| RenderError::Window(error.to_string()))?;
            runner.systems.truncate(original);
            runner.systems.extend(extra.systems);
        }
        Ok(())
    }

    fn finish(&mut self) {
        if let Some(world) = self.world.take() {
            self.owner.control.complete();
            if let Some(runner) = self.runner.as_mut() {
                runner.join();
            }
            *self.owner.owner.lock().unwrap() = Some(world);
        }
    }
}

impl DesktopApplication for WindowFrames<'_> {
    fn start(
        &mut self,
        _window: &Window,
        renderer: &mut NativeRenderer,
        info: &DesktopInfo,
        _: &wgpu::Adapter,
    ) -> Result<bool, RenderError> {
        self.display = Some((info.clone(), renderer.adapter().to_json()));
        let active = Python::attach(|py| self.startup.call0(py)?.extract::<bool>(py))
            .map_err(|error| RenderError::Window(error.to_string()))?;
        if !active {
            return Ok(false);
        }
        let registrations = self
            .registrations
            .take()
            .ok_or_else(|| RenderError::Window("cannot restart the desktop scheduler".into()))?;
        let (world, state) = self
            .owner
            .begin(&registrations, None)
            .map_err(RenderError::Window)?;
        self.world = Some(world);
        if let Some(position) = self.position.take() {
            renderer.prepare_resident_positions()?;
            if registrations.iter().any(|(_, hz, reads, writes, participant)| {
                reads.contains(&position.column)
                    || hz.is_some() && writes.contains(&position.column)
                    || matches!(participant, Participant::Kernel(column, _) if *column == position.column)
            }) {
                return Err(RenderError::Window("resident column cannot use CPU reads, async writes or a CPU kernel".into()));
            }
            self.resident = Some(ResidentKernel::new(
                renderer,
                self.world.as_ref().unwrap(),
                position,
            )?);
        }
        self.runner = Some(
            Loop::new(registrations, None, state, self.frame_limit.is_some())
                .map_err(|error| RenderError::Window(error.to_string()))?,
        );
        Ok(true)
    }

    fn draw(
        &mut self,
        window: &Window,
        renderer: &mut NativeRenderer,
        encoder: &mut wgpu::CommandEncoder,
        target: FrameTarget<'_>,
    ) -> Result<Option<RenderStats>, RenderError> {
        if self
            .frame_limit
            .is_some_and(|limit| self.clock.tick() >= limit)
        {
            return Ok(None);
        }
        let draw_start = Instant::now();
        self.application_frame([target.width, target.height], window.scale_factor())?;
        let world = self.world.as_mut().expect("desktop start owns the World");
        let control = &self.owner.control;
        let resident = &mut self.resident;
        let mut commit = |world: &mut World, command| match resident {
            Some(resident) => resident.apply(world, command),
            None => apply(world, command),
        };
        if !control.frame_boundary_with(
            self.clock.tick(),
            self.owner.batches_per_frame,
            |command| commit(world, command),
        ) {
            return Ok(None);
        }
        self.runner
            .as_mut()
            .expect("desktop start owns the scheduler")
            .systems_at_barrier(
                world,
                &self.clock,
                &mut self.cache,
                &mut self.due,
                &mut self.report,
                &mut commit,
            )
            .map_err(RenderError::Window)?;
        // The desktop Engine owns this queue and has no external receipt consumer.
        // Return credit after each barrier and surface any rejected scene edit.
        while let Some(outcome) = control.take() {
            if let Err(error) = outcome.result {
                return Err(RenderError::Window(format!(
                    "scene command {} at frame {}: {error}",
                    outcome.sequence, outcome.frame
                )));
            }
        }
        if let Some((_, after)) = &self.application {
            Python::attach(|py| after.call0(py))
                .map_err(|error| RenderError::Window(error.to_string()))?;
        }
        self.report.events.clear();
        let update_ns = nanos(draw_start.elapsed());
        let (systems, over_budget) = if self.debug {
            let runner = self
                .runner
                .as_ref()
                .expect("desktop start owns the scheduler");
            let systems = runner
                .systems
                .iter()
                .filter(|system| system.ran)
                .map(|system| (system.name.clone(), system.last_ns))
                .collect();
            // A synchronous Python callback over the frame budget is the
            // documented HUD/bench signal, so only that lane is marked here.
            let over_budget = runner
                .systems
                .iter()
                .filter(|system| {
                    system.ran && matches!(system.work, Work::Sync(_)) && system.last_ns > 1_000_000
                })
                .map(|system| (system.name.clone(), system.last_ns))
                .collect();
            (systems, over_budget)
        } else {
            (Vec::new(), Vec::new())
        };
        let render_start = Instant::now();
        let dispatch = graphics::compute_overrides(&self.scene)?;
        let camera = {
            let mut scene = self.scene.lock().unwrap();
            scene.run_compute(renderer, world, self.clock.dt() as f32, &dispatch)?;
            scene.apply(renderer, world)?
        };
        renderer.set_frame_time(self.clock.time() + self.clock.dt());
        if let Some(resident) = &mut self.resident {
            resident.dispatch(world, encoder, self.clock.dt() as f32, self.skip_dispatch)?;
        }
        let target = FrameTarget {
            positions: self.resident.as_ref().map(ResidentKernel::positions),
            clear: renderer.background(),
            ..target
        };
        let draw = renderer.encode_world_to_target(
            encoder,
            target,
            world,
            graphics::source(world)?,
            &camera,
        )?;
        if self.debug {
            self.pending_metrics = Some(FrameMetrics {
                update_ns,
                render_ns: nanos(render_start.elapsed()),
                systems,
                over_budget,
                ..FrameMetrics::default()
            });
        }
        control.record_frame();
        self.clock.advance();
        Ok(Some(draw))
    }

    fn event(&mut self, _: &Window, event: &WindowEvent) -> Result<(), RenderError> {
        let mut data = [0.0; 4];
        let mut key = String::new();
        let mut button = 0;
        let name = match event {
            WindowEvent::Focused(false) | WindowEvent::CloseRequested | WindowEvent::Destroyed => {
                self.cursor = None;
                self.modifiers = ModifiersState::empty();
                "focus_lost"
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                self.modifiers = modifiers.state();
                return Ok(());
            }
            WindowEvent::KeyboardInput { event, .. } => {
                key = match &event.logical_key {
                    Key::Character(name) => name.to_lowercase(),
                    Key::Named(NamedKey::Space) => "Space".into(),
                    Key::Named(name) => format!("{name:?}"),
                    _ => return Ok(()),
                };
                if event.state.is_pressed() {
                    "key_down"
                } else {
                    "key_up"
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let current = [position.x, position.y];
                let previous = self.cursor.replace(current).unwrap_or(current);
                data = [
                    current[0],
                    current[1],
                    current[0] - previous[0],
                    current[1] - previous[1],
                ];
                "pointer_move"
            }
            WindowEvent::MouseInput {
                state,
                button: mouse,
                ..
            } => {
                let [x, y] = self.cursor.unwrap_or_default();
                data = [x, y, 0.0, 0.0];
                button = match mouse {
                    MouseButton::Left => 1,
                    MouseButton::Right => 2,
                    MouseButton::Middle => 3,
                    MouseButton::Back => 4,
                    MouseButton::Forward => 5,
                    MouseButton::Other(value) => *value,
                };
                if state.is_pressed() {
                    "pointer_down"
                } else {
                    "pointer_up"
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let [x, y] = self.cursor.unwrap_or_default();
                let (dx, dy) = match delta {
                    MouseScrollDelta::LineDelta(dx, dy) => {
                        (-f64::from(*dx) * 100.0, -f64::from(*dy) * 100.0)
                    }
                    MouseScrollDelta::PixelDelta(point) => (-point.x, -point.y),
                };
                data = [x, y, dx, dy];
                "wheel"
            }
            _ => return Ok(()),
        };
        if self.application.is_some() {
            let modifiers = [
                (ModifiersState::SHIFT, "Shift"),
                (ModifiersState::CONTROL, "Control"),
                (ModifiersState::ALT, "Alt"),
                (ModifiersState::SUPER, "Meta"),
            ]
            .into_iter()
            .filter_map(|(flag, name)| self.modifiers.contains(flag).then_some(name))
            .collect();
            self.events.push((name, key, data, button, modifiers));
        } else if matches!(name, "focus_lost" | "key_down" | "key_up") {
            let key = (name != "focus_lost").then_some(key);
            Python::attach(|py| self.keyboard.call1(py, (key, name == "key_down")))
                .map_err(|error| RenderError::Window(error.to_string()))?;
        }
        Ok(())
    }

    fn presented(&mut self, frame: PresentedFrame) -> Result<(), RenderError> {
        if self.debug {
            if let Some(mut metrics) = self.pending_metrics.take() {
                metrics.interval_ns = frame.wall_interval.map(nanos).unwrap_or(0);
                metrics.acquire_ns = nanos(frame.acquire);
                metrics.submit_ns = nanos(frame.submit);
                metrics.present_ns = nanos(frame.present);
                *self.metrics.lock().unwrap() = Some(metrics);
            }
        }
        if self.frame_limit.is_some() {
            self.frames.push(frame);
        }
        Ok(())
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
