//! Calling-thread native execution. No Python dependency, callbacks or threads.
//! Boundary order: admitted FIFO operations (including snapshots), integration,
//! publish counters, optional interruptible pacing. Stop cancels remaining FIFO.
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::ecs::{
    ColumnData, ComponentPayload, ComponentReadout, EntityId, Scalar, Selection, World,
    MAX_COMPONENT_WIDTH,
};
use crate::vehicles::{Controls, Sim, DT};

/// Maximum owned input or selected readback bytes per operation.
pub const MAX_COMMAND_BYTES: usize = 4 * 1024 * 1024;

/// One atomic World transaction or selected owned readback.
#[derive(Debug)]
pub enum Command {
    /// Replace one component's rows, validated together at application time.
    Update {
        component: usize,
        ids: Vec<EntityId>,
        data: ColumnData,
    },
    Set {
        selection: Selection,
        component: usize,
        field: usize,
        value: Scalar,
    },
    Spawn {
        count: u32,
        payload: Vec<ComponentPayload>,
    },
    /// Destroy identities, validated together at application time.
    Destroy(Vec<EntityId>),
    /// Read selected identities at this command's FIFO position.
    Read {
        component: usize,
        ids: Vec<EntityId>,
    },
    /// Replace the attached vehicle's bounded controls at this command's FIFO
    /// boundary. It is rejected when the runtime has no vehicle.
    Controls {
        steer: f64,
        throttle: f64,
        brake: f64,
        handbrake: f64,
    },
}
impl Command {
    fn rows(&self) -> usize {
        match self {
            Self::Update { ids, .. } | Self::Read { ids, .. } | Self::Destroy(ids) => ids.len(),
            Self::Set { .. } | Self::Controls { .. } => 0,
            Self::Spawn { count, .. } => *count as usize,
        }
    }
    fn bytes(&self) -> Option<usize> {
        let row_bytes = self.rows().checked_mul(16)?;
        match self {
            Self::Update { data, .. } => row_bytes.checked_add(data.byte_len()),
            Self::Set { selection, .. } => {
                32usize.checked_add(selection.predicates.len().checked_mul(32)?)
            }
            Self::Spawn { payload, .. } => payload.iter().try_fold(row_bytes, |bytes, part| {
                bytes.checked_add(8)?.checked_add(part.data.byte_len())
            }),
            Self::Controls { .. } => Some(4 * std::mem::size_of::<f64>()),
            _ => Some(row_bytes),
        }
    }
    fn valid_controls(&self) -> bool {
        match self {
            Self::Controls {
                steer,
                throttle,
                brake,
                handbrake,
            } => {
                steer.is_finite()
                    && (-1.0..=1.0).contains(steer)
                    && throttle.is_finite()
                    && (0.0..=1.0).contains(throttle)
                    && brake.is_finite()
                    && (0.0..=1.0).contains(brake)
                    && handbrake.is_finite()
                    && (0.0..=1.0).contains(handbrake)
            }
            _ => true,
        }
    }
}

/// Apply the same validated operation in every caller and scheduler lane.
pub fn apply(world: &mut World, command: Command) -> Result<Option<ComponentReadout>, String> {
    match command {
        Command::Update {
            component,
            ids,
            data,
        } => world.update_rows(component, &ids, data).map(|_| None),
        Command::Destroy(ids) => world.destroy(&ids).map(|_| None),
        Command::Read { component, ids } => world.read_component(component, Some(&ids)).map(Some),
        Command::Set {
            selection,
            component,
            field,
            value,
        } => {
            return world
                .set_selected(&selection, component, field, value)
                .map(|_| None);
        }
        Command::Spawn { count, payload } => world.spawn(count, &payload).map(|_| None),
        Command::Controls { .. } => return Err("controls require an attached vehicle".into()),
    }
    .map_err(|e| e.to_string())
}

/// Admission is nonblocking. IDs are allocated only for admitted commands.
#[derive(Debug, PartialEq, Eq)]
pub enum Admission {
    Admitted(u64),
    Full,
    Stopped,
    Error(&'static str),
}
/// Runtime lifecycle; a Runtime runs at most once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Ready,
    Running,
    Stopped,
}
/// Owned command completion. Taking it frees one queue credit.
#[derive(Debug)]
pub struct Outcome {
    pub sequence: u64,
    /// Zero-based boundary before this frame's system; cancellation uses frames completed.
    pub frame: u64,
    pub result: Result<Option<ComponentReadout>, String>,
}
/// Cumulative counters, copied without reading any World rows.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub frames: u64,
    pub native_steps: u64,
    pub rows_integrated: u64,
    pub admitted: u64,
    pub applied: u64,
    pub rejected: u64,
    pub cancelled: u64,
    pub full: u64,
    pub backlog: usize,
    pub outcomes: usize,
    /// Copied handle/payload bytes attempted, including Full/Stopped submissions.
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub readback_bytes: u64,
    pub native_ns: u64,
    pub frame_ns: u64,
    pub max_frame_ns: u64,
    pub paced_ns: u64,
    pub wall_ns: u64,
    /// Fixed vehicle substeps executed by the attached native system.
    pub vehicle_steps: u64,
    /// Time inside the vehicle solver only.
    pub vehicle_ns: u64,
    /// Time publishing the vehicle pose into the World.
    pub publish_ns: u64,
    /// First apply rejection or fatal execution failure, retained after outcomes are taken.
    pub first_error: Option<String>,
}
struct State {
    phase: Phase,
    stop: bool,
    next: u64,
    queue: VecDeque<(u64, Command)>,
    staged: usize,
    outcomes: VecDeque<Outcome>,
    stats: Stats,
}
struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    capacity: usize,
    max_rows: usize,
    read_row_bytes: Vec<usize>,
    has_vehicle: bool,
}
impl Shared {
    fn command_bytes(&self, command: &Command) -> Result<usize, Admission> {
        let limit = || Admission::Error("command exceeds row or scalar limit");
        if !command.valid_controls()
            || (matches!(command, Command::Controls { .. }) && !self.has_vehicle)
            || command.rows() > self.max_rows
            || matches!(command, Command::Update { data, ids, .. }
                if data.len().div_ceil(MAX_COMPONENT_WIDTH as usize) > ids.len())
        {
            return Err(limit());
        }
        if let Command::Read { component, ids } = command {
            let row_bytes = self.read_row_bytes.get(*component).ok_or_else(limit)?;
            if ids.len().checked_mul(*row_bytes).ok_or_else(limit)? > MAX_COMMAND_BYTES {
                return Err(limit());
            }
        }
        command
            .bytes()
            .filter(|&bytes| bytes <= MAX_COMMAND_BYTES)
            .ok_or_else(limit)
    }
}
/// Cloneable producer handle; owns no World and launches no worker.
#[derive(Clone)]
pub struct Control(Arc<Shared>);
impl Control {
    pub fn new(world: &World, capacity: usize, max_rows: usize) -> Self {
        Self::with_vehicle_support(world, capacity, max_rows, false)
    }
    fn with_vehicle_support(
        world: &World,
        capacity: usize,
        max_rows: usize,
        has_vehicle: bool,
    ) -> Self {
        Self(Arc::new(Shared {
            state: Mutex::new(State {
                phase: Phase::Ready,
                stop: false,
                next: 1,
                queue: VecDeque::new(),
                staged: 0,
                outcomes: VecDeque::new(),
                stats: Stats::default(),
            }),
            wake: Condvar::new(),
            capacity,
            max_rows,
            read_row_bytes: world.schemas().map(|s| 17 + s.width as usize * 4).collect(),
            has_vehicle,
        }))
    }
    pub fn start(&self) -> Result<(), &'static str> {
        let mut s = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        if s.phase != Phase::Ready {
            return Err("runtime is single-shot or stopped");
        }
        s.phase = Phase::Running;
        Ok(())
    }
    pub fn complete(&self) {
        let mut state = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        Self::finish(&mut state);
        self.0.wake.notify_all();
    }
    pub fn frame_boundary(&self, world: &mut World, frame: u64, limit: usize) -> bool {
        self.frame_boundary_with(frame, limit, |command| apply(world, command))
    }

    pub fn frame_boundary_with(
        &self,
        frame: u64,
        limit: usize,
        mut apply_command: impl FnMut(Command) -> Result<Option<ComponentReadout>, String>,
    ) -> bool {
        let commands = {
            let mut s = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
            if s.stop {
                return false;
            }
            let count = limit.min(s.queue.len());
            s.staged += count;
            s.queue
                .drain(..count)
                .map(|(sequence, command)| Staged {
                    control: self.clone(),
                    sequence,
                    command: Some(command),
                    settled: false,
                })
                .collect::<Vec<_>>()
        };
        for command in commands {
            command.commit_with(frame, &mut apply_command);
        }
        true
    }
    fn record_outcome(
        s: &mut State,
        sequence: u64,
        frame: u64,
        result: Result<Option<ComponentReadout>, String>,
    ) {
        match &result {
            Ok(read) => {
                s.stats.applied += 1;
                if let Some(read) = read {
                    s.stats.readback_bytes += (read.ids.len() * 17 + read.data.byte_len()) as u64;
                }
            }
            Err(error) => {
                s.stats.rejected += 1;
                if s.stats.first_error.is_none() {
                    s.stats.first_error = Some(error.clone());
                }
            }
        }
        s.outcomes.push_back(Outcome {
            sequence,
            frame,
            result,
        });
    }

    /// Reserve shared queue capacity before touching an input buffer.
    pub fn reserve(&self) -> Result<Credit, Admission> {
        let mut s = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        if s.stop || s.phase == Phase::Stopped {
            return Err(Admission::Stopped);
        }
        if s.queue.len() + s.outcomes.len() + s.staged >= self.0.capacity {
            s.stats.full += 1;
            return Err(Admission::Full);
        }
        s.staged += 1;
        Ok(Credit {
            control: self.clone(),
            active: true,
        })
    }

    pub fn stage(&self, command: Command) -> Result<Staged, Admission> {
        self.reserve()?.stage(command)
    }

    pub fn record_frame(&self) {
        self.0
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .stats
            .frames += 1;
    }
    pub fn wait_until(&self, deadline: Instant) {
        let s = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        let _ = self.0.wake.wait_timeout_while(
            s,
            deadline.saturating_duration_since(Instant::now()),
            |s| !s.stop,
        );
    }
    pub fn submit(&self, command: Command) -> Admission {
        let bytes = match self.0.command_bytes(&command) {
            Ok(bytes) => bytes,
            Err(error) => return error,
        };
        let mut s = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        s.stats.bytes_in += bytes as u64;
        if s.stop || s.phase == Phase::Stopped {
            return Admission::Stopped;
        }
        if s.queue.len() + s.outcomes.len() + s.staged >= self.0.capacity {
            s.stats.full += 1;
            return Admission::Full;
        }
        let Some(next) = s.next.checked_add(1) else {
            return Admission::Error("sequence exhausted");
        };
        let sequence = s.next;
        s.next = next;
        s.stats.admitted += 1;
        s.queue.push_back((sequence, command));
        Admission::Admitted(sequence)
    }
    pub fn take(&self) -> Option<Outcome> {
        let mut s = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        let outcome = s.outcomes.pop_front()?;
        if let Ok(Some(read)) = &outcome.result {
            s.stats.bytes_out += (read.ids.len() * 17 + read.data.byte_len()) as u64;
        }
        Some(outcome)
    }
    pub fn phase(&self) -> Phase {
        self.0.state.lock().unwrap_or_else(|e| e.into_inner()).phase
    }
    pub fn stats(&self) -> Stats {
        let s = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        let mut result = s.stats.clone();
        result.backlog = s.queue.len() + s.staged;
        result.outcomes = s.outcomes.len();
        result
    }
    pub fn stop(&self) {
        let mut s = self.0.state.lock().unwrap_or_else(|e| e.into_inner());
        s.stop = true;
        if s.phase == Phase::Ready {
            Self::finish(&mut s);
        }
        self.0.wake.notify_all();
    }
    fn finish(s: &mut State) {
        while let Some((sequence, _)) = s.queue.pop_front() {
            s.stats.cancelled += 1;
            s.outcomes.push_back(Outcome {
                sequence,
                frame: s.stats.frames,
                result: Err("cancelled on stop".into()),
            });
        }
        s.phase = Phase::Stopped;
        s.stop = true;
    }
}

/// Reserved admission with no sequence number until its payload is valid.
pub struct Credit {
    control: Control,
    active: bool,
}

impl Credit {
    fn accept(&mut self, state: &mut State, bytes: usize) -> Result<u64, Admission> {
        if state.stop || state.phase == Phase::Stopped {
            return Err(Admission::Stopped);
        }
        let sequence = state.next;
        state.next = sequence
            .checked_add(1)
            .ok_or(Admission::Error("sequence exhausted"))?;
        state.stats.admitted += 1;
        state.stats.bytes_in += bytes as u64;
        self.active = false;
        Ok(sequence)
    }

    pub fn stage(mut self, command: Command) -> Result<Staged, Admission> {
        if matches!(command, Command::Controls { .. }) {
            return Err(Admission::Error(
                "vehicle controls must use the runtime queue",
            ));
        }
        let bytes = self.control.0.command_bytes(&command)?;
        let control = self.control.clone();
        let mut state = control.0.state.lock().unwrap_or_else(|e| e.into_inner());
        let sequence = self.accept(&mut state, bytes)?;
        drop(state);
        Ok(Staged {
            control,
            sequence,
            command: Some(command),
            settled: false,
        })
    }

    pub fn submit(mut self, command: Command) -> Result<u64, Admission> {
        let bytes = self.control.0.command_bytes(&command)?;
        let control = self.control.clone();
        let mut state = control.0.state.lock().unwrap_or_else(|e| e.into_inner());
        let sequence = self.accept(&mut state, bytes)?;
        state.staged -= 1;
        state.queue.push_back((sequence, command));
        Ok(sequence)
    }
}

impl Drop for Credit {
    fn drop(&mut self) {
        if self.active {
            self.control
                .0
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .staged -= 1;
        }
    }
}

/// Work owned by a callback until the scheduler commits it at a barrier.
pub struct Staged {
    control: Control,
    sequence: u64,
    command: Option<Command>,
    settled: bool,
}

impl Staged {
    pub fn id(&self) -> u64 {
        self.sequence
    }

    pub fn commit(self, world: &mut World, frame: u64) -> bool {
        self.commit_with(frame, |command| apply(world, command))
    }

    pub fn commit_with(
        mut self,
        frame: u64,
        apply_command: impl FnOnce(Command) -> Result<Option<ComponentReadout>, String>,
    ) -> bool {
        let result = apply_command(self.command.take().unwrap());
        let success = result.is_ok();
        self.settle(frame, result);
        success
    }

    pub fn reject(mut self, frame: u64, error: &str) {
        self.settle(frame, Err(error.into()));
    }

    fn settle(&mut self, frame: u64, result: Result<Option<ComponentReadout>, String>) {
        let mut s = self
            .control
            .0
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        s.staged -= 1;
        Control::record_outcome(&mut s, self.sequence, frame, result);
        self.settled = true;
    }
}

impl Drop for Staged {
    fn drop(&mut self) {
        if !self.settled {
            let mut s = self
                .control
                .0
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            s.staged -= 1;
            s.stats.cancelled += 1;
            let frame = s.stats.frames;
            s.outcomes.push_back(Outcome {
                sequence: self.sequence,
                frame,
                result: Err("cancelled on stop".into()),
            });
        }
    }
}

/// The four `f32` columns the native vehicle system publishes each frame.
#[derive(Clone, Copy, Debug)]
pub struct VehicleColumns {
    pub position: usize,
    pub rotation: usize,
    pub velocity: usize,
    pub angular: usize,
}

struct VehicleSystem {
    sim: Sim,
    columns: VehicleColumns,
    entity: EntityId,
}

/// Owns the actual World, not a copy. Run it on the intended OS/main thread.
/// Dropping it stops admission and cancels pending work; there is no join.
pub struct Runtime {
    world: Option<World>,
    control: Control,
    position: usize,
    velocity: usize,
    batches_per_frame: usize,
    vehicle: Option<VehicleSystem>,
}
impl Runtime {
    /// Validate the fixed native system and queue limits before taking ownership.
    pub fn validate(
        world: &World,
        position: usize,
        velocity: usize,
        capacity: usize,
        max_rows: usize,
        batches_per_frame: usize,
    ) -> Result<(), &'static str> {
        if capacity == 0 || max_rows == 0 || batches_per_frame == 0 {
            return Err("runtime limits must be positive");
        }
        let p = world.schema(position).ok_or("unknown position column")?;
        let v = world.schema(velocity).ok_or("unknown velocity column")?;
        if position == velocity
            || p.element != crate::ecs::ElementType::F32
            || v.element != p.element
            || p.width == 0
            || p.width != v.width
        {
            return Err("integration requires distinct equal-width non-marker f32 columns");
        }
        Ok(())
    }
    /// Caller must first call validate with these same arguments.
    pub fn new(
        world: World,
        position: usize,
        velocity: usize,
        capacity: usize,
        max_rows: usize,
        batches_per_frame: usize,
    ) -> Result<Self, Box<(World, &'static str)>> {
        if let Err(error) = Self::validate(
            &world,
            position,
            velocity,
            capacity,
            max_rows,
            batches_per_frame,
        ) {
            return Err(Box::new((world, error)));
        }
        let control = Control::new(&world, capacity, max_rows);
        Ok(Self {
            world: Some(world),
            control,
            position,
            velocity,
            batches_per_frame,
            vehicle: None,
        })
    }
    /// Validate a World with the four pose columns one vehicle publishes into.
    pub fn validate_vehicle(
        world: &World,
        columns: VehicleColumns,
        entity: EntityId,
    ) -> Result<(), &'static str> {
        let position = world
            .schema(columns.position)
            .ok_or("unknown position column")?;
        let rotation = world
            .schema(columns.rotation)
            .ok_or("unknown rotation column")?;
        let velocity = world
            .schema(columns.velocity)
            .ok_or("unknown velocity column")?;
        let angular = world
            .schema(columns.angular)
            .ok_or("unknown angular column")?;
        if position.element != crate::ecs::ElementType::F32
            || rotation.element != crate::ecs::ElementType::F32
            || velocity.element != crate::ecs::ElementType::F32
            || angular.element != crate::ecs::ElementType::F32
        {
            return Err("vehicle pose columns must be f32");
        }
        if position.width != 3 || rotation.width != 4 || velocity.width != 3 || angular.width != 3 {
            return Err("vehicle pose columns must be width 3, 4, 3 and 3");
        }
        let indices = [
            columns.position,
            columns.rotation,
            columns.velocity,
            columns.angular,
        ];
        if (1..indices.len()).any(|i| indices[..i].contains(&indices[i])) {
            return Err("vehicle pose columns must be distinct");
        }
        for index in indices {
            if !world
                .is_member(index, entity)
                .map_err(|_| "invalid vehicle entity")?
            {
                return Err("vehicle entity must carry every pose column");
            }
        }
        Ok(())
    }
    /// Validate and take ownership for one vehicle stepped by the fixed native
    /// system. Caller must first call `validate_vehicle` with these arguments.
    pub fn with_vehicle(
        world: World,
        columns: VehicleColumns,
        sim: Sim,
        entity: EntityId,
        capacity: usize,
        max_rows: usize,
        batches_per_frame: usize,
    ) -> Result<Self, Box<(World, &'static str)>> {
        if capacity == 0 || max_rows == 0 || batches_per_frame == 0 {
            return Err(Box::new((world, "runtime limits must be positive")));
        }
        if let Err(error) = Self::validate_vehicle(&world, columns, entity) {
            return Err(Box::new((world, error)));
        }
        let control = Control::with_vehicle_support(&world, capacity, max_rows, true);
        Ok(Self {
            world: Some(world),
            control,
            position: columns.position,
            velocity: columns.velocity,
            batches_per_frame,
            vehicle: Some(VehicleSystem {
                sim,
                columns,
                entity,
            }),
        })
    }
    pub fn control(&self) -> Control {
        self.control.clone()
    }
    /// Owned copy of the attached vehicle's current state, if there is one.
    pub fn vehicle_state(&self) -> Option<crate::vehicles::Snapshot> {
        self.vehicle.as_ref().map(|vehicle| vehicle.sim.snapshot())
    }
    /// Fixed simulation dt; pacing only adds wall time and never changes dt.
    pub fn run(&mut self, frames: u64, dt: f32, pace: Option<Duration>) -> Result<Stats, String> {
        if frames == 0 || !dt.is_finite() || dt <= 0.0 {
            return Err("frames and finite dt must be positive".into());
        }
        if self.vehicle.is_some() && (dt as f64 - DT).abs() > 1e-6 {
            return Err("a vehicle runtime steps at the fixed 120 Hz dt".into());
        }
        self.control.start()?;
        let start = Instant::now();
        // Always close admission and expose failure even if a future native system panics.
        let execution = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.execute(frames, dt, pace)
        }));
        let mut s = self
            .control
            .0
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if execution.is_err() && s.stats.first_error.is_none() {
            s.stats.first_error = Some("native runtime panicked".into());
        }
        s.stats.wall_ns = nanos(start.elapsed());
        Control::finish(&mut s);
        drop(s);
        match execution {
            Ok(()) => Ok(self.control.stats()),
            Err(_) => Err("native runtime panicked".into()),
        }
    }
    fn execute(&mut self, frames: u64, dt: f32, pace: Option<Duration>) {
        let world = self.world.as_mut().unwrap();
        for frame in 0..frames {
            let begin = Instant::now();
            let vehicle = &mut self.vehicle;
            if !self
                .control
                .frame_boundary_with(frame, self.batches_per_frame, |command| match command {
                    Command::Controls {
                        steer,
                        throttle,
                        brake,
                        handbrake,
                    } => {
                        let vehicle = vehicle
                            .as_mut()
                            .expect("controls require an attached vehicle");
                        vehicle.sim.set_controls(Controls {
                            steer,
                            throttle,
                            brake,
                            handbrake,
                        })?;
                        Ok(None)
                    }
                    command => apply(world, command),
                })
            {
                break;
            }
            let vehicle_entity = self.vehicle.as_ref().map(|vehicle| vehicle.entity);
            let is_vehicle = vehicle_entity.is_some();
            let mut publish_error = None;
            let (vehicle_ns, publish_ns) = if let Some(vehicle) = self.vehicle.as_mut() {
                let native_start = Instant::now();
                vehicle.sim.step();
                let vehicle_ns = nanos(native_start.elapsed());
                let publish_start = Instant::now();
                let (position, rotation) = vehicle.sim.pose();
                let velocity = vehicle.sim.velocity();
                let angular = vehicle.sim.angular();
                let entity = [vehicle.entity];
                let rows = [
                    (
                        vehicle.columns.position,
                        position.iter().map(|value| *value as f32).collect(),
                    ),
                    (
                        vehicle.columns.rotation,
                        rotation.iter().map(|value| *value as f32).collect(),
                    ),
                    (
                        vehicle.columns.velocity,
                        velocity.iter().map(|value| *value as f32).collect(),
                    ),
                    (
                        vehicle.columns.angular,
                        angular.iter().map(|value| *value as f32).collect(),
                    ),
                ];
                for (component, data) in rows {
                    if let Err(error) = world.update_rows(component, &entity, ColumnData::F32(data))
                    {
                        publish_error = Some(error.to_string());
                        break;
                    }
                }
                let publish_ns = nanos(publish_start.elapsed());
                (vehicle_ns, publish_ns)
            } else {
                (0, 0)
            };
            let linear_start = Instant::now();
            let linear_rows = world
                .f32_rows_except(self.position, self.velocity, vehicle_entity)
                .expect("validated fixed schema")
                .for_each(|p, v| {
                    for (p, v) in p.iter_mut().zip(v) {
                        *p += *v * dt;
                    }
                });
            let native_ns = vehicle_ns + publish_ns + nanos(linear_start.elapsed());
            let rows = linear_rows + usize::from(is_vehicle);
            let frame_ns = nanos(begin.elapsed());
            let mut s = self
                .control
                .0
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            s.stats.frames += 1;
            s.stats.native_steps += 1;
            s.stats.rows_integrated += rows as u64;
            s.stats.native_ns += native_ns;
            if let Some(error) = publish_error {
                if s.stats.first_error.is_none() {
                    s.stats.first_error = Some(error);
                }
            }
            if is_vehicle {
                s.stats.vehicle_steps += 1;
                s.stats.vehicle_ns += vehicle_ns;
                s.stats.publish_ns += publish_ns;
            }
            s.stats.frame_ns += frame_ns;
            s.stats.max_frame_ns = s.stats.max_frame_ns.max(frame_ns);
            if let Some(period) = pace {
                let remaining = period.saturating_sub(begin.elapsed());
                let wait_start = Instant::now();
                let (mut waited, _) = self
                    .control
                    .0
                    .wake
                    .wait_timeout_while(s, remaining, |s| !s.stop)
                    .unwrap();
                waited.stats.paced_ns += nanos(wait_start.elapsed());
            }
        }
    }
    /// Stop and recover the identical World. No pending batch is applied here.
    pub fn into_world(mut self) -> World {
        self.control.stop();
        self.world.take().unwrap()
    }
}
impl Drop for Runtime {
    fn drop(&mut self) {
        self.control.stop();
    }
}
fn nanos(d: Duration) -> u64 {
    d.as_nanos().min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::{ComponentPayload, ElementType};
    fn setup(capacity: usize, per_frame: usize) -> (Runtime, Vec<EntityId>) {
        let mut w = World::new(4).unwrap();
        let p = w.register("position", ElementType::F32, 1).unwrap();
        let v = w.register("velocity", ElementType::F32, 1).unwrap();
        let ids = w
            .spawn(
                2,
                &[
                    ComponentPayload {
                        component: p,
                        data: ColumnData::F32(vec![0., 10.]),
                    },
                    ComponentPayload {
                        component: v,
                        data: ColumnData::F32(vec![1., 2.]),
                    },
                ],
            )
            .unwrap();
        (
            Runtime::new(w, p, v, capacity, 4, per_frame).ok().unwrap(),
            ids,
        )
    }
    fn update(ids: &[EntityId], values: &[f32]) -> Command {
        Command::Update {
            component: 0,
            ids: ids.to_vec(),
            data: ColumnData::F32(values.to_vec()),
        }
    }
    fn read(ids: &[EntityId]) -> Command {
        Command::Read {
            component: 0,
            ids: ids.to_vec(),
        }
    }
    #[test]
    fn command_bounds_match_across_admission_routes() {
        let width = MAX_COMPONENT_WIDTH;
        let read_row_bytes = 17 + width as usize * 4;
        let allowed_rows = MAX_COMMAND_BYTES / read_row_bytes;
        let mut world = World::new((allowed_rows + 1) as u32).unwrap();
        let wide = world.register("wide", ElementType::F32, width).unwrap();
        let ids = world.spawn((allowed_rows + 1) as u32, &[]).unwrap();
        let control = Control::new(&world, 1, allowed_rows + 1);
        fn settle(staged: Result<Staged, Admission>, world: &mut World) -> Admission {
            match staged {
                Ok(staged) => {
                    let id = staged.id();
                    assert!(staged.commit(world, 0));
                    Admission::Admitted(id)
                }
                Err(error) => error,
            }
        }
        type Route = fn(&Control, &mut World, Command) -> Admission;
        let routes: [Route; 4] = [
            |c, world, command| {
                let admission = c.submit(command);
                assert!(c.frame_boundary(world, 0, 1));
                admission
            },
            |c, world, command| settle(c.stage(command), world),
            |c, world, command| settle(c.reserve().unwrap().stage(command), world),
            |c, world, command| {
                let admission = c.reserve().unwrap().submit(command);
                assert!(c.frame_boundary(world, 0, 1));
                admission.map_or_else(|error| error, Admission::Admitted)
            },
        ];
        for admit in routes {
            let error = Admission::Error("command exceeds row or scalar limit");
            assert_eq!(
                admit(
                    &control,
                    &mut world,
                    Command::Read {
                        component: wide,
                        ids: ids.clone(),
                    },
                ),
                error
            );
            assert_eq!(
                admit(
                    &control,
                    &mut world,
                    Command::Destroy(vec![ids[0]; allowed_rows + 2]),
                ),
                error
            );
            assert_eq!(
                admit(
                    &control,
                    &mut world,
                    update(&ids[..1], &vec![0.; width as usize + 1]),
                ),
                error
            );
            let input_control = Control::new(&world, 1, usize::MAX);
            assert_eq!(
                admit(
                    &input_control,
                    &mut world,
                    Command::Spawn {
                        count: (MAX_COMMAND_BYTES / 16 + 1) as u32,
                        payload: vec![],
                    },
                ),
                error
            );
            assert!(control.frame_boundary(&mut world, 0, 1));
            assert!(control.take().is_none());
            let stats = control.stats();
            assert_eq!((stats.backlog, stats.admitted, stats.bytes_in), (0, 0, 0));
            assert_eq!(world.alive_count() as usize, ids.len());
        }

        for (i, admit) in routes.into_iter().enumerate() {
            let command = Command::Read {
                component: wide,
                ids: ids[..allowed_rows].to_vec(),
            };
            let sequence = i as u64 + 1;
            assert_eq!(
                admit(&control, &mut world, command),
                Admission::Admitted(sequence)
            );
            let read = control.take().unwrap().result.unwrap().unwrap();
            assert_eq!(read.ids.len(), allowed_rows);
        }
        let stats = control.stats();
        assert_eq!(stats.bytes_in, (routes.len() * allowed_rows * 16) as u64);
        assert_eq!(
            stats.bytes_out,
            (routes.len() * allowed_rows * read_row_bytes) as u64
        );
    }
    fn values(r: &Runtime, ids: &[EntityId]) -> Vec<f32> {
        let ColumnData::F32(data) = r
            .world
            .as_ref()
            .unwrap()
            .read_component(0, Some(ids))
            .unwrap()
            .data
        else {
            panic!()
        };
        data
    }
    #[test]
    fn deterministic_boundary_fifo_and_native_world_recovery() {
        let (mut r, ids) = setup(8, 1);
        let c = r.control();
        assert_eq!(
            c.submit(update(&ids, &[100., 200.])),
            Admission::Admitted(1)
        );
        assert_eq!(c.submit(read(&ids)), Admission::Admitted(2));
        assert_eq!(c.submit(update(&ids, &[5., 6.])), Admission::Admitted(3));
        let s = r.run(3, 1., None).unwrap();
        assert_eq!(
            (s.frames, s.native_steps, s.rows_integrated, s.applied),
            (3, 3, 6, 3)
        );
        assert_eq!(values(&r, &ids), [6., 8.]);
        assert_eq!((c.take().unwrap().sequence, c.stats().outcomes), (1, 2));
        let snapshot = c.take().unwrap();
        assert_eq!(snapshot.frame, 1);
        let read = snapshot.result.unwrap().unwrap();
        assert_eq!(read.data, ColumnData::F32(vec![101., 202.]));
        assert_eq!(c.take().unwrap().frame, 2);
        assert!(c.take().is_none());
        let w = r.into_world();
        assert_eq!(w.alive_count(), 2);
        assert_eq!(
            w.read_component(0, Some(&ids)).unwrap().data,
            ColumnData::F32(vec![6., 8.])
        );
        assert_eq!(read.data, ColumnData::F32(vec![101., 202.]));
    }
    #[test]
    fn stale_mixed_batch_is_atomic_and_later_valid_work_survives() {
        let (mut r, ids) = setup(8, 8);
        let c = r.control();
        let bad = vec![
            ids[0],
            EntityId::new(ids[1].slot(), ids[1].generation() + 1),
        ];
        c.submit(update(&bad, &[999., 999.]));
        c.submit(read(&ids));
        c.submit(update(&ids, &[3., 4.]));
        let stats = r.run(1, 1., None).unwrap();
        assert_eq!((stats.rejected, stats.applied), (1, 2));
        assert!(stats.first_error.is_some());
        assert!(c.take().unwrap().result.is_err());
        assert_eq!(
            c.take().unwrap().result.unwrap().unwrap().data,
            ColumnData::F32(vec![0., 10.])
        );
        assert_eq!(values(&r, &ids), [4., 6.]);
    }
    #[test]
    fn full_includes_outcomes_and_shutdown_cancels_fifo() {
        let (mut r, ids) = setup(2, 1);
        let c = r.control();
        assert_eq!(c.submit(read(&ids)), Admission::Admitted(1));
        assert_eq!(c.submit(update(&ids, &[5., 6.])), Admission::Admitted(2));
        assert_eq!(c.submit(read(&ids)), Admission::Full);
        let stats = r.run(1, 1., None).unwrap();
        assert_eq!(
            (
                stats.applied,
                stats.cancelled,
                stats.backlog,
                stats.outcomes
            ),
            (1, 1, 0, 2)
        );
        assert_eq!(values(&r, &ids), [1., 12.]);
        assert_eq!(c.take().unwrap().sequence, 1);
        let cancelled = c.take().unwrap();
        assert_eq!(cancelled.sequence, 2);
        assert!(cancelled.result.is_err());
        assert_eq!(c.submit(read(&ids)), Admission::Stopped);
        assert!(r.run(1, 1., None).is_err());
    }
    #[test]
    fn stop_before_start_drop_and_limits_have_no_worker() {
        let (r, ids) = setup(1, 1);
        let c = r.control();
        c.submit(read(&ids));
        drop(r);
        assert_eq!(c.phase(), Phase::Stopped);
        assert_eq!(c.stats().cancelled, 1);
        assert_eq!(c.take().unwrap().sequence, 1);
        let (mut r, ids) = setup(1, 1);
        let c = r.control();
        assert!(matches!(c.submit(read(&[ids[0]; 5])), Admission::Error(_)));
        assert_eq!(c.stats().admitted, 0);
        c.stop();
        assert!(r.run(1, 1., None).is_err());
    }
    #[test]
    fn malformed_payload_nonmember_and_destroy_validate_at_apply() {
        let (mut r, ids) = setup(8, 8);
        let c = r.control();
        c.submit(update(&ids, &[9.]));
        c.submit(Command::Destroy(vec![ids[1]]));
        c.submit(update(&ids, &[8., 8.]));
        c.submit(update(&ids[..1], &[7.]));
        let stats = r.run(1, 1., None).unwrap();
        assert_eq!(
            (stats.rejected, stats.applied, stats.rows_integrated),
            (2, 2, 1)
        );
        assert_eq!(values(&r, &ids[..1]), [8.]);
    }
    #[test]
    fn invalid_run_does_not_consume_runtime() {
        let (mut r, _) = setup(1, 1);
        assert!(r.run(0, 1., None).is_err());
        assert!(r.run(1, f32::NAN, None).is_err());
        assert_eq!(r.control().phase(), Phase::Ready);
        assert_eq!(r.run(1, 1., None).unwrap().frames, 1);
    }

    #[test]
    fn mixed_world_steps_vehicle_at_fixed_dt_and_integrates_other_rows() {
        use crate::vehicles::sim::test_car;
        use crate::vehicles::{Ground, Sim, DT};

        let mut w = World::new(2).unwrap();
        let position = w.register("position", ElementType::F32, 3).unwrap();
        let rotation = w.register("rotation", ElementType::F32, 4).unwrap();
        let velocity = w.register("velocity", ElementType::F32, 3).unwrap();
        let angular = w.register("angular", ElementType::F32, 3).unwrap();
        let ids = w
            .spawn(
                1,
                &[
                    ComponentPayload {
                        component: position,
                        data: ColumnData::F32(vec![0.0; 3]),
                    },
                    ComponentPayload {
                        component: rotation,
                        data: ColumnData::F32(vec![0.0, 0.0, 0.0, 1.0]),
                    },
                    ComponentPayload {
                        component: velocity,
                        data: ColumnData::F32(vec![0.0; 3]),
                    },
                    ComponentPayload {
                        component: angular,
                        data: ColumnData::F32(vec![0.0; 3]),
                    },
                ],
            )
            .unwrap();
        let other = w
            .spawn(
                1,
                &[
                    ComponentPayload {
                        component: position,
                        data: ColumnData::F32(vec![10.0, 0.0, 0.0]),
                    },
                    ComponentPayload {
                        component: velocity,
                        data: ColumnData::F32(vec![12.0, 0.0, 0.0]),
                    },
                ],
            )
            .unwrap();
        let mut sim = Sim::new(test_car()).unwrap();
        sim.set_ground(Ground::flat(0.0, 1.0).unwrap());
        sim.place(0.0, 0.0, 0.0, 0.0).unwrap();
        let columns = VehicleColumns {
            position,
            rotation,
            velocity,
            angular,
        };
        let mut runtime = Runtime::with_vehicle(w, columns, sim, ids[0], 8, 8, 8).unwrap();
        let control = runtime.control();
        assert!(matches!(
            control.stage(Command::Controls {
                steer: 0.0,
                throttle: 1.0,
                brake: 0.0,
                handbrake: 0.0,
            }),
            Err(Admission::Error(
                "vehicle controls must use the runtime queue"
            ))
        ));
        assert_eq!(
            control.submit(Command::Controls {
                steer: 0.0,
                throttle: 1.0,
                brake: 0.0,
                handbrake: 0.0,
            }),
            Admission::Admitted(1)
        );
        assert!(matches!(
            control.submit(Command::Controls {
                steer: 2.0,
                throttle: 1.0,
                brake: 0.0,
                handbrake: 0.0,
            }),
            Admission::Error(_)
        ));
        let stats = runtime.run(120, DT as f32, None).unwrap();
        assert_eq!(stats.vehicle_steps, 120);
        assert!(stats.vehicle_ns > 0);
        assert!(stats.publish_ns > 0);
        assert!(control.take().unwrap().result.is_ok());
        let solver_position = runtime.vehicle.as_ref().unwrap().sim.pose().0;
        let w = runtime.into_world();
        let ColumnData::F32(other_position) =
            w.read_component(position, Some(&other)).unwrap().data
        else {
            panic!("position must stay f32");
        };
        assert!(
            (other_position[0] - 22.0).abs() < 1e-4,
            "{other_position:?}"
        );
        assert_eq!(stats.rows_integrated, 240);
        let ColumnData::F32(position) = w.read_component(position, Some(&ids)).unwrap().data else {
            panic!("position must stay f32");
        };
        assert!(position[0] > 0.1, "{position:?}");
        assert_eq!(position, solver_position.map(|value| value as f32));
        let ColumnData::F32(rotation) = w.read_component(rotation, Some(&ids)).unwrap().data else {
            panic!("rotation must stay f32");
        };
        let norm: f32 = rotation.iter().map(|value| value * value).sum();
        assert!((norm - 1.0).abs() < 1e-4, "{norm}");
    }

    #[test]
    fn invalid_vehicle_dt_and_a_recycled_target_gets_only_linear_motion() {
        use crate::vehicles::sim::test_car;
        use crate::vehicles::{Ground, Sim, DT};

        let mut w = World::new(2).unwrap();
        let position = w.register("position", ElementType::F32, 3).unwrap();
        let rotation = w.register("rotation", ElementType::F32, 4).unwrap();
        let velocity = w.register("velocity", ElementType::F32, 3).unwrap();
        let angular = w.register("angular", ElementType::F32, 3).unwrap();
        let ids = w
            .spawn(
                1,
                &[
                    ComponentPayload {
                        component: position,
                        data: ColumnData::F32(vec![1.0; 3]),
                    },
                    ComponentPayload {
                        component: rotation,
                        data: ColumnData::F32(vec![0.0, 0.0, 0.0, 1.0]),
                    },
                    ComponentPayload {
                        component: velocity,
                        data: ColumnData::F32(vec![0.0; 3]),
                    },
                    ComponentPayload {
                        component: angular,
                        data: ColumnData::F32(vec![0.0; 3]),
                    },
                ],
            )
            .unwrap();
        let mut sim = Sim::new(test_car()).unwrap();
        sim.set_ground(Ground::flat(0.0, 1.0).unwrap());
        sim.place(0.0, 0.0, 0.0, 0.0).unwrap();
        let mut runtime = Runtime::with_vehicle(
            w,
            VehicleColumns {
                position,
                rotation,
                velocity,
                angular,
            },
            sim,
            ids[0],
            8,
            8,
            8,
        )
        .unwrap();
        assert!(runtime.run(120, 1.0 / 60.0, None).is_err());
        let control = runtime.control();
        assert_eq!(
            control.submit(Command::Destroy(vec![ids[0]])),
            Admission::Admitted(1)
        );
        assert_eq!(
            control.submit(Command::Spawn {
                count: 1,
                payload: vec![
                    ComponentPayload {
                        component: position,
                        data: ColumnData::F32(vec![10.0, 0.0, 0.0]),
                    },
                    ComponentPayload {
                        component: velocity,
                        data: ColumnData::F32(vec![120.0, 0.0, 0.0]),
                    },
                ],
            }),
            Admission::Admitted(2)
        );
        let stats = runtime.run(1, DT as f32, None).unwrap();
        assert_eq!(stats.vehicle_steps, 1);
        assert!(stats.first_error.is_some());
        assert!(stats.publish_ns > 0);
        assert_eq!(stats.applied, 2);
        let w = runtime.into_world();
        assert!(w.read_component(position, Some(&ids)).is_err());
        let replacement = w.read_component(position, None).unwrap();
        assert_eq!(replacement.ids.len(), 1);
        assert_eq!(replacement.ids[0].slot(), ids[0].slot());
        assert_ne!(replacement.ids[0].generation(), ids[0].generation());
        assert_eq!(replacement.data, ColumnData::F32(vec![11.0, 0.0, 0.0]));
    }

    #[test]
    fn controls_are_rejected_without_a_vehicle() {
        let (r, _) = setup(4, 4);
        let c = r.control();
        assert!(matches!(
            c.submit(Command::Controls {
                steer: 0.0,
                throttle: 0.0,
                brake: 0.0,
                handbrake: 0.0,
            }),
            Admission::Error(_)
        ));
    }

    #[test]
    fn typed_kernel_skips_nonmembers_and_accepts_reversed_columns() {
        let mut w = World::new(2).unwrap();
        let v = w.register("v", ElementType::F32, 1).unwrap();
        let p = w.register("p", ElementType::F32, 1).unwrap();
        let a = w
            .spawn(
                1,
                &[ComponentPayload {
                    component: p,
                    data: ColumnData::F32(vec![10.]),
                }],
            )
            .unwrap();
        let b = w
            .spawn(
                1,
                &[
                    ComponentPayload {
                        component: p,
                        data: ColumnData::F32(vec![20.]),
                    },
                    ComponentPayload {
                        component: v,
                        data: ColumnData::F32(vec![3.]),
                    },
                ],
            )
            .unwrap();
        assert!(Runtime::validate(&w, p, p, 1, 2, 1).is_err());
        let mut r = Runtime::new(w, p, v, 1, 2, 1).ok().unwrap();
        assert_eq!(r.run(2, 1., None).unwrap().rows_integrated, 2);
        let w = r.into_world();
        assert_eq!(
            w.read_component(p, Some(&[a[0], b[0]])).unwrap().data,
            ColumnData::F32(vec![10., 26.])
        );
    }
}
