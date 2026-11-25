//! One upright vehicle stepped in f64 at a fixed 120 Hz.
//!
//! The equations are the scalar Python model in
//! `src/manifoldx/vehicles/single.py`: ray suspension, Pacejka slip with
//! relaxation, a three-point engine curve and automatic gearbox, Ackermann
//! steering, gravity, drag, quaternion integration and body-box ground
//! contact. The solver owns no clock and starts no thread; callers decide how
//! many fixed steps run.

use std::f64::consts::PI;

use super::ground::Ground;
use super::spec::{magic, peak_slip, VehicleSpec};

/// The fixed substep: 120 Hz.
pub const DT: f64 = 1.0 / 120.0;

const G: f64 = 9.81;
const AIR: f64 = 1.2;
const RPM: f64 = 60.0 / (2.0 * PI);
const CLUTCH_RPM: f64 = 1200.0;
const REDLINE: f64 = 1.15;
const UP_AT: f64 = 0.9;
const LIGHT_UP: f64 = 0.4;
const OFF_PEDAL: f64 = 0.1;
const DOWN_AT: f64 = 0.75;
const REVERSE_BELOW: f64 = 0.5;

const RELAX: f64 = 0.3;
const V_MIN: f64 = 3.0;
const LOW_SPEED_DAMPING: f64 = 0.5;

const SAMPLES: [f64; 3] = [-0.6, 0.0, 0.6];
const NORMAL_STEP: f64 = 0.3;
const DAMPER_SPEED: f64 = 2.0;
const BUMP_STOP: f64 = 10.0;

const RESTITUTION: f64 = 0.2;
const FRICTION: f64 = 0.4;
const STEP: f64 = 0.25;
const RESTING: f64 = 1.0;
const SLOP: f64 = 0.01;
const PUSH: f64 = 0.8;

pub type V3 = [f64; 3];
pub type Quaternion = [f64; 4];

/// Bounded driver inputs. Values outside their range are rejected, never
/// clamped silently.
#[derive(Clone, Copy, Debug, Default)]
pub struct Controls {
    pub steer: f64,
    pub throttle: f64,
    pub brake: f64,
    pub handbrake: f64,
}

impl Controls {
    pub fn new(steer: f64, throttle: f64, brake: f64, handbrake: f64) -> Result<Self, String> {
        let controls = Self {
            steer,
            throttle,
            brake,
            handbrake,
        };
        controls.validate()?;
        Ok(controls)
    }

    pub fn validate(&self) -> Result<(), String> {
        let values = [
            ("steer", self.steer),
            ("throttle", self.throttle),
            ("brake", self.brake),
            ("handbrake", self.handbrake),
        ];
        for (name, value) in values {
            if !value.is_finite() {
                return Err(format!("{name} must be finite"));
            }
        }
        if self.steer < -1.0 || self.steer > 1.0 {
            return Err("steer must lie in [-1, 1]".into());
        }
        for (name, value) in [
            ("throttle", self.throttle),
            ("brake", self.brake),
            ("handbrake", self.handbrake),
        ] {
            if !(0.0..=1.0).contains(&value) {
                return Err(format!("{name} must lie in [0, 1]"));
            }
        }
        Ok(())
    }
}

/// Owned copy of every output the solver keeps between steps.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub position: V3,
    pub velocity: V3,
    pub angular: V3,
    pub rotation: Quaternion,
    pub load: Vec<f64>,
    pub compression: Vec<f64>,
    pub ground_y: Vec<f64>,
    pub omega: Vec<f64>,
    pub wheel_turn: Vec<f64>,
    pub slip: Vec<[f64; 2]>,
    pub steer_angle: Vec<f64>,
    pub gear: i32,
    pub shift_timer: f64,
    pub rpm: f64,
    pub impulse: f64,
    pub wheel_centres: Vec<V3>,
    pub chassis_contacts: u32,
    pub drive_cut: f64,
    pub brake_scale: Vec<f64>,
    pub lit: [bool; 4],
}

/// Derived geometry and per-wheel curves, computed once from the spec.
#[derive(Clone)]
struct Constants {
    n_driven: f64,
    n_hand: f64,
    driven_radius: f64,
    ack_base: f64,
    ack_track: f64,
    zsign: Vec<f64>,
    kappa_peak: Vec<f64>,
    alpha_peak: Vec<f64>,
    arb: Vec<f64>,
    pair: Vec<usize>,
    lo: f64,
}

impl Constants {
    fn new(spec: &VehicleSpec) -> Self {
        let n = spec.wheels.len();
        let mut zsign = Vec::with_capacity(n);
        let mut kappa_peak = Vec::with_capacity(n);
        let mut alpha_peak = Vec::with_capacity(n);
        let mut arb = Vec::with_capacity(n);
        let mut pair = Vec::with_capacity(n);
        let (mut ack_max_x, mut ack_min_x) = (f64::NEG_INFINITY, f64::INFINITY);
        let (mut ack_max_z, mut ack_min_z) = (f64::NEG_INFINITY, f64::INFINITY);
        let mut lo = f64::INFINITY;
        let mut driven_count = 0usize;
        let mut hand_count = 0usize;
        let mut driven_radius = 0.0;
        for (index, wheel) in spec.wheels.iter().enumerate() {
            zsign.push(sign(wheel.mount[2]));
            let (long, lat) = (
                peak_slip(wheel.tyre.b_long, wheel.tyre.c_long, wheel.tyre.e_long),
                peak_slip(wheel.tyre.b_lat, wheel.tyre.c_lat, wheel.tyre.e_lat),
            );
            kappa_peak.push(long);
            alpha_peak.push(lat);
            let is_front = wheel.mount[0] > 0.0;
            arb.push(if is_front { spec.arb[0] } else { spec.arb[1] });
            if wheel.steered {
                ack_max_x = ack_max_x.max(wheel.mount[0]);
                ack_max_z = ack_max_z.max(wheel.mount[2]);
                ack_min_z = ack_min_z.min(wheel.mount[2]);
            } else {
                ack_min_x = ack_min_x.min(wheel.mount[0]);
            }
            lo = lo.min(wheel.mount[1]);
            if wheel.driven {
                driven_count += 1;
                driven_radius += wheel.radius;
            }
            if wheel.handbrake {
                hand_count += 1;
            }
            let mut other = index;
            for (candidate, wheel_other) in spec.wheels.iter().enumerate() {
                if candidate != index
                    && (wheel_other.mount[0] - wheel.mount[0]).abs() < 1e-6
                    && wheel_other.mount[2] * wheel.mount[2] < 0.0
                {
                    other = candidate;
                }
            }
            pair.push(other);
        }
        Self {
            n_driven: driven_count.max(1) as f64,
            n_hand: hand_count.max(1) as f64,
            driven_radius: driven_radius / driven_count.max(1) as f64,
            ack_base: ack_max_x - ack_min_x,
            ack_track: ack_max_z - ack_min_z,
            zsign,
            kappa_peak,
            alpha_peak,
            arb,
            pair,
            lo,
        }
    }
}

#[derive(Clone)]
struct State {
    position: V3,
    velocity: V3,
    angular: V3,
    rotation: Quaternion,
    load: Vec<f64>,
    compression: Vec<f64>,
    ground_y: Vec<f64>,
    omega: Vec<f64>,
    wheel_turn: Vec<f64>,
    slip: Vec<[f64; 2]>,
    steer_angle: Vec<f64>,
    gear: i32,
    shift_timer: f64,
    rpm: f64,
    drive_cut: f64,
    impulse: f64,
    wheel_centres: Vec<V3>,
    chassis_contacts: u32,
    lit: [bool; 4],
    brake_scale: Vec<f64>,
    extra_brake: Vec<f64>,
    forward: V3,
}

impl State {
    fn new(spec: &VehicleSpec) -> Self {
        let n = spec.wheels.len();
        Self {
            position: [0.0; 3],
            velocity: [0.0; 3],
            angular: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            load: vec![0.0; n],
            compression: vec![0.0; n],
            ground_y: vec![0.0; n],
            omega: vec![0.0; n],
            wheel_turn: vec![0.0; n],
            slip: vec![[0.0; 2]; n],
            steer_angle: vec![0.0; n],
            gear: 1,
            shift_timer: 0.0,
            rpm: spec.engine.idle_rpm,
            drive_cut: 0.0,
            impulse: 0.0,
            wheel_centres: vec![[0.0; 3]; n],
            chassis_contacts: 0,
            lit: [false; 4],
            brake_scale: vec![1.0; n],
            extra_brake: vec![0.0; n],
            forward: [1.0, 0.0, 0.0],
        }
    }
}

/// One vehicle and its ground.
#[derive(Clone)]
pub struct Sim {
    spec: VehicleSpec,
    constants: Constants,
    state: State,
    ground: Ground,
    controls: Controls,
}

impl Sim {
    pub fn new(spec: VehicleSpec) -> Result<Self, String> {
        spec.validate()?;
        let constants = Constants::new(&spec);
        let state = State::new(&spec);
        Ok(Self {
            spec,
            constants,
            state,
            ground: Ground::default(),
            controls: Controls::default(),
        })
    }

    pub fn spec(&self) -> &VehicleSpec {
        &self.spec
    }

    pub fn set_ground(&mut self, ground: Ground) {
        self.ground = ground;
    }

    pub fn ground(&self) -> &Ground {
        &self.ground
    }

    pub fn set_controls(&mut self, controls: Controls) -> Result<(), String> {
        controls.validate()?;
        self.controls = controls;
        Ok(())
    }

    pub fn controls(&self) -> Controls {
        self.controls
    }

    /// Put the car at rest at `(x, y, z)` facing `yaw`, its wheels on the level
    /// nearest `y`. Non-finite input is rejected before the state changes.
    pub fn place(&mut self, x: f64, y: f64, z: f64, yaw: f64) -> Result<(), String> {
        if !(x.is_finite() && y.is_finite() && z.is_finite() && yaw.is_finite()) {
            return Err("place coordinates and yaw must be finite".into());
        }
        let rotation = [0.0, (yaw / 2.0).sin(), 0.0, (yaw / 2.0).cos()];
        let (ax, ay, az) = rotation_columns(rotation);
        let n = self.spec.wheels.len();
        let mut heights = vec![0.0; n];
        let mut body_y = f64::NEG_INFINITY;
        for (index, wheel) in self.spec.wheels.iter().enumerate() {
            let mount = [
                ax[0] * wheel.mount[0] + ay[0] * wheel.mount[1] + az[0] * wheel.mount[2],
                ax[1] * wheel.mount[0] + ay[1] * wheel.mount[1] + az[1] * wheel.mount[2],
                ax[2] * wheel.mount[0] + ay[2] * wheel.mount[1] + az[2] * wheel.mount[2],
            ];
            let height = self
                .ground
                .height_if_present(x + mount[0], z + mount[2], y)
                .ok_or_else(|| format!("no registered ground under wheel {index}"))?;
            heights[index] = height;
            let rest = self.spec.mass * G / n as f64 / wheel.spring.max(1.0);
            let reach = wheel.radius + wheel.travel - rest.min(wheel.travel);
            body_y = body_y.max(height + reach - mount[1]);
        }
        self.state.position = [x, body_y, z];
        self.state.velocity = [0.0; 3];
        self.state.angular = [0.0; 3];
        self.state.rotation = rotation;
        self.state.ground_y = heights;
        self.state.load.fill(0.0);
        self.state.compression.fill(0.0);
        self.state.omega.fill(0.0);
        self.state.wheel_turn.fill(0.0);
        self.state.slip.fill([0.0; 2]);
        self.state.steer_angle.fill(0.0);
        self.state.gear = 1;
        self.state.shift_timer = 0.0;
        self.state.rpm = self.spec.engine.idle_rpm;
        self.state.drive_cut = 0.0;
        self.state.impulse = 0.0;
        self.state.chassis_contacts = 0;
        for (index, wheel) in self.spec.wheels.iter().enumerate() {
            self.state.wheel_centres[index] = [
                x + ax[0] * wheel.mount[0] + az[0] * wheel.mount[2],
                self.state.ground_y[index] + wheel.radius,
                z + ax[2] * wheel.mount[0] + az[2] * wheel.mount[2],
            ];
        }
        self.clear_assists();
        self.state.forward = ax;
        Ok(())
    }

    /// Advance one fixed 1/120 s step with the last controls.
    pub fn step(&mut self) {
        step_inner(
            &self.spec,
            &self.constants,
            &mut self.state,
            &self.ground,
            self.controls,
        );
    }

    /// Run `substeps` fixed steps. Zero or an unbounded count is rejected.
    pub fn step_many(&mut self, substeps: u32) -> Result<u32, String> {
        if substeps == 0 || substeps > 4096 {
            return Err("substeps must be between 1 and 4096".into());
        }
        for _ in 0..substeps {
            self.step();
        }
        Ok(substeps)
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            position: self.state.position,
            velocity: self.state.velocity,
            angular: self.state.angular,
            rotation: self.state.rotation,
            load: self.state.load.clone(),
            compression: self.state.compression.clone(),
            ground_y: self.state.ground_y.clone(),
            omega: self.state.omega.clone(),
            wheel_turn: self.state.wheel_turn.clone(),
            slip: self.state.slip.clone(),
            steer_angle: self.state.steer_angle.clone(),
            gear: self.state.gear,
            shift_timer: self.state.shift_timer,
            rpm: self.state.rpm,
            impulse: self.state.impulse,
            wheel_centres: self.state.wheel_centres.clone(),
            chassis_contacts: self.state.chassis_contacts,
            drive_cut: self.state.drive_cut,
            brake_scale: self.state.brake_scale.clone(),
            lit: self.state.lit,
        }
    }

    /// The body's position and quaternion, for pose publication.
    pub fn pose(&self) -> (V3, Quaternion) {
        (self.state.position, self.state.rotation)
    }

    pub fn velocity(&self) -> V3 {
        self.state.velocity
    }

    pub fn angular(&self) -> V3 {
        self.state.angular
    }

    pub(super) fn set_velocity(&mut self, velocity: V3) {
        self.state.velocity = velocity;
    }

    pub(super) fn rebase(&mut self, delta: V3) {
        self.state.position = add(self.state.position, scale(delta, -1.0));
        for height in &mut self.state.ground_y {
            *height -= delta[1];
        }
        for centre in &mut self.state.wheel_centres {
            *centre = add(*centre, scale(delta, -1.0));
        }
    }

    pub(super) fn set_assists(
        &mut self,
        drive_cut: f64,
        brake_scale: &[f64],
        extra_brake: &[f64],
        lit: [bool; 4],
    ) {
        self.state.drive_cut = drive_cut;
        self.state.lit = lit;
        self.state.brake_scale.copy_from_slice(brake_scale);
        self.state.extra_brake.copy_from_slice(extra_brake);
    }

    pub(super) fn clear_assists(&mut self) {
        self.state.drive_cut = 0.0;
        self.state.lit = [false; 4];
        self.state.brake_scale.fill(1.0);
        self.state.extra_brake.fill(0.0);
    }

    pub(super) fn loaded_wheels(&self) -> usize {
        self.state.load.iter().filter(|load| **load > 0.0).count()
    }

    pub(super) fn chassis_contacts(&self) -> (u32, f64) {
        (self.state.chassis_contacts, self.state.impulse)
    }

    pub(super) fn body(&self) -> Body {
        Body {
            pos: self.state.position,
            vel: self.state.velocity,
            ang: self.state.angular,
            inv_mass: 1.0 / self.spec.mass,
            inv: inverse_inertia(rotation_columns(self.state.rotation), self.spec.inertia),
            impulse: self.state.impulse,
            contacts: self.state.chassis_contacts,
        }
    }

    pub(super) fn set_body(&mut self, body: Body) {
        self.state.position = body.pos;
        self.state.velocity = body.vel;
        self.state.angular = body.ang;
        self.state.impulse = body.impulse;
    }

    pub(super) fn body_bounds(&self) -> ([V3; 3], [f64; 2]) {
        let (x, y, z) = rotation_columns(self.state.rotation);
        (
            [x, y, z],
            [self.constants.lo, self.constants.lo + self.spec.size[1]],
        )
    }
}

fn step_inner(
    spec: &VehicleSpec,
    k: &Constants,
    state: &mut State,
    ground: &Ground,
    controls: Controls,
) {
    let dt = DT;
    let n = spec.wheels.len();
    let pos = state.position;
    let vel = state.velocity;
    let ang = state.angular;
    let (ax, ay, az) = rotation_columns(state.rotation);
    state.forward = ax;
    let near = state.ground_y.clone();

    let mounts: Vec<V3> = spec
        .wheels
        .iter()
        .map(|wheel| {
            [
                pos[0] + ax[0] * wheel.mount[0] + ay[0] * wheel.mount[1] + az[0] * wheel.mount[2],
                pos[1] + ax[1] * wheel.mount[0] + ay[1] * wheel.mount[1] + az[1] * wheel.mount[2],
                pos[2] + ax[2] * wheel.mount[0] + ay[2] * wheel.mount[1] + az[2] * wheel.mount[2],
            ]
        })
        .collect();

    // suspension.contact: every ground read for one step
    let mut heights = vec![0.0; n];
    let mut dist = vec![0.0; n];
    let mut normal = vec![[0.0, 1.0, 0.0]; n];
    let tiled = matches!(ground, Ground::Tiles { .. });
    let mut tile_grip = vec![0.0; n];
    let tilt = ay[1].max(0.2);
    let upright = ay[1] > 0.3;
    for (index, wheel) in spec.wheels.iter().enumerate() {
        let mount = mounts[index];
        if tiled {
            let mut best: Option<(f64, _)> = None;
            for along in SAMPLES {
                let x = mount[0] + ax[0] * wheel.radius * along;
                let z = mount[2] + ax[2] * wheel.radius * along;
                if let Some(surface) = ground.surface(x, z, near[index]) {
                    let centre = surface.height + wheel.radius * (1.0 - along * along).sqrt();
                    if best.is_none_or(|(height, _)| centre > height) {
                        best = Some((centre, surface));
                    }
                }
            }
            if let Some((centre, surface)) = best {
                dist[index] = (mount[1] - centre) / tilt;
                heights[index] = surface.height;
                normal[index] = surface.normal;
                tile_grip[index] = surface.grip;
            } else {
                heights[index] = near[index];
                dist[index] = wheel.travel;
            }
            continue;
        }
        let mut samples = [0.0; 3];
        let mut best = f64::NEG_INFINITY;
        let mut best_sample = 0usize;
        for (sample_index, along) in SAMPLES.iter().enumerate() {
            let x = mount[0] + ax[0] * wheel.radius * along;
            let z = mount[2] + ax[2] * wheel.radius * along;
            samples[sample_index] = ground.height(x, z, near[index]);
            let centre =
                samples[sample_index] + wheel.radius * (1.0 - along * along).max(0.0).sqrt();
            if centre > best {
                best = centre;
                best_sample = sample_index;
            }
        }
        heights[index] = samples[best_sample];
        dist[index] = (mount[1] - best) / tilt;
        let h0 = samples[1];
        let dx = -(ground.height(mount[0] + NORMAL_STEP, mount[2], near[index]) - h0) / NORMAL_STEP;
        let dz = -(ground.height(mount[0], mount[2] + NORMAL_STEP, near[index]) - h0) / NORMAL_STEP;
        let length = (dx * dx + 1.0 + dz * dz).sqrt();
        normal[index] = [dx / length, 1.0 / length, dz / length];
    }
    let mut compression = vec![0.0; n];
    for (index, wheel) in spec.wheels.iter().enumerate() {
        compression[index] = if upright {
            (wheel.travel - dist[index]).max(0.0)
        } else {
            0.0
        };
    }
    let previous = state.compression.clone();
    let mut load = vec![0.0; n];
    for (index, wheel) in spec.wheels.iter().enumerate() {
        if compression[index] > 0.0 {
            let rate =
                ((compression[index] - previous[index]) / dt).clamp(-DAMPER_SPEED, DAMPER_SPEED);
            let force = wheel.spring * compression[index]
                + wheel.spring * BUMP_STOP * (compression[index] - wheel.travel).max(0.0)
                + wheel.damper * rate
                + k.arb[index] * (compression[index] - compression[k.pair[index]]);
            load[index] = force.max(0.0);
        }
    }
    state.ground_y = heights;
    state.compression = compression;
    state.load = load.clone();

    let down = [-ay[0], -ay[1], -ay[2]];
    let mut contact = vec![[0.0; 3]; n];
    for index in 0..n {
        let wheel = &spec.wheels[index];
        state.wheel_centres[index] = [
            mounts[index][0] + down[0] * dist[index],
            mounts[index][1] + down[1] * dist[index],
            mounts[index][2] + down[2] * dist[index],
        ];
        contact[index] = [
            mounts[index][0] + down[0] * dist[index] - normal[index][0] * wheel.radius,
            mounts[index][1] + down[1] * dist[index] - normal[index][1] * wheel.radius,
            mounts[index][2] + down[2] * dist[index] - normal[index][2] * wheel.radius,
        ];
    }

    let steer = ackermann(spec, k, controls.steer);
    state.steer_angle = steer.clone();
    let mu_ground = if tiled {
        tile_grip
    } else {
        contact.iter().map(|p| ground.grip(p[0], p[2])).collect()
    };
    let mut heading = vec![[1.0, 0.0, 0.0]; n];
    let mut side = vec![[0.0, 0.0, 1.0]; n];
    let mut v_long = vec![0.0; n];
    let mut v_lat = vec![0.0; n];
    let mut mu = vec![0.0; n];
    for index in 0..n {
        let (cs, sn) = (steer[index].cos(), steer[index].sin());
        let nn = normal[index];
        let mut hd = [
            ax[0] * cs + az[0] * sn,
            ax[1] * cs + az[1] * sn,
            ax[2] * cs + az[2] * sn,
        ];
        let dn = dot(hd, nn);
        hd = [hd[0] - dn * nn[0], hd[1] - dn * nn[1], hd[2] - dn * nn[2]];
        let length = dot(hd, hd).sqrt().max(1e-9);
        hd = [hd[0] / length, hd[1] / length, hd[2] / length];
        let sd = cross(hd, nn);
        let lever = sub(contact[index], pos);
        let point_velocity = add(vel, cross(ang, lever));
        heading[index] = hd;
        side[index] = sd;
        v_long[index] = dot(point_velocity, hd);
        v_lat[index] = dot(point_velocity, sd);
        mu[index] = spec.wheels[index].tyre.mu * mu_ground[index];
    }

    let (drive, brakes) = torques(spec, k, state, &controls);
    let mut omega = state.omega.clone();
    let mut turn = state.wheel_turn.clone();
    let slip = state.slip.clone();
    let mut next_slip = vec![[0.0; 2]; n];
    let mut tyre_force = vec![[0.0; 3]; n];
    for index in 0..n {
        let wheel = &spec.wheels[index];
        let radius = wheel.radius;
        let tyre = wheel.tyre;
        let grip = mu[index] * load[index];
        let on = load[index] > 0.0;
        let den = 1.0 + dt * v_long[index].abs() / RELAX;
        let a = (slip[index][0] - dt * v_long[index] / RELAX) / den;
        let b = dt * radius / RELAX / den;
        let s0 = a + b * omega[index];
        let fx0 = grip * magic(s0, tyre.b_long, tyre.c_long, tyre.e_long);
        let h = 1e-4;
        let slope = (magic(s0 + h, tyre.b_long, tyre.c_long, tyre.e_long)
            - magic(s0 - h, tyre.b_long, tyre.c_long, tyre.e_long))
            / (2.0 * h);
        let stiffness = grip * slope.max(0.0);
        let effective = wheel.inertia + if on { dt * radius * stiffness * b } else { 0.0 };
        let free =
            omega[index] + dt * (drive[index] - if on { radius * fx0 } else { 0.0 }) / effective;
        let stop = dt * brakes[index] / effective;
        let next_omega = if free.abs() <= stop {
            0.0
        } else {
            free - sign(free) * stop
        };
        omega[index] = next_omega;
        turn[index] += next_omega * dt;
        if !on {
            next_slip[index] = [0.0, 0.0];
            continue;
        }
        let s_long = a + b * next_omega;
        let s_lat = (slip[index][1] + dt * v_lat[index] / RELAX) / den;
        next_slip[index] = [s_long, s_lat];
        let kn = s_long / k.kappa_peak[index];
        let an = s_lat / k.alpha_peak[index];
        let rho = (kn * kn + an * an).sqrt();
        let (mut fx, mut fy) = if rho > 1e-12 {
            (
                magic(
                    rho * k.kappa_peak[index],
                    tyre.b_long,
                    tyre.c_long,
                    tyre.e_long,
                ) * kn
                    / rho
                    * grip,
                -magic(
                    rho * k.alpha_peak[index],
                    tyre.b_lat,
                    tyre.c_lat,
                    tyre.e_lat,
                ) * an
                    / rho
                    * grip,
            )
        } else {
            (0.0, 0.0)
        };
        let fade = (1.0 - v_long[index].abs() / V_MIN).clamp(0.0, 1.0);
        let stiff = grip * tyre.b_lat * tyre.c_lat / RELAX;
        let damping = 2.0 * LOW_SPEED_DAMPING * (stiff * load[index] / G).max(0.0).sqrt() * fade;
        fx += damping * (next_omega * radius - v_long[index]);
        fy -= damping * v_lat[index];
        let total = (fx * fx + fy * fy).sqrt();
        if total > grip {
            let cap = grip / total.max(1e-9);
            fx *= cap;
            fy *= cap;
        }
        tyre_force[index] = [
            fx * heading[index][0] + fy * side[index][0],
            fx * heading[index][1] + fy * side[index][1],
            fx * heading[index][2] + fy * side[index][2],
        ];
    }
    state.omega = omega;
    state.wheel_turn = turn;
    state.slip = next_slip;

    let mut force = [0.0; 3];
    let mut torque = [0.0; 3];
    for index in 0..n {
        let push = scale(normal[index], load[index]);
        let total = add(push, tyre_force[index]);
        force = add(force, total);
        torque = add(torque, cross(sub(contact[index], pos), total));
    }
    let speed = dot(vel, vel).sqrt();
    let drag = -0.5 * AIR * spec.cd * spec.area * speed;
    let total = [
        force[0] + drag * vel[0],
        force[1] + drag * vel[1] - spec.mass * G,
        force[2] + drag * vel[2],
    ];
    integrate(spec, k, state, total, torque, ground);
}

fn torques(
    spec: &VehicleSpec,
    k: &Constants,
    state: &mut State,
    controls: &Controls,
) -> (Vec<f64>, Vec<f64>) {
    let dt = DT;
    let n = spec.wheels.len();
    let throttle = controls.throttle;
    let brake = controls.brake;
    let hand = controls.handbrake;
    let v = dot(state.velocity, state.forward);
    let slow = v.abs() < REVERSE_BELOW;
    let mut gear = state.gear;
    let mut back = gear < 0;
    if !back && slow && brake > 0.0 && throttle == 0.0 {
        gear = -1;
    }
    if ((back && slow) || gear == 0) && throttle > 0.0 && brake == 0.0 {
        gear = 1;
    }
    back = gear < 0;
    let neutral = gear == 0;
    let (pedal, stop) = if back {
        (brake, throttle)
    } else {
        (throttle, brake)
    };

    let g = (gear.max(1) - 1) as usize;
    let ratio = (if back {
        -spec.gearbox.reverse
    } else {
        spec.gearbox.ratios[g]
    }) * spec.gearbox.final_drive;
    let spin: f64 = state
        .omega
        .iter()
        .zip(spec.wheels.iter())
        .filter(|(_, wheel)| wheel.driven)
        .map(|(omega, _)| *omega)
        .sum();
    let wheel_rpm = (spin / k.n_driven * ratio).abs() * RPM;
    let engine = &spec.engine;
    let rpm = wheel_rpm.max(if pedal > 0.0 {
        CLUTCH_RPM
    } else {
        engine.idle_rpm
    });
    let road_rpm = (v / k.driven_radius.max(1e-3) * ratio).abs() * RPM;

    let mut timer = (state.shift_timer - dt).max(0.0);
    let up_rpm = engine.peak_power_rpm * (LIGHT_UP + (UP_AT - LIGHT_UP) * pedal * pedal);
    let ready = timer == 0.0 && !back;
    let up = ready && !neutral && road_rpm > up_rpm && gear < spec.gearbox.ratios.len() as i32;
    let lower = spec.gearbox.ratios[(g as i32 - 1).max(0) as usize];
    let down_rpm = road_rpm * lower / spec.gearbox.ratios[g].max(1e-9);
    let down = ready && !up && gear > 1 && down_rpm < DOWN_AT * up_rpm;
    gear += up as i32 - down as i32;
    if up || down {
        timer = spec.gearbox.shift_time;
    }
    state.gear = gear;
    state.shift_timer = timer;
    state.rpm = rpm;

    let top = engine.peak_power_rpm * REDLINE;
    let r = rpm.clamp(engine.idle_rpm, top);
    let (x0, x1, y0, y1) = if r >= engine.peak_power_rpm {
        (engine.peak_power_rpm, top, engine.power_torque, 0.0)
    } else if r >= engine.peak_torque_rpm {
        (
            engine.peak_torque_rpm,
            engine.peak_power_rpm,
            engine.peak_torque,
            engine.power_torque,
        )
    } else {
        (
            engine.idle_rpm,
            engine.peak_torque_rpm,
            engine.idle_torque,
            engine.peak_torque,
        )
    };
    let curve = y0 + (y1 - y0) * (r - x0) / (x1 - x0).max(1e-9);
    let off = (1.0 - pedal / OFF_PEDAL).clamp(0.0, 1.0);
    let mut engine_torque = (curve * pedal
        - engine.engine_brake * (rpm - engine.idle_rpm).max(0.0) * off)
        * (1.0 - state.drive_cut);
    if timer > 0.0 || neutral {
        engine_torque = if neutral { 0.0 } else { engine_torque.min(0.0) };
    }
    if engine_torque > 0.0 {
        engine_torque *= spec.gearbox.efficiency;
    }
    let per = engine_torque * ratio / k.n_driven;
    let mut drive = vec![0.0; n];
    for (index, wheel) in spec.wheels.iter().enumerate() {
        if wheel.driven {
            drive[index] = per;
        }
    }
    let handbrake = hand * spec.brake_torque / k.n_hand;
    let pedal_brake = stop * spec.brake_torque;
    let mut brakes = vec![0.0; n];
    for (index, wheel) in spec.wheels.iter().enumerate() {
        brakes[index] = pedal_brake * wheel.brake_share * state.brake_scale[index]
            + if wheel.handbrake { handbrake } else { 0.0 }
            + spec.rolling * state.load[index] * wheel.radius
            + state.extra_brake[index];
    }
    (drive, brakes)
}

fn integrate(
    spec: &VehicleSpec,
    k: &Constants,
    state: &mut State,
    force: V3,
    torque: V3,
    ground: &Ground,
) {
    let dt = DT;
    let mass = spec.mass;
    let velocity = [
        state.velocity[0] + force[0] / mass * dt,
        state.velocity[1] + force[1] / mass * dt,
        state.velocity[2] + force[2] / mass * dt,
    ];
    let (ax, ay, az) = rotation_columns(state.rotation);
    let inertia = spec.inertia;
    let mut body = [
        dot(ax, state.angular),
        dot(ay, state.angular),
        dot(az, state.angular),
    ];
    let body_torque = [dot(ax, torque), dot(ay, torque), dot(az, torque)];
    let gyro = cross(
        body,
        [
            inertia[0] * body[0],
            inertia[1] * body[1],
            inertia[2] * body[2],
        ],
    );
    for index in 0..3 {
        body[index] += dt * (body_torque[index] - gyro[index]) / inertia[index];
    }
    let angular = [
        ax[0] * body[0] + ay[0] * body[1] + az[0] * body[2],
        ax[1] * body[0] + ay[1] * body[1] + az[1] * body[2],
        ax[2] * body[0] + ay[2] * body[1] + az[2] * body[2],
    ];

    let [x, y, z, w] = state.rotation;
    let [wx, wy, wz] = angular;
    let hd = 0.5 * dt;
    let mut q = [
        x + hd * (wx * w + wy * z - wz * y),
        y + hd * (wy * w + wz * x - wx * z),
        z + hd * (wz * w + wx * y - wy * x),
        w - hd * (wx * x + wy * y + wz * z),
    ];
    let length = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    for value in &mut q {
        *value /= length;
    }
    let (ax, ay, az) = rotation_columns(q);
    let inv = inverse_inertia((ax, ay, az), inertia);
    let mut body = Body {
        pos: state.position,
        vel: velocity,
        ang: angular,
        inv_mass: 1.0 / mass,
        inv,
        impulse: 0.0,
        contacts: 0,
    };
    let speed = dot(velocity, velocity).sqrt();
    let substeps = ((speed * dt / STEP).ceil() as usize).clamp(1, 64);
    let loose = ay[1] < 0.87
        || spec.wheels.iter().enumerate().any(|(index, wheel)| {
            state.load[index] <= 0.0 || state.compression[index] >= wheel.travel
        });
    for _ in 0..substeps {
        let fraction = dt / substeps as f64;
        body.pos = [
            body.pos[0] + body.vel[0] * fraction,
            body.pos[1] + body.vel[1] * fraction,
            body.pos[2] + body.vel[2] * fraction,
        ];
        if loose {
            body_on_ground(&mut body, spec, k, q, ground);
        }
    }
    state.position = body.pos;
    state.velocity = body.vel;
    state.angular = body.ang;
    state.rotation = q;
    state.impulse = body.impulse;
    state.chassis_contacts = body.contacts;
}

fn inverse_inertia((ax, ay, az): (V3, V3, V3), inertia: V3) -> [[f64; 3]; 3] {
    [
        [
            ax[0] * ax[0] / inertia[0] + ay[0] * ay[0] / inertia[1] + az[0] * az[0] / inertia[2],
            ax[0] * ax[1] / inertia[0] + ay[0] * ay[1] / inertia[1] + az[0] * az[1] / inertia[2],
            ax[0] * ax[2] / inertia[0] + ay[0] * ay[2] / inertia[1] + az[0] * az[2] / inertia[2],
        ],
        [
            ax[1] * ax[0] / inertia[0] + ay[1] * ay[0] / inertia[1] + az[1] * az[0] / inertia[2],
            ax[1] * ax[1] / inertia[0] + ay[1] * ay[1] / inertia[1] + az[1] * az[1] / inertia[2],
            ax[1] * ax[2] / inertia[0] + ay[1] * ay[2] / inertia[1] + az[1] * az[2] / inertia[2],
        ],
        [
            ax[2] * ax[0] / inertia[0] + ay[2] * ay[0] / inertia[1] + az[2] * az[0] / inertia[2],
            ax[2] * ax[1] / inertia[0] + ay[2] * ay[1] / inertia[1] + az[2] * az[1] / inertia[2],
            ax[2] * ax[2] / inertia[0] + ay[2] * ay[2] / inertia[1] + az[2] * az[2] / inertia[2],
        ],
    ]
}

pub(super) struct Body {
    pub pos: V3,
    pub vel: V3,
    pub ang: V3,
    pub inv_mass: f64,
    pub inv: [[f64; 3]; 3],
    pub impulse: f64,
    pub contacts: u32,
}

fn body_on_ground(
    body: &mut Body,
    spec: &VehicleSpec,
    k: &Constants,
    rotation: Quaternion,
    ground: &Ground,
) {
    let (ax, ay, az) = rotation_columns(rotation);
    let half = [spec.size[0] / 2.0, spec.size[2] / 2.0];
    let mut offsets = Vec::with_capacity(8);
    for sx in [-1.0, 1.0] {
        for sz in [-1.0, 1.0] {
            for y in [k.lo, k.lo + spec.size[1]] {
                let corner = [sx * half[0], y, sz * half[1]];
                offsets.push([
                    ax[0] * corner[0] + ay[0] * corner[1] + az[0] * corner[2],
                    ax[1] * corner[0] + ay[1] * corner[1] + az[1] * corner[2],
                    ax[2] * corner[0] + ay[2] * corner[1] + az[2] * corner[2],
                ]);
            }
        }
    }
    let mut heights = Vec::with_capacity(offsets.len());
    let mut touching = false;
    for offset in &offsets {
        let height = ground
            .height_if_present(
                body.pos[0] + offset[0],
                body.pos[2] + offset[2],
                body.pos[1] + offset[1],
            )
            .unwrap_or(f64::NEG_INFINITY);
        heights.push(height);
        if height > body.pos[1] + offset[1] {
            touching = true;
        }
    }
    if !touching {
        return;
    }
    for _ in 0..2 {
        for (offset, height) in offsets.iter().zip(heights.iter()) {
            let depth = height - (body.pos[1] + offset[1]);
            if depth > 0.0 {
                apply(body, *offset, [0.0, 1.0, 0.0], depth);
            }
        }
    }
}

fn apply(body: &mut Body, lever: V3, normal: V3, depth: f64) {
    let point_velocity = add(body.vel, cross(body.ang, lever));
    let normal_speed = dot(point_velocity, normal);
    if normal_speed < 0.0 {
        let lever_normal = cross(lever, normal);
        let normal_mass =
            body.inv_mass + dot(normal, cross(multiply(body.inv, lever_normal), lever));
        let bounce = if normal_speed < -RESTING {
            RESTITUTION
        } else {
            0.0
        };
        let impulse_normal = -(1.0 + bounce) * normal_speed / normal_mass;
        let tangent = [
            point_velocity[0] - normal_speed * normal[0],
            point_velocity[1] - normal_speed * normal[1],
            point_velocity[2] - normal_speed * normal[2],
        ];
        let speed = dot(tangent, tangent).sqrt();
        let tangent = scale(tangent, 1.0 / speed.max(1e-9));
        let lever_tangent = cross(lever, tangent);
        let tangent_mass =
            body.inv_mass + dot(tangent, cross(multiply(body.inv, lever_tangent), lever));
        let impulse_tangent = (speed / tangent_mass).min(FRICTION * impulse_normal);
        let impulse = [
            impulse_normal * normal[0] - impulse_tangent * tangent[0],
            impulse_normal * normal[1] - impulse_tangent * tangent[1],
            impulse_normal * normal[2] - impulse_tangent * tangent[2],
        ];
        body.vel = add(body.vel, scale(impulse, body.inv_mass));
        body.ang = add(body.ang, multiply(body.inv, cross(lever, impulse)));
        body.impulse += dot(impulse, impulse).sqrt();
        body.contacts += 1;
    }
    let push = (depth - SLOP).max(0.0) * PUSH;
    body.pos = add(body.pos, scale(normal, push));
}

fn ackermann(spec: &VehicleSpec, k: &Constants, steer: f64) -> Vec<f64> {
    let n = spec.wheels.len();
    let outer = steer.abs() * spec.max_steer;
    if steer.abs() <= 1e-6 {
        return vec![0.0; n];
    }
    let radius = k.ack_base / outer.max(1e-6).tan() - k.ack_track / 2.0;
    let inner = (k.ack_base / (radius - k.ack_track / 2.0).max(1e-6)).atan();
    let side = sign(steer);
    (0..n)
        .map(|index| {
            let wheel = &spec.wheels[index];
            if !wheel.steered {
                0.0
            } else if k.zsign[index] == side {
                inner * side
            } else {
                outer * side
            }
        })
        .collect()
}

pub(super) fn rotation_columns(q: Quaternion) -> (V3, V3, V3) {
    let [x, y, z, w] = q;
    (
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ],
        [
            2.0 * (x * y - z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + x * w),
        ],
        [
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    )
}

pub(super) fn multiply(matrix: [[f64; 3]; 3], vector: V3) -> V3 {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

pub(super) fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(super) fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(super) fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(super) fn scale(a: V3, factor: f64) -> V3 {
    [a[0] * factor, a[1] * factor, a[2] * factor]
}

fn sign(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[cfg(test)]
pub(crate) fn test_car() -> VehicleSpec {
    use super::spec::{Engine, Gearbox, Wheel};
    let mass = 1200.0;
    let size = [4.4, 1.45, 1.75];
    let corner = mass * G / 4.0;
    let spring = corner / G * (2.0 * PI * 1.4) * (2.0 * PI * 1.4);
    let damper = 2.0 * 0.35 * (spring * corner / G).sqrt();
    let peak = 85_000.0 / (5500.0 * 2.0 * PI / 60.0);
    let wheel = |x: f64, z: f64, front: bool| {
        let mut wheel = Wheel::new([x, -0.15, z], 0.31, 0.2, spring, damper);
        wheel.driven = !front;
        wheel.steered = front;
        wheel.brake_share = if front { 0.3 } else { 0.2 };
        wheel.handbrake = !front;
        wheel
    };
    VehicleSpec {
        mass,
        inertia: [
            mass / 12.0 * (size[1] * size[1] + size[2] * size[2]),
            mass / 12.0 * (size[0] * size[0] + size[2] * size[2]),
            mass / 12.0 * (size[0] * size[0] + size[1] * size[1]),
        ],
        size,
        wheels: vec![
            wheel(1.15, -0.75, true),
            wheel(1.15, 0.75, true),
            wheel(-1.45, -0.75, false),
            wheel(-1.45, 0.75, false),
        ],
        engine: Engine {
            idle_rpm: 850.0,
            peak_torque_rpm: 3500.0,
            peak_power_rpm: 5500.0,
            idle_torque: 0.6 * peak,
            peak_torque: 1.3 * peak,
            power_torque: peak,
            engine_brake: 0.02,
        },
        gearbox: Gearbox {
            ratios: vec![3.6, 2.1, 1.4, 1.0, 0.8],
            final_drive: 3.9,
            reverse: 3.5,
            shift_time: 0.3,
            efficiency: 0.85,
        },
        arb: [9000.0, 4000.0],
        cd: 0.35,
        area: 2.0,
        rolling: 0.015,
        max_steer: 0.6,
        brake_torque: 5000.0,
        model_origin: [0.0, 0.0, 0.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vehicles::spec::Tyre;

    fn flat_sim() -> Sim {
        let mut sim = Sim::new(test_car()).unwrap();
        sim.set_ground(Ground::flat(0.0, 1.0).unwrap());
        sim.place(0.0, 0.0, 0.0, 0.0).unwrap();
        sim
    }

    #[test]
    fn rejects_invalid_wheel_inertia_before_creating_a_sim() {
        for inertia in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut spec = test_car();
            spec.wheels[2].inertia = inertia;
            assert_eq!(
                Sim::new(spec).err().as_deref(),
                Some("wheel 2 inertia must be finite and positive"),
                "inertia: {inertia}"
            );
        }
    }

    #[test]
    fn driven_airborne_wheels_spin_with_finite_state() {
        let mut sim = flat_sim();
        sim.set_ground(Ground::flat(-10.0, 1.0).unwrap());
        sim.set_controls(Controls::new(0.0, 1.0, 0.0, 0.0).unwrap())
            .unwrap();
        sim.step();

        let state = sim.snapshot();
        assert_eq!(state.load, vec![0.0; 4]);
        assert!(state
            .omega
            .iter()
            .chain(&state.wheel_turn)
            .all(|v| v.is_finite()));
        assert!(state.omega[2] > 0.0 && state.wheel_turn[2] > 0.0);
    }

    #[test]
    fn place_rejects_non_finite_inputs_before_mutating() {
        let mut sim = flat_sim();
        sim.step_many(12).unwrap();
        let before = format!("{:?}", sim.snapshot());
        let finite = [1.0, 0.5, -2.0, 0.25];
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for axis in 0..4 {
                let mut args = finite;
                args[axis] = bad;
                let result = sim.place(args[0], args[1], args[2], args[3]);
                assert!(result.is_err(), "place accepted {bad} at axis {axis}");
                assert_eq!(
                    format!("{:?}", sim.snapshot()),
                    before,
                    "place mutated state after rejecting {bad} at axis {axis}"
                );
            }
        }
        sim.place(4.0, 0.0, 0.0, 0.0).unwrap();
        assert!((sim.snapshot().position[0] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn parked_on_the_flat_settles_without_creeping() {
        let mut sim = flat_sim();
        sim.step_many(240).unwrap();
        let start = sim.snapshot();
        sim.step_many(600).unwrap();
        let end = sim.snapshot();
        let dx = end.position[0] - start.position[0];
        let dz = end.position[2] - start.position[2];
        assert!((dx * dx + dz * dz).sqrt() < 1e-3, "{dx} {dz}");
        let norm = end.rotation.iter().map(|v| v * v).sum::<f64>();
        assert!((norm - 1.0).abs() < 1e-9);
        for value in end.position.iter().chain(end.velocity.iter()) {
            assert!(value.is_finite());
        }
    }

    #[test]
    fn full_throttle_accelerates_and_changes_gear() {
        let mut sim = flat_sim();
        sim.set_controls(Controls::new(0.0, 1.0, 0.0, 0.0).unwrap())
            .unwrap();
        sim.step_many(480).unwrap();
        let state = sim.snapshot();
        assert!(state.velocity[0] > 10.0, "{}", state.velocity[0]);
        assert!(state.gear >= 2, "{}", state.gear);
        assert!(state.rpm > 800.0);
        assert!(state.position[0] > 10.0, "{}", state.position[0]);
    }

    #[test]
    fn brakes_to_rest_after_accelerating() {
        let mut sim = flat_sim();
        sim.set_controls(Controls::new(0.0, 1.0, 0.0, 0.0).unwrap())
            .unwrap();
        sim.step_many(480).unwrap();
        let speed = sim.snapshot().velocity[0];
        sim.set_controls(Controls::new(0.0, 0.0, 1.0, 0.0).unwrap())
            .unwrap();
        let mut slowest = speed;
        for _ in 0..360 {
            sim.step();
            slowest = slowest.min(sim.snapshot().velocity[0].abs());
        }
        let state = sim.snapshot();
        assert!(speed > 8.0);
        assert!(slowest < 0.2, "brakes never brought it to rest: {slowest}");
        assert!(state.impulse >= 0.0);
        assert!(state.velocity[0].is_finite());
    }

    #[test]
    fn steering_turns_the_car_and_stays_finite() {
        let mut sim = flat_sim();
        sim.set_controls(Controls::new(0.0, 1.0, 0.0, 0.0).unwrap())
            .unwrap();
        sim.step_many(240).unwrap();
        sim.set_controls(Controls::new(0.3, 1.0, 0.0, 0.0).unwrap())
            .unwrap();
        sim.step_many(240).unwrap();
        let state = sim.snapshot();
        assert!(state.position[2].abs() > 1.0, "{}", state.position[2]);
        assert!(state.angular[1].abs() > 0.05, "{}", state.angular[1]);
        assert!(state.rotation.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn coasting_stays_straight_and_side_slip_stops() {
        let mut sim = flat_sim();
        sim.place(0.0, 0.0, 0.0, 0.0).unwrap();
        sim.step_many(120).unwrap();
        sim.set_controls(Controls::new(0.0, 0.0, 0.0, 0.0).unwrap())
            .unwrap();
        sim.state.velocity = [20.0, 0.0, 0.0];
        sim.state
            .omega
            .iter_mut()
            .for_each(|omega| *omega = 20.0 / 0.31);
        while sim.snapshot().position[0] < 50.0 {
            sim.step_many(12).unwrap();
        }
        assert!(sim.snapshot().position[2].abs() < 0.05);

        let mut slide = flat_sim();
        slide.step_many(120).unwrap();
        slide.state.velocity = [0.0, 0.0, 3.0];
        slide.step_many(240).unwrap();
        assert!(slide.snapshot().velocity[2].abs() < 0.05);
        assert!(slide.snapshot().position[2] < 3.0);
    }

    #[test]
    fn rolls_down_a_slope_without_brakes() {
        let mut sim = Sim::new(test_car()).unwrap();
        sim.set_ground(
            Ground::profile(vec![(-200.0, 10.0), (200.0, -10.0)], vec![], 0.0, None, 1.0).unwrap(),
        );
        sim.place(0.0, 0.0, 0.0, 0.0).unwrap();
        sim.step_many(360).unwrap();
        assert!(
            sim.snapshot().velocity[0] > 0.5,
            "{}",
            sim.snapshot().velocity[0]
        );
    }

    #[test]
    fn a_bridge_is_selected_by_the_level_nearest_the_car() {
        let ground = Ground::profile(
            vec![(-100.0, 0.0), (100.0, 0.0)],
            vec![],
            0.0,
            Some((5.0, 15.0, 6.0)),
            1.0,
        )
        .unwrap();
        let mut above = Sim::new(test_car()).unwrap();
        above.set_ground(ground.clone());
        above.place(10.0, 6.0, 0.0, 0.0).unwrap();
        above.step_many(120).unwrap();
        assert!((above.snapshot().ground_y[0] - 6.0).abs() < 1e-9);
        assert!(above.snapshot().velocity.iter().map(|v| v * v).sum::<f64>() < 0.05 * 0.05);

        let mut below = Sim::new(test_car()).unwrap();
        below.set_ground(ground);
        below.place(10.0, 0.0, 0.0, 0.0).unwrap();
        below.step_many(120).unwrap();
        assert!(below.snapshot().ground_y[0].abs() < 1e-9);
        assert!(below.snapshot().velocity.iter().map(|v| v * v).sum::<f64>() < 0.05 * 0.05);
    }

    #[test]
    fn a_kerb_lifts_the_car_and_the_profile_can_be_driven() {
        let mut sim = Sim::new(test_car()).unwrap();
        sim.set_ground(
            Ground::profile(
                vec![
                    (-50.0, 0.0),
                    (20.0, 0.0),
                    (40.0, 1.5),
                    (60.0, 0.0),
                    (400.0, 0.0),
                ],
                vec![30.0, 70.0],
                0.15,
                Some((90.0, 110.0, 3.0)),
                1.0,
            )
            .unwrap(),
        );
        sim.place(0.0, 0.0, 0.0, 0.0).unwrap();
        sim.set_controls(Controls::new(0.0, 1.0, 0.0, 0.0).unwrap())
            .unwrap();
        sim.step_many(960).unwrap();
        let state = sim.snapshot();
        assert!(state.position[0] > 50.0, "{}", state.position[0]);
        assert!(state.rotation.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn replay_of_fixed_substeps_is_bit_identical() {
        let mut first = flat_sim();
        let mut second = flat_sim();
        first
            .set_controls(Controls::new(0.2, 0.8, 0.0, 0.0).unwrap())
            .unwrap();
        second
            .set_controls(Controls::new(0.2, 0.8, 0.0, 0.0).unwrap())
            .unwrap();
        first.step_many(240).unwrap();
        second.step_many(240).unwrap();
        let a = first.snapshot();
        let b = second.snapshot();
        assert_eq!(a.position, b.position);
        assert_eq!(a.velocity, b.velocity);
        assert_eq!(a.angular, b.angular);
        assert_eq!(a.rotation, b.rotation);
        assert_eq!(a.slip, b.slip);
        assert_eq!(a.gear, b.gear);
        assert_eq!(a.rpm, b.rpm);
        assert!(first.step_many(0).is_err());
        assert!(first.step_many(4097).is_err());
    }

    #[test]
    fn zero_grip_is_a_negative_control_for_acceleration() {
        let mut spec = test_car();
        for wheel in &mut spec.wheels {
            wheel.tyre = Tyre {
                mu: 0.0,
                ..Tyre::default()
            };
        }
        let mut sim = Sim::new(spec).unwrap();
        sim.set_ground(Ground::flat(0.0, 1.0).unwrap());
        sim.place(0.0, 0.0, 0.0, 0.0).unwrap();
        sim.set_controls(Controls::new(0.0, 1.0, 0.0, 0.0).unwrap())
            .unwrap();
        sim.step_many(480).unwrap();
        let speed = sim.snapshot().velocity[0];
        assert!(speed.abs() < 0.05, "zero grip still moved: {speed}");
    }
}
