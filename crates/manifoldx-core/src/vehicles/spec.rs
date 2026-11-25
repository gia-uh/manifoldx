//! What a vehicle is made of: body, wheels, tyres, engine and gearbox, as owned
//! f64 values. Axes are the car's own: x forward, y up, z to the right.

/// A simplified Pacejka tyre: stiffness, shape and curvature per direction,
/// plus its grip.
#[derive(Clone, Copy, Debug)]
pub struct Tyre {
    pub mu: f64,
    pub b_lat: f64,
    pub c_lat: f64,
    pub e_lat: f64,
    pub b_long: f64,
    pub c_long: f64,
    pub e_long: f64,
}

impl Default for Tyre {
    fn default() -> Self {
        Self {
            mu: 1.0,
            b_lat: 11.0,
            c_lat: 1.5,
            e_lat: 0.05,
            b_long: 18.0,
            c_long: 1.45,
            e_long: -0.3,
        }
    }
}

impl Tyre {
    pub fn peak_slips(&self) -> (f64, f64) {
        (
            peak_slip(self.b_long, self.c_long, self.e_long),
            peak_slip(self.b_lat, self.c_lat, self.e_lat),
        )
    }
}

/// A wheel hung from the body at `mount`: a ray of `travel + radius` straight
/// down the car's -y.
#[derive(Clone, Debug)]
pub struct Wheel {
    pub mount: [f64; 3],
    pub radius: f64,
    pub travel: f64,
    pub spring: f64,
    pub damper: f64,
    pub tyre: Tyre,
    pub driven: bool,
    pub steered: bool,
    pub brake_share: f64,
    pub handbrake: bool,
    pub inertia: f64,
}

impl Wheel {
    pub fn new(mount: [f64; 3], radius: f64, travel: f64, spring: f64, damper: f64) -> Self {
        Self {
            mount,
            radius,
            travel,
            spring,
            damper,
            tyre: Tyre::default(),
            driven: false,
            steered: false,
            brake_share: 0.25,
            handbrake: false,
            inertia: 1.2,
        }
    }
}

/// A torque curve through three points, in N·m at each rpm.
#[derive(Clone, Debug)]
pub struct Engine {
    pub idle_rpm: f64,
    pub peak_torque_rpm: f64,
    pub peak_power_rpm: f64,
    pub idle_torque: f64,
    pub peak_torque: f64,
    pub power_torque: f64,
    pub engine_brake: f64,
}

/// Forward ratios from first up, the final drive, reverse, and the shift cut.
#[derive(Clone, Debug)]
pub struct Gearbox {
    pub ratios: Vec<f64>,
    pub final_drive: f64,
    pub reverse: f64,
    pub shift_time: f64,
    pub efficiency: f64,
}

/// One vehicle. `size` is the body box (length, height, width); `inertia` is
/// diagonal in car axes.
#[derive(Clone, Debug)]
pub struct VehicleSpec {
    pub mass: f64,
    pub inertia: [f64; 3],
    pub size: [f64; 3],
    pub wheels: Vec<Wheel>,
    pub engine: Engine,
    pub gearbox: Gearbox,
    /// Front and rear anti-roll bar rates, N/m.
    pub arb: [f64; 2],
    pub cd: f64,
    pub area: f64,
    /// Rolling resistance per newton of load.
    pub rolling: f64,
    /// Radians of the outer front wheel at full lock.
    pub max_steer: f64,
    /// N·m over all wheels at full brake.
    pub brake_torque: f64,
    pub model_origin: [f64; 3],
}

impl VehicleSpec {
    /// Reject anything the solver cannot integrate before allocating state.
    pub fn validate(&self) -> Result<(), String> {
        if !(self.mass.is_finite() && self.mass > 0.0) {
            return Err("mass must be finite and positive".into());
        }
        for (name, value) in [
            ("inertia x", self.inertia[0]),
            ("inertia y", self.inertia[1]),
            ("inertia z", self.inertia[2]),
            ("size length", self.size[0]),
            ("size height", self.size[1]),
            ("size width", self.size[2]),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(format!("{name} must be finite and positive"));
            }
        }
        if self.wheels.is_empty() || self.wheels.len() > 16 {
            return Err("a vehicle needs between 1 and 16 wheels".into());
        }
        let share: f64 = self.wheels.iter().map(|wheel| wheel.brake_share).sum();
        if (share - 1.0).abs() > 1e-6 {
            return Err(format!("brake_share sums to {share:.3}, not 1"));
        }
        for (index, wheel) in self.wheels.iter().enumerate() {
            for value in wheel.mount {
                if !value.is_finite() {
                    return Err(format!("wheel {index} mount must be finite"));
                }
            }
            for (name, value) in [("radius", wheel.radius), ("inertia", wheel.inertia)] {
                if !(value.is_finite() && value > 0.0) {
                    return Err(format!("wheel {index} {name} must be finite and positive"));
                }
            }
            for (name, value) in [
                ("travel", wheel.travel),
                ("damper", wheel.damper),
                ("brake_share", wheel.brake_share),
                ("tyre mu", wheel.tyre.mu),
                ("tyre b_lat", wheel.tyre.b_lat),
                ("tyre c_lat", wheel.tyre.c_lat),
                ("tyre b_long", wheel.tyre.b_long),
                ("tyre c_long", wheel.tyre.c_long),
            ] {
                if !(value.is_finite() && value >= 0.0) {
                    return Err(format!(
                        "wheel {index} {name} must be finite and non-negative"
                    ));
                }
            }
            for (name, value) in [
                ("tyre e_lat", wheel.tyre.e_lat),
                ("tyre e_long", wheel.tyre.e_long),
            ] {
                if !value.is_finite() {
                    return Err(format!("wheel {index} {name} must be finite"));
                }
            }
            if !(wheel.spring.is_finite() && wheel.spring > 0.0) {
                return Err(format!("wheel {index} spring must be finite and positive"));
            }
            if wheel.brake_share > 1.0 {
                return Err(format!("wheel {index} brake_share must be at most 1"));
            }
        }
        let engine = &self.engine;
        for (name, value) in [
            ("idle_rpm", engine.idle_rpm),
            ("peak_torque_rpm", engine.peak_torque_rpm),
            ("peak_power_rpm", engine.peak_power_rpm),
            ("idle_torque", engine.idle_torque),
            ("peak_torque", engine.peak_torque),
            ("power_torque", engine.power_torque),
            ("engine_brake", engine.engine_brake),
        ] {
            if !(value.is_finite() && value >= 0.0) {
                return Err(format!("engine {name} must be finite and non-negative"));
            }
        }
        if engine.idle_rpm >= engine.peak_torque_rpm
            || engine.peak_torque_rpm >= engine.peak_power_rpm
        {
            return Err("engine rpm points must increase".into());
        }
        let gearbox = &self.gearbox;
        if gearbox.ratios.is_empty() || gearbox.ratios.len() > 16 {
            return Err("gearbox needs between 1 and 16 forward ratios".into());
        }
        for (index, ratio) in gearbox.ratios.iter().enumerate() {
            if !(ratio.is_finite() && *ratio > 0.0) {
                return Err(format!("gear ratio {index} must be finite and positive"));
            }
        }
        for (name, value) in [
            ("final drive", gearbox.final_drive),
            ("reverse", gearbox.reverse),
            ("shift time", gearbox.shift_time),
            ("max steer", self.max_steer),
            ("brake torque", self.brake_torque),
            ("cd", self.cd),
            ("area", self.area),
            ("rolling", self.rolling),
        ] {
            if !(value.is_finite() && value >= 0.0) {
                return Err(format!("{name} must be finite and non-negative"));
            }
        }
        if gearbox.final_drive <= 0.0 {
            return Err("final drive must be positive".into());
        }
        if !(gearbox.efficiency.is_finite()
            && gearbox.efficiency > 0.0
            && gearbox.efficiency <= 1.0)
        {
            return Err("gearbox efficiency must be finite, positive and at most 1".into());
        }
        for (name, value) in [("arb front", self.arb[0]), ("arb rear", self.arb[1])] {
            if !(value.is_finite() && value >= 0.0) {
                return Err(format!("{name} must be finite and non-negative"));
            }
        }
        if self.max_steer <= 0.0 {
            return Err("max steer must be positive".into());
        }
        for value in self.model_origin {
            if !value.is_finite() {
                return Err("model origin must be finite".into());
            }
        }
        Ok(())
    }
}

/// `sin(c·atan(b·s − e·(b·s − atan(b·s))))`: the share of `mu * load` a tyre
/// gives at slip `s`, odd in `s`.
pub fn magic(s: f64, b: f64, c: f64, e: f64) -> f64 {
    let bs = b * s;
    (c * (bs - e * (bs - bs.atan())).atan()).sin()
}

/// The slip where `magic` peaks, on the same 2001-point grid the Python model
/// uses.
pub fn peak_slip(b: f64, c: f64, e: f64) -> f64 {
    let mut best = f64::NEG_INFINITY;
    let mut at = 0.0;
    for index in 0..=2000 {
        let s = index as f64 / 2000.0;
        let value = magic(s, b, c, e);
        if value > best {
            best = value;
            at = s;
        }
    }
    at
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tyre_peaks_between_the_documented_slips() {
        let tyre = Tyre::default();
        let (long, lat) = tyre.peak_slips();
        assert!((0.08..=0.15).contains(&long), "{long}");
        assert!((0.12..=0.20).contains(&lat), "{lat}");
    }

    #[test]
    fn validation_rejects_bad_brake_shares_before_state_exists() {
        let mut wheel = Wheel::new([1.0, 0.0, 0.0], 0.3, 0.2, 20_000.0, 1_000.0);
        wheel.brake_share = 0.5;
        let spec = VehicleSpec {
            mass: 1_000.0,
            inertia: [1.0, 1.0, 1.0],
            size: [4.0, 1.0, 2.0],
            wheels: vec![wheel],
            engine: Engine {
                idle_rpm: 800.0,
                peak_torque_rpm: 3_000.0,
                peak_power_rpm: 5_000.0,
                idle_torque: 50.0,
                peak_torque: 100.0,
                power_torque: 80.0,
                engine_brake: 0.02,
            },
            gearbox: Gearbox {
                ratios: vec![3.0],
                final_drive: 4.0,
                reverse: 3.0,
                shift_time: 0.3,
                efficiency: 0.85,
            },
            arb: [0.0, 0.0],
            cd: 0.3,
            area: 2.0,
            rolling: 0.01,
            max_steer: 0.6,
            brake_torque: 1_000.0,
            model_origin: [0.0, 0.0, 0.0],
        };
        assert!(spec.validate().unwrap_err().contains("brake_share"));
    }
}
