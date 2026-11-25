//! Fixed-step clock and owned state for a batch of vehicles.

use std::sync::Arc;
use std::time::Instant;

use super::collision::{self, Contact};
use super::sim::{add, Quaternion, V3};
use super::terrain::{Surface, Tile, TileKey};
use super::{Controls, Ground, Sim, Snapshot, VehicleSpec, DT};

const MAX_FRAME: f64 = 0.05;

pub type Pose = (V3, Quaternion);

#[derive(Clone, Debug)]
pub struct Assist {
    pub drive_cut: f64,
    pub brake_scale: Vec<f64>,
    pub extra_brake: Vec<f64>,
    pub lit: [bool; 4],
}

#[derive(Debug, Default)]
pub struct Frame {
    pub substeps: u32,
    pub bodies: usize,
    pub loaded_wheels: usize,
    pub chassis_contacts: u32,
    pub chassis_impulse: f64,
    pub contacts: Vec<Contact>,
    pub impulse: f64,
    pub solver_ns: u64,
    pub dropped_seconds: f64,
    pub remainder_seconds: f64,
}

#[derive(Clone)]
pub struct Fleet {
    sims: Vec<Sim>,
    ground: Ground,
    pending_ground: Option<Ground>,
    pending_origin: Option<V3>,
    previous: Vec<Pose>,
    residual: f64,
    friction: f64,
    identity: Arc<()>,
}

impl Fleet {
    pub fn new(specs: Vec<VehicleSpec>, ground: Ground, friction: f64) -> Result<Self, String> {
        if !(0.0..=1.0).contains(&friction) {
            return Err("contact friction must lie in [0, 1]".into());
        }
        let mut fleet = Self {
            sims: Vec::with_capacity(specs.len()),
            previous: Vec::with_capacity(specs.len()),
            ground,
            pending_ground: None,
            pending_origin: None,
            residual: 0.0,
            friction,
            identity: Arc::new(()),
        };
        for spec in specs {
            fleet.add(spec)?;
        }
        Ok(fleet)
    }

    pub fn add(&mut self, spec: VehicleSpec) -> Result<usize, String> {
        let mut sim = Sim::new(spec)?;
        sim.set_ground(self.ground.clone());
        self.previous.push(sim.pose());
        self.sims.push(sim);
        Ok(self.sims.len() - 1)
    }

    pub fn remove(&mut self, index: usize) -> Result<(), String> {
        self.sim(index)?;
        self.sims.remove(index);
        self.previous.remove(index);
        Ok(())
    }

    fn sim(&mut self, index: usize) -> Result<&mut Sim, String> {
        self.sims
            .get_mut(index)
            .ok_or_else(|| format!("no vehicle at index {index}"))
    }

    pub fn place(
        &mut self,
        index: usize,
        position: V3,
        yaw: f64,
        velocity: V3,
    ) -> Result<(), String> {
        if !velocity.iter().all(|u| u.is_finite()) {
            return Err("vehicle velocity must be finite".into());
        }
        let sim = self.sim(index)?;
        sim.place(position[0], position[1], position[2], yaw)?;
        sim.set_velocity(velocity);
        self.previous[index] = sim.pose();
        Ok(())
    }

    /// Stage changes against the latest admitted tile batch. They become visible
    /// together immediately before the next actual fixed step.
    pub fn update(&mut self, tiles: Vec<Tile>, removed: &[(TileKey, u64)]) -> Result<(), String> {
        let ground = self.pending_ground.as_ref().unwrap_or(&self.ground);
        let Ground::Tiles { terrain, origin } = ground else {
            return Err("tile edits require tiled ground".into());
        };
        let terrain = Arc::new(terrain.updated(tiles, removed)?);
        self.pending_ground = Some(Ground::Tiles {
            terrain,
            origin: *origin,
        });
        Ok(())
    }

    /// Stage an origin shift in application axes. A world delta is subtracted
    /// from vehicle poses and added to future terrain lookup coordinates.
    pub fn rebase(&mut self, delta: V3) -> Result<(), String> {
        let Ground::Tiles { origin, .. } = &self.ground else {
            return Err("origin changes require tiled ground".into());
        };
        let pending = add(self.pending_origin.unwrap_or([0.0; 3]), delta);
        if !add(*origin, pending).iter().all(|v| v.is_finite()) {
            return Err("origin must remain finite".into());
        }
        self.pending_origin = Some(pending);
        Ok(())
    }

    fn publish_ground(&mut self) {
        if let Some(ground) = self.pending_ground.take() {
            self.ground = ground;
        } else if self.pending_origin.is_none() {
            return;
        }
        if let Some(delta) = self.pending_origin.take() {
            let Ground::Tiles { origin, .. } = &mut self.ground else {
                unreachable!()
            };
            *origin = add(*origin, delta);
            for sim in &mut self.sims {
                sim.rebase(delta);
            }
        }
        for sim in &mut self.sims {
            sim.set_ground(self.ground.clone());
        }
    }

    pub fn sample(&self, x: f64, z: f64, near: f64) -> Result<Option<Surface>, String> {
        if ![x, z, near].iter().all(|v| v.is_finite()) {
            return Err("ground query coordinates must be finite".into());
        }
        Ok(self.ground.surface(x, z, near))
    }

    pub fn state(&self) -> Vec<Snapshot> {
        self.sims.iter().map(Sim::snapshot).collect()
    }

    pub fn restore(&mut self, checkpoint: &Self) -> Result<(), String> {
        if !Arc::ptr_eq(&self.identity, &checkpoint.identity) {
            return Err("checkpoint belongs to another fleet".into());
        }
        *self = checkpoint.clone();
        Ok(())
    }

    pub fn pose(&self, alpha: Option<f64>) -> Result<Vec<Pose>, String> {
        let alpha = alpha.unwrap_or(self.residual / DT);
        if !alpha.is_finite() {
            return Err("interpolation alpha must be finite".into());
        }
        let alpha = alpha.clamp(0.0, 1.0);
        Ok(self
            .previous
            .iter()
            .zip(&self.sims)
            .map(|(&(p0, q0), sim)| {
                let (p1, q1) = sim.pose();
                (
                    std::array::from_fn(|i| p0[i] + (p1[i] - p0[i]) * alpha),
                    slerp(q0, q1, alpha),
                )
            })
            .collect())
    }

    pub fn advance(
        &mut self,
        dt: f64,
        controls: &[Controls],
        assists: Option<&[Assist]>,
    ) -> Result<Frame, String> {
        if !dt.is_finite() || dt < 0.0 {
            return Err("frame dt must be finite and non-negative".into());
        }
        if controls.len() != self.sims.len() {
            return Err("expected one controls row per vehicle".into());
        }
        for control in controls {
            control.validate()?;
        }
        if let Some(assists) = assists {
            if assists.len() != self.sims.len() {
                return Err("expected one assist row per vehicle".into());
            }
            for (assist, sim) in assists.iter().zip(&self.sims) {
                let n = sim.spec().wheels.len();
                if !(0.0..=1.0).contains(&assist.drive_cut)
                    || assist.brake_scale.len() != n
                    || assist.extra_brake.len() != n
                    || !assist.brake_scale.iter().all(|v| (0.0..=1.0).contains(v))
                    || !assist
                        .extra_brake
                        .iter()
                        .all(|v| v.is_finite() && *v >= 0.0)
                {
                    return Err("invalid per-wheel brake or drive assist".into());
                }
            }
        }
        for (index, (sim, control)) in self.sims.iter_mut().zip(controls).enumerate() {
            sim.set_controls(*control)?;
            if let Some(assists) = assists {
                let row = &assists[index];
                sim.set_assists(row.drive_cut, &row.brake_scale, &row.extra_brake, row.lit);
            } else {
                sim.clear_assists();
            }
        }
        let mut frame = Frame {
            bodies: self.sims.len(),
            dropped_seconds: (dt - MAX_FRAME).max(0.0),
            ..Frame::default()
        };
        self.residual += dt.min(MAX_FRAME);
        while self.residual >= DT - 1e-12 {
            let start = Instant::now();
            self.publish_ground();
            for (index, sim) in self.sims.iter_mut().enumerate() {
                self.previous[index] = sim.pose();
                sim.step();
                let (count, impulse) = sim.chassis_contacts();
                frame.chassis_contacts += count;
                frame.chassis_impulse += impulse;
                frame.loaded_wheels += sim.loaded_wheels();
                collision::walls(sim, &self.ground, index, self.friction, &mut frame.contacts);
            }
            collision::pairs(&mut self.sims, self.friction, &mut frame.contacts);
            self.residual = (self.residual - DT).max(0.0);
            frame.substeps += 1;
            frame.solver_ns += start.elapsed().as_nanos() as u64;
        }
        frame.impulse = frame.chassis_impulse
            + frame
                .contacts
                .iter()
                .map(|c| c.impulse[0].hypot(c.impulse[1]).hypot(c.impulse[2]))
                .sum::<f64>();
        frame.remainder_seconds = self.residual;
        Ok(frame)
    }
}

fn slerp(a: Quaternion, mut b: Quaternion, t: f64) -> Quaternion {
    let mut dot = a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>();
    if dot < 0.0 {
        b = b.map(|x| -x);
        dot = -dot;
    }
    if dot > 0.9995 {
        let q = std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t);
        let magnitude = q.iter().map(|x| x * x).sum::<f64>().sqrt();
        return q.map(|x| x / magnitude);
    }
    let angle = dot.acos();
    let left = ((1.0 - t) * angle).sin() / angle.sin();
    let right = (t * angle).sin() / angle.sin();
    std::array::from_fn(|i| left * a[i] + right * b[i])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vehicles::sim::test_car;

    #[test]
    fn pose_midpoint_uses_the_short_arc() {
        let mut fleet = Fleet::new(vec![test_car()], Ground::default(), 0.4).unwrap();
        let yaw = 170.0_f64.to_radians();
        fleet.place(0, [0.0; 3], yaw, [0.0; 3]).unwrap();
        let previous = fleet.sims[0].pose();
        fleet.place(0, [0.0; 3], -yaw, [0.0; 3]).unwrap();
        fleet.previous[0] = previous;
        let a = previous.1;
        let b = fleet.sims[0].pose().1;
        assert!(a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>() < 0.0);

        // The short path from +170° to -170° crosses 180° around +Y.
        // The long path crosses 0° instead, where |w| is 1.
        let q = fleet.pose(Some(0.5)).unwrap()[0].1;
        let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() <= 1e-9, "midpoint norm: {norm}");
        assert!(
            q[0].abs() <= 1e-9
                && (q[1].abs() - 1.0).abs() <= 1e-9
                && q[2].abs() <= 1e-9
                && q[3].abs() <= 1e-9,
            "short-arc midpoint must face back: {q:?}"
        );
    }

    #[test]
    fn checkpoint_restores_already_admitted_controls() {
        let mut fleet = Fleet::new(vec![test_car()], Ground::default(), 0.4).unwrap();
        let drive = Controls::new(-0.25, 0.75, 0.125, 0.5).unwrap();
        let frame = fleet.advance(DT / 2.0, &[drive], None).unwrap();
        assert_eq!(frame.substeps, 0);
        let checkpoint = fleet.clone();

        let frame = fleet.advance(0.0, &[Controls::default()], None).unwrap();
        assert_eq!(frame.substeps, 0);
        assert_eq!(fleet.sims[0].controls().throttle, 0.0);
        fleet.restore(&checkpoint).unwrap();

        // Observe restored controls before advance can overwrite them.
        let restored = fleet.sims[0].controls();
        assert_eq!(
            [
                restored.steer,
                restored.throttle,
                restored.brake,
                restored.handbrake
            ],
            [-0.25, 0.75, 0.125, 0.5],
            "already-admitted controls must survive restore"
        );
        assert_eq!(fleet.residual, DT / 2.0);
    }
}
