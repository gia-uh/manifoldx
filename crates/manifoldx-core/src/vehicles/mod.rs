//! Native vehicle physics: a fixed 120 Hz solver with native ground.
//!
//! The model is the scalar Python vehicle model in `src/manifoldx/vehicles`
//! (suspension, tyre, powertrain, body and ground), ported to owned `f64`
//! values. It has no Python dependency and owns no clock.

pub mod ground;
pub mod sim;
pub mod spec;

pub use ground::Ground;
pub use sim::{Controls, Sim, Snapshot, DT};
pub use spec::{Engine, Gearbox, Tyre, VehicleSpec, Wheel};
