//! Native vehicle physics: a fixed 120 Hz solver with native ground.
//!
//! The model is the scalar Python vehicle model in `src/manifoldx/vehicles`
//! (suspension, tyre, powertrain, body and ground), ported to owned `f64`
//! values. It has no Python dependency and owns no clock.

mod collision;
pub mod fleet;
pub mod ground;
pub mod sim;
pub mod spec;
pub mod terrain;

pub use collision::Contact;
pub use fleet::{Assist, Fleet, Frame, Pose};
pub use ground::Ground;
pub use sim::{Controls, Sim, Snapshot, DT};
pub use spec::{Engine, Gearbox, Tyre, VehicleSpec, Wheel};
