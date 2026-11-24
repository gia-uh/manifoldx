//! Entity storage primitives for the native runtime.
//!
//! Phase 1a ships one fixed-capacity slot allocator. It owns the identity
//! lifecycle only: reservation, commit, cancellation, destruction, reuse and
//! generation validation. Components, queries and the Engine runtime are not
//! part of this module.

mod entity_slots;

pub use entity_slots::{EntityError, EntityId, EntitySlots, SlotState, Snapshot};
