//! Native ECS storage primitives.
//!
//! The phase 1a slot allocator owns identity lifetimes: reservation, commit,
//! cancellation, destruction, reuse and generation validation. The World adds
//! the explicit component schema, typed columns, membership and validated bulk
//! transactions on top of the same allocator. Queries, scheduling and the
//! Engine runtime are not part of this module.

mod entity_slots;
mod world;

pub use entity_slots::{EntityError, EntityId, EntitySlots, SlotState, Snapshot};
pub use world::{
    ColumnData, ComponentPayload, ComponentReadout, ComponentSchema, ElementType, F32Rows, World,
    WorldError, WorldSnapshot, MAX_COMPONENT_WIDTH,
};
