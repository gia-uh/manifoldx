//! Fixed-capacity component storage built on the generation-bearing entity
//! slots.
//!
//! This is the first native World/store slice. It owns the entity lifetimes,
//! an explicit schema, one typed column per component and membership. The
//! supported element types are `f32` and `u32`; a width of zero is a marker.
//! There is no byte-addressed storage and no export of writable native memory.
//!
//! # Contract
//!
//! - **Schemas are explicit and frozen at first publish.** A component is
//!   registered with a name, an element type and a fixed row width. Names are
//!   unique; a duplicate is rejected without touching live state. Once the
//!   first entity batch has been published no further schema can be registered.
//! - **Fixed capacity.** The entity table and every column are allocated once
//!   for the declared capacity and never grow.
//! - **Rows always exist.** Every live entity has a zeroed row in every
//!   registered column. Membership says whether the entity currently carries
//!   the component. Only members carry meaningful values; a non-member row is
//!   zeroed and is not a place to read stale data from.
//! - **Spawn initialises the whole row set.** For each new entity every column
//!   starts as non-member with a zeroed row; each payload component then gets
//!   the submitted values and becomes a member. A recycled slot therefore never
//!   inherits a previous life's values or membership.
//! - **Batch mutations are atomic.** A malformed descriptor, a length
//!   mismatch, a duplicate component or identity, a stale generation, a
//!   non-member update or an over-capacity spawn rejects the complete batch
//!   before any entity, membership or value changes.
//! - **Updates require membership.** Row and field updates write only entities
//!   that already carry the component; they never attach one implicitly.
//! - **Destroy clears the row.** A successful destroy marks the entities dead
//!   and zeroes every column's row and membership for those slots before they
//!   can be reserved again. A stale destroy is rejected by the slot allocator.
//! - **Owned readback.** Snapshots and component readouts are copies. No
//!   pointer into Rust storage crosses the boundary.
//!
//! Costs are linear in the submitted batch and the selected rows, never in
//! `capacity × batch`; no mutation clones the whole World.

use std::collections::HashMap;
use std::fmt;

use super::entity_slots::{EntityError, EntityId, EntitySlots, SlotState};

/// The widest row a component can declare, in scalars.
///
/// This bounds one column's allocation and keeps length arithmetic explicit.
pub const MAX_COMPONENT_WIDTH: u32 = 256;

/// A scalar element type supported by native component columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementType {
    /// A 32-bit IEEE float.
    F32,
    /// A 32-bit unsigned integer.
    U32,
}

impl ElementType {
    /// The stable lowercase name reported by snapshots and bindings.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::U32 => "u32",
        }
    }

    /// The size of one element in bytes.
    pub const fn byte_width(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::U32 => 4,
        }
    }
}

/// One registered component: a name, an element type and a fixed row width.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentSchema {
    /// Unique name, as registered.
    pub name: String,
    /// Element type of every scalar in the row.
    pub element: ElementType,
    /// Number of scalars per entity; zero is a marker component.
    pub width: u32,
}

/// A batch of typed values laid out one row per submitted entity.
#[derive(Clone, Debug, PartialEq)]
pub enum ColumnData {
    /// Float rows.
    F32(Vec<f32>),
    /// Unsigned integer rows.
    U32(Vec<u32>),
}

impl ColumnData {
    /// The element type held by this payload.
    pub const fn element_type(&self) -> ElementType {
        match self {
            Self::F32(_) => ElementType::F32,
            Self::U32(_) => ElementType::U32,
        }
    }

    /// Number of scalars in the payload.
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::U32(values) => values.len(),
        }
    }

    /// Whether the payload holds no scalars.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of bytes in the payload.
    pub fn byte_len(&self) -> usize {
        self.len() * self.element_type().byte_width()
    }

    fn try_zeros(element: ElementType, len: usize, component: &str) -> Result<Self, WorldError> {
        match element {
            ElementType::F32 => {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(len)
                    .map_err(|_| WorldError::Allocation {
                        component: component.to_string(),
                        values: len,
                    })?;
                values.resize(len, 0.0);
                Ok(Self::F32(values))
            }
            ElementType::U32 => {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(len)
                    .map_err(|_| WorldError::Allocation {
                        component: component.to_string(),
                        values: len,
                    })?;
                values.resize(len, 0);
                Ok(Self::U32(values))
            }
        }
    }

    fn copy_row_into(&self, row: usize, width: usize, target: &mut Self, target_start: usize) {
        debug_assert_eq!(self.element_type(), target.element_type());
        let source_start = row * width;
        match (self, target) {
            (Self::F32(source), Self::F32(target)) => target[target_start..target_start + width]
                .copy_from_slice(&source[source_start..source_start + width]),
            (Self::U32(source), Self::U32(target)) => target[target_start..target_start + width]
                .copy_from_slice(&source[source_start..source_start + width]),
            _ => unreachable!("payload kind is validated before a row copy"),
        }
    }
}

/// One component attached to a spawn batch.
#[derive(Clone, Debug, PartialEq)]
pub struct ComponentPayload {
    /// Index returned by [`World::register`].
    pub component: usize,
    /// Rows for the new entities, in the order they are spawned.
    pub data: ColumnData,
}

/// An owned readback of one component for a set of live entities.
#[derive(Clone, Debug, PartialEq)]
pub struct ComponentReadout {
    /// The component's registered schema.
    pub schema: ComponentSchema,
    /// Committed column version.
    pub version: u64,
    /// The entities read, in request order or ascending slot order.
    pub ids: Vec<EntityId>,
    /// Membership flags, parallel to `ids`.
    pub membership: Vec<bool>,
    /// Rows parallel to `ids`; non-member rows are zeroed.
    pub data: ColumnData,
}

/// An owned snapshot of the World's entities and schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorldSnapshot {
    /// Fixed entity capacity.
    pub capacity: u32,
    /// Number of live entities.
    pub alive_count: u32,
    /// One past the highest slot ever published.
    pub high_water: u32,
    /// World mutation version.
    pub version: u64,
    /// Live identities in ascending slot order.
    pub entities: Vec<EntityId>,
    /// Registered schemas in registration order.
    pub schemas: Vec<ComponentSchema>,
}

/// A rejected World operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorldError {
    /// The entity slot allocator rejected the batch.
    Entity(EntityError),
    /// A component name was empty.
    EmptyName,
    /// A component with this name is already registered.
    DuplicateSchema {
        /// The duplicate name.
        name: String,
    },
    /// Schemas may only be registered before the first published entity.
    SchemaLocked {
        /// The rejected name.
        name: String,
    },
    /// The row width is above [`MAX_COMPONENT_WIDTH`].
    WidthTooLarge {
        /// The component name.
        name: String,
        /// Requested width.
        width: u32,
        /// Maximum accepted width.
        max: u32,
    },
    /// A column allocation failed.
    Allocation {
        /// The component name.
        component: String,
        /// Number of scalars requested.
        values: usize,
    },
    /// A component name is not registered.
    UnknownComponent {
        /// The unknown name.
        name: String,
    },
    /// A component index is outside the registered schema.
    UnknownComponentIndex {
        /// The unknown index.
        index: usize,
    },
    /// A payload's element type does not match its component.
    PayloadKind {
        /// The component name.
        component: String,
        /// The registered element type.
        expected: ElementType,
        /// The payload element type.
        got: ElementType,
    },
    /// A payload's length does not match the batch and schema.
    PayloadLength {
        /// The component name.
        component: String,
        /// Required scalar count.
        expected: usize,
        /// Submitted scalar count.
        got: usize,
        /// Rows in the batch.
        rows: usize,
        /// Scalars per row.
        width: u32,
    },
    /// The same component appears twice in one spawn batch.
    DuplicatePayload {
        /// The repeated component name.
        component: String,
    },
    /// A zero-entity spawn carried a payload.
    PayloadForEmptyBatch,
    /// A length computation overflowed `usize`.
    LengthOverflow {
        /// The component name.
        component: String,
        /// Rows in the batch.
        rows: usize,
        /// Scalars per row.
        width: u32,
    },
    /// An update named a component the entity does not carry.
    NotMember {
        /// The rejected identity.
        id: EntityId,
        /// The component name.
        component: String,
    },
    /// A field index is outside the component row.
    FieldOutOfRange {
        /// The component name.
        component: String,
        /// The requested field.
        field: u32,
        /// The component width.
        width: u32,
    },
}

impl fmt::Display for WorldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Entity(error) => write!(f, "{error}"),
            Self::EmptyName => write!(f, "component name must not be empty"),
            Self::DuplicateSchema { name } => {
                write!(f, "component schema '{name}' is already registered")
            }
            Self::SchemaLocked { name } => write!(
                f,
                "component schema '{name}' cannot be registered after entities are published"
            ),
            Self::WidthTooLarge { name, width, max } => write!(
                f,
                "component '{name}' width {width} exceeds the maximum {max}"
            ),
            Self::Allocation { component, values } => write!(
                f,
                "cannot allocate {values} values for component '{component}'"
            ),
            Self::UnknownComponent { name } => write!(f, "unknown component '{name}'"),
            Self::UnknownComponentIndex { index } => {
                write!(f, "unknown component index {index}")
            }
            Self::PayloadKind {
                component,
                expected,
                got,
            } => write!(
                f,
                "component '{component}' is {} but the payload is {}",
                expected.as_str(),
                got.as_str()
            ),
            Self::PayloadLength {
                component,
                expected,
                got,
                rows,
                width,
            } => write!(
                f,
                "component '{component}' payload has {got} values for {rows} rows of width {width}; expected {expected}"
            ),
            Self::DuplicatePayload { component } => {
                write!(f, "component '{component}' appears twice in one spawn batch")
            }
            Self::PayloadForEmptyBatch => {
                write!(f, "a zero-entity spawn must not carry component payloads")
            }
            Self::LengthOverflow {
                component,
                rows,
                width,
            } => write!(
                f,
                "component '{component}' row count overflows: {rows} rows of width {width}"
            ),
            Self::NotMember { id, component } => {
                write!(f, "entity {id} has no '{component}' component")
            }
            Self::FieldOutOfRange {
                component,
                field,
                width,
            } => write!(
                f,
                "field {field} is outside component '{component}' width {width}"
            ),
        }
    }
}

impl std::error::Error for WorldError {}

impl From<EntityError> for WorldError {
    fn from(error: EntityError) -> Self {
        Self::Entity(error)
    }
}

#[derive(Debug)]
struct Column {
    version: u64,
    schema: ComponentSchema,
    data: ColumnData,
    /// One bit per slot, in slot order.
    members: Vec<u64>,
}

impl Column {
    fn new(schema: ComponentSchema, capacity: u32) -> Result<Self, WorldError> {
        let len = (capacity as usize)
            .checked_mul(schema.width as usize)
            .ok_or_else(|| WorldError::LengthOverflow {
                component: schema.name.clone(),
                rows: capacity as usize,
                width: schema.width,
            })?;
        let data = ColumnData::try_zeros(schema.element, len, &schema.name)?;
        let words = (capacity as usize).div_ceil(64);
        let mut members = Vec::new();
        members
            .try_reserve_exact(words)
            .map_err(|_| WorldError::Allocation {
                component: schema.name.clone(),
                values: words,
            })?;
        members.resize(words, 0);
        Ok(Self {
            version: 0,
            schema,
            data,
            members,
        })
    }

    fn is_member(&self, slot: u32) -> bool {
        let index = slot as usize;
        (self.members[index / 64] >> (index % 64)) & 1 != 0
    }

    fn set_member(&mut self, slot: u32, member: bool) {
        let index = slot as usize;
        let mask = 1u64 << (index % 64);
        if member {
            self.members[index / 64] |= mask;
        } else {
            self.members[index / 64] &= !mask;
        }
    }

    fn clear_row(&mut self, slot: u32) {
        let width = self.schema.width as usize;
        let start = slot as usize * width;
        match &mut self.data {
            ColumnData::F32(values) => values[start..start + width].fill(0.0),
            ColumnData::U32(values) => values[start..start + width].fill(0),
        }
        self.set_member(slot, false);
    }
}

/// Borrowed native rows from two distinct float columns of equal width.
/// Only rows carrying both components are visited. No allocation or readback.
pub struct F32Rows<'a> {
    target: &'a mut [f32],
    source: &'a [f32],
    target_members: &'a [u64],
    source_members: &'a [u64],
    width: usize,
    excluded_slot: Option<usize>,
}

impl F32Rows<'_> {
    /// Visit the intersection in ascending slot order and return its row count.
    pub fn for_each(&mut self, mut visit: impl FnMut(&mut [f32], &[f32])) -> usize {
        let mut count = 0;
        for (slot, (target, source)) in self
            .target
            .chunks_exact_mut(self.width)
            .zip(self.source.chunks_exact(self.width))
            .enumerate()
        {
            let bit = 1u64 << (slot % 64);
            if Some(slot) != self.excluded_slot
                && self.target_members[slot / 64] & self.source_members[slot / 64] & bit != 0
            {
                visit(target, source);
                count += 1;
            }
        }
        count
    }
}

/// A fixed-capacity component World built on [`EntitySlots`].
#[derive(Debug)]
pub struct World {
    slots: EntitySlots,
    columns: Vec<Column>,
    index: HashMap<String, usize>,
    published: bool,
    version: u64,
}

impl World {
    /// Lend two distinct, equally wide non-marker f32 columns to a native kernel.
    /// Dead rows never carry membership; callers cannot retain the borrow.
    pub fn f32_rows(&mut self, target: usize, source: usize) -> Result<F32Rows<'_>, &'static str> {
        self.f32_rows_except(target, source, None)
    }

    /// Exclude a solver-owned identity, allowing a later occupant of its slot.
    pub(crate) fn f32_rows_except(
        &mut self,
        target: usize,
        source: usize,
        excluded: Option<EntityId>,
    ) -> Result<F32Rows<'_>, &'static str> {
        let excluded_slot = excluded
            .filter(|id| self.slots.validate_alive(std::slice::from_ref(id)).is_ok())
            .map(|id| id.slot() as usize);
        if target == source {
            return Err("native columns must be distinct");
        }
        let t = self.columns.get(target).ok_or("unknown target column")?;
        let s = self.columns.get(source).ok_or("unknown source column")?;
        if t.schema.element != ElementType::F32
            || s.schema.element != ElementType::F32
            || t.schema.width == 0
            || t.schema.width != s.schema.width
        {
            return Err("native columns require equal nonzero f32 widths");
        }
        self.version += 1;
        let (t, s) = if target < source {
            let (left, right) = self.columns.split_at_mut(source);
            (&mut left[target], &right[0])
        } else {
            let (left, right) = self.columns.split_at_mut(target);
            (&mut right[0], &left[source])
        };
        t.version = self.version;
        let (ColumnData::F32(target), ColumnData::F32(source)) = (&mut t.data, &s.data) else {
            unreachable!()
        };
        Ok(F32Rows {
            target,
            source,
            target_members: &t.members,
            source_members: &s.members,
            width: t.schema.width as usize,
            excluded_slot,
        })
    }

    /// Visit contiguous runs of native members, in ascending slot order.
    pub(crate) fn visit_kernel_runs(
        &mut self,
        index: usize,
        writes: bool,
        mut visit: impl FnMut(&mut [f32]),
    ) -> Result<usize, &'static str> {
        let column = self.columns.get_mut(index).ok_or("unknown kernel column")?;
        let ColumnData::F32(data) = &mut column.data else {
            return Err("kernel column requires float32 rows");
        };
        let width = column.schema.width as usize;
        if width == 0 {
            return Err("kernel column cannot be a marker");
        }
        let mut count = 0;
        for (word_id, &members) in column.members.iter().enumerate() {
            let mut active = members;
            while active != 0 {
                let first = active.trailing_zeros();
                let len = (active >> first).trailing_ones();
                let start = (word_id * 64 + first as usize) * width;
                let end = start + len as usize * width;
                visit(&mut data[start..end]);
                count += len as usize;
                active &= !((u64::MAX >> (64 - len)) << first);
            }
        }
        if writes && count != 0 {
            self.version += 1;
            column.version = self.version;
        }
        Ok(count)
    }

    /// Allocate the fixed slot table and an empty schema.
    pub fn new(capacity: u32) -> Result<Self, WorldError> {
        Ok(Self {
            slots: EntitySlots::new(capacity)?,
            columns: Vec::new(),
            index: HashMap::new(),
            published: false,
            version: 0,
        })
    }

    /// Fixed entity capacity.
    pub const fn capacity(&self) -> u32 {
        self.slots.capacity()
    }

    /// Current identity of a live slot, if any.
    pub fn live_identity(&self, slot: u32) -> Option<EntityId> {
        if self.slots.state_of(slot) != Some(SlotState::Alive) {
            return None;
        }
        self.slots.current_identity(slot).ok()
    }

    /// Number of live entities.
    pub fn alive_count(&self) -> u32 {
        self.slots.alive_count()
    }

    /// One past the highest slot ever published.
    pub const fn high_water(&self) -> u32 {
        self.slots.high_water()
    }

    /// World mutation version.
    pub const fn version(&self) -> u64 {
        self.version
    }

    /// Whether an entity has ever been published.
    pub const fn published(&self) -> bool {
        self.published
    }

    /// Register one component schema and return its index.
    ///
    /// The name must be new and must be non-empty, the width must not exceed
    /// [`MAX_COMPONENT_WIDTH`], and the World must not have published an
    /// entity yet. A rejection leaves the World unchanged.
    pub fn register(
        &mut self,
        name: &str,
        element: ElementType,
        width: u32,
    ) -> Result<usize, WorldError> {
        if name.trim().is_empty() {
            return Err(WorldError::EmptyName);
        }
        if self.index.contains_key(name) {
            return Err(WorldError::DuplicateSchema {
                name: name.to_string(),
            });
        }
        if self.published {
            return Err(WorldError::SchemaLocked {
                name: name.to_string(),
            });
        }
        if width > MAX_COMPONENT_WIDTH {
            return Err(WorldError::WidthTooLarge {
                name: name.to_string(),
                width,
                max: MAX_COMPONENT_WIDTH,
            });
        }
        let schema = ComponentSchema {
            name: name.to_string(),
            element,
            width,
        };
        let column = Column::new(schema, self.slots.capacity())?;
        let index = self.columns.len();
        self.columns.push(column);
        self.index.insert(name.to_string(), index);
        self.version += 1;
        Ok(index)
    }

    /// The index of a registered component, or `None`.
    pub fn component_index(&self, name: &str) -> Option<usize> {
        self.index.get(name).copied()
    }

    /// One registered schema by index.
    pub fn schema(&self, component: usize) -> Option<&ComponentSchema> {
        self.columns.get(component).map(|column| &column.schema)
    }

    /// Registered schemas in registration order.
    pub fn schemas(&self) -> impl Iterator<Item = &ComponentSchema> {
        self.columns.iter().map(|column| &column.schema)
    }

    /// Publish `count` new entities with the submitted component payloads.
    ///
    /// Every payload is validated against its schema and the requested batch
    /// size before the slot reservation. On success the new entities are live,
    /// and every registered component starts zeroed and non-member except the
    /// payload components, which become members with the submitted rows.
    pub fn spawn(
        &mut self,
        count: u32,
        payloads: &[ComponentPayload],
    ) -> Result<Vec<EntityId>, WorldError> {
        self.spawn_batches([(count, payloads)])
    }

    /// Publish all batches with one reservation after validating every payload.
    pub fn spawn_batches<'a>(
        &mut self,
        batches: impl IntoIterator<Item = (u32, &'a [ComponentPayload])>,
    ) -> Result<Vec<EntityId>, WorldError> {
        let batches: Vec<_> = batches.into_iter().collect();
        let mut count = 0u32;
        for &(n, payloads) in &batches {
            self.validate_spawn(n, payloads)?;
            count = count.checked_add(n).ok_or(WorldError::LengthOverflow {
                component: "spawn batch".into(),
                rows: count as usize,
                width: n as usize,
            })?;
        }
        if count == 0 {
            return Ok(Vec::new());
        }
        let ids = self.slots.reserve(count)?;
        if let Err(error) = self.slots.commit(&ids) {
            let _ = self.slots.cancel(&ids);
            return Err(error.into());
        }
        self.published = true;

        for id in &ids {
            for column in &mut self.columns {
                column.clear_row(id.slot());
            }
        }
        let mut start = 0;
        for (count, payloads) in batches {
            let end = start + count as usize;
            let batch = &ids[start..end];
            for payload in payloads {
                let column = &mut self.columns[payload.component];
                column.version = self.version + 1;
                let width = column.schema.width as usize;
                for (row, id) in batch.iter().enumerate() {
                    let start = id.slot() as usize * width;
                    payload
                        .data
                        .copy_row_into(row, width, &mut column.data, start);
                    column.set_member(id.slot(), true);
                }
            }
            start = end;
        }
        self.version += 1;
        Ok(ids)
    }

    fn validate_spawn(&self, count: u32, payloads: &[ComponentPayload]) -> Result<(), WorldError> {
        if count == 0 {
            if payloads.is_empty() {
                return Ok(());
            }
            return Err(WorldError::PayloadForEmptyBatch);
        }

        let mut seen = vec![false; self.columns.len()];
        for payload in payloads {
            let column =
                self.columns
                    .get(payload.component)
                    .ok_or(WorldError::UnknownComponentIndex {
                        index: payload.component,
                    })?;
            if seen[payload.component] {
                return Err(WorldError::DuplicatePayload {
                    component: column.schema.name.clone(),
                });
            }
            seen[payload.component] = true;
            if payload.data.element_type() != column.schema.element {
                return Err(WorldError::PayloadKind {
                    component: column.schema.name.clone(),
                    expected: column.schema.element,
                    got: payload.data.element_type(),
                });
            }
            let expected = (count as usize)
                .checked_mul(column.schema.width as usize)
                .ok_or_else(|| WorldError::LengthOverflow {
                    component: column.schema.name.clone(),
                    rows: count as usize,
                    width: column.schema.width,
                })?;
            if payload.data.len() != expected {
                return Err(WorldError::PayloadLength {
                    component: column.schema.name.clone(),
                    expected,
                    got: payload.data.len(),
                    rows: count as usize,
                    width: column.schema.width,
                });
            }
        }
        Ok(())
    }

    /// Overwrite whole rows of an existing component on selected entities.
    ///
    /// Every identity must be live and carry the component. The complete batch
    /// is checked before any row changes.
    pub fn update_rows(
        &mut self,
        component: usize,
        ids: &[EntityId],
        data: ColumnData,
    ) -> Result<usize, WorldError> {
        let (element, width, name) = {
            let column = self
                .columns
                .get(component)
                .ok_or(WorldError::UnknownComponentIndex { index: component })?;
            (
                column.schema.element,
                column.schema.width,
                column.schema.name.clone(),
            )
        };
        if data.element_type() != element {
            return Err(WorldError::PayloadKind {
                component: name,
                expected: element,
                got: data.element_type(),
            });
        }
        let expected =
            ids.len()
                .checked_mul(width as usize)
                .ok_or_else(|| WorldError::LengthOverflow {
                    component: name.clone(),
                    rows: ids.len(),
                    width,
                })?;
        if data.len() != expected {
            return Err(WorldError::PayloadLength {
                component: name,
                expected,
                got: data.len(),
                rows: ids.len(),
                width,
            });
        }
        self.slots.validate_alive(ids)?;
        {
            let column = &self.columns[component];
            for id in ids {
                if !column.is_member(id.slot()) {
                    return Err(WorldError::NotMember {
                        id: *id,
                        component: column.schema.name.clone(),
                    });
                }
            }
        }
        let column = &mut self.columns[component];
        for (row, id) in ids.iter().enumerate() {
            data.copy_row_into(
                row,
                width as usize,
                &mut column.data,
                id.slot() as usize * width as usize,
            );
        }
        self.version += 1;
        column.version = self.version;
        Ok(ids.len())
    }

    /// Overwrite one scalar field of an existing component on selected
    /// entities.
    ///
    /// The field must be inside the row, every identity must be live and carry
    /// the component, and the full batch is checked before any value changes.
    pub fn update_field(
        &mut self,
        component: usize,
        field: u32,
        ids: &[EntityId],
        data: ColumnData,
    ) -> Result<usize, WorldError> {
        let (element, width, name) = {
            let column = self
                .columns
                .get(component)
                .ok_or(WorldError::UnknownComponentIndex { index: component })?;
            (
                column.schema.element,
                column.schema.width,
                column.schema.name.clone(),
            )
        };
        if field >= width {
            return Err(WorldError::FieldOutOfRange {
                component: name,
                field,
                width,
            });
        }
        if data.element_type() != element {
            return Err(WorldError::PayloadKind {
                component: name,
                expected: element,
                got: data.element_type(),
            });
        }
        if data.len() != ids.len() {
            return Err(WorldError::PayloadLength {
                component: name,
                expected: ids.len(),
                got: data.len(),
                rows: ids.len(),
                width: 1,
            });
        }
        self.slots.validate_alive(ids)?;
        {
            let column = &self.columns[component];
            for id in ids {
                if !column.is_member(id.slot()) {
                    return Err(WorldError::NotMember {
                        id: *id,
                        component: column.schema.name.clone(),
                    });
                }
            }
        }
        let column = &mut self.columns[component];
        match (&mut column.data, &data) {
            (ColumnData::F32(target), ColumnData::F32(source)) => {
                for (row, id) in ids.iter().enumerate() {
                    target[id.slot() as usize * width as usize + field as usize] = source[row];
                }
            }
            (ColumnData::U32(target), ColumnData::U32(source)) => {
                for (row, id) in ids.iter().enumerate() {
                    target[id.slot() as usize * width as usize + field as usize] = source[row];
                }
            }
            _ => unreachable!("payload kind is validated before a field write"),
        }
        self.version += 1;
        column.version = self.version;
        Ok(ids.len())
    }

    /// Destroy live entities and clear their rows and membership.
    ///
    /// The slot allocator validates the whole batch first; a stale identity
    /// rejects the batch with every entity and value intact.
    pub fn destroy(&mut self, ids: &[EntityId]) -> Result<usize, WorldError> {
        if ids.is_empty() {
            return Ok(0);
        }
        self.slots.destroy(ids)?;
        for id in ids {
            for column in &mut self.columns {
                if column.is_member(id.slot()) {
                    column.version = self.version + 1;
                    column.clear_row(id.slot());
                }
            }
        }
        self.version += 1;
        Ok(ids.len())
    }

    /// Membership flags for `ids` in request order.
    pub fn memberships(&self, component: usize, ids: &[EntityId]) -> Result<Vec<bool>, WorldError> {
        let column = self
            .columns
            .get(component)
            .ok_or(WorldError::UnknownComponentIndex { index: component })?;
        self.slots.validate_alive(ids)?;
        Ok(ids.iter().map(|id| column.is_member(id.slot())).collect())
    }

    /// Whether one live entity carries `component`.
    pub fn is_member(&self, component: usize, id: EntityId) -> Result<bool, WorldError> {
        let flags = self.memberships(component, std::slice::from_ref(&id))?;
        Ok(flags[0])
    }

    /// Owned readback of one component.
    ///
    /// `ids` selects and orders rows; `None` reads every live entity in
    /// ascending slot order. Non-member rows are returned zeroed with their
    /// membership flag false.
    pub fn read_component(
        &self,
        component: usize,
        ids: Option<&[EntityId]>,
    ) -> Result<ComponentReadout, WorldError> {
        let column = self
            .columns
            .get(component)
            .ok_or(WorldError::UnknownComponentIndex { index: component })?;
        let selected;
        let ids: &[EntityId] = match ids {
            Some(ids) => {
                self.slots.validate_alive(ids)?;
                ids
            }
            None => {
                selected = self.slots.alive_ids();
                &selected
            }
        };
        let width = column.schema.width as usize;
        let len = ids
            .len()
            .checked_mul(width)
            .ok_or_else(|| WorldError::LengthOverflow {
                component: column.schema.name.clone(),
                rows: ids.len(),
                width: column.schema.width,
            })?;
        let mut data = ColumnData::try_zeros(column.schema.element, len, &column.schema.name)?;
        match (&column.data, &mut data) {
            (ColumnData::F32(source), ColumnData::F32(target)) => {
                for (row, id) in ids.iter().enumerate() {
                    let start = id.slot() as usize * width;
                    target[row * width..(row + 1) * width]
                        .copy_from_slice(&source[start..start + width]);
                }
            }
            (ColumnData::U32(source), ColumnData::U32(target)) => {
                for (row, id) in ids.iter().enumerate() {
                    let start = id.slot() as usize * width;
                    target[row * width..(row + 1) * width]
                        .copy_from_slice(&source[start..start + width]);
                }
            }
            _ => unreachable!("column and readout share the registered element type"),
        }
        Ok(ComponentReadout {
            schema: column.schema.clone(),
            version: column.version,
            ids: ids.to_vec(),
            membership: ids.iter().map(|id| column.is_member(id.slot())).collect(),
            data,
        })
    }

    /// An owned snapshot of entities and schema, without component values.
    pub fn snapshot(&self) -> WorldSnapshot {
        WorldSnapshot {
            capacity: self.capacity(),
            alive_count: self.alive_count(),
            high_water: self.high_water(),
            version: self.version,
            entities: self.slots.alive_ids(),
            schemas: self.schemas().cloned().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_rows(rows: &[&[f32]]) -> ColumnData {
        ColumnData::F32(rows.iter().flat_map(|row| row.iter().copied()).collect())
    }

    fn u32_rows(rows: &[&[u32]]) -> ColumnData {
        ColumnData::U32(rows.iter().flat_map(|row| row.iter().copied()).collect())
    }

    fn world(capacity: u32) -> (World, usize, usize, usize) {
        let mut world = World::new(capacity).unwrap();
        let position = world
            .register("transform.position", ElementType::F32, 3)
            .unwrap();
        let mesh = world.register("mesh.id", ElementType::U32, 1).unwrap();
        let marker = world.register("flags.marker", ElementType::U32, 0).unwrap();
        (world, position, mesh, marker)
    }

    #[test]
    fn schemas_are_explicit_unique_and_frozen_at_publish() {
        let mut world = World::new(8).unwrap();
        assert_eq!(world.register("a", ElementType::F32, 3).unwrap(), 0);
        assert_eq!(
            world.register("a", ElementType::U32, 1).unwrap_err(),
            WorldError::DuplicateSchema { name: "a".into() }
        );
        assert_eq!(
            world.register("", ElementType::F32, 1).unwrap_err(),
            WorldError::EmptyName
        );
        assert!(matches!(
            world.register("wide", ElementType::F32, MAX_COMPONENT_WIDTH + 1),
            Err(WorldError::WidthTooLarge { .. })
        ));
        let _ = world.spawn(1, &[]).unwrap();
        assert_eq!(
            world.register("late", ElementType::F32, 1).unwrap_err(),
            WorldError::SchemaLocked {
                name: "late".into()
            }
        );
    }

    #[test]
    fn spawn_reads_values_defaults_and_membership() {
        let (mut world, position, mesh, marker) = world(8);
        let ids = world
            .spawn(
                3,
                &[
                    ComponentPayload {
                        component: position,
                        data: f32_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]),
                    },
                    ComponentPayload {
                        component: mesh,
                        data: u32_rows(&[&[10], &[11], &[12]]),
                    },
                    ComponentPayload {
                        component: marker,
                        data: ColumnData::U32(Vec::new()),
                    },
                ],
            )
            .unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(world.alive_count(), 3);

        let read = world.read_component(position, Some(&ids)).unwrap();
        assert_eq!(read.membership, vec![true, true, true]);
        assert_eq!(
            read.data,
            f32_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]])
        );
        assert_eq!(
            world.read_component(mesh, Some(&ids)).unwrap().data,
            u32_rows(&[&[10], &[11], &[12]])
        );
        assert_eq!(
            world.read_component(marker, Some(&ids)).unwrap().membership,
            vec![true, true, true]
        );
        assert_eq!(
            world.read_component(marker, Some(&ids)).unwrap().data.len(),
            0
        );
    }

    #[test]
    fn spawn_initialises_columns_without_payload() {
        let (mut world, position, mesh, marker) = world(4);
        let ids = world
            .spawn(
                2,
                &[ComponentPayload {
                    component: position,
                    data: f32_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]),
                }],
            )
            .unwrap();
        let mesh_read = world.read_component(mesh, Some(&ids)).unwrap();
        assert_eq!(mesh_read.membership, vec![false, false]);
        assert_eq!(mesh_read.data, u32_rows(&[&[0], &[0]]));
        let marker_read = world.read_component(marker, Some(&ids)).unwrap();
        assert_eq!(marker_read.membership, vec![false, false]);

        // A update cannot attach a missing component.
        let before = world.snapshot();
        let error = world
            .update_rows(mesh, &ids, u32_rows(&[&[9], &[9]]))
            .unwrap_err();
        assert!(matches!(error, WorldError::NotMember { .. }));
        assert_eq!(world.snapshot(), before);
    }

    #[test]
    fn row_and_field_updates_change_only_selected_members() {
        let (mut world, position, mesh, _) = world(6);
        let ids = world
            .spawn(
                3,
                &[
                    ComponentPayload {
                        component: position,
                        data: f32_rows(&[&[0.0; 3], &[0.0; 3], &[0.0; 3]]),
                    },
                    ComponentPayload {
                        component: mesh,
                        data: u32_rows(&[&[1], &[2], &[3]]),
                    },
                ],
            )
            .unwrap();
        world
            .update_rows(
                position,
                &[ids[2], ids[0]],
                f32_rows(&[&[30.0, 31.0, 32.0], &[10.0, 11.0, 12.0]]),
            )
            .unwrap();
        world
            .update_field(mesh, 0, &[ids[1]], ColumnData::U32(vec![20]))
            .unwrap();
        assert_eq!(
            world.read_component(position, Some(&ids)).unwrap().data,
            f32_rows(&[&[10.0, 11.0, 12.0], &[0.0, 0.0, 0.0], &[30.0, 31.0, 32.0]])
        );
        assert_eq!(
            world.read_component(mesh, Some(&ids)).unwrap().data,
            u32_rows(&[&[1], &[20], &[3]])
        );
    }

    #[test]
    fn malformed_batches_are_atomic_and_change_nothing() {
        let (mut world, position, mesh, _) = world(4);
        let ids = world
            .spawn(
                2,
                &[
                    ComponentPayload {
                        component: position,
                        data: f32_rows(&[&[0.0; 3], &[0.0; 3]]),
                    },
                    ComponentPayload {
                        component: mesh,
                        data: u32_rows(&[&[1], &[2]]),
                    },
                ],
            )
            .unwrap();
        let before = world.snapshot();
        let before_position = world.read_component(position, Some(&ids)).unwrap();

        // Wrong row length.
        let error = world
            .update_rows(position, &ids, f32_rows(&[&[1.0, 2.0, 3.0]]))
            .unwrap_err();
        assert!(matches!(error, WorldError::PayloadLength { .. }));
        assert_eq!(world.snapshot(), before);

        // Wrong element kind.
        let error = world
            .update_rows(position, &ids, ColumnData::U32(vec![0; 6]))
            .unwrap_err();
        assert!(matches!(error, WorldError::PayloadKind { .. }));
        assert_eq!(world.snapshot(), before);

        // Field outside the row.
        let error = world
            .update_field(position, 3, &ids, ColumnData::F32(vec![1.0, 2.0]))
            .unwrap_err();
        assert!(matches!(error, WorldError::FieldOutOfRange { .. }));
        assert_eq!(world.snapshot(), before);

        // One stale identity rejects the whole batch, including a good first
        // row and its data. This is the batch-validation defect control.
        let stale = EntityId::new(ids[1].slot(), ids[1].generation() + 1);
        let error = world
            .update_rows(
                position,
                &[ids[0], stale],
                f32_rows(&[&[9.0; 3], &[9.0; 3]]),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            WorldError::Entity(EntityError::Stale { .. })
        ));
        assert_eq!(world.snapshot(), before);
        assert_eq!(
            world.read_component(position, Some(&ids)).unwrap(),
            before_position
        );

        // Duplicate component payload on spawn rejects before reservation.
        let error = world
            .spawn(
                1,
                &[
                    ComponentPayload {
                        component: position,
                        data: f32_rows(&[&[1.0; 3]]),
                    },
                    ComponentPayload {
                        component: position,
                        data: f32_rows(&[&[2.0; 3]]),
                    },
                ],
            )
            .unwrap_err();
        assert!(matches!(error, WorldError::DuplicatePayload { .. }));
        assert_eq!(world.snapshot(), before);
    }

    #[test]
    fn destroy_clears_and_reuse_starts_from_defaults() {
        let (mut world, position, mesh, _) = world(4);
        let ids = world
            .spawn(
                2,
                &[
                    ComponentPayload {
                        component: position,
                        data: f32_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]),
                    },
                    ComponentPayload {
                        component: mesh,
                        data: u32_rows(&[&[7], &[8]]),
                    },
                ],
            )
            .unwrap();
        world.destroy(&[ids[0]]).unwrap();
        assert_eq!(world.alive_count(), 1);
        let reused = world
            .spawn(
                1,
                &[ComponentPayload {
                    component: position,
                    data: f32_rows(&[&[9.0, 10.0, 11.0]]),
                }],
            )
            .unwrap();
        assert_eq!(reused[0].slot(), ids[0].slot());
        assert_ne!(reused[0].generation(), ids[0].generation());
        let reused_read = world.read_component(position, Some(&reused)).unwrap();
        assert_eq!(reused_read.data, f32_rows(&[&[9.0, 10.0, 11.0]]));
        let mesh_read = world.read_component(mesh, Some(&reused)).unwrap();
        assert_eq!(mesh_read.membership, vec![false]);
        assert_eq!(mesh_read.data, u32_rows(&[&[0]]));

        // The old identity is stale for every command, including destroy.
        let before = world.snapshot();
        assert!(matches!(
            world.destroy(&[ids[0]]).unwrap_err(),
            WorldError::Entity(EntityError::Stale { .. })
        ));
        assert_eq!(world.snapshot(), before);
    }

    #[test]
    fn capacity_overflow_is_bounded_and_atomic() {
        let mut world = World::new(2).unwrap();
        world.register("value", ElementType::F32, 1).unwrap();
        world.spawn(2, &[]).unwrap();
        let before = world.snapshot();
        let error = world.spawn(1, &[]).unwrap_err();
        assert!(matches!(
            error,
            WorldError::Entity(EntityError::Capacity { .. })
        ));
        assert_eq!(world.snapshot(), before);

        let error = world.spawn(u32::MAX, &[]).unwrap_err();
        assert!(matches!(
            error,
            WorldError::Entity(EntityError::Capacity { .. })
        ));
        assert_eq!(world.snapshot(), before);
    }

    #[test]
    fn readouts_and_snapshots_are_owned_copies() {
        let (mut world, position, _, _) = world(4);
        let ids = world
            .spawn(
                1,
                &[ComponentPayload {
                    component: position,
                    data: f32_rows(&[&[1.0, 2.0, 3.0]]),
                }],
            )
            .unwrap();
        let mut read = world.read_component(position, Some(&ids)).unwrap();
        if let ColumnData::F32(values) = &mut read.data {
            values[0] = -99.0;
        }
        read.membership[0] = false;
        let fresh = world.read_component(position, Some(&ids)).unwrap();
        assert_eq!(fresh.data, f32_rows(&[&[1.0, 2.0, 3.0]]));
        assert_eq!(fresh.membership, vec![true]);

        let mut snap = world.snapshot();
        snap.entities.clear();
        assert_eq!(world.snapshot().entities, ids);
    }

    #[test]
    fn entity_lifecycle_regression_reuses_lifo_slots() {
        let mut world = World::new(8).unwrap();
        world.register("value", ElementType::U32, 1).unwrap();
        let ids = world.spawn(5, &[]).unwrap();
        assert_eq!(
            ids.iter().map(|id| id.slot()).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
        world.destroy(&[ids[1], ids[3]]).unwrap();
        let reused = world.spawn(2, &[]).unwrap();
        assert_eq!(
            reused.iter().map(|id| id.slot()).collect::<Vec<_>>(),
            vec![3, 1]
        );
        assert!(reused[0].generation() > ids[3].generation());
        assert_eq!(world.high_water(), 5);
    }
}

/// A typed scalar in selection and command IR.
#[derive(Clone, Copy, Debug)]
pub enum Scalar {
    F32(f32),
    U32(u32),
}

/// Supported scalar comparisons at a command boundary.
#[derive(Clone, Copy, Debug)]
pub enum Compare {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}
impl Compare {
    fn test<T: PartialOrd>(self, lhs: T, rhs: T) -> bool {
        match self {
            Self::Eq => lhs == rhs,
            Self::Ne => lhs != rhs,
            Self::Lt => lhs < rhs,
            Self::Le => lhs <= rhs,
            Self::Gt => lhs > rhs,
            Self::Ge => lhs >= rhs,
        }
    }
}

/// A lazy conjunction over fields of registered component rows.
#[derive(Debug)]
pub struct Selection {
    pub component: usize,
    pub predicates: Vec<(usize, usize, Compare, Scalar)>,
}

impl Column {
    fn validate_scalar(&self, field: usize, scalar: Scalar) -> Result<(), String> {
        if field >= self.schema.width as usize {
            return Err(format!("field outside '{}' row", self.schema.name));
        }
        match (&self.data, scalar) {
            (ColumnData::F32(_), Scalar::F32(value)) if value.is_finite() => Ok(()),
            (ColumnData::U32(_), Scalar::U32(_)) => Ok(()),
            _ => Err(format!("invalid scalar for '{}' field", self.schema.name)),
        }
    }
}

impl World {
    fn ir_column(&self, index: usize) -> Result<&Column, String> {
        self.columns
            .get(index)
            .ok_or_else(|| format!("unknown component {index}"))
    }

    pub fn entity_version(&self) -> u64 {
        self.slots.version()
    }

    pub fn column_version(&self, component: usize) -> Result<u64, String> {
        Ok(self.ir_column(component)?.version)
    }

    pub fn select(&self, selection: &Selection) -> Result<Vec<EntityId>, String> {
        let root = self.ir_column(selection.component)?;
        for &(index, field, _, value) in &selection.predicates {
            self.ir_column(index)?.validate_scalar(field, value)?;
        }
        let mut ids = self.slots.alive_ids();
        ids.retain(|id| {
            root.is_member(id.slot())
                && selection
                    .predicates
                    .iter()
                    .all(|&(index, field, cmp, value)| {
                        let column = &self.columns[index];
                        if !column.is_member(id.slot()) {
                            return false;
                        }
                        let offset = id.slot() as usize * column.schema.width as usize + field;
                        match (&column.data, value) {
                            (ColumnData::F32(data), Scalar::F32(value)) => {
                                cmp.test(data[offset], value)
                            }
                            (ColumnData::U32(data), Scalar::U32(value)) => {
                                cmp.test(data[offset], value)
                            }
                            _ => unreachable!("validated selection"),
                        }
                    })
        });
        Ok(ids)
    }

    pub fn set_selected(
        &mut self,
        selection: &Selection,
        component: usize,
        field: usize,
        value: Scalar,
    ) -> Result<usize, String> {
        let target = self.ir_column(component)?;
        target.validate_scalar(field, value)?;
        let ids = self.select(selection)?;
        for id in &ids {
            if !target.is_member(id.slot()) {
                return Err(format!("{id:?} does not carry '{}'", target.schema.name));
            }
        }
        if ids.is_empty() {
            return Ok(0);
        }
        let width = target.schema.width as usize;
        let target = &mut self.columns[component];
        for id in &ids {
            let offset = id.slot() as usize * width + field;
            match (&mut target.data, value) {
                (ColumnData::F32(data), Scalar::F32(value)) => data[offset] = value,
                (ColumnData::U32(data), Scalar::U32(value)) => data[offset] = value,
                _ => unreachable!("validated assignment"),
            }
        }
        self.version += 1;
        target.version = self.version;
        Ok(ids.len())
    }
}
