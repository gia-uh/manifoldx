//! Fixed-capacity entity slots with non-aliasing generations.
//!
//! The allocator never grows. `reserve` hands out free slots, most recently
//! freed first (LIFO), then the lowest never-used slots. A reservation is
//! invisible to callers until `commit` makes it live. `cancel` returns a
//! reservation to the free set; `destroy` does the same for a live entity.
//!
//! Every slot carries a generation. A new reservation increments it, so an
//! identity from an earlier life never matches the replacement in that slot.
//! When a slot would have to wrap past `u64::MAX`, it is retired instead: no
//! handle is ever reused with the same generation.
//!
//! All batch mutations validate the whole batch first. A failed call leaves
//! slots, free order, high-water, generations and `version` exactly as they
//! were.

use std::fmt;

/// One entity identity: a slot plus the generation that slot had when the
/// identity was issued.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntityId {
    slot: u32,
    generation: u64,
}

impl EntityId {
    /// Build an identity from its slot and generation.
    pub const fn new(slot: u32, generation: u64) -> Self {
        Self { slot, generation }
    }

    /// The slot this identity refers to.
    pub const fn slot(self) -> u32 {
        self.slot
    }

    /// The generation this identity was issued with.
    pub const fn generation(self) -> u64 {
        self.generation
    }
}

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "slot {} generation {}", self.slot, self.generation)
    }
}

/// The lifecycle state of one slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotState {
    /// Available for a future reservation.
    Free,
    /// Handed out by `reserve`, not yet live or canceled.
    Reserved,
    /// Live, after a successful `commit`.
    Alive,
    /// Permanently unavailable because its generation cannot be incremented.
    Retired,
}

impl SlotState {
    /// A stable lowercase name, for snapshots and diagnostics.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Free => "free",
            Self::Reserved => "reserved",
            Self::Alive => "alive",
            Self::Retired => "retired",
        }
    }
}

/// An owned copy of the allocator's observable state.
///
/// The vectors are copies: dropping or mutating the store, or mutating a
/// snapshot, cannot affect the other.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    /// Fixed capacity declared at construction.
    pub capacity: u32,
    /// Every slot index, `0..capacity`.
    pub slots: Vec<u32>,
    /// The current generation of each slot, parallel to `slots`.
    pub generations: Vec<u64>,
    /// The lifecycle state of each slot, parallel to `slots`.
    pub states: Vec<SlotState>,
    /// Free slots in reuse order; `reserve` pops from the end first.
    pub free_order: Vec<u32>,
    /// One past the highest slot ever committed live. Never decreases.
    pub high_water: u32,
    /// Incremented by each successful mutation; unchanged by failures.
    pub version: u64,
}

/// A rejected slot operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EntityError {
    /// The requested batch is larger than the usable free set.
    Capacity {
        /// Slots requested.
        requested: u32,
        /// Usable slots available without changing state.
        available: u32,
        /// Declared fixed capacity.
        capacity: u32,
    },
    /// A slot index is outside `0..capacity`.
    SlotOutOfRange {
        /// The rejected slot.
        slot: u32,
        /// Declared fixed capacity.
        capacity: u32,
    },
    /// The identity's generation does not match the slot's current generation.
    Stale {
        /// The rejected identity.
        id: EntityId,
        /// The slot's current generation.
        current: u64,
    },
    /// The identity is current but the slot is not reserved.
    NotReserved {
        /// The rejected identity.
        id: EntityId,
    },
    /// The identity is current but the slot is not alive.
    NotAlive {
        /// The rejected identity.
        id: EntityId,
    },
    /// Two identities in one batch name the same slot.
    Duplicate {
        /// The repeated slot.
        slot: u32,
    },
    /// The fixed slot table could not be allocated.
    Allocation {
        /// Requested capacity.
        capacity: u32,
    },
}

impl fmt::Display for EntityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Capacity {
                requested,
                available,
                capacity,
            } => write!(
                f,
                "Max entities {capacity} reached: requested {requested} with {available} free slots"
            ),
            Self::SlotOutOfRange { slot, capacity } => {
                write!(f, "slot {slot} is outside capacity {capacity}")
            }
            Self::Stale { id, current } => write!(
                f,
                "stale entity slot {} generation {}: current generation {current}",
                id.slot, id.generation
            ),
            Self::NotReserved { id } => {
                write!(f, "entity {id} is not reserved")
            }
            Self::NotAlive { id } => {
                write!(f, "entity {id} is not alive")
            }
            Self::Duplicate { slot } => write!(f, "duplicate slot {slot} in batch"),
            Self::Allocation { capacity } => {
                write!(f, "cannot allocate {capacity} fixed entity slots")
            }
        }
    }
}

impl std::error::Error for EntityError {}

#[derive(Clone, Copy, Debug)]
struct Slot {
    generation: u64,
    state: SlotState,
}

impl Slot {
    const fn free() -> Self {
        Self {
            generation: 0,
            state: SlotState::Free,
        }
    }
}

/// A fixed-capacity entity slot allocator.
#[derive(Debug)]
pub struct EntitySlots {
    capacity: u32,
    slots: Vec<Slot>,
    /// Free slots in reuse order; `reserve` pops from the end.
    free: Vec<u32>,
    /// Lowest slot never handed out by `reserve`.
    next_fresh: u32,
    /// One past the highest slot ever committed live.
    high_water: u32,
    /// Changes on every successful mutation.
    version: u64,
}

impl EntitySlots {
    /// Allocate the fixed slot table.
    pub fn new(capacity: u32) -> Result<Self, EntityError> {
        let len = capacity as usize;
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(len)
            .map_err(|_| EntityError::Allocation { capacity })?;
        slots.resize(len, Slot::free());
        Ok(Self {
            capacity,
            slots,
            free: Vec::new(),
            next_fresh: 0,
            high_water: 0,
            version: 0,
        })
    }

    /// Fixed capacity.
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Current mutation version.
    pub const fn version(&self) -> u64 {
        self.version
    }

    /// One past the highest slot ever committed live.
    pub const fn high_water(&self) -> u32 {
        self.high_water
    }

    /// Number of live entities.
    pub fn alive_count(&self) -> u32 {
        self.slots
            .iter()
            .filter(|slot| slot.state == SlotState::Alive)
            .count() as u32
    }

    /// Number of reserved but not yet committed entities.
    pub fn reserved_count(&self) -> u32 {
        self.slots
            .iter()
            .filter(|slot| slot.state == SlotState::Reserved)
            .count() as u32
    }

    /// The state of one slot, or `None` when it is outside the table.
    pub fn state_of(&self, slot: u32) -> Option<SlotState> {
        self.slots.get(slot as usize).map(|slot| slot.state)
    }

    /// Validate that a batch names unique, live, generation-current entities.
    ///
    /// The whole batch is checked without changing any slot. A stale
    /// generation, a duplicate slot, an out-of-range slot or a non-live slot
    /// rejects the batch with the allocator untouched.
    pub fn validate_alive(&self, ids: &[EntityId]) -> Result<(), EntityError> {
        if ids.is_empty() {
            return Ok(());
        }
        self.validate_unique(ids)?;
        for id in ids {
            self.require_state(*id, SlotState::Alive, EntityError::NotAlive { id: *id })?;
        }
        Ok(())
    }

    /// Every live identity in ascending slot order, as an owned list.
    pub fn alive_ids(&self) -> Vec<EntityId> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.state == SlotState::Alive)
            .map(|(slot, state)| EntityId::new(slot as u32, state.generation))
            .collect()
    }

    /// The current identity of one slot, including free slots.
    pub fn current_identity(&self, slot: u32) -> Result<EntityId, EntityError> {
        let current = self
            .slots
            .get(slot as usize)
            .ok_or(EntityError::SlotOutOfRange {
                slot,
                capacity: self.capacity,
            })?;
        Ok(EntityId::new(slot, current.generation))
    }

    /// Reserve `n` slots. Free slots are consumed LIFO, then the lowest
    /// never-used slots are consumed in ascending order.
    pub fn reserve(&mut self, n: u32) -> Result<Vec<EntityId>, EntityError> {
        if n == 0 {
            return Ok(Vec::new());
        }

        let requested = n as usize;
        // Grow with the slots actually found, never with the untrusted request:
        // an impossible batch must not allocate its own length before the
        // capacity check below.
        let mut selected = Vec::new();
        let mut retired = Vec::new();

        // The free list is a stack, so a reservation consumes one contiguous
        // suffix. Record where that suffix starts and truncate it only after
        // the whole batch validates, leaving a failed reserve untouched.
        let mut consumed = self.free.len();
        for index in (0..self.free.len()).rev() {
            if selected.len() == requested {
                break;
            }
            consumed = index;
            let slot = self.free[index];
            let generation = self.slots[slot as usize].generation;
            if generation == u64::MAX {
                retired.push(slot);
            } else {
                selected.push(slot);
            }
        }

        let mut next_fresh = self.next_fresh;
        while next_fresh < self.capacity && selected.len() < requested {
            let slot = &self.slots[next_fresh as usize];
            match slot.state {
                SlotState::Free => {
                    if slot.generation == u64::MAX {
                        retired.push(next_fresh);
                    } else {
                        selected.push(next_fresh);
                    }
                }
                // A retired slot was already handed out and withdrawn; a
                // reserved or alive slot above the frontier is an invariant
                // break, not something this allocator hands out again.
                SlotState::Retired | SlotState::Reserved | SlotState::Alive => {}
            }
            next_fresh += 1;
        }

        if selected.len() != requested {
            return Err(EntityError::Capacity {
                requested: n,
                available: selected.len() as u32,
                capacity: self.capacity,
            });
        }

        self.free.truncate(consumed);
        for slot in retired {
            self.slots[slot as usize].state = SlotState::Retired;
        }

        let mut ids = Vec::with_capacity(requested);
        for slot in selected {
            let state = &mut self.slots[slot as usize];
            state.generation += 1;
            state.state = SlotState::Reserved;
            ids.push(EntityId::new(slot, state.generation));
        }
        self.next_fresh = next_fresh;
        self.version += 1;
        Ok(ids)
    }

    /// Make a reserved batch live. The whole batch is validated before any
    /// slot changes.
    pub fn commit(&mut self, ids: &[EntityId]) -> Result<(), EntityError> {
        if ids.is_empty() {
            return Ok(());
        }
        self.validate_unique(ids)?;
        for id in ids {
            self.require_state(
                *id,
                SlotState::Reserved,
                EntityError::NotReserved { id: *id },
            )?;
        }
        for id in ids {
            self.slots[id.slot as usize].state = SlotState::Alive;
            self.high_water = self.high_water.max(id.slot + 1);
        }
        self.version += 1;
        Ok(())
    }

    /// Return a reserved batch to the free set without making it live.
    pub fn cancel(&mut self, ids: &[EntityId]) -> Result<(), EntityError> {
        if ids.is_empty() {
            return Ok(());
        }
        self.validate_unique(ids)?;
        for id in ids {
            self.require_state(
                *id,
                SlotState::Reserved,
                EntityError::NotReserved { id: *id },
            )?;
        }
        let mut slots: Vec<u32> = ids.iter().map(|id| id.slot).collect();
        slots.sort_unstable();
        for slot in slots {
            let state = &mut self.slots[slot as usize];
            if state.generation == u64::MAX {
                state.state = SlotState::Retired;
            } else {
                state.state = SlotState::Free;
                self.free.push(slot);
            }
        }
        self.version += 1;
        Ok(())
    }

    /// Mark a live batch dead and return its slots to the free set.
    pub fn destroy(&mut self, ids: &[EntityId]) -> Result<(), EntityError> {
        if ids.is_empty() {
            return Ok(());
        }
        self.validate_unique(ids)?;
        for id in ids {
            self.require_state(*id, SlotState::Alive, EntityError::NotAlive { id: *id })?;
        }
        let mut slots: Vec<u32> = ids.iter().map(|id| id.slot).collect();
        slots.sort_unstable();
        for slot in slots {
            let state = &mut self.slots[slot as usize];
            if state.generation == u64::MAX {
                state.state = SlotState::Retired;
            } else {
                state.state = SlotState::Free;
                self.free.push(slot);
            }
        }
        self.version += 1;
        Ok(())
    }

    /// Reserve and commit in one call.
    pub fn spawn(&mut self, n: u32) -> Result<Vec<EntityId>, EntityError> {
        let ids = self.reserve(n)?;
        self.commit(&ids)?;
        Ok(ids)
    }

    /// An owned snapshot of slots, generations, states, free order and counters.
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            capacity: self.capacity,
            slots: (0..self.capacity).collect(),
            generations: self.slots.iter().map(|slot| slot.generation).collect(),
            states: self.slots.iter().map(|slot| slot.state).collect(),
            free_order: self.free.clone(),
            high_water: self.high_water,
            version: self.version,
        }
    }

    fn validate_unique(&self, ids: &[EntityId]) -> Result<(), EntityError> {
        let mut slots: Vec<u32> = ids.iter().map(|id| id.slot).collect();
        slots.sort_unstable();
        for pair in slots.windows(2) {
            if pair[0] == pair[1] {
                return Err(EntityError::Duplicate { slot: pair[0] });
            }
        }
        for id in ids {
            if id.slot >= self.capacity {
                return Err(EntityError::SlotOutOfRange {
                    slot: id.slot,
                    capacity: self.capacity,
                });
            }
        }
        Ok(())
    }

    fn require_state(
        &self,
        id: EntityId,
        expected: SlotState,
        wrong_state: EntityError,
    ) -> Result<(), EntityError> {
        let slot = &self.slots[id.slot as usize];
        if slot.generation != id.generation {
            return Err(EntityError::Stale {
                id,
                current: slot.generation,
            });
        }
        if slot.state != expected {
            return Err(wrong_state);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slots_of(ids: &[EntityId]) -> Vec<u32> {
        ids.iter().map(|id| id.slot()).collect()
    }

    fn id_at(store: &EntitySlots, slot: u32) -> EntityId {
        store.current_identity(slot).unwrap()
    }

    #[test]
    fn reserve_commit_destroy_and_reuse() {
        let mut store = EntitySlots::new(20).unwrap();
        let first = store.spawn(10).unwrap();
        assert_eq!(slots_of(&first), (0..10).collect::<Vec<_>>());
        assert_eq!(store.alive_count(), 10);
        assert_eq!(store.high_water(), 10);

        store
            .destroy(&[id_at(&store, 2), id_at(&store, 5), id_at(&store, 7)])
            .unwrap();
        let ids = store.spawn(5).unwrap();
        assert_eq!(slots_of(&ids), vec![7, 5, 2, 10, 11]);
        assert_eq!(store.alive_count(), 12);
        assert_eq!(store.high_water(), 12);

        store.destroy(&[id_at(&store, 11)]).unwrap();
        assert_eq!(store.high_water(), 12);
        assert_eq!(store.state_of(11), Some(SlotState::Free));
    }

    #[test]
    fn zero_capacity_and_zero_batches() {
        let mut empty = EntitySlots::new(0).unwrap();
        assert_eq!(empty.capacity(), 0);
        assert!(empty.reserve(0).unwrap().is_empty());
        assert!(empty.spawn(0).unwrap().is_empty());
        let before = empty.snapshot();
        let error = empty.reserve(1).unwrap_err();
        assert!(matches!(error, EntityError::Capacity { .. }));
        assert_eq!(empty.snapshot(), before);

        let mut one = EntitySlots::new(1).unwrap();
        let first = one.spawn(1).unwrap();
        assert_eq!(slots_of(&first), vec![0]);
        one.destroy(&first).unwrap();
        let second = one.spawn(1).unwrap();
        assert_eq!(slots_of(&second), vec![0]);
        assert_ne!(first[0].generation(), second[0].generation());
    }

    #[test]
    fn reservations_are_disjoint_and_invisible() {
        let mut store = EntitySlots::new(20).unwrap();
        let first = store.reserve(2).unwrap();
        let second = store.reserve(3).unwrap();
        assert_eq!(slots_of(&first), vec![0, 1]);
        assert_eq!(slots_of(&second), vec![2, 3, 4]);
        assert_eq!(store.alive_count(), 0);
        assert_eq!(store.reserved_count(), 5);
        assert_eq!(store.high_water(), 0);
        assert_eq!(store.state_of(0), Some(SlotState::Reserved));

        store.commit(&second).unwrap();
        assert_eq!(store.alive_count(), 3);
        assert_eq!(store.reserved_count(), 2);
        assert_eq!(store.high_water(), 5);
        store.commit(&first).unwrap();
        assert_eq!(store.alive_count(), 5);
        assert_eq!(store.reserved_count(), 0);
        assert_eq!(store.high_water(), 5);
    }

    #[test]
    fn cancel_returns_slots_in_deterministic_lifo_order() {
        let mut store = EntitySlots::new(6).unwrap();
        let handles = store.reserve(3).unwrap();
        assert_eq!(slots_of(&handles), vec![0, 1, 2]);
        store.cancel(&handles).unwrap();
        assert_eq!(store.alive_count(), 0);
        assert_eq!(store.high_water(), 0);
        let reused = store.reserve(3).unwrap();
        assert_eq!(slots_of(&reused), vec![2, 1, 0]);
    }

    #[test]
    fn failed_batches_change_nothing() {
        let mut store = EntitySlots::new(10).unwrap();
        store.spawn(8).unwrap();
        store.destroy(&[id_at(&store, 0)]).unwrap();
        let before = store.snapshot();

        let error = store.spawn(4).unwrap_err();
        assert!(matches!(error, EntityError::Capacity { .. }));
        assert_eq!(store.snapshot(), before);

        let stale = EntityId::new(0, before.generations[0] + 1);
        let error = store.destroy(&[id_at(&store, 1), stale]).unwrap_err();
        assert!(matches!(error, EntityError::Stale { .. }));
        assert_eq!(store.snapshot(), before);

        let reserved = store.reserve(1).unwrap();
        let after_reserve = store.snapshot();
        let stale_reserved = EntityId::new(reserved[0].slot(), reserved[0].generation() + 1);
        let error = store.commit(&[stale_reserved]).unwrap_err();
        assert!(matches!(error, EntityError::Stale { .. }));
        assert_eq!(store.snapshot(), after_reserve);

        let error = store.commit(&[reserved[0], reserved[0]]).unwrap_err();
        assert!(matches!(error, EntityError::Duplicate { .. }));
        assert_eq!(store.snapshot(), after_reserve);

        store.cancel(&reserved).unwrap();
        let after_cancel = store.snapshot();
        let error = store.cancel(&reserved).unwrap_err();
        assert!(matches!(error, EntityError::NotReserved { .. }));
        assert_eq!(store.snapshot(), after_cancel);
    }

    #[test]
    fn reserve_rejects_extreme_batches_without_allocating_them() {
        let mut store = EntitySlots::new(1).unwrap();
        let before = store.snapshot();
        let error = store.reserve(u32::MAX).unwrap_err();
        assert!(matches!(error, EntityError::Capacity { .. }));
        assert_eq!(store.snapshot(), before);

        let ids = store.reserve(1).unwrap();
        assert_eq!(slots_of(&ids), vec![0]);
        assert_eq!(store.reserved_count(), 1);
    }

    #[test]
    fn reserve_pops_a_free_stack_suffix_then_fresh_and_fails_atomically() {
        let mut store = EntitySlots::new(8).unwrap();
        store.free = vec![0, 1, 2, 3];
        store.next_fresh = 4;
        store.slots[0].generation = u64::MAX;

        // The free suffix is 3, 2; slot 0 stays below the consumed boundary.
        let ids = store.reserve(2).unwrap();
        assert_eq!(slots_of(&ids), vec![3, 2]);
        assert_eq!(store.snapshot().free_order, vec![0, 1]);

        // Too large: the scan reaches the retired slot 0 but must not apply it.
        let before = store.snapshot();
        let error = store.reserve(9).unwrap_err();
        assert!(matches!(
            error,
            EntityError::Capacity {
                requested: 9,
                available: 5,
                capacity: 8
            }
        ));
        assert_eq!(store.snapshot(), before);
        assert_eq!(store.state_of(0), Some(SlotState::Free));

        // The next batch consumes slot 1, retires 0, then extends into the
        // fresh range in ascending order.
        let ids = store.reserve(5).unwrap();
        assert_eq!(slots_of(&ids), vec![1, 4, 5, 6, 7]);
        assert!(store.snapshot().free_order.is_empty());
        assert_eq!(store.state_of(0), Some(SlotState::Retired));
    }

    #[test]
    fn stale_recycled_and_canceled_identities_are_rejected() {
        let mut store = EntitySlots::new(2).unwrap();

        let canceled = store.reserve(1).unwrap()[0];
        store.cancel(&[canceled]).unwrap();
        let replacement = store.reserve(1).unwrap()[0];
        assert_eq!(replacement.slot(), canceled.slot());
        assert_ne!(replacement.generation(), canceled.generation());

        let before = store.snapshot();
        assert!(matches!(
            store.commit(&[canceled]).unwrap_err(),
            EntityError::Stale { .. }
        ));
        assert_eq!(store.snapshot(), before);

        store.commit(&[replacement]).unwrap();
        let live_before = store.snapshot();
        assert!(matches!(
            store.destroy(&[canceled]).unwrap_err(),
            EntityError::Stale { .. }
        ));
        assert_eq!(store.snapshot(), live_before);

        store.destroy(&[replacement]).unwrap();
        let recycled = store.reserve(1).unwrap()[0];
        assert_eq!(recycled.slot(), canceled.slot());
        assert_ne!(recycled.generation(), replacement.generation());
        let before = store.snapshot();
        assert!(matches!(
            store.cancel(&[replacement]).unwrap_err(),
            EntityError::Stale { .. }
        ));
        assert_eq!(store.snapshot(), before);
        store.cancel(&[recycled]).unwrap();
    }

    #[test]
    fn duplicate_identities_are_rejected_before_mutation() {
        let mut store = EntitySlots::new(4).unwrap();
        let ids = store.spawn(2).unwrap();
        let before = store.snapshot();
        assert!(matches!(
            store.destroy(&[ids[0], ids[0]]).unwrap_err(),
            EntityError::Duplicate { .. }
        ));
        assert_eq!(store.snapshot(), before);

        let reserved = store.reserve(2).unwrap();
        let before = store.snapshot();
        let same_slot_other_generation =
            EntityId::new(reserved[0].slot(), reserved[0].generation() + 1);
        assert!(matches!(
            store
                .commit(&[reserved[0], same_slot_other_generation])
                .unwrap_err(),
            EntityError::Duplicate { .. }
        ));
        assert_eq!(store.snapshot(), before);
        assert!(matches!(
            store.cancel(&[reserved[1], reserved[1]]).unwrap_err(),
            EntityError::Duplicate { .. }
        ));
        assert_eq!(store.snapshot(), before);
        store.cancel(&reserved).unwrap();
    }

    #[test]
    fn exhausted_generation_retires_the_slot_without_wrapping() {
        let mut store = EntitySlots::new(2).unwrap();
        store.slots[0].generation = u64::MAX - 1;

        let last = store.spawn(1).unwrap()[0];
        assert_eq!(last.slot(), 0);
        assert_eq!(last.generation(), u64::MAX);
        store.destroy(&[last]).unwrap();
        assert_eq!(store.state_of(0), Some(SlotState::Retired));

        let before = store.snapshot();
        assert!(matches!(
            store.destroy(&[last]).unwrap_err(),
            EntityError::NotAlive { .. }
        ));
        assert_eq!(store.snapshot(), before);

        let next = store.spawn(1).unwrap()[0];
        assert_eq!(next.slot(), 1);
        assert_ne!(next.generation(), 0);
        assert_eq!(store.state_of(0), Some(SlotState::Retired));
        assert_eq!(store.state_of(1), Some(SlotState::Alive));
    }

    #[test]
    fn an_exhausted_free_slot_is_retired_only_by_a_successful_reserve() {
        let mut store = EntitySlots::new(2).unwrap();
        store.slots[0].generation = u64::MAX;

        let before = store.snapshot();
        assert!(matches!(
            store.reserve(2).unwrap_err(),
            EntityError::Capacity { .. }
        ));
        assert_eq!(store.snapshot(), before);
        assert_eq!(store.state_of(0), Some(SlotState::Free));

        let ids = store.reserve(1).unwrap();
        assert_eq!(ids[0].slot(), 1);
        assert_eq!(store.state_of(0), Some(SlotState::Retired));
    }

    #[test]
    fn snapshots_are_owned_copies() {
        let mut store = EntitySlots::new(3).unwrap();
        store.spawn(2).unwrap();
        let snapshot = store.snapshot();
        drop(store);

        assert_eq!(snapshot.capacity, 3);
        assert_eq!(snapshot.slots, vec![0, 1, 2]);
        assert_eq!(snapshot.states[0], SlotState::Alive);
        assert_eq!(snapshot.states[2], SlotState::Free);
        assert_eq!(snapshot.high_water, 2);

        let mut copy = snapshot.clone();
        copy.slots[0] = 99;
        copy.states[0] = SlotState::Retired;
        assert_eq!(snapshot.slots[0], 0);
        assert_eq!(snapshot.states[0], SlotState::Alive);
    }
}
