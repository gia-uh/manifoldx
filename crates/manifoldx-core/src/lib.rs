//! The ManifoldX production runtime core.
//!
//! This crate owns the engine internals the migration plan (2026-11-24)
//! moves to Rust: the entity store, scheduler, physics and renderer. It must
//! stay free of any Python dependency; the bindings live in
//! `crates/manifoldx-python` and consume this crate.
//!
//! The runtime itself is not ported yet. Phase 0 ships the packaging only:
//! an installable wheel whose extension reports this crate's real state, so
//! nothing downstream can mistake it for a working native runtime.

pub mod ecs;
pub mod runtime;
pub mod vehicles;

/// The runtime's name, as the bindings' bootstrap reports it.
pub const RUNTIME_NAME: &str = "manifoldx-core";

/// This crate's version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The runtime's status, straight from the crate's own build.
pub const RUNTIME_STATUS: &str = "not implemented";

#[cfg(test)]
mod tests {
    #[test]
    fn runtime_is_named_and_not_claimed() {
        assert_eq!(super::RUNTIME_NAME, "manifoldx-core");
        assert_eq!(super::RUNTIME_STATUS, "not implemented");
        assert!(!super::VERSION.is_empty());
    }
}
