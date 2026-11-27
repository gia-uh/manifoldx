//! The C boundary between manifoldx and a game's native crate.
//!
//! Both wheels compile against these `#[repr(C)]` types and the versioned
//! capsule name. No Rust type with a Rust layout crosses: the engine owns the
//! output buffers and the game only calls the sinks it is handed.

use std::ffi::{c_void, CStr};

/// The capsule a game exports for a cell builder. The engine checks this name
/// before it calls the pointer; a mismatch fails at registration, not in a frame.
pub const BUILDER_NAME: &CStr = c"manifoldx.builder.v1";

/// What the engine asks for. `x` and `y` are the cell's world origin, `size` its
/// side in metres, `lod` a level of detail and `seed` the world seed.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CellRequest {
    pub x: f64,
    pub y: f64,
    pub size: f32,
    pub lod: u32,
    pub seed: u64,
}

/// The sinks the engine hands a builder. The builder never returns memory of its
/// own: it pushes into these and the engine copies into its store or the GPU.
#[repr(C)]
pub struct BuildOutput {
    pub ctx: *mut c_void,
    /// positions: `vertices` triples of f32; normals: the same count of triples;
    /// indices: `index_count` u32. Returns a mesh id.
    pub push_mesh: unsafe extern "C" fn(
        ctx: *mut c_void,
        positions: *const f32,
        vertices: usize,
        normals: *const f32,
        indices: *const u32,
        index_count: usize,
        material: u32,
    ) -> u32,
    /// `rows` is `row_count` rows of `row_bytes` bytes for the given archetype.
    pub spawn: unsafe extern "C" fn(
        ctx: *mut c_void,
        archetype: u32,
        rows: *const u8,
        row_count: usize,
        row_bytes: usize,
    ),
    /// A human-readable error, copied by the engine before the builder returns.
    pub error: unsafe extern "C" fn(ctx: *mut c_void, message: *const u8, len: usize),
}

/// The function a capsule points at. `0` is success; anything else is a failure
/// whose message went through `error`.
pub type BuilderFn = unsafe extern "C" fn(*const CellRequest, *mut BuildOutput) -> i32;

/// A safe wrapper over the engine's sinks, for the game side.
pub struct Output {
    ptr: *mut BuildOutput,
}

impl Output {
    /// # Safety
    /// `ptr` must point at a live `BuildOutput` for the duration of the call.
    pub unsafe fn new(ptr: *mut BuildOutput) -> Self {
        Self { ptr }
    }

    pub fn mesh(
        &mut self,
        positions: &[f32],
        normals: &[f32],
        indices: &[u32],
        material: u32,
    ) -> u32 {
        if positions.len() != normals.len() || !positions.len().is_multiple_of(3) {
            self.error("positions and normals must contain equal numbers of triples");
            return u32::MAX;
        }
        unsafe {
            ((*self.ptr).push_mesh)(
                (*self.ptr).ctx,
                positions.as_ptr(),
                positions.len() / 3,
                normals.as_ptr(),
                indices.as_ptr(),
                indices.len(),
                material,
            )
        }
    }

    pub fn spawn(&mut self, archetype: u32, rows: &[u8], row_bytes: usize) {
        if row_bytes == 0 || !rows.len().is_multiple_of(row_bytes) {
            self.error("row bytes must be a whole number of nonzero-width rows");
            return;
        }
        unsafe {
            ((*self.ptr).spawn)(
                (*self.ptr).ctx,
                archetype,
                rows.as_ptr(),
                rows.len() / row_bytes,
                row_bytes,
            )
        }
    }

    pub fn error(&mut self, message: &str) {
        unsafe { ((*self.ptr).error)((*self.ptr).ctx, message.as_ptr(), message.len()) }
    }
}

/// A convenience for the wrapper a game exports: run `body`, catch a panic, and
/// turn both failures into the error sink and a non-zero code.
pub fn guarded<F>(out: &mut Output, body: F) -> i32
where
    F: FnOnce(&mut Output) -> Result<(), String> + std::panic::UnwindSafe,
{
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(out))) {
        Ok(Ok(())) => 0,
        Ok(Err(message)) => {
            out.error(&message);
            1
        }
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "native builder panicked".to_string());
            out.error(&format!("panic: {message}"));
            2
        }
    }
}
