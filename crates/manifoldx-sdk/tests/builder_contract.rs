use std::ffi::c_void;

use manifoldx_sdk::{guarded, BuildOutput, BuilderFn, CellRequest, Output, BUILDER_NAME};

#[derive(Default)]
struct Staged {
    rows: Vec<u8>,
    error: Vec<u8>,
}

unsafe extern "C" fn mesh(
    _: *mut c_void,
    _: *const f32,
    _: usize,
    _: *const f32,
    _: *const u32,
    _: usize,
    _: u32,
) -> u32 {
    0
}

unsafe extern "C" fn spawn(ctx: *mut c_void, _: u32, rows: *const u8, count: usize, width: usize) {
    let staged = unsafe { &mut *ctx.cast::<Staged>() };
    let rows = unsafe { std::slice::from_raw_parts(rows, count * width) };
    staged.rows.extend_from_slice(rows);
}

unsafe extern "C" fn error(ctx: *mut c_void, message: *const u8, len: usize) {
    let staged = unsafe { &mut *ctx.cast::<Staged>() };
    staged
        .error
        .extend_from_slice(unsafe { std::slice::from_raw_parts(message, len) });
}

unsafe extern "C" fn build(req: *const CellRequest, out: *mut BuildOutput) -> i32 {
    let req = unsafe { &*req };
    let mut out = unsafe { Output::new(out) };
    guarded(&mut out, |out| {
        let mut bytes = req.seed.to_le_bytes();
        out.spawn(1, &bytes, bytes.len());
        bytes.fill(0);
        match req.lod {
            0 => Ok(()),
            1 => Err("rejected".into()),
            _ => panic!("fixture panic"),
        }
    })
}

fn call(lod: u32) -> (i32, Staged) {
    let mut staged = Staged::default();
    let mut out = BuildOutput {
        ctx: (&mut staged as *mut Staged).cast(),
        push_mesh: mesh,
        spawn,
        error,
    };
    let req = CellRequest {
        x: 0.0,
        y: 0.0,
        size: 1.0,
        lod,
        seed: 42,
    };
    let func: BuilderFn = build;
    let status = unsafe { func(&req, &mut out) };
    (status, staged)
}

#[test]
fn builder_v1_copies_output_and_contains_failures() {
    assert_eq!(BUILDER_NAME, c"manifoldx.builder.v1");
    let (status, staged) = call(0);
    assert_eq!(status, 0);
    assert_eq!(staged.rows, 42_u64.to_le_bytes());
    assert!(staged.error.is_empty());

    let (status, staged) = call(1);
    assert_eq!(status, 1);
    assert_eq!(staged.error, b"rejected");
    let (status, staged) = call(2);
    assert_eq!(status, 2);
    assert_eq!(staged.error, b"panic: fixture panic");

    let (status, staged) = call(0);
    assert_eq!(status, 0);
    assert_eq!(staged.rows, 42_u64.to_le_bytes());
}
