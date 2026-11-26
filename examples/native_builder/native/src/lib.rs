//! A game's native cell builder for the S-b spike.
//!
//! It generates a 1 km² cell: a 25 m street grid, one box per block (some blocks
//! are parks) and a lamp every third intersection. The Python builder in
//! `examples/native_builder/cell.py` implements the same algorithm, and the spike
//! compares their bytes.
//!
//! The module exposes three capsules: the real builder, the same builder under a
//! wrong version name, and one that panics.

use std::ffi::c_void;
use std::ptr::NonNull;

use manifoldx_sdk::{guarded, BuildOutput, CellRequest, Output};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyModule};

const BLOCK: f32 = 25.0;
const ROAD: f32 = 8.0;
const HALF: f32 = ROAD / 2.0;
const INSET: f32 = HALF + 2.0;

fn hash(i: usize, j: usize, seed: u64) -> u64 {
    (i as u64)
        .wrapping_mul(73_856_093)
        .wrapping_add((j as u64).wrapping_mul(19_349_663))
        .wrapping_add(seed)
}

fn face(
    positions: &mut Vec<f32>,
    normals: &mut Vec<f32>,
    indices: &mut Vec<u32>,
    verts: [[f32; 3]; 4],
    normal: [f32; 3],
) {
    let base = (positions.len() / 3) as u32;
    for vertex in verts {
        positions.extend_from_slice(&vertex);
        normals.extend_from_slice(&normal);
    }
    indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn build_cell_impl(req: &CellRequest, out: &mut Output) -> Result<(), String> {
    if !(req.size > 0.0) {
        return Err(format!("size must be positive, got {}", req.size));
    }
    let n = (req.size / BLOCK) as usize;
    let size = req.size;
    let y = 0.02_f32;

    // Roads: a quad per grid line, both directions.
    let mut rp = Vec::new();
    let mut rn = Vec::new();
    let mut ri = Vec::new();
    for i in 0..=n {
        let z = i as f32 * BLOCK;
        face(
            &mut rp, &mut rn, &mut ri,
            [[0.0, y, z - HALF], [size, y, z - HALF], [size, y, z + HALF], [0.0, y, z + HALF]],
            [0.0, 1.0, 0.0],
        );
        let x = i as f32 * BLOCK;
        face(
            &mut rp, &mut rn, &mut ri,
            [[x - HALF, y, 0.0], [x + HALF, y, 0.0], [x + HALF, y, size], [x - HALF, y, size]],
            [0.0, 1.0, 0.0],
        );
    }
    out.mesh(&rp, &rn, &ri, 0);

    // Buildings: one box per block, with a deterministic height and a park every 11th block.
    let mut bp = Vec::new();
    let mut bn = Vec::new();
    let mut bi = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let h = hash(i, j, req.seed);
            if h % 11 == 0 {
                continue;
            }
            let height = 8.0 + (h % 32) as f32;
            let x0 = i as f32 * BLOCK + INSET;
            let x1 = (i + 1) as f32 * BLOCK - INSET;
            let z0 = j as f32 * BLOCK + INSET;
            let z1 = (j + 1) as f32 * BLOCK - INSET;

            face(&mut bp, &mut bn, &mut bi,
                 [[x0, 0.0, z0], [x1, 0.0, z0], [x1, 0.0, z1], [x0, 0.0, z1]], [0.0, -1.0, 0.0]);
            face(&mut bp, &mut bn, &mut bi,
                 [[x0, height, z0], [x1, height, z0], [x1, height, z1], [x0, height, z1]], [0.0, 1.0, 0.0]);
            face(&mut bp, &mut bn, &mut bi,
                 [[x0, 0.0, z0], [x0, height, z0], [x0, height, z1], [x0, 0.0, z1]], [-1.0, 0.0, 0.0]);
            face(&mut bp, &mut bn, &mut bi,
                 [[x1, 0.0, z0], [x1, height, z0], [x1, height, z1], [x1, 0.0, z1]], [1.0, 0.0, 0.0]);
            face(&mut bp, &mut bn, &mut bi,
                 [[x0, 0.0, z0], [x1, 0.0, z0], [x1, height, z0], [x0, height, z0]], [0.0, 0.0, -1.0]);
            face(&mut bp, &mut bn, &mut bi,
                 [[x0, 0.0, z1], [x1, 0.0, z1], [x1, height, z1], [x0, height, z1]], [0.0, 0.0, 1.0]);
        }
    }
    out.mesh(&bp, &bn, &bi, 1);

    // Lamps: a row of 12 bytes (x f32, z f32, kind u32) every third intersection.
    let mut rows = Vec::new();
    for i in 0..=n {
        for j in 0..=n {
            if (i as u64 + j as u64 + req.seed) % 3 != 0 {
                continue;
            }
            rows.extend_from_slice(&(i as f32 * BLOCK).to_le_bytes());
            rows.extend_from_slice(&(j as f32 * BLOCK).to_le_bytes());
            rows.extend_from_slice(&1_u32.to_le_bytes());
        }
    }
    out.spawn(0, &rows, 12);

    Ok(())
}

unsafe extern "C" fn build_cell_c(req: *const CellRequest, out: *mut BuildOutput) -> i32 {
    if req.is_null() || out.is_null() {
        return 3;
    }
    let req = unsafe { &*req };
    let mut output = unsafe { Output::new(out) };
    guarded(&mut output, |output| build_cell_impl(req, output))
}

unsafe extern "C" fn build_cell_panics_c(_req: *const CellRequest, out: *mut BuildOutput) -> i32 {
    let mut output = unsafe { Output::new(out) };
    guarded(&mut output, |_| panic!("boom from the native builder"))
}

unsafe extern "C" fn build_cell_malformed_c(_req: *const CellRequest, out: *mut BuildOutput) -> i32 {
    let output = unsafe { &mut *out };
    unsafe {
        (output.push_mesh)(output.ctx, std::ptr::null(), 3, std::ptr::null(), std::ptr::null(), 3, 0);
    }
    0
}

fn capsule<'py>(
    py: Python<'py>,
    function: unsafe extern "C" fn(*const CellRequest, *mut BuildOutput) -> i32,
    name: &'static std::ffi::CStr,
) -> PyResult<Bound<'py, PyCapsule>> {
    let pointer = NonNull::new(function as *mut c_void).expect("a function pointer is never null");
    unsafe { PyCapsule::new_with_pointer(py, pointer, name) }
}

#[pymodule]
fn native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("build_cell", capsule(py, build_cell_c, manifoldx_sdk::BUILDER_NAME)?)?;
    m.add(
        "build_cell_panics",
        capsule(py, build_cell_panics_c, manifoldx_sdk::BUILDER_NAME)?,
    )?;
    m.add(
        "build_cell_v2",
        capsule(py, build_cell_c, c"manifoldx.builder.v2")?,
    )?;
    m.add(
        "build_cell_irregular",
        capsule(py, build_cell_irregular_c, manifoldx_sdk::BUILDER_NAME)?,
    )?;
    m.add("build_cell_malformed", capsule(py, build_cell_malformed_c, manifoldx_sdk::BUILDER_NAME)?)?;
    m.add("sdk_version", "1")?;
    Ok(())
}

// --- S-b2: the irregular cell -------------------------------------------------

const MAX_DIST: u32 = u32::MAX;

fn hash_edge(i: usize, j: usize, dir: u64, seed: u64) -> u64 {
    (i as u64)
        .wrapping_mul(73_856_093)
        .wrapping_add((j as u64).wrapping_mul(19_349_663))
        .wrapping_add(dir.wrapping_mul(83_492_791))
        .wrapping_add(seed)
}

fn hash_block(i: usize, j: usize, seed: u64) -> u64 {
    (i as u64)
        .wrapping_mul(83_492_791)
        .wrapping_add((j as u64).wrapping_mul(73_856_093))
        .wrapping_add(seed)
}

fn build_cell_irregular_impl(req: &CellRequest, out: &mut Output) -> Result<(), String> {
    if !(req.size > 0.0) {
        return Err(format!("size must be positive, got {}", req.size));
    }
    let n = (req.size / BLOCK) as usize;
    let size = req.size;
    let nodes = n + 1;
    let index = |i: usize, j: usize| i * nodes + j;

    // The graph: a street grid with a ninth of its edges missing.
    let mut adjacency: Vec<Vec<u32>> = vec![Vec::new(); nodes * nodes];
    let mut horizontal = Vec::with_capacity(n * nodes);
    let mut vertical = Vec::with_capacity(nodes * n);
    for i in 0..nodes {
        for j in 0..nodes {
            if i < n && hash_edge(i, j, 0, req.seed) % 9 != 0 {
                let a = index(i, j);
                let b = index(i + 1, j);
                adjacency[a].push(b as u32);
                adjacency[b].push(a as u32);
                horizontal.push((i, j));
            }
            if j < n && hash_edge(i, j, 1, req.seed) % 9 != 0 {
                let a = index(i, j);
                let b = index(i, j + 1);
                adjacency[a].push(b as u32);
                adjacency[b].push(a as u32);
                vertical.push((i, j));
            }
        }
    }

    // BFS from the centre; the distance reaches building heights and lamp rows.
    let mut dist = vec![MAX_DIST; nodes * nodes];
    let centre = index(n / 2, n / 2);
    dist[centre] = 0;
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(centre);
    while let Some(node) = queue.pop_front() {
        let d = dist[node];
        for &neighbour in &adjacency[node] {
            let neighbour = neighbour as usize;
            if dist[neighbour] == MAX_DIST {
                dist[neighbour] = d + 1;
                queue.push_back(neighbour);
            }
        }
    }

    // Roads: one quad per surviving edge, horizontal first, then vertical.
    let y = 0.02_f32;
    let mut rp = Vec::with_capacity((horizontal.len() + vertical.len()) * 12);
    let mut rn = Vec::with_capacity((horizontal.len() + vertical.len()) * 12);
    let mut ri = Vec::with_capacity((horizontal.len() + vertical.len()) * 6);
    for &(i, j) in &horizontal {
        let x0 = i as f32 * BLOCK;
        let x1 = (i + 1) as f32 * BLOCK;
        let z = j as f32 * BLOCK;
        face(
            &mut rp, &mut rn, &mut ri,
            [[x0, y, z - HALF], [x1, y, z - HALF], [x1, y, z + HALF], [x0, y, z + HALF]],
            [0.0, 1.0, 0.0],
        );
    }
    for &(i, j) in &vertical {
        let x = i as f32 * BLOCK;
        let z0 = j as f32 * BLOCK;
        let z1 = (j + 1) as f32 * BLOCK;
        face(
            &mut rp, &mut rn, &mut ri,
            [[x - HALF, y, z0], [x + HALF, y, z0], [x + HALF, y, z1], [x - HALF, y, z1]],
            [0.0, 1.0, 0.0],
        );
    }
    out.mesh(&rp, &rn, &ri, 0);

    // Lots: four per block, kept by the block's hash, block-major inside each lot index.
    let mut bp = Vec::new();
    let mut bn = Vec::new();
    let mut bi = Vec::new();
    for k in 0..4usize {
        for i in 0..n {
            for j in 0..n {
                let h = hash_block(i, j, req.seed);
                if h % 13 == 0 || (h >> (k + 8)) & 1 == 0 {
                    continue;
                }
                let d = match dist[index(i, j)] {
                    MAX_DIST => 0,
                    value => value,
                };
                let height = 4.0 + ((d.wrapping_add(((h >> (4 * k)) & 31) as u32)) % 28) as f32;
                let qx = (k % 2) as f32;
                let qz = (k / 2) as f32;
                let x0 = i as f32 * BLOCK + qx * 12.5 + 1.5;
                let x1 = x0 + 9.5;
                let z0 = j as f32 * BLOCK + qz * 12.5 + 1.5;
                let z1 = z0 + 9.5;

                face(&mut bp, &mut bn, &mut bi,
                     [[x0, 0.0, z0], [x1, 0.0, z0], [x1, 0.0, z1], [x0, 0.0, z1]], [0.0, -1.0, 0.0]);
                face(&mut bp, &mut bn, &mut bi,
                     [[x0, height, z0], [x1, height, z0], [x1, height, z1], [x0, height, z1]], [0.0, 1.0, 0.0]);
                face(&mut bp, &mut bn, &mut bi,
                     [[x0, 0.0, z0], [x0, height, z0], [x0, height, z1], [x0, 0.0, z1]], [-1.0, 0.0, 0.0]);
                face(&mut bp, &mut bn, &mut bi,
                     [[x1, 0.0, z0], [x1, height, z0], [x1, height, z1], [x1, 0.0, z1]], [1.0, 0.0, 0.0]);
                face(&mut bp, &mut bn, &mut bi,
                     [[x0, 0.0, z0], [x1, 0.0, z0], [x1, height, z0], [x0, height, z0]], [0.0, 0.0, -1.0]);
                face(&mut bp, &mut bn, &mut bi,
                     [[x0, 0.0, z1], [x1, 0.0, z1], [x1, height, z1], [x0, height, z1]], [0.0, 0.0, 1.0]);
            }
        }
    }
    out.mesh(&bp, &bn, &bi, 1);

    // Lamps: one row of 16 bytes (x, z, kind, BFS distance) per reachable node every third step.
    let mut rows = Vec::new();
    for i in 0..nodes {
        for j in 0..nodes {
            let d = dist[index(i, j)];
            if d == MAX_DIST || d % 3 != 0 {
                continue;
            }
            rows.extend_from_slice(&(i as f32 * BLOCK).to_le_bytes());
            rows.extend_from_slice(&(j as f32 * BLOCK).to_le_bytes());
            rows.extend_from_slice(&1_u32.to_le_bytes());
            rows.extend_from_slice(&d.to_le_bytes());
        }
    }
    out.spawn(0, &rows, 16);

    let _ = size;
    Ok(())
}

unsafe extern "C" fn build_cell_irregular_c(req: *const CellRequest, out: *mut BuildOutput) -> i32 {
    if req.is_null() || out.is_null() {
        return 3;
    }
    let req = unsafe { &*req };
    let mut output = unsafe { Output::new(out) };
    guarded(&mut output, |output| build_cell_irregular_impl(req, output))
}
