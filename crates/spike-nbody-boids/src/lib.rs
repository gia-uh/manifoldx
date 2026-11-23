//! Native steps for the nbody and boids examples, for comparing with numpy.
//!
//! The math is the examples' math: all-pairs gravity with `max(dist, softening)`
//! and the boid rules with a uniform grid at the perception radius. The point is
//! to see where a native step wins and where numpy is already enough.

use numpy::{PyReadonlyArray1, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use rayon::prelude::*;

fn check2(array: &PyReadwriteArray2<f32>, n: usize, name: &str) -> PyResult<()> {
    let shape = array.shape();
    if shape != [n, 3] {
        return Err(PyValueError::new_err(format!("{name} must be ({n}, 3), got {shape:?}")));
    }
    Ok(())
}

#[pyfunction]
fn nbody_step(
    mut positions: PyReadwriteArray2<f32>,
    mut velocities: PyReadwriteArray2<f32>,
    masses: PyReadonlyArray1<f32>,
    dt: f32,
    g: f32,
    softening: f32,
) -> PyResult<()> {
    let n = masses.len();
    check2(&positions, n, "positions")?;
    check2(&velocities, n, "velocities")?;
    let masses = masses.as_slice()?;
    let pos = positions
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("positions must be contiguous"))?;
    let vel = velocities
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("velocities must be contiguous"))?;
    let eps = softening.max(0.0);

    {
        let p: &[f32] = pos;
        vel.par_chunks_mut(3).enumerate().for_each(|(i, v)| {
            let (xi, yi, zi) = (p[3 * i], p[3 * i + 1], p[3 * i + 2]);
            let (mut ax, mut ay, mut az) = (0.0f32, 0.0f32, 0.0f32);
            for j in 0..n {
                if j == i {
                    continue;
                }
                let dx = p[3 * j] - xi;
                let dy = p[3 * j + 1] - yi;
                let dz = p[3 * j + 2] - zi;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                let ds = if dist > eps { dist } else { eps };
                if ds <= 0.0 {
                    continue;
                }
                let inv = 1.0 / (ds * ds * ds);
                let w = g * masses[j] * inv;
                ax += w * dx;
                ay += w * dy;
                az += w * dz;
            }
            v[0] += ax * dt;
            v[1] += ay * dt;
            v[2] += az * dt;
        });
    }

    pos.par_chunks_mut(3)
        .zip(vel.par_chunks(3))
        .for_each(|(p, v)| {
            p[0] += v[0] * dt;
            p[1] += v[1] * dt;
            p[2] += v[2] * dt;
        });
    Ok(())
}

const PERCEPTION_SQ: f32 = 16.0;
const FEAR_RADIUS: f32 = 10.0;
const SEP_W: f32 = 1.5;
const ALI_W: f32 = 1.0;
const COH_W: f32 = 0.8;
const FEAR_W: f32 = 20.0;
const MAX_SPEED: f32 = 10.0;
const PANIC_SPEED: f32 = 15.0;
const MIN_SPEED: f32 = 4.0;
const BOUND_RADIUS: f32 = 12.0;
const PULL_STRENGTH: f32 = 0.02;
const CELL: f32 = 4.0;

#[derive(Default, Clone, Copy)]
struct Forces {
    sep: [f32; 3],
    ali: [f32; 3],
    coh: [f32; 3],
    fear: [f32; 3],
    count: u32,
    scared: bool,
}

fn norm3(v: [f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[pyfunction]
fn boids_step(
    mut positions: PyReadwriteArray2<f32>,
    mut velocities: PyReadwriteArray2<f32>,
    mut pred_positions: PyReadwriteArray2<f32>,
    mut pred_directions: PyReadwriteArray2<f32>,
    dt: f32,
) -> PyResult<()> {
    let n = positions.shape()[0];
    let np = pred_positions.shape()[0];
    check2(&positions, n, "positions")?;
    check2(&velocities, n, "velocities")?;
    check2(&pred_positions, np, "pred_positions")?;
    check2(&pred_directions, np, "pred_directions")?;
    let pos = positions
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("positions must be contiguous"))?;
    let vel = velocities
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("velocities must be contiguous"))?;
    let pp = pred_positions
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("pred_positions must be contiguous"))?;
    let pd = pred_directions
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("pred_directions must be contiguous"))?;

    // A uniform grid at the perception radius: any neighbour within 4 m is in one
    // of the 27 cells around a boid.
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for i in 0..n {
        for k in 0..3 {
            min[k] = min[k].min(pos[3 * i + k]);
            max[k] = max[k].max(pos[3 * i + k]);
        }
    }
    let nx = (((max[0] - min[0]) / CELL).floor() as i32 + 1).max(1);
    let ny = (((max[1] - min[1]) / CELL).floor() as i32 + 1).max(1);
    let nz = (((max[2] - min[2]) / CELL).floor() as i32 + 1).max(1);
    let cell_of = |x: f32, y: f32, z: f32| -> (i32, i32, i32) {
        (
            ((x - min[0]) / CELL).floor() as i32,
            ((y - min[1]) / CELL).floor() as i32,
            ((z - min[2]) / CELL).floor() as i32,
        )
    };
    let index = |c: (i32, i32, i32)| -> usize {
        ((c.2 as usize * ny as usize) + c.1 as usize) * nx as usize + c.0 as usize
    };
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); (nx * ny * nz) as usize];
    for i in 0..n {
        let c = cell_of(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]);
        buckets[index(c)].push(i as u32);
    }

    let forces: Vec<Forces> = (0..n)
        .into_par_iter()
        .map(|i| {
            let pi = [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]];
            let c = cell_of(pi[0], pi[1], pi[2]);
            let mut f = Forces::default();
            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let cx = c.0 + dx;
                        let cy = c.1 + dy;
                        let cz = c.2 + dz;
                        if cx < 0 || cy < 0 || cz < 0 || cx >= nx || cy >= ny || cz >= nz {
                            continue;
                        }
                        for &j in &buckets[index((cx, cy, cz))] {
                            let j = j as usize;
                            if j == i {
                                continue;
                            }
                            let d = [pi[0] - pos[3 * j], pi[1] - pos[3 * j + 1], pi[2] - pos[3 * j + 2]];
                            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                            if r2 >= PERCEPTION_SQ || r2 <= 1e-6 {
                                continue;
                            }
                            let inv = 1.0 / r2;
                            f.sep[0] += d[0] * inv;
                            f.sep[1] += d[1] * inv;
                            f.sep[2] += d[2] * inv;
                            f.ali[0] += vel[3 * j];
                            f.ali[1] += vel[3 * j + 1];
                            f.ali[2] += vel[3 * j + 2];
                            f.coh[0] += pos[3 * j];
                            f.coh[1] += pos[3 * j + 1];
                            f.coh[2] += pos[3 * j + 2];
                            f.count += 1;
                        }
                    }
                }
            }
            for p in 0..np {
                let d = [pi[0] - pp[3 * p], pi[1] - pp[3 * p + 1], pi[2] - pp[3 * p + 2]];
                let dist = norm3(d);
                if dist < FEAR_RADIUS && dist > 1e-6 {
                    f.scared = true;
                    let inv = 1.0 / dist;
                    f.fear[0] += d[0] * inv;
                    f.fear[1] += d[1] * inv;
                    f.fear[2] += d[2] * inv;
                }
            }
            f
        })
        .collect();

    for i in 0..n {
        let f = forces[i];
        let safe = if f.count == 0 { 1.0 } else { f.count as f32 };
        let pi = [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]];
        let vi = [vel[3 * i], vel[3 * i + 1], vel[3 * i + 2]];
        let ali = [f.ali[0] / safe - vi[0], f.ali[1] / safe - vi[1], f.ali[2] / safe - vi[2]];
        let coh = [f.coh[0] / safe - pi[0], f.coh[1] / safe - pi[1], f.coh[2] / safe - pi[2]];
        let mut v = [
            vi[0] + (SEP_W * f.sep[0] + ALI_W * ali[0] + COH_W * coh[0] + FEAR_W * f.fear[0] - PULL_STRENGTH * pi[0]) * dt,
            vi[1] + (SEP_W * f.sep[1] + ALI_W * ali[1] + COH_W * coh[1] + FEAR_W * f.fear[1] - PULL_STRENGTH * pi[1]) * dt,
            vi[2] + (SEP_W * f.sep[2] + ALI_W * ali[2] + COH_W * coh[2] + FEAR_W * f.fear[2] - PULL_STRENGTH * pi[2]) * dt,
        ];

        let r = norm3(pi);
        let overshoot = (r - BOUND_RADIUS).max(0.0);
        if overshoot > 0.0 {
            let inv = 1.0 / r.max(1e-9);
            let push = overshoot * 3.0;
            v[0] -= pi[0] * inv * push;
            v[1] -= pi[1] * inv * push;
            v[2] -= pi[2] * inv * push;
        }

        let speed = norm3(v);
        let max_s = if f.scared { PANIC_SPEED } else { MAX_SPEED };
        let min_s = if f.scared { MIN_SPEED * 1.5 } else { MIN_SPEED };
        if speed > max_s {
            let k = max_s / speed;
            v = [v[0] * k, v[1] * k, v[2] * k];
        } else if speed < min_s && speed > 1e-6 {
            let k = min_s / speed;
            v = [v[0] * k, v[1] * k, v[2] * k];
        }

        vel[3 * i] = v[0];
        vel[3 * i + 1] = v[1];
        vel[3 * i + 2] = v[2];
        pos[3 * i] += v[0] * dt;
        pos[3 * i + 1] += v[1] * dt;
        pos[3 * i + 2] += v[2] * dt;
    }

    // Predators: a deterministic wander, the same shape as the example's random
    // turn; the boid comparison runs with predators far away.
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut noise = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / 8_388_608.0) - 1.0
    };
    for p in 0..np {
        let mut d = [pd[3 * p], pd[3 * p + 1], pd[3 * p + 2]];
        d[0] += noise() * dt;
        d[1] += noise() * dt;
        d[2] += noise() * dt;
        let len = norm3(d).max(1e-6);
        d = [d[0] / len, d[1] / len, d[2] / len];
        pd[3 * p] = d[0];
        pd[3 * p + 1] = d[1];
        pd[3 * p + 2] = d[2];

        let r = norm3([pp[3 * p], pp[3 * p + 1], pp[3 * p + 2]]);
        let over = (r - BOUND_RADIUS * 0.8).max(0.0);
        let inv = 1.0 / r.max(1e-6);
        let v = [
            d[0] * 3.0 - pp[3 * p] * inv * over * 0.5,
            d[1] * 3.0 - pp[3 * p + 1] * inv * over * 0.5,
            d[2] * 3.0 - pp[3 * p + 2] * inv * over * 0.5,
        ];
        pp[3 * p] += v[0] * dt;
        pp[3 * p + 1] += v[1] * dt;
        pp[3 * p + 2] += v[2] * dt;
    }
    Ok(())
}

#[pymodule]
fn _spike_nb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nbody_step, m)?)?;
    m.add_function(wrap_pyfunction!(boids_step, m)?)?;
    Ok(())
}
