//! S-d spike: selections and commands over 100k rows, with the store in Rust.
//!
//! Python builds a selection and enqueues a set and a spawn; it never reads a
//! row. The frame loop runs five kernels in Rust and applies the pending
//! commands at the frame boundary, measuring the apply time.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};

const X: usize = 0;
const Z: usize = 1;
const VX: usize = 2;
const VZ: usize = 3;
const SPEED: usize = 4;
const TARGET: usize = 5;
const TTL: usize = 6;
const ENERGY: usize = 7;
const COLS: usize = 8;

fn column_of(name: &str) -> Option<usize> {
    Some(match name {
        "x" => X,
        "z" => Z,
        "vx" => VX,
        "vz" => VZ,
        "speed" => SPEED,
        "target_speed" => TARGET,
        "ttl" => TTL,
        "energy" => ENERGY,
        _ => return None,
    })
}

fn op_code(op: &str) -> Option<u8> {
    Some(match op {
        "<" => 0,
        "<=" => 1,
        ">" => 2,
        ">=" => 3,
        "==" => 4,
        _ => return None,
    })
}

fn compare(value: f32, op: u8, against: f32) -> bool {
    match op {
        0 => value < against,
        1 => value <= against,
        2 => value > against,
        3 => value >= against,
        _ => value == against,
    }
}

#[derive(Clone, Copy)]
struct Selection {
    column: usize,
    op: u8,
    value: f32,
}

enum Command {
    Set { selection: u32, column: usize, value: f32 },
    Spawn { count: usize },
}

struct Store {
    rows: Vec<[f32; COLS]>,
    pending: Vec<Command>,
    rows_read: u64,
}

impl Store {
    fn new() -> Self {
        Self { rows: Vec::new(), pending: Vec::new(), rows_read: 0 }
    }

    fn kernels(&mut self, dt: f32) {
        for row in &mut self.rows {
            row[X] += row[VX] * dt;
            row[Z] += row[VZ] * dt;
        }
        for row in &mut self.rows {
            row[SPEED] += (row[TARGET] - row[SPEED]).clamp(-3.0 * dt, 2.0 * dt);
        }
        for row in &mut self.rows {
            row[TTL] = (row[TTL] - dt).max(0.0);
        }
        for row in &mut self.rows {
            row[VX] += (row[TARGET] - row[SPEED]) * 0.01 * dt;
        }
        for row in &mut self.rows {
            row[ENERGY] = (row[ENERGY] - 0.01 * dt).max(0.0);
        }
    }

    fn apply(&mut self, selections: &[Selection]) -> (usize, f64, usize, usize) {
        let commands = std::mem::take(&mut self.pending);
        let started = Instant::now();
        let mut affected = 0usize;
        let mut sets = 0usize;
        let mut spawns = 0usize;
        for command in commands {
            match command {
                Command::Set { selection, column, value } => {
                    let selection = selections[selection as usize];
                    for row in &mut self.rows {
                        if compare(row[selection.column], selection.op, selection.value) {
                            row[column] = value;
                            affected += 1;
                        }
                    }
                    sets += 1;
                }
                Command::Spawn { count } => {
                    for _ in 0..count {
                        let n = self.rows.len() as f32;
                        self.rows.push([n * 0.01, 0.0, 1.0, 0.0, 0.0, 0.0, 100.0, 1.0]);
                    }
                    spawns += 1;
                }
            }
        }
        (affected, started.elapsed().as_secs_f64() * 1000.0, sets, spawns)
    }
}

#[pyclass]
struct World {
    store: Arc<Mutex<Store>>,
    selections: Arc<Mutex<Vec<Selection>>>,
}

#[pymethods]
impl World {
    #[new]
    fn new() -> Self {
        Self { store: Arc::new(Mutex::new(Store::new())), selections: Arc::new(Mutex::new(Vec::new())) }
    }

    /// Fill the store with `n` deterministic agents.
    fn populate(&self, n: usize) {
        let mut store = self.store.lock().unwrap();
        store.rows.clear();
        store.rows.reserve(n);
        for i in 0..n {
            let speed = (i % 1000) as f32 / 100.0;
            store.rows.push([i as f32 * 0.01, 0.0, 1.0, 0.0, speed, speed, 100.0, 1.0]);
        }
    }

    /// Build a selection. It stores a spec in Rust and returns its id; no row is read.
    fn select(&self, column: &str, op: &str, value: f64) -> PyResult<u32> {
        let column = column_of(column).ok_or_else(|| PyValueError::new_err(format!("unknown column {column:?}")))?;
        let op = op_code(op).ok_or_else(|| PyValueError::new_err(format!("unknown operator {op:?}")))?;
        let mut selections = self.selections.lock().unwrap();
        selections.push(Selection { column, op, value: value as f32 });
        Ok((selections.len() - 1) as u32)
    }

    /// How many rows a selection matches, evaluated in Rust; it does not read
    /// rows into Python.
    fn count(&self, selection: u32) -> PyResult<usize> {
        let selections = self.selections.lock().unwrap();
        let selection = *selections
            .get(selection as usize)
            .ok_or_else(|| PyValueError::new_err(format!("no selection {selection}")))?;
        drop(selections);
        let store = self.store.lock().unwrap();
        Ok(store.rows.iter().filter(|row| compare(row[selection.column], selection.op, selection.value)).count())
    }

    /// Enqueue a set over a selection; it applies at the next frame boundary.
    fn set(&self, selection: u32, column: &str, value: f64) -> PyResult<()> {
        let column = column_of(column).ok_or_else(|| PyValueError::new_err(format!("unknown column {column:?}")))?;
        let selections = self.selections.lock().unwrap();
        if selection as usize >= selections.len() {
            return Err(PyValueError::new_err(format!("no selection {selection}")));
        }
        drop(selections);
        self.store.lock().unwrap().pending.push(Command::Set { selection, column, value: value as f32 });
        Ok(())
    }

    /// Enqueue a spawn; it applies at the next frame boundary.
    fn spawn(&self, count: usize) {
        self.store.lock().unwrap().pending.push(Command::Spawn { count });
    }

    /// The naive path: read a column into Python. Increments the row counter.
    fn read_speeds(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let mut store = self.store.lock().unwrap();
        let list = PyList::empty(py);
        for row in &store.rows {
            list.append(row[SPEED])?;
        }
        store.rows_read += store.rows.len() as u64;
        Ok(list.unbind())
    }

    fn rows_read(&self) -> u64 {
        self.store.lock().unwrap().rows_read
    }

    fn reset_rows_read(&self) {
        self.store.lock().unwrap().rows_read = 0;
    }

    fn entity_count(&self) -> usize {
        self.store.lock().unwrap().rows.len()
    }

    /// Run `frames` paced frames; every `every` frames call the Python system,
    /// then apply what it enqueued. Returns the per-frame periods, the apply
    /// times, the async call times and the command counts.
    fn run(&self, py: Python<'_>, frames: usize, every: usize, system: Py<PyAny>) -> PyResult<Py<PyDict>> {
        let store = self.store.clone();
        let selections = self.selections.clone();
        let out = py.detach(move || {
            let budget = Duration::from_secs_f64(1.0 / 60.0);
            let mut periods = Vec::with_capacity(frames);
            let mut applies = Vec::new();
            let mut batches = Vec::new();
            let mut asyncs = Vec::new();
            let mut sets = 0usize;
            let mut spawns = 0usize;
            let mut affected_total = 0usize;
            let mut prev = Instant::now();
            let mut next = prev + budget;

            for frame in 0..frames {
                let t0 = Instant::now();
                {
                    let mut store = store.lock().unwrap();
                    store.kernels(1.0 / 60.0);
                }
                let (affected, apply_ms, set_count, spawn_count) = {
                    let mut store = store.lock().unwrap();
                    let selections = selections.lock().unwrap();
                    store.apply(&selections)
                };
                applies.push(apply_ms);
                if set_count + spawn_count > 0 {
                    batches.push(apply_ms);
                }
                sets += set_count;
                spawns += spawn_count;
                affected_total += affected;

                if every > 0 && frame % every == 0 {
                    let a0 = Instant::now();
                    Python::attach(|py| {
                        let _ = system.call0(py);
                    });
                    asyncs.push(a0.elapsed().as_secs_f64() * 1000.0);
                }

                let now = Instant::now();
                if next > now {
                    let wait = next - now;
                    if wait > Duration::from_micros(1200) {
                        std::thread::sleep(wait - Duration::from_micros(1200));
                    }
                    while Instant::now() < next {
                        std::hint::spin_loop();
                    }
                } else {
                    next = now;
                }
                let t1 = Instant::now();
                periods.push((t1 - prev).as_secs_f64() * 1000.0);
                prev = t1;
                next += budget;
                let _ = t0;
            }
            (periods, applies, batches, asyncs, sets, spawns, affected_total, store.lock().unwrap().rows_read)
        });

        let result = PyDict::new(py);
        result.set_item("periods", out.0)?;
        result.set_item("applies", out.1)?;
        result.set_item("batches", out.2)?;
        result.set_item("asyncs", out.3)?;
        result.set_item("sets", out.4)?;
        result.set_item("spawns", out.5)?;
        result.set_item("affected", out.6)?;
        result.set_item("rows_read", out.7)?;
        Ok(result.unbind())
    }
}

#[pymodule]
fn _spike_sd(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<World>()?;
    Ok(())
}
