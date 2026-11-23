//! S-c spike: a store that lives in Rust and survives a Python reload.
//!
//! It holds the rows, the mesh handles and the frame counter; Python holds the
//! systems. A reload reimports the app module and swaps the system list; the
//! store is never rebuilt.

use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pyclass]
struct World {
    entities: Vec<[f32; 4]>, // x, z, vx, vz
    meshes: Vec<u64>,
    systems: Vec<Py<PyAny>>,
    frame: u64,
}

#[pymethods]
impl World {
    #[new]
    fn new() -> Self {
        Self { entities: Vec::new(), meshes: Vec::new(), systems: Vec::new(), frame: 0 }
    }

    fn spawn(&mut self, n: usize) {
        let start = self.entities.len();
        self.entities.reserve(n);
        for i in 0..n {
            let k = (start + i) as f32;
            self.entities.push([k * 0.01, 0.0, 1.0, 0.5]);
        }
    }

    fn mesh(&mut self, id: u64) {
        self.meshes.push(id);
    }

    fn entity_count(&self) -> usize {
        self.entities.len()
    }

    fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    fn frame(&self) -> u64 {
        self.frame
    }

    fn set_systems(&mut self, systems: Vec<Py<PyAny>>) {
        self.systems = systems;
    }

    fn step(&mut self, py: Python<'_>, dt: f64) -> PyResult<()> {
        for system in &self.systems {
            system.call1(py, (dt,))?;
        }
        let dt = dt as f32;
        for entity in &mut self.entities {
            entity[0] += entity[2] * dt;
            entity[1] += entity[3] * dt;
        }
        self.frame += 1;
        Ok(())
    }
}

#[pymodule]
fn _spike_sc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<World>()?;
    Ok(())
}
