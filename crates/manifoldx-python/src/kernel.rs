//! Private binding for validated kernel IR and the v1 dispatch arguments.
use super::*;
use manifoldx_core::kernel::{Program, Word, MAX_CODE};
use pyo3::types::PyList;

#[pyclass(frozen, name = "_KernelProgram")]
pub(crate) struct KernelProgram(pub Arc<Program>);

#[pymethods]
impl KernelProgram {
    #[new]
    fn new(
        version: u32,
        name: String,
        component: String,
        width: usize,
        words: &Bound<'_, PyList>,
    ) -> PyResult<Self> {
        if words.len() > MAX_CODE {
            return Err(PyValueError::new_err("kernel code exceeds limits"));
        }
        let words = words.extract::<Vec<Word>>()?;
        Program::decode(version, name, component, width, words)
            .map(|program| Self(Arc::new(program)))
            .map_err(PyValueError::new_err)
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<KernelProgram>()
}
