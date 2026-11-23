//! S-a spike: a Rust frame loop with Python callbacks, for measuring whether a
//! slow async Python system can move the frame.
//!
//! This is not the real core. It is the smallest loop that answers the question:
//! a paced 60 Hz loop in Rust that releases the GIL, calls zero or more Python
//! "sync systems" per frame, and runs Python "async systems" on their own threads
//! at a rate.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pyclass]
struct SpikeLoop {
    fps: f64,
    syncs: Arc<Mutex<Vec<Py<PyAny>>>>,
    asyncs: Arc<Mutex<Vec<(Py<PyAny>, f64)>>>,
    handles: Arc<Mutex<Vec<thread::JoinHandle<()>>>>,
    stop: Arc<AtomicBool>,
    started: Arc<AtomicBool>,
    errors: Arc<AtomicUsize>,
}

#[pymethods]
impl SpikeLoop {
    #[new]
    fn new(fps: f64) -> Self {
        Self {
            fps,
            syncs: Arc::new(Mutex::new(Vec::new())),
            asyncs: Arc::new(Mutex::new(Vec::new())),
            handles: Arc::new(Mutex::new(Vec::new())),
            stop: Arc::new(AtomicBool::new(false)),
            started: Arc::new(AtomicBool::new(false)),
            errors: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// A Python callable the frame waits for, once per frame.
    fn add_sync(&self, f: Py<PyAny>) {
        self.syncs.lock().unwrap().push(f);
    }

    /// A Python callable on its own thread, called at `hz`, never waited for.
    fn add_async(&self, f: Py<PyAny>, hz: f64) {
        self.asyncs.lock().unwrap().push((f, hz));
    }

    fn start_async(&self, py: Python<'_>) {
        if self.started.swap(true, Ordering::SeqCst) {
            return;
        }
        let list: Vec<(Py<PyAny>, f64)> = self
            .asyncs
            .lock()
            .unwrap()
            .iter()
            .map(|(f, hz)| (f.clone_ref(py), *hz))
            .collect();
        let mut handles = self.handles.lock().unwrap();
        for (f, hz) in list {
            let stop = self.stop.clone();
            let errors = self.errors.clone();
            handles.push(thread::spawn(move || {
                let period = Duration::from_secs_f64(1.0 / hz.max(0.001));
                let mut next = Instant::now() + period;
                while !stop.load(Ordering::Relaxed) {
                    Python::attach(|py| {
                        if f.call0(py).is_err() {
                            errors.fetch_add(1, Ordering::Relaxed);
                        }
                    });
                    let now = Instant::now();
                    if next > now {
                        thread::sleep(next - now);
                    }
                    next = Instant::now() + period;
                }
            }));
        }
    }

    /// Run `frames` paced frames. Returns (period_ms, work_ms, frames_over_33, callback_errors).
    fn run(
        &self,
        py: Python<'_>,
        frames: usize,
    ) -> PyResult<(Vec<f64>, Vec<f64>, usize, usize)> {
        self.stop.store(false, Ordering::SeqCst);
        self.start_async(py);
        let syncs: Vec<Py<PyAny>> = self
            .syncs
            .lock()
            .unwrap()
            .iter()
            .map(|f| f.clone_ref(py))
            .collect();
        let stop = self.stop.clone();
        let errors = self.errors.clone();
        let fps = self.fps;

        let out = py.detach(move || {
            let budget = Duration::from_secs_f64(1.0 / fps.max(1.0));
            let mut periods = Vec::with_capacity(frames);
            let mut works = Vec::with_capacity(frames);
            let mut over = 0usize;
            let mut prev = Instant::now();
            let mut next = prev + budget;

            for _ in 0..frames {
                let t0 = Instant::now();
                if !syncs.is_empty() {
                    Python::attach(|py| {
                        for f in &syncs {
                            let _ = f.call0(py);
                        }
                    });
                }
                let work = t0.elapsed();

                let now = Instant::now();
                if next > now {
                    let wait = next - now;
                    if wait > Duration::from_micros(1200) {
                        thread::sleep(wait - Duration::from_micros(1200));
                    }
                    while Instant::now() < next {
                        std::hint::spin_loop();
                    }
                } else {
                    next = now;
                }

                let t1 = Instant::now();
                let period = (t1 - prev).as_secs_f64() * 1000.0;
                periods.push(period);
                works.push(work.as_secs_f64() * 1000.0);
                if period > 33.0 {
                    over += 1;
                }
                prev = t1;
                next += budget;
            }
            (periods, works, over, errors.load(Ordering::Relaxed))
        });

        stop.store(true, Ordering::SeqCst);
        let handles: Vec<_> = self.handles.lock().unwrap().drain(..).collect();
        for handle in handles {
            let _ = handle.join();
        }
        self.started.store(false, Ordering::SeqCst);
        Ok(out)
    }
}

#[pymodule]
fn _spike_sa(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SpikeLoop>()?;
    Ok(())
}
