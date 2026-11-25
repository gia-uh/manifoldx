//! Fixed simulation time, wall-clock pacing, and periodic system deadlines.
use std::time::{Duration, Instant};

/// Only the native calling thread advances the frame and simulation clock.
pub struct FrameClock {
    tick: u64,
    step: Duration,
    next_frame: Instant,
}

impl FrameClock {
    pub fn new(step: Duration) -> Self {
        Self {
            tick: 0,
            step,
            next_frame: Instant::now(),
        }
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn time(&self) -> f64 {
        self.tick as f64 * self.dt()
    }

    pub fn dt(&self) -> f64 {
        self.step.as_secs_f64()
    }

    pub fn advance(&mut self) -> Instant {
        self.tick += 1;
        self.next_frame += self.step;
        self.next_frame
    }
}

/// A periodic registration drops all elapsed starts when it cannot run.
pub struct Cadence {
    period: f64,
    next: f64,
}

impl Cadence {
    pub fn new(hz: f64) -> Self {
        Self {
            period: 1.0 / hz,
            next: 0.0,
        }
    }

    pub fn due(&mut self, time: f64) -> bool {
        if time < self.next {
            return false;
        }
        self.next += ((time - self.next) / self.period)
            .floor()
            .mul_add(self.period, self.period);
        true
    }
}

/// A column conflict exists when either participant writes what the other uses.
pub fn conflicts(
    a_reads: &[usize],
    a_writes: &[usize],
    b_reads: &[usize],
    b_writes: &[usize],
) -> bool {
    a_writes
        .iter()
        .any(|c| b_reads.contains(c) || b_writes.contains(c))
        || b_writes.iter().any(|c| a_reads.contains(c))
}
