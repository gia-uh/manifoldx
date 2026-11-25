//! What the wheels stand on, sampled natively. A flat plane, or a road along
//! x with a piecewise-linear height profile, square kerbs and an optional
//! bridge deck selected by the level nearest `near`. No callback crosses into
//! Python.

use std::sync::Arc;

use super::sim::V3;
use super::terrain::{Obstacle, Surface, Terrain};

/// One native ground. `near` picks the level at a place where several exist.
#[derive(Clone, Debug)]
pub enum Ground {
    Tiles {
        terrain: Arc<Terrain>,
        origin: V3,
    },
    Flat {
        level: f64,
        grip: f64,
    },
    Profile {
        points: Vec<(f64, f64)>,
        kerbs: Vec<f64>,
        kerb_height: f64,
        bridge: Option<(f64, f64, f64)>,
        grip: f64,
    },
}

impl Default for Ground {
    fn default() -> Self {
        Self::Flat {
            level: 0.0,
            grip: 1.0,
        }
    }
}

impl Ground {
    pub fn flat(level: f64, grip: f64) -> Result<Self, String> {
        if !(level.is_finite() && grip.is_finite() && grip >= 0.0) {
            return Err("flat ground needs a finite level and a non-negative grip".into());
        }
        Ok(Self::Flat { level, grip })
    }

    pub fn profile(
        points: Vec<(f64, f64)>,
        kerbs: Vec<f64>,
        kerb_height: f64,
        bridge: Option<(f64, f64, f64)>,
        grip: f64,
    ) -> Result<Self, String> {
        if points.len() < 2 {
            return Err("a profile needs at least two points".into());
        }
        let mut previous = f64::NEG_INFINITY;
        for (x, y) in &points {
            if !(x.is_finite() && y.is_finite()) {
                return Err("profile points must be finite".into());
            }
            if *x <= previous {
                return Err("profile x values must increase strictly".into());
            }
            previous = *x;
        }
        if kerbs.iter().any(|x| !x.is_finite()) {
            return Err("kerbs must be finite".into());
        }
        if !(kerb_height.is_finite() && kerb_height >= 0.0) {
            return Err("kerb height must be finite and non-negative".into());
        }
        if let Some((x0, x1, deck)) = bridge {
            if !(x0.is_finite() && x1.is_finite() && deck.is_finite() && x0 <= x1) {
                return Err("bridge needs finite x0 <= x1 and a finite deck".into());
            }
        }
        if !(grip.is_finite() && grip >= 0.0) {
            return Err("grip must be finite and non-negative".into());
        }
        Ok(Self::Profile {
            points,
            kerbs,
            kerb_height,
            bridge,
            grip,
        })
    }

    pub fn terrain(terrain: Arc<Terrain>) -> Self {
        Self::Tiles {
            terrain,
            origin: [0.0; 3],
        }
    }

    pub fn surface(&self, x: f64, z: f64, near: f64) -> Option<Surface> {
        let Self::Tiles { terrain, origin } = self else {
            return None;
        };
        let mut sample = terrain.sample(x + origin[0], z + origin[2], near + origin[1])?;
        sample.height -= origin[1];
        Some(sample)
    }

    pub(super) fn obstacles(&self, low: [f64; 2], high: [f64; 2]) -> Vec<Obstacle> {
        let Self::Tiles { terrain, origin } = self else {
            return Vec::new();
        };
        terrain
            .obstacles(
                [low[0] + origin[0], low[1] + origin[2]],
                [high[0] + origin[0], high[1] + origin[2]],
            )
            .map(|mut wall| {
                for p in &mut wall.ends {
                    p[0] -= origin[0];
                    p[1] -= origin[2];
                }
                wall.y[0] -= origin[1];
                wall.y[1] -= origin[1];
                wall
            })
            .collect()
    }

    pub(super) fn height_if_present(&self, x: f64, z: f64, near: f64) -> Option<f64> {
        match self {
            Self::Tiles { .. } => self.surface(x, z, near).map(|sample| sample.height),
            _ => Some(self.height(x, z, near)),
        }
    }

    /// The surface height under `(x, z)`; where several levels exist, the one
    /// nearest `near`.
    pub fn height(&self, x: f64, z: f64, near: f64) -> f64 {
        match self {
            Self::Tiles { .. } => self
                .surface(x, z, near)
                .map_or(f64::NEG_INFINITY, |s| s.height),
            Self::Flat { level, .. } => *level,
            Self::Profile {
                points,
                kerbs,
                kerb_height,
                bridge,
                ..
            } => {
                let mut base = interpolate(points, x);
                if !kerbs.is_empty() {
                    let steps = kerbs.iter().filter(|kerb| x >= **kerb).count();
                    base += *kerb_height * steps as f64;
                }
                match bridge {
                    Some((x0, x1, deck)) if x >= *x0 && x <= *x1 => {
                        if (near - deck).abs() < (near - base).abs() {
                            *deck
                        } else {
                            base
                        }
                    }
                    _ => base,
                }
            }
        }
    }

    /// The surface's factor on each tyre's `mu`.
    pub fn grip(&self, _x: f64, _z: f64) -> f64 {
        match self {
            Self::Tiles { .. } => self.surface(_x, _z, 0.0).map_or(0.0, |s| s.grip),
            Self::Flat { grip, .. } | Self::Profile { grip, .. } => *grip,
        }
    }
}

/// `numpy.interp` for one value: clamp outside the profile.
fn interpolate(points: &[(f64, f64)], x: f64) -> f64 {
    let first = points[0];
    if x <= first.0 {
        return first.1;
    }
    let last = points[points.len() - 1];
    if x >= last.0 {
        return last.1;
    }
    let mut low = 0usize;
    let mut high = points.len() - 1;
    while high - low > 1 {
        let mid = (low + high) / 2;
        if points[mid].0 <= x {
            low = mid;
        } else {
            high = mid;
        }
    }
    let (x0, y0) = points[low];
    let (x1, y1) = points[high];
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_profile_clamps_kerbs_and_chooses_the_bridge_by_near() {
        let ground = Ground::profile(
            vec![
                (-50.0, 0.0),
                (20.0, 0.0),
                (40.0, 1.5),
                (60.0, 0.0),
                (400.0, 0.0),
            ],
            vec![30.0, 70.0],
            0.15,
            Some((90.0, 110.0, 3.0)),
            1.0,
        )
        .unwrap();
        assert_eq!(ground.height(-100.0, 0.0, 0.0), 0.0);
        assert_eq!(ground.height(30.0, 0.0, 0.0), 0.9);
        assert_eq!(ground.height(80.0, 0.0, 0.0), 0.3);
        assert_eq!(ground.height(100.0, 0.0, 3.0), 3.0);
        assert_eq!(ground.height(100.0, 0.0, 0.0), 0.3);
        assert_eq!(ground.height(400.0, 0.0, 0.0), 0.3);
    }

    #[test]
    fn validation_rejects_a_bad_profile() {
        assert!(Ground::profile(vec![(0.0, 0.0)], vec![], 0.0, None, 1.0).is_err());
        assert!(Ground::profile(vec![(0.0, 0.0), (0.0, 1.0)], vec![], 0.0, None, 1.0).is_err());
        assert!(Ground::flat(0.0, -1.0).is_err());
    }
}
