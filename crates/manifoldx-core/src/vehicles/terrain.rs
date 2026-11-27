//! Versioned, immutable terrain batches indexed in application coordinates.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::sim::V3;

pub type TileKey = (i32, i32);

#[derive(Clone, Copy, Debug)]
pub struct Surface {
    pub height: f64,
    pub normal: V3,
    pub grip: f64,
    pub level: u64,
    pub tile: TileKey,
    pub version: u64,
}

#[derive(Clone, Debug)]
pub struct Heightfield {
    pub origin: [f64; 2],
    pub spacing: [f64; 2],
    pub shape: [usize; 2],
    pub heights: Vec<f64>,
    pub grip: f64,
    pub level: u64,
}

#[derive(Clone, Debug)]
pub struct Triangle {
    pub vertices: [V3; 3],
    pub grip: f64,
    pub level: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct Obstacle {
    pub ends: [[f64; 2]; 2],
    pub y: [f64; 2],
    pub strength: f64,
}

#[derive(Clone, Debug)]
pub struct Tile {
    pub key: TileKey,
    pub version: u64,
    pub heightfields: Vec<Heightfield>,
    pub triangles: Vec<Triangle>,
    pub obstacles: Vec<Obstacle>,
}

#[derive(Clone, Debug)]
pub struct Terrain {
    size: f64,
    tiles: BTreeMap<TileKey, Arc<Tile>>,
    versions: BTreeMap<TileKey, u64>,
}

fn nonnegative(x: f64) -> bool {
    x.is_finite() && x >= 0.0
}

fn normal(dx: f64, dz: f64) -> V3 {
    let magnitude = dx.hypot(dz).hypot(1.0);
    [-dx / magnitude, 1.0 / magnitude, -dz / magnitude]
}

impl Heightfield {
    fn sample(&self, x: f64, z: f64) -> Option<(f64, V3)> {
        let uv = [
            (x - self.origin[0]) / self.spacing[0],
            (z - self.origin[1]) / self.spacing[1],
        ];
        if uv
            .iter()
            .zip(self.shape)
            .any(|(u, n)| *u < 0.0 || *u > (n - 1) as f64)
        {
            return None;
        }
        let i = (uv[0] as usize).min(self.shape[0] - 2);
        let j = (uv[1] as usize).min(self.shape[1] - 2);
        let u = uv[0] - i as f64;
        let v = uv[1] - j as f64;
        let width = self.shape[0];
        let at = j * width + i;
        let [a, b, c, d] = [
            self.heights[at],
            self.heights[at + 1],
            self.heights[at + width],
            self.heights[at + width + 1],
        ];
        let low = a + (b - a) * u;
        let high = c + (d - c) * u;
        let dx = ((b - a) * (1.0 - v) + (d - c) * v) / self.spacing[0];
        let dz = (high - low) / self.spacing[1];
        Some((low + (high - low) * v, normal(dx, dz)))
    }
}

impl Triangle {
    fn edges(&self) -> (V3, V3, f64) {
        let [a, b, c] = self.vertices;
        let ab = std::array::from_fn(|i| b[i] - a[i]);
        let ac = std::array::from_fn(|i| c[i] - a[i]);
        (ab, ac, ab[0] * ac[2] - ab[2] * ac[0])
    }

    fn sample(&self, x: f64, z: f64) -> Option<(f64, V3)> {
        let (ab, ac, det) = self.edges();
        let [ax, ay, az] = self.vertices[0];
        let x = x - ax;
        let z = z - az;
        let u = (x * ac[2] - z * ac[0]) / det;
        let v = (ab[0] * z - ab[2] * x) / det;
        if u < -1e-12 || v < -1e-12 || u + v > 1.0 + 1e-12 {
            return None;
        }
        let dx = (ab[1] * ac[2] - ac[1] * ab[2]) / det;
        let dz = (ab[0] * ac[1] - ac[0] * ab[1]) / det;
        Some((ay + ab[1] * u + ac[1] * v, normal(dx, dz)))
    }
}

impl Terrain {
    pub fn new(size: f64, tiles: Vec<Tile>) -> Result<Self, String> {
        if !size.is_finite() || size <= 0.0 {
            return Err("tile size must be finite and positive".into());
        }
        Self {
            size,
            tiles: BTreeMap::new(),
            versions: BTreeMap::new(),
        }
        .updated(tiles, &[])
    }

    /// Return a new index. Unchanged tiles retain the same allocation and older
    /// readers keep their version until their last reference is dropped.
    pub fn updated(&self, tiles: Vec<Tile>, removed: &[(TileKey, u64)]) -> Result<Self, String> {
        let mut next = self.clone();
        let mut seen = std::collections::BTreeSet::new();
        for (key, version) in removed {
            if !seen.insert(*key) || next.tiles.get(key).map(|tile| tile.version) != Some(*version)
            {
                return Err(format!("cannot remove tile {key:?} at version {version}"));
            }
            next.tiles.remove(key);
        }
        for tile in tiles {
            if !seen.insert(tile.key)
                || tile.version == 0
                || next
                    .versions
                    .get(&tile.key)
                    .is_some_and(|v| *v >= tile.version)
            {
                return Err(format!(
                    "tile {:?} requires a new positive version",
                    tile.key
                ));
            }
            self.validate(&tile)?;
            next.versions.insert(tile.key, tile.version);
            next.tiles.insert(tile.key, Arc::new(tile));
        }
        Ok(next)
    }

    pub fn tile_version(&self, key: TileKey) -> Option<u64> {
        self.tiles.get(&key).map(|tile| tile.version)
    }

    fn validate(&self, tile: &Tile) -> Result<(), String> {
        let start = [tile.key.0 as f64 * self.size, tile.key.1 as f64 * self.size];
        let in_tile = |x: f64, z: f64| {
            [x, z]
                .iter()
                .zip(start)
                .all(|(p, lo)| p.is_finite() && lo.is_finite() && *p >= lo && *p <= lo + self.size)
        };
        for grid in &tile.heightfields {
            let [nx, nz] = grid.shape;
            let [x, z] = grid.origin;
            let [dx, dz] = grid.spacing;
            if nx < 2
                || nz < 2
                || nx.checked_mul(nz) != Some(grid.heights.len())
                || !(dx > 0.0 && dz > 0.0)
                || !in_tile(x, z)
                || !in_tile(x + (nx - 1) as f64 * dx, z + (nz - 1) as f64 * dz)
                || !grid.heights.iter().all(|v| v.is_finite())
                || !nonnegative(grid.grip)
            {
                return Err(format!("invalid heightfield in tile {:?}", tile.key));
            }
        }
        for tri in &tile.triangles {
            let det = tri.edges().2;
            if !det.is_finite()
                || det == 0.0
                || !tri
                    .vertices
                    .iter()
                    .all(|p| in_tile(p[0], p[2]) && p[1].is_finite())
                || !nonnegative(tri.grip)
            {
                return Err(format!("invalid ground triangle in tile {:?}", tile.key));
            }
        }
        for wall in &tile.obstacles {
            if !wall.ends.iter().all(|p| in_tile(p[0], p[1]))
                || wall.ends[0] == wall.ends[1]
                || !wall.y.iter().all(|v| v.is_finite())
                || wall.y[0] >= wall.y[1]
                || !(0.0..=1.0).contains(&wall.strength)
            {
                return Err(format!("invalid obstacle in tile {:?}", tile.key));
            }
        }
        Ok(())
    }

    fn index(&self, x: f64, z: f64) -> TileKey {
        (
            (x / self.size).floor() as i32,
            (z / self.size).floor() as i32,
        )
    }

    pub fn sample(&self, x: f64, z: f64, near: f64) -> Option<Surface> {
        let (i, j) = self.index(x, z);
        let mut best: Option<Surface> = None;
        let xs = [
            Some(i),
            (x == i as f64 * self.size)
                .then(|| i.checked_sub(1))
                .flatten(),
        ];
        let zs = [
            Some(j),
            (z == j as f64 * self.size)
                .then(|| j.checked_sub(1))
                .flatten(),
        ];
        for tx in xs.into_iter().flatten() {
            for tz in zs.into_iter().flatten() {
                let Some(tile) = self.tiles.get(&(tx, tz)) else {
                    continue;
                };
                let grids = tile.heightfields.iter().filter_map(|grid| {
                    grid.sample(x, z)
                        .map(|(h, n)| (h, n, grid.grip, grid.level))
                });
                let triangles = tile
                    .triangles
                    .iter()
                    .filter_map(|tri| tri.sample(x, z).map(|(h, n)| (h, n, tri.grip, tri.level)));
                for (height, normal, grip, level) in grids.chain(triangles) {
                    if best.is_none_or(|old| (height - near).abs() < (old.height - near).abs()) {
                        best = Some(Surface {
                            height,
                            normal,
                            grip,
                            level,
                            tile: tile.key,
                            version: tile.version,
                        });
                    }
                }
            }
        }
        best
    }

    pub fn obstacles(&self, low: [f64; 2], high: [f64; 2]) -> impl Iterator<Item = Obstacle> + '_ {
        // Include obstacles exactly on the lower edge, owned by the tile to
        // its left. The caller tests their actual segment and vertical extent.
        let (x0, z0) = self.index(low[0], low[1]);
        let (x1, z1) = self.index(high[0], high[1]);
        let xmin = if low[0] == x0 as f64 * self.size {
            x0.saturating_sub(1)
        } else {
            x0
        };
        let zmin = if low[1] == z0 as f64 * self.size {
            z0.saturating_sub(1)
        } else {
            z0
        };
        self.tiles
            .range((xmin, i32::MIN)..=(x1, i32::MAX))
            .filter(move |((_, z), _)| (zmin..=z1).contains(z))
            .flat_map(|(_, tile)| tile.obstacles.iter().copied())
    }
}
