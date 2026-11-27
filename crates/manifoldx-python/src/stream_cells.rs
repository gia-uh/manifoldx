//! Cell publications at the graphics barrier. Workers only prepare owned data.

use std::collections::{HashMap, HashSet};

use manifoldx_core::ecs::{ColumnData, ComponentPayload, EntityId, World, WorldError};
use manifoldx_core::vehicles::terrain::{Surface, Terrain, Tile, TileKey};
use manifoldx_render::{NativeRenderer, RenderError};

pub(crate) type CellKey = (u64, u32, i64, i64);
pub(crate) type CellDraw = (u32, u32, Option<f32>, Option<f32>);
pub(crate) type CellColumn = (String, Option<Vec<f32>>, Option<Vec<u32>>);
pub(crate) type CellBatch = (u32, Vec<CellColumn>);

pub(crate) struct CellCommit {
    pub key: CellKey,
    pub version: u64,
    pub meshes: Vec<CellDraw>,
    pub origin: [f64; 3],
    pub rows: Vec<CellBatch>,
    pub tiles: Vec<Tile>,
}

struct Cell {
    version: u64,
    entities: Vec<EntityId>,
    tiles: Vec<(TileKey, u64)>,
}

#[derive(Default)]
pub(crate) struct StreamCells {
    cells: HashMap<CellKey, Cell>,
    terrain: Option<Terrain>,
    last_version: u64,
}

impl StreamCells {
    pub fn configure(&mut self, size: f64) -> Result<(), RenderError> {
        if self.terrain.is_some() {
            return Err(resource("stream collision terrain is already configured"));
        }
        self.terrain = Some(Terrain::new(size, Vec::new()).map_err(resource)?);
        Ok(())
    }

    pub fn sample(&self, x: f64, z: f64, near: f64) -> Option<Surface> {
        self.terrain.as_ref()?.sample(x, z, near)
    }

    pub fn count(&self) -> usize {
        self.cells.len()
    }

    pub fn commit(
        &mut self,
        world: &mut World,
        renderer: &NativeRenderer,
        commit: CellCommit,
        origin: [f64; 2],
    ) -> Result<(), RenderError> {
        let CellCommit {
            key,
            version,
            meshes,
            origin: pos,
            rows,
            tiles,
        } = commit;
        let old = self.cells.get(&key);
        if version <= self.last_version {
            return Err(resource("stale streamed cell version"));
        }
        let staged = self.stage_terrain(old, tiles, version)?;
        let p = [
            (pos[0] - origin[0]) as f32,
            pos[1] as f32,
            (pos[2] - origin[1]) as f32,
        ];
        if p.iter().any(|x| !x.is_finite()) {
            return Err(resource("invalid streamed position"));
        }
        let transform = vec![p[0], p[1], p[2], 0., 0., 0., 1., 1., 1., 1.];
        let mut batches = Vec::with_capacity(meshes.len() + rows.len());
        for (mesh, material, fade, baked) in meshes {
            if !renderer.has_stream_mesh(mesh, material) {
                return Err(resource("unavailable streamed mesh or material"));
            }
            let mut data = vec![
                payload(world, "Transform", ColumnData::F32(transform.clone()))?,
                payload(world, "Mesh", ColumnData::U32(vec![mesh]))?,
                payload(world, "Material", ColumnData::U32(vec![material]))?,
            ];
            for (name, value) in [("Fade", fade), ("Baked", baked)] {
                if let Some(value) = value {
                    data.push(payload(world, name, ColumnData::F32(vec![value]))?);
                }
            }
            batches.push((1, data));
        }
        for (count, columns) in rows {
            let mut data = Vec::with_capacity(columns.len());
            for (name, floats, integers) in columns {
                if matches!(name.as_str(), "Mesh" | "Material") {
                    return Err(resource("scene geometry belongs to the streamed mesh list"));
                }
                let column = match (floats, integers) {
                    (Some(values), None) if values.iter().all(|value| value.is_finite()) => {
                        ColumnData::F32(values)
                    }
                    (None, Some(values)) => ColumnData::U32(values),
                    _ => return Err(resource("invalid streamed ECS column data")),
                };
                data.push(payload(world, &name, column)?);
            }
            batches.push((count, data));
        }
        // All potentially rejected payloads were staged. Reserve every row together.
        let entities = world
            .spawn_batches(
                batches
                    .iter()
                    .map(|(count, data)| (*count, data.as_slice())),
            )
            .map_err(world_error)?;
        if let Some(old) = old {
            destroy_live(world, &old.entities)?;
        }
        let mut tiles = Vec::new();
        if let Some((terrain, ids)) = staged {
            self.terrain = Some(terrain);
            tiles = ids;
        }
        self.last_version = version;
        self.cells.insert(
            key,
            Cell {
                version,
                entities,
                tiles,
            },
        );
        Ok(())
    }

    pub fn hide(
        &mut self,
        world: &mut World,
        key: CellKey,
        version: u64,
    ) -> Result<(), RenderError> {
        let Some(cell) = self.cells.get(&key).filter(|cell| cell.version == version) else {
            return Ok(());
        };
        let terrain = self.stage_terrain(Some(cell), Vec::new(), version)?;
        destroy_live(world, &cell.entities)?;
        if let Some((terrain, _)) = terrain {
            self.terrain = Some(terrain);
        }
        self.cells.remove(&key);
        Ok(())
    }

    pub fn fade(
        &self,
        world: &mut World,
        key: CellKey,
        index: usize,
        value: f32,
    ) -> Result<(), RenderError> {
        if let Some(entity) = self
            .cells
            .get(&key)
            .and_then(|cell| cell.entities.get(index))
        {
            let fade = payload(world, "Fade", ColumnData::F32(vec![value]))?;
            world
                .update_rows(fade.component, &[*entity], fade.data)
                .map_err(world_error)?;
        }
        Ok(())
    }

    fn stage_terrain(
        &self,
        old: Option<&Cell>,
        mut tiles: Vec<Tile>,
        version: u64,
    ) -> Result<Option<TerrainEdit>, RenderError> {
        let old = old.map_or(&[][..], |cell| cell.tiles.as_slice());
        if old.is_empty() && tiles.is_empty() {
            return Ok(None);
        }
        let terrain = self
            .terrain
            .as_ref()
            .ok_or_else(|| resource("configure streamed collision before publishing tiles"))?;
        let old_keys: HashSet<_> = old.iter().map(|&(key, _)| key).collect();
        let mut ids = Vec::with_capacity(tiles.len());
        let mut new_keys = HashSet::new();
        for tile in &mut tiles {
            if terrain.tile_version(tile.key).is_some() && !old_keys.contains(&tile.key) {
                return Err(resource(
                    "collision tile belongs to a different streamed cell",
                ));
            }
            tile.version = version;
            ids.push((tile.key, version));
            new_keys.insert(tile.key);
        }
        let removed: Vec<_> = old
            .iter()
            .copied()
            .filter(|(key, _)| !new_keys.contains(key))
            .collect();
        terrain
            .updated(tiles, &removed)
            .map(|terrain| Some((terrain, ids)))
            .map_err(resource)
    }
}

type TerrainEdit = (Terrain, Vec<(TileKey, u64)>);

fn destroy_live(world: &mut World, ids: &[EntityId]) -> Result<(), RenderError> {
    let live: Vec<_> = ids
        .iter()
        .copied()
        .filter(|id| world.live_identity(id.slot()) == Some(*id))
        .collect();
    world.destroy(&live).map_err(world_error)?;
    Ok(())
}

fn payload(world: &World, name: &str, data: ColumnData) -> Result<ComponentPayload, RenderError> {
    let component = world
        .component_index(name)
        .ok_or_else(|| resource(format!("missing stream column {name}")))?;
    Ok(ComponentPayload { component, data })
}

fn resource(message: impl Into<String>) -> RenderError {
    RenderError::Resource(message.into())
}

fn world_error(error: WorldError) -> RenderError {
    resource(error.to_string())
}
