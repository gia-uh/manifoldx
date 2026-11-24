//! One standalone native scene: build a real World with the production
//! Transform/Mesh/Material schemas, upload a cube mesh once, render two states
//! offscreen on the GPU, and read the pixels back.
//!
//! `--check` asserts coverage, depth ordering, a transform-driven image change
//! and the two negative controls. `--no-draw` and `--ignore-transform` are the
//! controls; with `--check` they deliberately exit non-zero.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use manifoldx_core::ecs::{ColumnData, ElementType, EntityId, World};
use manifoldx_render::camera::Camera;
use manifoldx_render::renderer::{
    json_string, write_ppm, NativeRenderer, RenderOutput, RendererOptions, MAX_SIDE,
};
use manifoldx_render::scene::{resolve_source, spawn_instances, MeshData, DEFAULT_MAX_INSTANCES};

const CUBE_GEOMETRY: u32 = 1;
const MAT_GROUND: u32 = 1;
const MAT_GREEN: u32 = 2;
const MAT_CYAN: u32 = 3;
const MAT_RED: u32 = 4;
const MAT_BLUE: u32 = 5;
const MAT_YELLOW: u32 = 6;

const DEFAULT_WIDTH: u32 = 800;
const DEFAULT_HEIGHT: u32 = 600;

#[derive(Clone, Debug)]
struct Options {
    out_dir: PathBuf,
    width: u32,
    height: u32,
    check: bool,
    no_draw: bool,
    ignore_transform: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            out_dir: PathBuf::from("."),
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            check: false,
            no_draw: false,
            ignore_transform: false,
        }
    }
}

impl Options {
    fn parse() -> Result<Self, String> {
        let mut options = Self::default();
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--check" => options.check = true,
                "--no-draw" => options.no_draw = true,
                "--ignore-transform" => options.ignore_transform = true,
                "--out-dir" => {
                    options.out_dir = PathBuf::from(args.next().ok_or("--out-dir needs a path")?);
                }
                "--width" => {
                    options.width = args
                        .next()
                        .ok_or("--width needs a number")?
                        .parse()
                        .map_err(|_| "--width must be a number")?;
                }
                "--height" => {
                    options.height = args
                        .next()
                        .ok_or("--height needs a number")?
                        .parse()
                        .map_err(|_| "--height must be a number")?;
                }
                "-h" | "--help" => {
                    println!(
                        "native-scene [--check] [--no-draw] [--ignore-transform]\n\
                         \x20            [--out-dir DIR] [--width N] [--height N]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument {other}")),
            }
        }
        if !(1..=MAX_SIDE).contains(&options.width) || !(1..=MAX_SIDE).contains(&options.height) {
            return Err(format!("width and height must be in 1..={MAX_SIDE}"));
        }
        Ok(options)
    }
}

fn main() -> ExitCode {
    let options = match Options::parse() {
        Ok(options) => options,
        Err(error) => {
            eprintln!("native-scene: {error}");
            return ExitCode::from(2);
        }
    };
    match run(options) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("CHECK FAIL: {error}");
            ExitCode::from(2)
        }
    }
}

fn run(options: Options) -> Result<(), String> {
    let started = Instant::now();
    std::fs::create_dir_all(&options.out_dir).map_err(|error| error.to_string())?;

    let mut rows = initial_rows();
    let (mut world, ids, transform_column) = build_world(&rows)?;
    let source = resolve_source(
        &world,
        "Transform",
        "Mesh",
        Some("Material"),
        DEFAULT_MAX_INSTANCES,
    )
    .map_err(|error| error.to_string())?;

    let mut renderer =
        NativeRenderer::new(RendererOptions::default()).map_err(|error| error.to_string())?;
    renderer
        .upload_mesh(CUBE_GEOMETRY, &unit_cube(), [0.82, 0.82, 0.82])
        .map_err(|error| error.to_string())?;
    for (id, color) in [
        (MAT_GROUND, [0.32, 0.34, 0.38]),
        (MAT_GREEN, [0.18, 0.78, 0.30]),
        (MAT_CYAN, [0.10, 0.72, 0.86]),
        (MAT_RED, [0.90, 0.16, 0.14]),
        (MAT_BLUE, [0.14, 0.28, 0.92]),
        (MAT_YELLOW, [0.92, 0.80, 0.18]),
    ] {
        renderer.register_material(id, color);
    }

    let camera = Camera {
        position: [0.0, 1.0, 7.0],
        target: [0.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0],
        fov_y_degrees: 55.0,
        near: 0.1,
        far: 100.0,
    };

    let state_a = if options.no_draw {
        renderer
            .render_without_draw(
                &world,
                source,
                &camera,
                options.width,
                options.height,
                options.ignore_transform,
            )
            .map_err(|error| error.to_string())?
    } else {
        renderer
            .render(
                &world,
                source,
                &camera,
                options.width,
                options.height,
                options.ignore_transform,
            )
            .map_err(|error| error.to_string())?
    };

    // State B changes the native World rows: the near red cube moves out of
    // the centre ray and the left green cube steps up and right. Nothing in
    // the renderer is told about the change except the World version bump.
    rows[3] = row([0.0, 2.7, 1.2], IDENTITY, [0.7, 0.7, 0.7]);
    rows[1] = row([-1.35, 0.85, 0.2], yaw(0.55), [0.7, 0.7, 0.7]);
    rows[2] = row([2.0, 0.0, 0.0], yaw(-0.95), [0.7, 0.7, 0.7]);
    let mut flat = Vec::with_capacity(ids.len() * 10);
    for row in &rows {
        flat.extend_from_slice(row);
    }
    world
        .update_rows(transform_column, &ids, ColumnData::F32(flat))
        .map_err(|error| error.to_string())?;

    let state_b = if options.no_draw {
        renderer
            .render_without_draw(
                &world,
                source,
                &camera,
                options.width,
                options.height,
                options.ignore_transform,
            )
            .map_err(|error| error.to_string())?
    } else {
        renderer
            .render(
                &world,
                source,
                &camera,
                options.width,
                options.height,
                options.ignore_transform,
            )
            .map_err(|error| error.to_string())?
    };

    write_ppm(
        &options.out_dir.join("state-a.ppm"),
        state_a.width,
        state_a.height,
        &state_a.rgba,
    )
    .map_err(|error| error.to_string())?;
    write_ppm(
        &options.out_dir.join("state-b.ppm"),
        state_b.width,
        state_b.height,
        &state_b.rgba,
    )
    .map_err(|error| error.to_string())?;

    let analysis_a = analyze(&state_a);
    let analysis_b = analyze(&state_b);
    let diff = diff(&state_a, &state_b);
    let check = check_scene(&options, &analysis_a, &analysis_b, &diff);

    let summary = summary_json(
        &options,
        &renderer,
        &state_a,
        &state_b,
        &analysis_a,
        &analysis_b,
        &diff,
        &check,
        elapsed_ms(started.elapsed()),
    );
    std::fs::write(options.out_dir.join("summary.json"), &summary)
        .map_err(|error| error.to_string())?;
    println!("{summary}");

    match check {
        CheckResult::Pass => Ok(()),
        CheckResult::Control(message) => Err(message),
        CheckResult::Fail(message) => Err(message),
    }
}

fn build_world(rows: &[[f32; 10]]) -> Result<(World, Vec<EntityId>, usize), String> {
    let mut world = World::new(16).map_err(|error| error.to_string())?;
    let transform = world
        .register("Transform", ElementType::F32, 10)
        .map_err(|error| error.to_string())?;
    world
        .register("Mesh", ElementType::U32, 1)
        .map_err(|error| error.to_string())?;
    world
        .register("Material", ElementType::U32, 1)
        .map_err(|error| error.to_string())?;
    let geometry = [CUBE_GEOMETRY; 6];
    let materials = [
        MAT_GROUND, MAT_GREEN, MAT_CYAN, MAT_RED, MAT_BLUE, MAT_YELLOW,
    ];
    let ids = spawn_instances(&mut world, rows, &geometry, &materials)
        .map_err(|error| error.to_string())?;
    Ok((world, ids, transform))
}

const IDENTITY: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

fn yaw(radians: f32) -> [f32; 4] {
    let half = radians * 0.5;
    [0.0, half.sin(), 0.0, half.cos()]
}

fn row(position: [f32; 3], rotation: [f32; 4], scale: [f32; 3]) -> [f32; 10] {
    [
        position[0],
        position[1],
        position[2],
        rotation[0],
        rotation[1],
        rotation[2],
        rotation[3],
        scale[0],
        scale[1],
        scale[2],
    ]
}

fn initial_rows() -> [[f32; 10]; 6] {
    [
        // A wide thin ground slab, so coverage and depth context are real.
        row([0.0, -1.0, 0.0], IDENTITY, [5.0, 0.08, 5.0]),
        // Left and right cubes, rotated to show two shaded faces.
        row([-2.0, 0.0, 0.0], yaw(0.6), [0.7, 0.7, 0.7]),
        row([2.0, 0.0, 0.0], yaw(-0.6), [0.7, 0.7, 0.7]),
        // Near red and far blue overlap on the centre ray. The near cube
        // wins the depth test; moving it in state B reveals the far cube.
        row([0.0, 0.0, 1.2], IDENTITY, [0.7, 0.7, 0.7]),
        row([0.0, 0.0, -1.2], IDENTITY, [0.7, 0.7, 0.7]),
        // One more instance above and behind, for silhouette variety.
        row([0.0, 1.35, -1.7], yaw(0.35), [0.5, 0.5, 0.5]),
    ]
}

/// A unit cube with outward normals, counter-clockwise from outside.
fn unit_cube() -> MeshData {
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        (
            [1.0, 0.0, 0.0],
            [
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0],
            ],
        ),
        (
            [-1.0, 0.0, 0.0],
            [
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, -1.0],
            ],
        ),
        (
            [0.0, 1.0, 0.0],
            [
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0],
            ],
        ),
        (
            [0.0, -1.0, 0.0],
            [
                [-1.0, -1.0, 1.0],
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, -1.0, 1.0],
            ],
        ),
        (
            [0.0, 0.0, 1.0],
            [
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ],
        ),
        (
            [0.0, 0.0, -1.0],
            [
                [1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0],
            ],
        ),
    ];
    let mut positions = Vec::with_capacity(24);
    let mut normals = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (normal, corners) in faces {
        let base = positions.len() as u32;
        for corner in corners {
            positions.push(corner);
            normals.push(normal);
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    MeshData {
        positions,
        normals,
        indices,
    }
}

#[derive(Clone, Debug, Default)]
struct ImageAnalysis {
    width: u32,
    height: u32,
    coverage: usize,
    red: usize,
    green: usize,
    blue: usize,
    cyan: usize,
    center: [u8; 4],
    bbox: Option<(u32, u32, u32, u32)>,
}

impl ImageAnalysis {
    fn coverage_fraction(&self) -> f64 {
        if self.width == 0 || self.height == 0 {
            return 0.0;
        }
        self.coverage as f64 / (self.width as f64 * self.height as f64)
    }

    fn center_is_red(&self) -> bool {
        let [r, g, b, _] = self.center;
        r > g.saturating_add(15) && r > b.saturating_add(15) && r > 30
    }

    fn center_is_blue(&self) -> bool {
        let [r, g, b, _] = self.center;
        b > r.saturating_add(20) && b > g.saturating_add(10) && b > 30
    }
}

#[allow(clippy::needless_range_loop)]
fn analyze(image: &RenderOutput) -> ImageAnalysis {
    let mut analysis = ImageAnalysis {
        width: image.width,
        height: image.height,
        ..Default::default()
    };
    let center_index =
        ((image.height / 2) as usize * image.width as usize + (image.width / 2) as usize) * 4;
    analysis
        .center
        .copy_from_slice(&image.rgba[center_index..center_index + 4]);
    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x = 0;
    let mut max_y = 0;
    for y in 0..image.height {
        for x in 0..image.width {
            let offset = ((y * image.width + x) * 4) as usize;
            let [r, g, b, _] = [
                image.rgba[offset],
                image.rgba[offset + 1],
                image.rgba[offset + 2],
                image.rgba[offset + 3],
            ];
            if r as u16 + g as u16 + b as u16 <= 8 {
                continue;
            }
            analysis.coverage += 1;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            let (ri, gi, bi) = (r as i32, g as i32, b as i32);
            if ri > gi + 20 && ri > bi + 20 {
                analysis.red += 1;
            } else if bi > ri + 25 && gi > ri + 25 && (bi - gi).abs() < 70 {
                analysis.cyan += 1;
            } else if bi > ri + 25 && bi > gi + 25 {
                analysis.blue += 1;
            } else if gi > ri + 20 && gi > bi + 20 {
                analysis.green += 1;
            }
        }
    }
    if analysis.coverage > 0 {
        analysis.bbox = Some((min_x, min_y, max_x, max_y));
    }
    analysis
}

#[derive(Clone, Copy, Debug, Default)]
struct DiffStats {
    pixels: usize,
    mean_abs: f64,
}

fn diff(a: &RenderOutput, b: &RenderOutput) -> DiffStats {
    if a.width != b.width || a.height != b.height || a.rgba.len() != b.rgba.len() {
        return DiffStats::default();
    }
    let mut pixels = 0;
    let mut total = 0u64;
    for (left, right) in a
        .rgba
        .as_chunks::<4>()
        .0
        .iter()
        .zip(b.rgba.as_chunks::<4>().0)
    {
        let delta = (0..3)
            .map(|channel| left[channel].abs_diff(right[channel]))
            .max()
            .unwrap_or(0);
        total += delta as u64;
        if delta > 8 {
            pixels += 1;
        }
    }
    let mean_abs = total as f64 / (a.width as f64 * a.height as f64 * 3.0);
    DiffStats { pixels, mean_abs }
}

enum CheckResult {
    Pass,
    Control(String),
    Fail(String),
}

fn check_scene(
    options: &Options,
    a: &ImageAnalysis,
    b: &ImageAnalysis,
    diff: &DiffStats,
) -> CheckResult {
    if options.no_draw {
        if a.coverage == 0 && b.coverage == 0 {
            return CheckResult::Control(
                "negative control --no-draw produced zero coverage; the positive check fails"
                    .into(),
            );
        }
        return CheckResult::Fail("negative control --no-draw still drew pixels".into());
    }
    if options.ignore_transform {
        if diff.pixels == 0 {
            return CheckResult::Control(
                "negative control --ignore-transform made both World states identical; \
                 the positive transform check fails"
                    .into(),
            );
        }
        return CheckResult::Fail(
            "negative control --ignore-transform still changed the image".into(),
        );
    }

    let pixels = a.width as f64 * a.height as f64;
    if a.coverage_fraction() < 0.05 || b.coverage_fraction() < 0.05 {
        return CheckResult::Fail(format!(
            "coverage too low: state A {:.3}, state B {:.3}",
            a.coverage_fraction(),
            b.coverage_fraction()
        ));
    }
    if diff.pixels as f64 / pixels < 0.01 {
        return CheckResult::Fail(format!(
            "transform change moved only {} pixels ({:.4} of the image)",
            diff.pixels,
            diff.pixels as f64 / pixels
        ));
    }
    if !a.center_is_red() {
        return CheckResult::Fail(format!(
            "state A centre is not the near red cube: {:?}",
            a.center
        ));
    }
    if !b.center_is_blue() {
        return CheckResult::Fail(format!(
            "state B centre is not the revealed far blue cube: {:?}",
            b.center
        ));
    }
    if a.green < 200 || a.cyan < 200 {
        return CheckResult::Fail(format!(
            "side instances missing: green {}, cyan {}",
            a.green, a.cyan
        ));
    }
    if b.blue < 200 || b.red < 200 {
        return CheckResult::Fail(format!(
            "state B instance colours missing: red {}, blue {}",
            b.red, b.blue
        ));
    }
    CheckResult::Pass
}

#[allow(clippy::too_many_arguments)]
fn summary_json(
    options: &Options,
    renderer: &NativeRenderer,
    a: &RenderOutput,
    b: &RenderOutput,
    analysis_a: &ImageAnalysis,
    analysis_b: &ImageAnalysis,
    diff: &DiffStats,
    check: &CheckResult,
    elapsed_ms: u64,
) -> String {
    let (result, message) = match check {
        CheckResult::Pass => ("pass", None),
        CheckResult::Control(message) => ("control", Some(message.as_str())),
        CheckResult::Fail(message) => ("fail", Some(message.as_str())),
    };
    let source_sha = option_env!("MANIFOLDX_SOURCE_SHA").unwrap_or("uncommitted");
    format!(
        concat!(
            "{{\"schema\":\"manifoldx-native-scene/1\",",
            "\"slice\":{},\"source_sha\":{},",
            "\"adapter\":{},\"width\":{},\"height\":{},",
            "\"format\":\"rgba8unorm-srgb\",\"depth_format\":\"depth32float\",",
            "\"states\":{{\"a\":{},\"b\":{}}},",
            "\"analysis\":{{\"a\":{},\"b\":{},\"diff_pixels\":{},\"mean_abs_diff\":{:.4}}},",
            "\"check\":{{\"requested\":{},\"mode\":{},\"result\":{},\"message\":{}}},",
            "\"elapsed_ms\":{},",
            "\"timings_note\":\"read_ns/upload_ns/encode_submit_ns are CPU wall; \
             wait_readback_ns is CPU wait around GPU execution and mapping; \
             no timestamp query is enabled\"}}"
        ),
        json_string(manifoldx_render::SLICE_NAME),
        json_string(source_sha),
        renderer.adapter().to_json(),
        a.width,
        a.height,
        state_json(a),
        state_json(b),
        analysis_json(analysis_a),
        analysis_json(analysis_b),
        diff.pixels,
        diff.mean_abs,
        options.check,
        json_string(if options.no_draw {
            "no-draw"
        } else if options.ignore_transform {
            "ignore-transform"
        } else {
            "normal"
        }),
        json_string(result),
        message.map_or_else(|| "null".to_string(), json_string),
        elapsed_ms
    )
}

fn state_json(image: &RenderOutput) -> String {
    let stats = &image.stats;
    format!(
        concat!(
            "{{\"entities_examined\":{},\"instances\":{},\"draw_calls\":{},",
            "\"triangles\":{},\"read_ns\":{},\"upload_ns\":{},",
            "\"encode_submit_ns\":{},\"wait_readback_ns\":{},\"total_ns\":{}}}"
        ),
        stats.entities_examined,
        stats.instances,
        stats.draw_calls,
        stats.triangles,
        stats.read_ns,
        stats.upload_ns,
        stats.encode_submit_ns,
        stats.wait_readback_ns,
        stats.total_ns
    )
}

fn analysis_json(analysis: &ImageAnalysis) -> String {
    let bbox = match analysis.bbox {
        Some((min_x, min_y, max_x, max_y)) => {
            format!("[{min_x},{min_y},{max_x},{max_y}]")
        }
        None => "null".to_string(),
    };
    format!(
        concat!(
            "{{\"coverage\":{},\"coverage_fraction\":{:.4},",
            "\"red\":{},\"green\":{},\"blue\":{},\"cyan\":{},",
            "\"center\":[{},{},{},{}],\"bbox\":{}}}"
        ),
        analysis.coverage,
        analysis.coverage_fraction(),
        analysis.red,
        analysis.green,
        analysis.blue,
        analysis.cyan,
        analysis.center[0],
        analysis.center[1],
        analysis.center[2],
        analysis.center[3],
        bbox
    )
}

fn elapsed_ms(duration: std::time::Duration) -> u64 {
    duration.as_millis().min(u64::MAX as u128) as u64
}
