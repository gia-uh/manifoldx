// The exported scene in three.js: the glTF's manifoldx extras (background, fog, environment,
// sun, shadows) applied once with manifoldx's calibration, the desktop's fixed light for
// BasicMaterial and PhongMaterial, and handles from entities and material ids to three.js
// meshes and materials. THREE is passed in, so this module imports nothing.

// three.js against the desktop at the same intensities, measured by tests/web/browser/test_visual.py
// on tests/web/apps/lit_yard.py (sky environment, sun, shadows, fog) at 320x200, lavapipe for the
// desktop and SwiftShader for Chromium. The desktop's mean colour is (131.7, 141.3, 157.9); the
// page gives (126.5, 133.6, 146.4) uncalibrated (4.0, 5.5, 7.3% darker), (132.3, 142.0, 159.7)
// at 1.5, the first consumer's factor (worst channel 1.1%), and (131.7, 141.2, 158.4) at 1.45,
// the best of a sweep from 1.30 to 1.70 in steps of 0.05 (worst channel 0.3%; each step moves
// the means about 0.5%). The sun needs none.
export const CALIBRATION = { environment: 1.45, sun: 1.0 };

// The desktop lights BasicMaterial and PhongMaterial with a fixed light from (1, 1, 1) in world
// space: 0.3 ambient + 0.7 Lambert (resources.py, the basic shader).
const BASIC_VERTEX = "\tvMxNormal = normalize( mat3( modelMatrix ) * normal );";
const BASIC_FRAGMENT = "\tdiffuseColor.rgb *= 0.3 + 0.7 * max( dot( normalize( vMxNormal ), normalize( vec3( 1.0 ) ) ), 0.0 );";

function mxBasic(THREE, base) {
  const m = new THREE.MeshBasicMaterial({
    name: base.name,
    color: base.color.clone(),
    transparent: base.transparent,
    opacity: base.opacity,
    side: base.side,
    fog: false, // the desktop fogs only StandardMaterial
  });
  m.userData.mxBasic = true;
  m.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace("void main() {", "varying vec3 vMxNormal;\nvoid main() {")
      .replace("#include <project_vertex>", `#include <project_vertex>\n${BASIC_VERTEX}`);
    shader.fragmentShader = shader.fragmentShader
      .replace("void main() {", "varying vec3 vMxNormal;\nvoid main() {")
      .replace("#include <color_fragment>", `#include <color_fragment>\n${BASIC_FRAGMENT}`);
  };
  return m;
}

// A glTF material as the desktop draws it, by the material class name the writer gave it.
function convert(THREE, base) {
  if (base.name === "BasicMaterial" || base.name === "PhongMaterial") return mxBasic(THREE, base);
  if (base.name !== "StandardMaterial") base.fog = false;
  return base;
}

// The exported sky (linear RGB float32, row 0 at the zenith) as a half-float equirectangular
// texture, whose rows run bottom-up.
export function equirectTexture(THREE, rgb, w, h) {
  const data = new Uint16Array(w * h * 4);
  const one = THREE.DataUtils.toHalfFloat(1);
  for (let r = 0; r < h; r++) {
    const dst = (h - 1 - r) * w;
    for (let c = 0; c < w; c++) {
      const i = (r * w + c) * 3, o = (dst + c) * 4;
      data[o] = THREE.DataUtils.toHalfFloat(rgb[i]);
      data[o + 1] = THREE.DataUtils.toHalfFloat(rgb[i + 1]);
      data[o + 2] = THREE.DataUtils.toHalfFloat(rgb[i + 2]);
      data[o + 3] = one;
    }
  }
  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.HalfFloatType);
  tex.mapping = THREE.EquirectangularReflectionMapping;
  tex.colorSpace = THREE.LinearSRGBColorSpace;
  tex.needsUpdate = true;
  return tex;
}

// Point the sun's shadow camera as the desktop does (renderer._sun_light_view_proj): with
// auto_fit, at the scene's bounding sphere padded by max(5%, 0.1), once; otherwise at target,
// extent, near and far, from far / 2 back. The sun keeps its direction.
function fitShadow(THREE, scene, root, sun, cfg, renderer) {
  root.updateMatrixWorld(true);
  const from = new THREE.Vector3().setFromMatrixPosition(sun.matrixWorld);
  const dir = new THREE.Vector3().setFromMatrixPosition(sun.target.matrixWorld).sub(from).normalize();
  let center, extent, near, far, back;
  if (cfg.auto_fit) {
    const sphere = new THREE.Box3().setFromObject(root).getBoundingSphere(new THREE.Sphere());
    const pad = Math.max(sphere.radius * 0.05, 0.1);
    center = sphere.center;
    extent = sphere.radius + pad;
    [near, far, back] = [0.01, 2 * extent + pad, extent + pad];
  } else {
    center = new THREE.Vector3(...cfg.target);
    [extent, near, far, back] = [cfg.extent, cfg.near, cfg.far, cfg.far * 0.5];
  }
  const target = new THREE.Object3D();
  target.position.copy(center);
  scene.add(target);
  sun.remove(sun.target);
  sun.target = target;
  const at = center.clone().addScaledVector(dir, -back);
  sun.position.copy(sun.parent ? sun.parent.worldToLocal(at) : at);

  const size = Math.min(cfg.resolution, renderer.capabilities.maxTextureSize);
  sun.castShadow = true;
  sun.shadow.mapSize.set(size, size);
  Object.assign(sun.shadow.camera, { left: -extent, right: extent, top: extent, bottom: -extent, near, far });
  sun.shadow.camera.updateProjectionMatrix();
  sun.shadow.bias = -0.0005;
  sun.shadow.normalBias = 0.05;
  sun.shadow.radius = cfg.pcf_radius;
  renderer.shadowMap.enabled = true;
}

// The glTF the loader parsed, as an MxScene: .scene and .camera to render, .entities (entity
// index to its meshes), .materials (material id to its three.js material), .setMaterial(entity,
// id) (the engine's setMaterial hook), .materialOf(entity), .resetMaterials() (every entity back
// to the glTF's material, as a rebuilt scene has it), .sync(engine) each frame,
// .setCalibration(partial) over a copy of CALIBRATION, and .dispose() (frees its GPU resources). The renderer is used for its
// capabilities, the shadow map switch and, when the scene has an environment, PMREM.
export async function buildScene(THREE, gltf, renderer) {
  const extras = gltf.scene.userData.manifoldx ?? {};
  // gltf.scene is a Group; background, fog and environment only apply to a Scene.
  const scene = new THREE.Scene();
  scene.add(gltf.scene);
  if (extras.background) scene.background = new THREE.Color().setRGB(...extras.background);
  if (extras.fog) scene.fog = new THREE.Fog(new THREE.Color().setRGB(...extras.fog.color), extras.fog.start, extras.fog.end);

  const converted = new Map(); // loader material -> the one drawn
  const drawn = (base) => {
    if (!converted.has(base)) converted.set(base, convert(THREE, base));
    return converted.get(base);
  };
  const materials = new Map();
  for (const [id, index] of Object.entries(extras.mx_materials ?? {})) {
    materials.set(Number(id), drawn(await gltf.parser.getDependency("material", index)));
  }

  const entities = new Map(), current = new Map(), original = new Map();
  gltf.scene.traverse((o) => {
    const mx = o.userData.mx;
    if (!mx) return;
    const meshes = [];
    o.traverse((m) => {
      if (!m.isMesh) return;
      m.material = materials.get(mx.material) ?? drawn(m.material);
      meshes.push(m);
    });
    entities.set(mx.entity, meshes);
    current.set(mx.entity, mx.material);
    original.set(mx.entity, [mx.material, meshes.map((m) => m.material)]);
  });

  if (extras.environment) {
    const env = extras.environment;
    const rgb = new Float32Array(await gltf.parser.getDependency("bufferView", env.bufferView));
    const tex = equirectTexture(THREE, rgb, env.width, env.height);
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromEquirectangular(tex).texture;
    tex.dispose();
    pmrem.dispose();
  }

  let sun = null;
  gltf.scene.traverse((o) => { if (!sun && o.isDirectionalLight) sun = o; });
  const sunIntensity = sun ? sun.intensity : 0;
  if (sun && extras.shadows) {
    fitShadow(THREE, scene, gltf.scene, sun, extras.shadows, renderer);
    scene.traverse((o) => { if (o.isMesh) o.castShadow = o.receiveShadow = true; });
  }

  const calibration = { ...CALIBRATION };
  const calibrate = () => {
    if (sun) sun.intensity = sunIntensity * calibration.sun;
    if (extras.environment) scene.environmentIntensity = extras.environment.intensity * calibration.environment;
  };
  calibrate();

  const camera = new THREE.PerspectiveCamera();
  return {
    scene,
    camera,
    entities,
    materials,
    setMaterial(entity, id) {
      const meshes = entities.get(entity), material = materials.get(id);
      if (meshes === undefined) throw new Error(`entity ${entity} is not in the exported scene`);
      if (material === undefined) throw new Error(`material ${id} is not in the exported scene`);
      for (const m of meshes) m.material = material;
      current.set(entity, id);
    },
    materialOf(entity) {
      return current.get(entity) ?? null;
    },
    resetMaterials() {
      for (const [entity, [id, mats]] of original) {
        entities.get(entity).forEach((m, i) => { m.material = mats[i]; });
        current.set(entity, id);
      }
    },
    dispose() {
      const done = new Set();
      const free = (thing) => {
        if (!thing || done.has(thing)) return;
        done.add(thing);
        if (thing.isMaterial) for (const v of Object.values(thing)) if (v?.isTexture) free(v);
        thing.dispose();
      };
      scene.traverse((o) => {
        free(o.geometry);
        for (const m of [o.material ?? []].flat()) free(m);
      });
      for (const m of materials.values()) free(m);
      for (const m of converted.values()) free(m);
      free(scene.environment);
    },
    sync(engine) {
      const c = engine.camera;
      camera.position.set(c.position[0], c.position[1], c.position[2]);
      camera.up.set(c.up[0], c.up[1], c.up[2]);
      camera.lookAt(c.target[0], c.target[1], c.target[2]);
      camera.fov = c.fov;
      camera.near = c.near;
      camera.far = c.far;
      camera.aspect = engine.w / engine.h;
      camera.updateProjectionMatrix();
    },
    setCalibration(partial) {
      Object.assign(calibration, partial);
      calibrate();
    },
  };
}
