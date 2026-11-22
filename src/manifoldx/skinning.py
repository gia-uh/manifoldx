"""GPU skinning: per-entity joint blocks and the skinned vertex-shader transform.

A skinned geometry carries one joint index per vertex (rigid), or two and a weight (blended): a vertex moves by
w * M[a] + (1 - w) * M[b]. Both travel in one 32-bit attribute (`pack_joints`). Each posed entity owns a
block of MAX_JOINTS matrices in one storage buffer; the mesh pass uploads a
block index per instance, and the vertex shader moves position and normal by
joint_mats[block * MAX_JOINTS + joint] before the material's own code runs.
"""

import re

import numpy as np

from manifoldx import _deferred
from manifoldx import counters

MAX_JOINTS = 32

JOINT_LOCATION = 7  # vertex attribute location of the joint index
JOINTS_BINDING = 8  # group(0) storage: array<mat4x4<f32>>, all joint blocks
BLOCKS_BINDING = 9  # group(0) storage: array<u32>, block index per instance

_BLOCK_BYTES = MAX_JOINTS * 64
FRAME_MASK = 0xFFFFF  # the block bits of the per-instance u32; bits 20-31 blend toward the next block
_IDENTITY_BLOCK = np.tile(np.eye(4, dtype=np.float32), (MAX_JOINTS, 1, 1))


def pack_joints(joints, weights=None) -> np.ndarray:
    """(N,) or (N, 2) joint indices and (N,) weights of the first -> (N,) uint32.

    Bits 0-7 hold joint a, 8-15 joint b, 16-31 round(weight_a * 65535). (N,) joints pack as a = b, weight 1.
    """
    j = np.asarray(joints, dtype=np.uint32)
    if j.ndim == 1:
        a, b, w = j, j, np.ones(len(j))
    else:
        a, b = j[:, 0], j[:, 1]
        w = np.ones(len(j)) if weights is None else np.asarray(weights, dtype=np.float64)
    bits = np.round(np.clip(w, 0.0, 1.0) * 65535).astype(np.uint32)
    return (a & 255) | ((b & 255) << 8) | (bits << 16)


def unpack_joints(packed) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The inverse of `pack_joints`: joint a, joint b and the weight of a."""
    p = np.asarray(packed, dtype=np.uint32)
    return p & 255, (p >> 8) & 255, (p >> 16).astype(np.float32) / 65535


def validate_weights(weights, joints: np.ndarray) -> np.ndarray | None:
    """Return the weights of a blended mesh's first joints as (N,) float32, or raise ValueError.

    Rigid (N,) joints take no weights; (N, 2) joints need one weight in [0, 1] per vertex.
    """
    if joints.ndim == 1:
        if weights is not None:
            raise ValueError("weights need joints of shape (N, 2); rigid (N,) joints take none")
        return None
    if weights is None:
        raise ValueError("joints of shape (N, 2) need weights of shape (N,)")
    w = np.asarray(weights, dtype=np.float32)
    if w.shape != (len(joints),):
        raise ValueError(f"weights must have shape ({len(joints)},), got {w.shape}")
    if len(w) and (w.min() < 0 or w.max() > 1):
        raise ValueError(f"weights must be in [0, 1], got [{w.min()}, {w.max()}]")
    return np.ascontiguousarray(w)


def validate_joints(joints, n_vertices: int) -> np.ndarray:
    """Return joints as (N,) or (N, 2) uint32 or raise ValueError."""
    j = np.asarray(joints)
    if j.ndim not in (1, 2) or len(j) != n_vertices or (j.ndim == 2 and j.shape[1] != 2):
        raise ValueError(f"joints must have shape ({n_vertices},) or ({n_vertices}, 2), got {j.shape}")
    if not np.issubdtype(j.dtype, np.integer):
        raise ValueError(f"joints must be integers, got dtype {j.dtype}")
    if len(j) and (j.min() < 0 or j.max() >= MAX_JOINTS):
        raise ValueError(f"joints must be in [0, {MAX_JOINTS}), got [{j.min()}, {j.max()}]")
    return np.ascontiguousarray(j, dtype=np.uint32)


def joint_count(geometry: dict) -> int:
    """Joints a skinned geometry uses: its largest joint index plus one."""
    return int(np.max(geometry["joints"])) + 1


def append_joint_column(interleaved: np.ndarray, joints, weights=None) -> np.ndarray:
    """(N, k) float32 vertex rows -> (N, k + 1), the last column the packed joint bits (`pack_joints`)."""
    out = np.empty((len(interleaved), interleaved.shape[1] + 1), dtype=np.float32)
    out[:, :-1] = interleaved
    out[:, -1] = pack_joints(joints, weights).view(np.float32)
    return out


class JointStore:
    """Joint blocks of every posed entity: a CPU mirror and its GPU storage buffer.

    Block 0 is all identity and never handed out; an entity that was never
    posed reads it. Matrices are stored transposed, so WGSL reads them as
    column-major mat4.

    A baked clip (`bake`) is a run of shared blocks, one per frame. An entity
    that `play`s it reads two neighbouring frames, blended on the GPU: the
    u32 the shader reads per instance holds the block in bits 0-19 and the
    blend fraction in bits 20-31 (0 for an entity posed with `set`).
    """

    def __init__(self, max_entities: int):
        self.block_of = np.zeros(max_entities, dtype=np.uint32)
        self.frame_of = np.zeros(max_entities, dtype=np.uint32)  # packed frame and fraction; 0 = own block
        self._clips: dict[int, int] = {}  # a clip's first block -> its frame count
        self.mirror = np.tile(_IDENTITY_BLOCK, (2, 1, 1, 1))  # (blocks, MAX_JOINTS, 4, 4)
        self._next = 1
        self._free: list[int] = []
        self._device = None
        self.buffer = None
        self._buffer_blocks = 0

    def _current(self) -> bool:
        return self.buffer is not None and self._buffer_blocks == len(self.mirror)

    def _alloc(self, k: int) -> np.ndarray:
        """k unused blocks: freed ones first, then new ones (the mirror doubles until they fit)."""
        reused = [self._free.pop() for _ in range(min(k, len(self._free)))]
        fresh = k - len(reused)
        while self._next + fresh > len(self.mirror):
            self.mirror = np.concatenate([self.mirror, np.tile(_IDENTITY_BLOCK, (len(self.mirror), 1, 1, 1))])
        self._next += fresh
        if self._next > FRAME_MASK:
            raise ValueError(f"joint blocks: {self._next} is over the {FRAME_MASK} a frame index can hold")
        return np.concatenate([np.array(reused, dtype=np.uint32),
                               np.arange(self._next - fresh, self._next, dtype=np.uint32)])

    @_deferred.deferred_in_pipeline
    def set(self, entity: int, matrices: np.ndarray):
        """Write (J, 4, 4) row-major matrices into the entity's block; upload J*64 bytes."""
        self.frame_of[entity] = 0
        block = int(self.block_of[entity])
        if block == 0:
            block = int(self._alloc(1)[0])
            self.block_of[entity] = block
        n = len(matrices)
        self.mirror[block, :n] = matrices.transpose(0, 2, 1)
        if self._current():
            counters.write_buffer(self._device.queue, self.buffer, block * _BLOCK_BYTES, self.mirror[block, :n].tobytes())

    @_deferred.deferred_in_pipeline
    def set_many(self, entities, matrices: np.ndarray):
        """Write (N, J, 4, 4) row-major matrices into the entities' blocks; upload the touched blocks once."""
        entities = np.asarray(entities, dtype=np.int64)
        self.frame_of[entities] = 0
        blocks = self.block_of[entities]
        new = np.unique(entities[blocks == 0])
        if len(new):
            self.block_of[new] = self._alloc(len(new))
            blocks = self.block_of[entities]
        self.mirror[blocks, : matrices.shape[1]] = matrices.transpose(0, 1, 3, 2)
        if self._current():
            lo, hi = int(blocks.min()), int(blocks.max()) + 1
            counters.write_buffer(self._device.queue, self.buffer, lo * _BLOCK_BYTES, self.mirror[lo:hi].tobytes())

    def pose(self, entity: int, j: int) -> np.ndarray:
        """The entity's first j joint matrices, (j, 4, 4) float32 row-major; identity when never posed."""
        return self.current(np.array([entity]))[0, :j].transpose(0, 2, 1).copy()

    def gpu_blocks(self, entities) -> np.ndarray:
        """The u32 per entity the shaders read: its playing frame and fraction, else its own block."""
        frames = self.frame_of[entities]
        return np.where(frames != 0, frames, self.block_of[entities])

    def current(self, entities) -> np.ndarray:
        """(N, MAX_JOINTS, 4, 4) transposed matrices each entity is drawn with, frames blended as the GPU does."""
        packed = self.gpu_blocks(entities)
        block = (packed & FRAME_MASK).astype(np.int64)
        f = ((packed >> 20) / 4095.0).astype(np.float32)[:, None, None, None]
        a = self.mirror[block]
        if not f.any():
            return a
        b = self.mirror[np.minimum(block + 1, len(self.mirror) - 1)]
        return a + f * (b - a)

    def bake(self, frames: np.ndarray) -> int:
        """Store (K, J, 4, 4) row-major poses as K consecutive shared blocks; return the first block.

        The blocks are numbered at once, so the caller can `play` them; the write is deferred with the engine.
        """
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 4 or frames.shape[2:] != (4, 4) or not 1 <= frames.shape[1] <= MAX_JOINTS or len(frames) < 1:
            raise ValueError(f"bake expects (K, J, 4, 4) poses with 1 <= J <= {MAX_JOINTS}, got {frames.shape}")
        k = len(frames)
        first = self._next
        if first + k > FRAME_MASK:
            raise ValueError(f"joint blocks: {first + k} is over the {FRAME_MASK} a frame index can hold")
        self._next += k
        self._clips[first] = k
        self._write_clip(first, frames)
        return first

    @_deferred.deferred_in_pipeline
    def _write_clip(self, first: int, frames: np.ndarray):
        k = len(frames)
        while first + k > len(self.mirror):
            self.mirror = np.concatenate([self.mirror, np.tile(_IDENTITY_BLOCK, (len(self.mirror), 1, 1, 1))])
        self.mirror[first : first + k, : frames.shape[1]] = frames.transpose(0, 1, 3, 2)
        if self._current():
            counters.write_buffer(self._device.queue, self.buffer, first * _BLOCK_BYTES,
                                  self.mirror[first : first + k].tobytes())
        return first

    def play(self, entities, first, positions):
        """Draw each entity at `positions` (frames, 0 to K - 1) of the clip baked at `first`, blending two frames.

        `first` is one clip for all or one per entity. Nothing is uploaded: the frame index rides in the
        per-instance block table the passes upload every frame.
        """
        entities = np.asarray(entities, dtype=np.int64)
        first = np.broadcast_to(np.asarray(first, dtype=np.int64), entities.shape)
        pos = np.asarray(positions, dtype=np.float64)
        lengths = np.array([self._clips.get(int(b), 0) for b in np.unique(first)])
        if (lengths == 0).any():
            raise ValueError("play: a first block is not a baked clip")
        last = np.vectorize(self._clips.get, otypes=[np.int64])(first) - 1
        if (pos < 0).any() or (pos > last).any():
            raise ValueError("play: a position is outside its clip's frames 0 to K - 1")
        whole = np.minimum(np.floor(pos).astype(np.int64), np.maximum(last - 1, 0))
        frac = np.rint((pos - whole) * 4095).astype(np.uint32)
        self._set_frames(entities, (first + whole).astype(np.uint32) | (frac << 20))

    @_deferred.deferred_in_pipeline
    def _set_frames(self, entities, packed):
        self.frame_of[entities] = packed

    @_deferred.deferred_in_pipeline
    def release(self, entities):
        """Free the blocks of destroyed or respawned entities, reset to identity."""
        entities = np.asarray(entities, dtype=np.int64)
        self.frame_of[entities] = 0
        blocks = self.block_of[entities]
        blocks = np.unique(blocks[blocks > 0])
        if len(blocks) == 0:
            return
        self.block_of[entities] = 0
        for b in blocks.tolist():
            self.mirror[b] = _IDENTITY_BLOCK
            if self._current():
                counters.write_buffer(self._device.queue, self.buffer, b * _BLOCK_BYTES, _IDENTITY_BLOCK.tobytes())
            self._free.append(b)

    @_deferred.deferred_in_pipeline
    def ensure_buffer(self, device):
        """The storage buffer holding every block; (re)created with the full mirror when it grew."""
        import wgpu

        if not self._current():
            self._device = device
            self.buffer = device.create_buffer(
                size=self.mirror.nbytes,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
            counters.write_buffer(device.queue, self.buffer, 0, self.mirror.tobytes())
            self._buffer_blocks = len(self.mirror)
        return self.buffer


# The joint storage and the one place that fetches an instance's joint matrix;
# every skinned shader (mesh materials, the shadow pass) appends it.
_SKIN_WGSL = f"""
@group(0) @binding({JOINTS_BINDING}) var<storage, read> skin_joints: array<mat4x4<f32>>;
@group(0) @binding({BLOCKS_BINDING}) var<storage, read> skin_blocks: array<u32>;

// Bits 0-19 the block, 20-31 how far toward the next block (a baked clip's next frame).
fn skin_matrix(instance: u32, joint: u32) -> mat4x4<f32> {{
    let packed = skin_blocks[instance];
    let a = skin_joints[(packed & {FRAME_MASK}u) * {MAX_JOINTS}u + joint];
    let f = f32(packed >> 20u) / 4095.0;
    if (f == 0.0) {{
        return a;
    }}
    return a + f * (skin_joints[((packed & {FRAME_MASK}u) + 1u) * {MAX_JOINTS}u + joint] - a);
}}

// The packed joint attribute: joint a in bits 0-7, joint b in 8-15, a's weight in 16-31.
fn skin_blend(instance: u32, packed: u32) -> mat4x4<f32> {{
    let w = f32(packed >> 16u) / 65535.0;
    return w * skin_matrix(instance, packed & 255u) + (1.0 - w) * skin_matrix(instance, (packed >> 8u) & 255u);
}}
"""

_STRUCT = re.compile(r"struct VertexInput\s*\{(?P<body>.*?)\};", re.S)
_ENTRY = re.compile(r"fn vs_main\(in: VertexInput\)(?P<ret>[^{]*)\{")


def skin_shader(source: str) -> str:
    """Rewrite a mesh material's WGSL so vs_main sees joint-posed position and normal.

    Expects `struct VertexInput` with a `position` field and an instance_index
    builtin, and an entry point `fn vs_main(in: VertexInput)`.
    """
    struct = _STRUCT.search(source)
    entries = _ENTRY.findall(source)
    if struct is None or len(entries) != 1:
        raise ValueError(
            "skinning needs a WGSL `struct VertexInput { ... };` and one "
            "`fn vs_main(in: VertexInput)` entry point"
        )
    body = struct.group("body")
    instance = re.search(r"@builtin\(instance_index\)\s*(\w+)\s*:", body)
    if instance is None or not re.search(r"\bposition\s*:", body):
        raise ValueError("skinning needs VertexInput.position and an @builtin(instance_index) field")
    if f"@location({JOINT_LOCATION})" in body:
        raise ValueError(f"VertexInput already uses @location({JOINT_LOCATION}), reserved for joints")
    has_normal = re.search(r"\bnormal\s*:", body) is not None

    fields = body.rstrip()
    if not fields.endswith(","):
        fields += ","
    fields += f"\n    @location({JOINT_LOCATION}) joint: u32,\n"
    out = source[: struct.start("body")] + fields + source[struct.end("body"):]
    out = _ENTRY.sub(
        lambda m: f"fn vs_main(in_raw: VertexInput){m.group('ret')}{{\n    var in = skin_vertex(in_raw);",
        out,
        count=1,
    )
    normal = "    o.normal = normalize(skin_normal_matrix(m) * v.normal);\n" if has_normal else ""
    return out + _SKIN_WGSL + f"""
// Inverse transpose of m's 3x3 up to a positive scale: the cofactor matrix times
// sign(det). A singular 3x3 keeps the normal as it is, so nothing divides by zero.
// Singular is relative to the columns' lengths, so a small uniform scale is not.
fn skin_normal_matrix(m: mat4x4<f32>) -> mat3x3<f32> {{
    let a = m[0].xyz;
    let b = m[1].xyz;
    let c = m[2].xyz;
    let det = dot(a, cross(b, c));
    if abs(det) <= 1e-6 * length(a) * length(b) * length(c) {{
        return mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
    }}
    return mat3x3<f32>(cross(b, c), cross(c, a), cross(a, b)) * sign(det);
}}

fn skin_vertex(v: VertexInput) -> VertexInput {{
    var o = v;
    let m = skin_blend(v.{instance.group(1)}, v.joint);
    o.position = (m * vec4<f32>(v.position, 1.0)).xyz;
{normal}    return o;
}}
"""


_SHADOW_ENTRY = re.compile(r"fn vs_main\((?P<args>.*?)\)(?P<ret>\s*->[^{]*)\{", re.S)


def skin_shadow_shader(source: str) -> str:
    """Rewrite the depth-only shadow WGSL so vs_main sees the joint-posed position.

    Expects one `fn vs_main(@location(0) position: vec3<f32>, ...)` entry point
    with an instance_index builtin; adds the joint attribute and the bindings.
    """
    entries = list(_SHADOW_ENTRY.finditer(source))
    if len(entries) != 1:
        raise ValueError("skinning needs one `fn vs_main(...)` entry point in the shadow WGSL")
    args = entries[0].group("args")
    instance = re.search(r"@builtin\(instance_index\)\s*(\w+)\s*:", args)
    if instance is None or not re.search(r"@location\(0\)\s*position\s*:\s*vec3<f32>", args):
        raise ValueError("skinning needs @location(0) position: vec3<f32> and an instance_index builtin")
    args = re.sub(r"\bposition\s*:", "position_raw:", args, count=1)
    args = args.rstrip() + f",\n           @location({JOINT_LOCATION}) joint: u32"
    body = (
        f"fn vs_main({args}){entries[0].group('ret')}{{\n"
        f"    let position = (skin_blend({instance.group(1)}, joint) * vec4<f32>(position_raw, 1.0)).xyz;"
    )
    return source[: entries[0].start()] + body + source[entries[0].end():] + _SKIN_WGSL
