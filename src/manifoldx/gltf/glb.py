"""Binary glTF (GLB): a JSON document plus one binary buffer, both 4-byte aligned."""

import json
import struct

import numpy as np

FLOAT, UINT8, UINT16, UINT32 = 5126, 5121, 5123, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
_COMPONENT = {np.dtype(np.float32): FLOAT, np.dtype(np.uint8): UINT8, np.dtype(np.uint16): UINT16,
              np.dtype(np.uint32): UINT32}
_TYPE = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4", 16: "MAT4"}


class GlbBuilder:
    """Accumulates glTF objects and binary data; to_bytes() returns the .glb."""

    def __init__(self):
        self.doc = {"asset": {"version": "2.0", "generator": "manifoldx"}}
        self.bin = bytearray()

    def add(self, key, item):
        """Append item to the document's top-level array `key`; return its index."""
        items = self.doc.setdefault(key, [])
        items.append(item)
        return len(items) - 1

    def view(self, data, target=None):
        self.bin.extend(b"\0" * (-len(self.bin) % 4))
        view = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.bin.extend(data)
        return self.add("bufferViews", view)

    def accessor(self, array, *, target=None, bounds=False):
        a = np.ascontiguousarray(array)
        width = 1 if a.ndim == 1 else a.shape[1]
        acc = {"bufferView": self.view(a.tobytes(), target), "componentType": _COMPONENT[a.dtype],
               "count": int(a.shape[0]), "type": _TYPE[width]}
        if bounds:
            flat = a.reshape(len(a), width)
            acc["min"] = flat.min(axis=0).tolist()
            acc["max"] = flat.max(axis=0).tolist()
        return self.add("accessors", acc)

    def use_extension(self, name):
        used = self.doc.setdefault("extensionsUsed", [])
        if name not in used:
            used.append(name)

    def to_bytes(self):
        doc = dict(self.doc)
        if self.bin:
            doc["buffers"] = [{"byteLength": len(self.bin)}]
        text = json.dumps(doc, separators=(",", ":"), allow_nan=False).encode()
        text += b" " * (-len(text) % 4)
        chunks = struct.pack("<II", len(text), 0x4E4F534A) + text
        if self.bin:
            binary = bytes(self.bin) + b"\0" * (-len(self.bin) % 4)
            chunks += struct.pack("<II", len(binary), 0x004E4942) + binary
        return struct.pack("<III", 0x46546C67, 2, 12 + len(chunks)) + chunks
