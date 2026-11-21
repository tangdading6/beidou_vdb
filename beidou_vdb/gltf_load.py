"""Minimal glTF 2.0 JSON reader.

Loads triangle positions into one world frame. Front faces stay the
exterior: a negative world-matrix determinant flips the geometric normal
so it still matches the glTF front face (counter-clockwise).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

_COMPONENT = {
    5120: (np.int8, 1),
    5121: (np.uint8, 1),
    5122: (np.int16, 2),
    5123: (np.uint16, 2),
    5125: (np.uint32, 4),
    5126: (np.float32, 4),
}
_NUM_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_TRIANGLES = 4


def load_gltf_triangles(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(triangles, normals)``.

    ``triangles`` has shape ``(T, 3, 3)`` in world meters.
    ``normals`` has shape ``(T, 3)`` and points out of the front face.
    """
    path = Path(path)
    raw = path.read_bytes()
    if raw[:4] == b"glTF":
        raise ValueError(f"{path} is a binary glTF; this reader expects JSON .gltf")
    gltf = json.loads(raw.decode("utf-8"))
    blobs = _buffer_bytes(gltf, path.parent)
    nodes = gltf.get("nodes", [])
    scenes = gltf.get("scenes", [])
    if not scenes:
        scene_nodes = list(range(len(nodes)))
    else:
        scene_index = gltf.get("scene", 0)
        scene_nodes = scenes[scene_index].get("nodes", [])

    triangles: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    identity = np.eye(4)

    def visit(index: int, parent: np.ndarray) -> None:
        node = nodes[index]
        world = parent @ _local_matrix(node)
        mesh_index = node.get("mesh")
        if mesh_index is not None:
            _append_mesh(gltf, blobs, mesh_index, world, triangles, normals)
        for child in node.get("children", []):
            visit(child, world)

    for root in scene_nodes:
        visit(root, identity)

    if not triangles:
        raise ValueError(f"{path} has no triangles")
    return np.stack(triangles, axis=0), np.stack(normals, axis=0)


def _buffer_bytes(gltf: dict, base_dir: Path) -> list[bytes]:
    blobs = []
    for buf in gltf.get("buffers", []):
        uri = buf.get("uri")
        if uri is None:
            raise ValueError("GLB-embedded buffers are not supported")
        if uri.startswith("data:"):
            payload = uri.split(",", 1)[1]
            data = base64.b64decode(payload)
        else:
            data = (base_dir / uri).read_bytes()
        expected = buf.get("byteLength")
        if expected is not None and len(data) < expected:
            raise ValueError("buffer shorter than byteLength")
        blobs.append(data)
    return blobs


def _append_mesh(gltf, blobs, mesh_index, world, triangles, normals) -> None:
    det = float(np.linalg.det(world[:3, :3]))
    flip = det < 0.0
    mesh = gltf["meshes"][mesh_index]
    for prim in mesh.get("primitives", []):
        mode = prim.get("mode", _TRIANGLES)
        if mode != _TRIANGLES:
            continue
        attrs = prim.get("attributes") or {}
        if "POSITION" not in attrs:
            continue
        positions = _read_accessor(gltf, blobs, attrs["POSITION"]).reshape(-1, 3)
        world_pos = _transform_points(positions, world)
        if "indices" in prim:
            indices = _read_accessor(gltf, blobs, prim["indices"]).astype(np.int64).reshape(-1)
        else:
            indices = np.arange(len(world_pos), dtype=np.int64)
        if len(indices) % 3 != 0:
            raise ValueError("triangle index count is not a multiple of 3")
        for a, b, c in indices.reshape(-1, 3):
            tri = np.stack([world_pos[a], world_pos[b], world_pos[c]], axis=0)
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            length = float(np.linalg.norm(normal))
            if length < 1e-12:
                continue
            normal = normal / length
            if flip:
                normal = -normal
            triangles.append(tri)
            normals.append(normal)


def _transform_points(points: np.ndarray, world: np.ndarray) -> np.ndarray:
    homo = np.concatenate([points.astype(np.float64), np.ones((len(points), 1))], axis=1)
    return (homo @ world.T)[:, :3]


def _read_accessor(gltf: dict, blobs: list[bytes], accessor_index: int) -> np.ndarray:
    accessor = gltf["accessors"][accessor_index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    dtype, comp_size = _COMPONENT[accessor["componentType"]]
    count = accessor["count"]
    ncomp = _NUM_COMPONENTS[accessor["type"]]
    buffer = blobs[view.get("buffer", 0)]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride", comp_size * ncomp)
    item = comp_size * ncomp
    if stride == item:
        raw = buffer[start : start + count * item]
        array = np.frombuffer(raw, dtype=dtype).copy()
        return array.reshape(count, ncomp) if ncomp > 1 else array
    rows = [
        np.frombuffer(buffer[start + i * stride : start + i * stride + item], dtype=dtype).copy()
        for i in range(count)
    ]
    array = np.stack(rows, axis=0)
    return array if ncomp > 1 else array.reshape(count)


def _local_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape((4, 4), order="F")
    translation = node.get("translation", [0.0, 0.0, 0.0])
    rotation = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    scale = node.get("scale", [1.0, 1.0, 1.0])
    matrix = np.eye(4)
    matrix[:3, :3] = _quat_matrix(rotation)[:3, :3] @ np.diag(np.asarray(scale, dtype=np.float64))
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def _quat_matrix(quat) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat)
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-20:
        return np.eye(4)
    s = 2.0 / norm
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    matrix = np.eye(4)
    matrix[0, 0] = 1.0 - (yy + zz)
    matrix[0, 1] = xy - wz
    matrix[0, 2] = xz + wy
    matrix[1, 0] = xy + wz
    matrix[1, 1] = 1.0 - (xx + zz)
    matrix[1, 2] = yz - wx
    matrix[2, 0] = xz - wy
    matrix[2, 1] = yz + wx
    matrix[2, 2] = 1.0 - (xx + yy)
    return matrix
