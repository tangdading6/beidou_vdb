"""Write the two synthetic building shells used by the occupancy check.

Both shells are an axis-aligned box. One omits the floor, the other omits the
+X wall. Walls, roof, and floor are separate meshes. A parent node translates
the building, and the roof node uses a matrix so the loader has to flatten
every mesh into one frame before filling.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

WORLD_OFFSET = (10.0, 0.0, 20.0)
LOCAL_SIZE = (4.0, 3.0, 5.0)
CASES_DIR = Path(__file__).resolve().parent.parent / "cases"

CASES = (
    {
        "name": "缺底面",
        "filename": "shell_missing_floor.gltf",
        "drop": "-Y",
        "axis": 1,
        "side": -1,
    },
    {
        "name": "缺一面侧墙",
        "filename": "shell_missing_wall.gltf",
        "drop": "+X",
        "axis": 0,
        "side": 1,
    },
)

_FACE_NORMAL = {
    "-X": np.array((-1.0, 0.0, 0.0)),
    "+X": np.array((1.0, 0.0, 0.0)),
    "-Y": np.array((0.0, -1.0, 0.0)),
    "+Y": np.array((0.0, 1.0, 0.0)),
    "-Z": np.array((0.0, 0.0, -1.0)),
    "+Z": np.array((0.0, 0.0, 1.0)),
}


def world_aabb() -> tuple[np.ndarray, np.ndarray]:
    offset = np.asarray(WORLD_OFFSET, dtype=np.float64)
    size = np.asarray(LOCAL_SIZE, dtype=np.float64)
    return offset, offset + size


def write_cases(directory: Path | None = None) -> list[Path]:
    directory = CASES_DIR if directory is None else Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for case in CASES:
        path = directory / case["filename"]
        path.write_text(json.dumps(_case_gltf(case), indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written


def _case_gltf(case: dict) -> dict:
    vertical = [name for name in ("-X", "+X", "-Z", "+Z") if name != case["drop"]]
    meshes = [_mesh_from_quads([_wall_quad(name) for name in vertical])]
    nodes = [
        {
            "name": "building",
            "translation": list(WORLD_OFFSET),
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "scale": [1.0, 1.0, 1.0],
            "children": [1, 2],
        },
        {"name": "walls", "mesh": 0},
        {"name": "roof", "mesh": 1, "matrix": _translation_matrix(0.0, LOCAL_SIZE[1], 0.0)},
    ]
    meshes.append(_mesh_from_quads([_roof_quad()]))
    if case["drop"] != "-Y":
        meshes.append(_mesh_from_quads([_wall_quad("-Y")]))
        nodes[0]["children"].append(3)
        nodes.append({"name": "floor", "mesh": 2})
    aabb_min, aabb_max = world_aabb()
    return _pack_gltf(
        meshes,
        nodes,
        {
            "case": case["filename"],
            "missing_face": case["drop"],
            "world_aabb_min": aabb_min.tolist(),
            "world_aabb_max": aabb_max.tolist(),
        },
    )


def _wall_quad(name: str):
    x1, y1, z1 = LOCAL_SIZE
    quads = {
        "-X": [(0.0, 0.0, 0.0), (0.0, 0.0, z1), (0.0, y1, z1), (0.0, y1, 0.0)],
        "+X": [(x1, 0.0, 0.0), (x1, y1, 0.0), (x1, y1, z1), (x1, 0.0, z1)],
        "-Y": [(0.0, 0.0, 0.0), (x1, 0.0, 0.0), (x1, 0.0, z1), (0.0, 0.0, z1)],
        "+Y": [(0.0, y1, 0.0), (0.0, y1, z1), (x1, y1, z1), (x1, y1, 0.0)],
        "-Z": [(0.0, 0.0, 0.0), (0.0, y1, 0.0), (x1, y1, 0.0), (x1, 0.0, 0.0)],
        "+Z": [(0.0, 0.0, z1), (x1, 0.0, z1), (x1, y1, z1), (0.0, y1, z1)],
    }
    return quads[name], _FACE_NORMAL[name]


def _roof_quad():
    """Roof rectangle in its own node, lying on local y = 0. The node matrix lifts it."""
    x1, _, z1 = LOCAL_SIZE
    corners = [(0.0, 0.0, 0.0), (0.0, 0.0, z1), (x1, 0.0, z1), (x1, 0.0, 0.0)]
    return corners, _FACE_NORMAL["+Y"]


def _mesh_from_quads(quads):
    positions = []
    indices = []
    for corners, outward in quads:
        a, b, c, d = (np.asarray(p, dtype=np.float64) for p in corners)
        geometric = np.cross(b - a, c - a)
        if float(np.dot(geometric, outward)) <= 0.0:
            raise RuntimeError(f"quad winding is not outward: {outward}")
        base = len(positions)
        positions.extend((a, b, c, d))
        indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))
    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(indices, dtype=np.uint16),
    )


def _translation_matrix(x: float, y: float, z: float) -> list[float]:
    # glTF matrices are column-major.
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        x, y, z, 1.0,
    ]


def _pack_gltf(meshes, nodes, extras) -> dict:
    blob = bytearray()
    buffer_views = []
    accessors = []
    mesh_json = []

    def add_bytes(data: bytes) -> int:
        while len(blob) % 4:
            blob.append(0)
        view = {
            "buffer": 0,
            "byteOffset": len(blob),
            "byteLength": len(data),
        }
        blob.extend(data)
        buffer_views.append(view)
        return len(buffer_views) - 1

    for positions, indices in meshes:
        pos = np.asarray(positions, dtype=np.float32)
        idx = np.asarray(indices, dtype=np.uint16)
        pos_view = add_bytes(pos.tobytes())
        idx_view = add_bytes(idx.tobytes())
        accessors.append(
            {
                "bufferView": pos_view,
                "componentType": 5126,
                "count": int(pos.shape[0]),
                "type": "VEC3",
                "min": pos.min(axis=0).astype(float).tolist(),
                "max": pos.max(axis=0).astype(float).tolist(),
            }
        )
        pos_index = len(accessors) - 1
        accessors.append(
            {
                "bufferView": idx_view,
                "componentType": 5123,
                "count": int(idx.shape[0]),
                "type": "SCALAR",
            }
        )
        idx_index = len(accessors) - 1
        mesh_json.append(
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": pos_index},
                        "indices": idx_index,
                        "mode": 4,
                    }
                ]
            }
        )

    encoded = base64.b64encode(blob).decode("ascii")
    return {
        "asset": {"version": "2.0", "generator": "beidou_vdb synthetic shells"},
        "scene": 0,
        "scenes": [{"nodes": [0], "extras": extras}],
        "nodes": nodes,
        "meshes": mesh_json,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [
            {
                "byteLength": len(blob),
                "uri": f"data:application/octet-stream;base64,{encoded}",
            }
        ],
    }


def main() -> None:
    paths = write_cases()
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
