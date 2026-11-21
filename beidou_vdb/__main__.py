"""Run the two synthetic shells and report how the occupancy plan behaved."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from beidou_vdb.gltf_load import load_gltf_triangles
from beidou_vdb.occupy import fill_occupancy, tri_box_overlap
from beidou_vdb.synthetic import CASES, CASES_DIR, _pack_gltf, world_aabb

_FACE_NORMAL = {
    "-X": np.array((-1.0, 0.0, 0.0)),
    "+X": np.array((1.0, 0.0, 0.0)),
    "-Y": np.array((0.0, -1.0, 0.0)),
    "+Y": np.array((0.0, 1.0, 0.0)),
    "-Z": np.array((0.0, 0.0, -1.0)),
    "+Z": np.array((0.0, 0.0, 1.0)),
}
_ALL_FACES = ("-X", "+X", "-Y", "+Y", "-Z", "+Z")


def main() -> None:
    _self_check()
    print("体素边长 1 m。两个壳体都先乘上节点变换，再在同一坐标系里填充。")
    print("顶点包围盒的六面是辅助表面，和建筑三角面一起挡住射线。")
    for case in CASES:
        path = CASES_DIR / case["filename"]
        triangles, normals = load_gltf_triangles(path)
        result = fill_occupancy(triangles, normals, voxel_size=1.0)
        print()
        print(_report(case, path, triangles, normals, result))


def _report(case, path, triangles, normals, result) -> str:
    expected_min, expected_max = world_aabb()
    transform_ok = np.allclose(result.aabb_min, expected_min, atol=1e-5) and np.allclose(
        result.aabb_max, expected_max, atol=1e-5
    )
    center = (result.aabb_min + result.aabb_max) * 0.5
    outward = all(
        float(np.dot(normal, tri.mean(axis=0) - center)) > 0.0
        for tri, normal in zip(triangles, normals)
    )
    grid = {
        (i, j, k)
        for i in range(result.dims[0])
        for j in range(result.dims[1])
        for k in range(result.dims[2])
    }
    holes = sorted(grid - result.occupied)
    outside = sorted(v for v in result.occupied if v not in grid)
    leak_centers = 0
    for ijk in result.occupied:
        voxel_center = result.origin + (np.asarray(ijk) + 0.5) * result.voxel_size
        if np.any(voxel_center < result.aabb_min - 1e-6) or np.any(voxel_center > result.aabb_max + 1e-6):
            leak_centers += 1
    escapes = [m for m in result.marches if m["escaped"]]
    between = {tuple(v) for m in result.marches for v in m["between"]}
    aux_only = result.auxiliary_only()
    axis = case["axis"]
    side = case["side"]
    bound = 0 if side < 0 else result.dims[axis] - 1
    aux_off_face = [v for v in aux_only if v[axis] != bound]
    gap_hits = [
        m
        for m in result.marches
        if m["hit"] is not None
        and m["hit"] in aux_only
        and m["hit"][axis] == bound
        and float(m["direction"][axis]) * side > 0.5
    ]
    shell_ok = _shell_matches_case(result, case["drop"])
    interior_filled = not holes and bool(between) and shell_ok
    stopped = (
        not escapes
        and not aux_off_face
        and bool(aux_only)
        and bool(gap_hits)
        and all(m["hit"][axis] == bound for m in gap_hits)
    )
    leak_count = len(outside) + leak_centers + len(escapes)
    lines = [
        f"## {case['name']}（{path.name}，缺 {case['drop']}）",
        (
            f"三角形 {len(triangles)} 个。世界包围盒 "
            f"{_fmt(result.aabb_min)} – {_fmt(result.aabb_max)}，"
            f"格子 {result.dims[0]}×{result.dims[1]}×{result.dims[2]}，"
            f"占用 {len(result.occupied)}/{result.cell_count}。"
        ),
        f"节点变换后的包围盒与壳体世界坐标一致：{_yes(transform_ok)}。正面法线朝外：{_yes(outward)}。",
        f"建筑表面 {len(result.building_surface)}，仅辅助面 {len(aux_only)}，射线写入的内部体素 {len(between)}。",
        f"内部是否填上：{_yes(interior_filled)}。",
        (
            f"缺口是否停在包围盒：{_yes(stopped)}。"
            f"朝缺口打到辅助面的射线 {len(gap_hits)} 条，走出格子的射线 {len(escapes)} 条。"
        ),
        (
            f"壳外渗漏：{_leak_text(leak_count)}。"
            if leak_count == 0
            else (
                f"壳外渗漏：有 {leak_count} 处"
                f"（盒外体素 {len(outside)}，盒外中心 {leak_centers}，逃逸射线 {len(escapes)}）。"
            )
        ),
    ]
    if holes:
        lines.append(f"空体素示例：{holes[:8]}")
    if not shell_ok:
        lines.append("壳体表面与预期缺口不一致。")
    lines.append(_slice_text(result))
    return "\n".join(lines)


def _shell_matches_case(result, drop: str) -> bool:
    for name in _ALL_FACES:
        voxels = _face_interior(result.dims, name)
        if not voxels:
            return False
        if name == drop:
            if any(v in result.building_surface for v in voxels):
                return False
            if any(v not in result.auxiliary_surface for v in voxels):
                return False
            continue
        normal = _FACE_NORMAL[name]
        for voxel in voxels:
            if voxel not in result.building_surface:
                return False
            normals = result.building_normals[voxel]
            if not any(float(np.dot(n, normal)) > 0.99 for n in normals):
                return False
    return True


def _face_interior(dims, name: str):
    axis = {"X": 0, "Y": 1, "Z": 2}[name[1]]
    side = -1 if name[0] == "-" else 1
    bound = 0 if side < 0 else dims[axis] - 1
    found = []
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                ijk = (i, j, k)
                if ijk[axis] != bound:
                    continue
                if all(0 < ijk[a] < dims[a] - 1 for a in range(3) if a != axis):
                    found.append(ijk)
    return found


def _slice_text(result) -> str:
    """One x-row per z line, grouped by y. Solid rows collapse to a count."""
    lines = ["水平切片（x 从左到右，z 从上到下，█ 占用，· 空）："]
    for j in range(result.dims[1]):
        rows = []
        for k in range(result.dims[2]):
            rows.append(
                "".join("█" if (i, j, k) in result.occupied else "·" for i in range(result.dims[0]))
            )
        if all(set(row) == {"█"} for row in rows):
            lines.append(f"  y={j}: 整层实心 {result.dims[0]}×{result.dims[2]}")
        else:
            lines.append(f"  y={j}:")
            lines.extend(f"    {row}" for row in rows)
    return "\n".join(lines)


def _fmt(point) -> str:
    return "(" + ", ".join(f"{float(v):.0f}" for v in point) + ")"


def _yes(flag: bool) -> str:
    return "是" if flag else "否"


def _leak_text(count: int) -> str:
    return "无" if count == 0 else f"有 {count} 处"


def _self_check() -> None:
    box_min = np.zeros(3)
    box_max = np.ones(3)
    on_face = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert tri_box_overlap(on_face, box_min, box_max)
    above = on_face + np.array([0.0, 0.0, 1.0])
    assert tri_box_overlap(above, box_min, box_max)
    outside = on_face + np.array([0.0, 0.0, 2.0])
    assert not tri_box_overlap(outside, box_min, box_max)
    cutting = np.array([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [0.0, 1.0, 0.5]])
    assert tri_box_overlap(cutting, box_min, box_max)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _expect_normal(root / "yaw.gltf", _yaw90_gltf(), np.array([1.0, 0.0, 0.0]))
        _expect_normal(root / "mirror.gltf", _mirror_gltf(), np.array([0.0, 0.0, 1.0]))


def _expect_normal(path: Path, gltf: dict, expected: np.ndarray) -> None:
    import json

    path.write_text(json.dumps(gltf), encoding="utf-8")
    triangles, normals = load_gltf_triangles(path)
    assert len(triangles) == 1
    assert np.allclose(normals[0], expected, atol=1e-6), normals[0]


def _unit_triangle_mesh():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    indices = np.array([0, 1, 2], dtype=np.uint16)
    return positions, indices


def _yaw90_gltf() -> dict:
    # +90 degrees about Y. The +Z front normal becomes +X.
    half = float(np.sin(np.pi / 4.0))
    return _pack_gltf(
        [_unit_triangle_mesh()],
        [{"mesh": 0, "rotation": [0.0, half, 0.0, half]}],
        {},
    )


def _mirror_gltf() -> dict:
    return _pack_gltf(
        [_unit_triangle_mesh()],
        [{"mesh": 0, "scale": [-1.0, 1.0, 1.0]}],
        {},
    )


if __name__ == "__main__":
    main()
