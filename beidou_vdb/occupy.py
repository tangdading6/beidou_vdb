"""Fill a building mesh into 1 m occupancy voxels.

Rules, followed as written:

1. Triangles that sweep a voxel mark it as a surface and store the outward
   normal. The six faces of the vertex AABB are auxiliary surfaces and block
   the same way.
2. From each building surface voxel, step one cell at a time opposite the
   outward normal.
3. Stop at the next surface (building or auxiliary). Voxels strictly between
   the start and the hit are occupied. Surface voxels themselves are occupied.
4. Rays start only from building surfaces. The box is not painted solid; it
   only supplies the missing face.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

VOXEL_SIZE_M = 1.0


@dataclass
class FillResult:
    origin: np.ndarray
    voxel_size: float
    dims: tuple[int, int, int]
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    occupied: set[tuple[int, int, int]]
    building_surface: set[tuple[int, int, int]]
    auxiliary_surface: set[tuple[int, int, int]]
    building_normals: dict[tuple[int, int, int], list[np.ndarray]]
    marches: list[dict] = field(default_factory=list)

    @property
    def cell_count(self) -> int:
        dx, dy, dz = self.dims
        return int(dx * dy * dz)

    def auxiliary_only(self) -> set[tuple[int, int, int]]:
        return self.auxiliary_surface - self.building_surface


def fill_occupancy(
    triangles: np.ndarray,
    normals: np.ndarray,
    voxel_size: float = VOXEL_SIZE_M,
) -> FillResult:
    """Voxel-fill ``triangles`` (T, 3, 3) using front-face ``normals`` (T, 3)."""
    if len(triangles) == 0:
        raise ValueError("no triangles")
    points = triangles.reshape(-1, 3)
    aabb_min = points.min(axis=0)
    aabb_max = points.max(axis=0)
    extent = aabb_max - aabb_min
    dims_arr = np.maximum(1, np.ceil(extent / voxel_size - 1e-9).astype(int))
    dims = (int(dims_arr[0]), int(dims_arr[1]), int(dims_arr[2]))
    origin = aabb_min.astype(np.float64)

    building_normals: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for tri, normal in zip(triangles, normals):
        _rasterize(tri, normal, origin, voxel_size, dims, building_normals)

    auxiliary_normals: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for tri, normal in _auxiliary_faces(aabb_min, aabb_max):
        _rasterize(tri, normal, origin, voxel_size, dims, auxiliary_normals)

    building_surface = set(building_normals)
    auxiliary_surface = set(auxiliary_normals)
    surface = building_surface | auxiliary_surface
    occupied = set(surface)
    marches: list[dict] = []

    for ijk, normal_list in building_normals.items():
        for normal in normal_list:
            march = _march(ijk, -normal, dims, surface)
            marches.append(march)
            occupied.update(march["between"])

    return FillResult(
        origin=origin,
        voxel_size=float(voxel_size),
        dims=dims,
        aabb_min=aabb_min.astype(np.float64),
        aabb_max=aabb_max.astype(np.float64),
        occupied=occupied,
        building_surface=building_surface,
        auxiliary_surface=auxiliary_surface,
        building_normals=building_normals,
        marches=marches,
    )


def _rasterize(tri, normal, origin, voxel_size, dims, bucket) -> None:
    i0, i1 = _index_range(tri, origin, voxel_size, dims)
    for i in range(int(i0[0]), int(i1[0]) + 1):
        for j in range(int(i0[1]), int(i1[1]) + 1):
            for k in range(int(i0[2]), int(i1[2]) + 1):
                box_min, box_max = _voxel_box((i, j, k), origin, voxel_size)
                if tri_box_overlap(tri, box_min, box_max):
                    _add_normal(bucket, (i, j, k), normal)


def _index_range(tri, origin, voxel_size, dims):
    """Voxel indices whose closed cells can touch ``tri``.

    A triangle that lies exactly on a grid plane touches the cells on both
    sides. The AABB max face is clamped back into the last cell, so the grid
    does not grow past the vertex box.
    """
    tmin = np.min(tri, axis=0)
    tmax = np.max(tri, axis=0)
    i0 = np.floor((tmin - origin) / voxel_size - 1e-9).astype(int)
    i1 = np.floor((tmax - origin) / voxel_size).astype(int)
    limit = np.asarray(dims) - 1
    return np.clip(i0, 0, limit), np.clip(i1, 0, limit)


def _voxel_box(ijk, origin, voxel_size):
    ijk_a = np.asarray(ijk, dtype=np.float64)
    box_min = origin + ijk_a * voxel_size
    box_max = box_min + voxel_size
    return box_min, box_max


def _add_normal(bucket, ijk, normal) -> None:
    normal = np.asarray(normal, dtype=np.float64)
    length = float(np.linalg.norm(normal))
    if length < 1e-12:
        return
    normal = normal / length
    existing = bucket.setdefault(ijk, [])
    for prior in existing:
        if float(np.dot(prior, normal)) > 1.0 - 1e-6:
            return
    existing.append(normal)


def _auxiliary_faces(aabb_min, aabb_max):
    x0, y0, z0 = (float(v) for v in aabb_min)
    x1, y1, z1 = (float(v) for v in aabb_max)
    quads = [
        ([(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)], (-1.0, 0.0, 0.0)),
        ([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], (1.0, 0.0, 0.0)),
        ([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], (0.0, -1.0, 0.0)),
        ([(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)], (0.0, 1.0, 0.0)),
        ([(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)], (0.0, 0.0, -1.0)),
        ([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], (0.0, 0.0, 1.0)),
    ]
    faces = []
    for corners, outward in quads:
        a, b, c, d = (np.asarray(p, dtype=np.float64) for p in corners)
        normal = np.asarray(outward, dtype=np.float64)
        faces.append((np.stack([a, b, c]), normal))
        faces.append((np.stack([a, c, d]), normal))
    return faces


def _march(start, direction, dims, surface) -> dict:
    direction = np.asarray(direction, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    direction = direction / length
    pos = [int(start[0]), int(start[1]), int(start[2])]
    step = [0, 0, 0]
    t_max = [np.inf, np.inf, np.inf]
    t_delta = [np.inf, np.inf, np.inf]
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            continue
        if direction[axis] > 0.0:
            step[axis] = 1
            t_max[axis] = (0.5) / direction[axis]
        else:
            step[axis] = -1
            t_max[axis] = (0.5) / -direction[axis]
        t_delta[axis] = 1.0 / abs(direction[axis])

    between: list[tuple[int, int, int]] = []
    limit = int(dims[0] + dims[1] + dims[2] + 3)
    for _ in range(limit):
        axis = min(range(3), key=lambda a: t_max[a])
        if step[axis] == 0 or not np.isfinite(t_max[axis]):
            break
        pos[axis] += step[axis]
        t_max[axis] += t_delta[axis]
        ijk = (pos[0], pos[1], pos[2])
        if not _inside(ijk, dims):
            return {
                "start": tuple(int(v) for v in start),
                "direction": direction.copy(),
                "between": between,
                "hit": None,
                "escaped": True,
            }
        if ijk in surface:
            return {
                "start": tuple(int(v) for v in start),
                "direction": direction.copy(),
                "between": between,
                "hit": ijk,
                "escaped": False,
            }
        between.append(ijk)
    return {
        "start": tuple(int(v) for v in start),
        "direction": direction.copy(),
        "between": between,
        "hit": None,
        "escaped": True,
    }


def _inside(ijk, dims) -> bool:
    return 0 <= ijk[0] < dims[0] and 0 <= ijk[1] < dims[1] and 0 <= ijk[2] < dims[2]


def tri_box_overlap(tri: np.ndarray, box_min: np.ndarray, box_max: np.ndarray, eps: float = 1e-8) -> bool:
    """True when a triangle intersects a closed axis-aligned box.

    Port of Akenine-Möller's separating-axis test. Comparisons are loosened by
    ``eps`` so a triangle that lies on a face still counts as sweeping that cell.
    """
    center = (box_min + box_max) * 0.5
    half = (box_max - box_min) * 0.5
    v0 = tri[0] - center
    v1 = tri[1] - center
    v2 = tri[2] - center
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2

    if _separated_by_edges(v0, v1, v2, e0, e1, e2, half, eps):
        return False
    if _separated_by_aabb(v0, v1, v2, half, eps):
        return False
    normal = np.cross(e0, e1)
    return _plane_overlaps_box(normal, v0, half, eps)


def _separated_by_aabb(v0, v1, v2, half, eps) -> bool:
    for axis in range(3):
        vmin = min(v0[axis], v1[axis], v2[axis])
        vmax = max(v0[axis], v1[axis], v2[axis])
        if vmin > half[axis] + eps or vmax < -half[axis] - eps:
            return True
    return False


def _separated_by_edges(v0, v1, v2, e0, e1, e2, half, eps) -> bool:
    # Pairing matches Akenine-Möller tribox3:
    # e0 uses X01, Y02, Z12; e1 uses X01, Y02, Z0; e2 uses X2, Y1, Z12.
    tests = (
        (e0, (_axis_x01, _axis_y02, _axis_z12)),
        (e1, (_axis_x01, _axis_y02, _axis_z0)),
        (e2, (_axis_x2, _axis_y1, _axis_z12)),
    )
    for edge, axes in tests:
        for test in axes:
            if test(edge, v0, v1, v2, half, eps):
                return True
    return False


def _project_separated(p_a, p_b, half_radius, eps) -> bool:
    lo = p_a if p_a < p_b else p_b
    hi = p_b if p_a < p_b else p_a
    return lo > half_radius + eps or hi < -half_radius - eps


def _axis_x01(edge, v0, v1, v2, half, eps) -> bool:
    a, b = edge[2], edge[1]
    fa, fb = abs(a), abs(b)
    p0 = a * v0[1] - b * v0[2]
    p2 = a * v2[1] - b * v2[2]
    rad = fa * half[1] + fb * half[2]
    return _project_separated(p0, p2, rad, eps)


def _axis_x2(edge, v0, v1, v2, half, eps) -> bool:
    a, b = edge[2], edge[1]
    fa, fb = abs(a), abs(b)
    p0 = a * v0[1] - b * v0[2]
    p1 = a * v1[1] - b * v1[2]
    rad = fa * half[1] + fb * half[2]
    return _project_separated(p0, p1, rad, eps)


def _axis_y02(edge, v0, v1, v2, half, eps) -> bool:
    a, b = edge[2], edge[0]
    fa, fb = abs(a), abs(b)
    p0 = -a * v0[0] + b * v0[2]
    p2 = -a * v2[0] + b * v2[2]
    rad = fa * half[0] + fb * half[2]
    return _project_separated(p0, p2, rad, eps)


def _axis_y1(edge, v0, v1, v2, half, eps) -> bool:
    a, b = edge[2], edge[0]
    fa, fb = abs(a), abs(b)
    p0 = -a * v0[0] + b * v0[2]
    p1 = -a * v1[0] + b * v1[2]
    rad = fa * half[0] + fb * half[2]
    return _project_separated(p0, p1, rad, eps)


def _axis_z12(edge, v0, v1, v2, half, eps) -> bool:
    a, b = edge[1], edge[0]
    fa, fb = abs(a), abs(b)
    p1 = a * v1[0] - b * v1[1]
    p2 = a * v2[0] - b * v2[1]
    rad = fa * half[0] + fb * half[1]
    return _project_separated(p1, p2, rad, eps)


def _axis_z0(edge, v0, v1, v2, half, eps) -> bool:
    a, b = edge[1], edge[0]
    fa, fb = abs(a), abs(b)
    p0 = a * v0[0] - b * v0[1]
    p1 = a * v1[0] - b * v1[1]
    rad = fa * half[0] + fb * half[1]
    return _project_separated(p0, p1, rad, eps)


def _plane_overlaps_box(normal, vert, half, eps) -> bool:
    vmin = np.empty(3, dtype=np.float64)
    vmax = np.empty(3, dtype=np.float64)
    for axis in range(3):
        v = vert[axis]
        if normal[axis] > 0.0:
            vmin[axis] = -half[axis] - v
            vmax[axis] = half[axis] - v
        else:
            vmin[axis] = half[axis] - v
            vmax[axis] = -half[axis] - v
    if float(np.dot(normal, vmin)) > eps:
        return False
    if float(np.dot(normal, vmax)) >= -eps:
        return True
    return False
