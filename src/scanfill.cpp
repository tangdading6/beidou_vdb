#include "scanfill.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace beidou {
namespace {

constexpr double kEps = 1e-8;
constexpr std::uint8_t kAux = 1;
constexpr std::uint8_t kBuilding = 2;

enum class Kind { Skip, Front, Back, Aux };

double dot3(const double a[3], const double b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

bool project_separated(double p_a, double p_b, double rad) {
    const double lo = p_a < p_b ? p_a : p_b;
    const double hi = p_a < p_b ? p_b : p_a;
    return lo > rad + kEps || hi < -rad - kEps;
}

bool axis_x01(const double e[3], const double v0[3], const double*, const double v2[3], const double h[3]) {
    const double a = e[2], b = e[1];
    return project_separated(a * v0[1] - b * v0[2], a * v2[1] - b * v2[2],
                             std::fabs(a) * h[1] + std::fabs(b) * h[2]);
}

bool axis_x2(const double e[3], const double v0[3], const double v1[3], const double*, const double h[3]) {
    const double a = e[2], b = e[1];
    return project_separated(a * v0[1] - b * v0[2], a * v1[1] - b * v1[2],
                             std::fabs(a) * h[1] + std::fabs(b) * h[2]);
}

bool axis_y02(const double e[3], const double v0[3], const double*, const double v2[3], const double h[3]) {
    const double a = e[2], b = e[0];
    return project_separated(-a * v0[0] + b * v0[2], -a * v2[0] + b * v2[2],
                             std::fabs(a) * h[0] + std::fabs(b) * h[2]);
}

bool axis_y1(const double e[3], const double v0[3], const double v1[3], const double*, const double h[3]) {
    const double a = e[2], b = e[0];
    return project_separated(-a * v0[0] + b * v0[2], -a * v1[0] + b * v1[2],
                             std::fabs(a) * h[0] + std::fabs(b) * h[2]);
}

bool axis_z12(const double e[3], const double*, const double v1[3], const double v2[3], const double h[3]) {
    const double a = e[1], b = e[0];
    return project_separated(a * v1[0] - b * v1[1], a * v2[0] - b * v2[1],
                             std::fabs(a) * h[0] + std::fabs(b) * h[1]);
}

bool axis_z0(const double e[3], const double v0[3], const double v1[3], const double*, const double h[3]) {
    const double a = e[1], b = e[0];
    return project_separated(a * v0[0] - b * v0[1], a * v1[0] - b * v1[1],
                             std::fabs(a) * h[0] + std::fabs(b) * h[1]);
}

bool separated_by_aabb(const double v0[3], const double v1[3], const double v2[3], const double h[3]) {
    for (int axis = 0; axis < 3; ++axis) {
        const double vmin = std::min(v0[axis], std::min(v1[axis], v2[axis]));
        const double vmax = std::max(v0[axis], std::max(v1[axis], v2[axis]));
        if (vmin > h[axis] + kEps || vmax < -h[axis] - kEps) return true;
    }
    return false;
}

bool separated_by_edges(const double v0[3], const double v1[3], const double v2[3], const double e0[3],
                        const double e1[3], const double e2[3], const double h[3]) {
    if (axis_x01(e0, v0, v1, v2, h) || axis_y02(e0, v0, v1, v2, h) || axis_z12(e0, v0, v1, v2, h)) return true;
    if (axis_x01(e1, v0, v1, v2, h) || axis_y02(e1, v0, v1, v2, h) || axis_z0(e1, v0, v1, v2, h)) return true;
    if (axis_x2(e2, v0, v1, v2, h) || axis_y1(e2, v0, v1, v2, h) || axis_z12(e2, v0, v1, v2, h)) return true;
    return false;
}

bool plane_overlaps(const double e0[3], const double e1[3], const double v0[3], const double h[3]) {
    const double n[3] = {e0[1] * e1[2] - e0[2] * e1[1], e0[2] * e1[0] - e0[0] * e1[2],
                         e0[0] * e1[1] - e0[1] * e1[0]};
    double vmin[3], vmax[3];
    for (int axis = 0; axis < 3; ++axis) {
        if (n[axis] > 0.0) {
            vmin[axis] = -h[axis] - v0[axis];
            vmax[axis] = h[axis] - v0[axis];
        } else {
            vmin[axis] = h[axis] - v0[axis];
            vmax[axis] = -h[axis] - v0[axis];
        }
    }
    if (dot3(n, vmin) > kEps) return false;
    if (dot3(n, vmax) >= -kEps) return true;
    return false;
}

int floor_index(double value, double origin, double voxel, bool loosen) {
    double q = (value - origin) / voxel;
    if (loosen) q -= 1e-9;
    return static_cast<int>(std::floor(q));
}

int clip_index(int index, int limit) { return std::max(0, std::min(limit, index)); }

void decode(const int dims[3], int slot, int& i, int& j, int& k) {
    const int plane = dims[1] * dims[2];
    i = slot / plane;
    const int rem = slot - i * plane;
    j = rem / dims[2];
    k = rem - j * dims[2];
}

void line_cell(int axis, int key_a, int key_b, int pos, int out[3]) {
    if (axis == 0) {
        out[0] = pos;
        out[1] = key_a;
        out[2] = key_b;
    } else if (axis == 1) {
        out[0] = key_a;
        out[1] = pos;
        out[2] = key_b;
    } else {
        out[0] = key_a;
        out[1] = key_b;
        out[2] = pos;
    }
}

struct Grid {
    int dims[3]{};
    double origin[3]{};
    double voxel = 1.0;
    std::vector<std::uint8_t> kind;
    std::vector<std::uint8_t> occupied;
    std::unordered_map<int, std::vector<std::array<double, 3>>> normals;
    std::vector<int> surface;

    int index(int i, int j, int k) const { return (i * dims[1] + j) * dims[2] + k; }
};

void add_building(Grid& grid, int slot, const double normal[3]) {
    const double len = std::sqrt(dot3(normal, normal));
    if (len < 1e-12) return;
    std::array<double, 3> unit{normal[0] / len, normal[1] / len, normal[2] / len};
    auto& list = grid.normals[slot];
    for (const auto& prior : list) {
        if (prior[0] * unit[0] + prior[1] * unit[1] + prior[2] * unit[2] > 1.0 - 1e-6) return;
    }
    list.push_back(unit);
    if (grid.kind[slot] == 0) grid.surface.push_back(slot);
    grid.kind[slot] = static_cast<std::uint8_t>(grid.kind[slot] | kBuilding);
    grid.occupied[slot] = 1;
}

void add_aux(Grid& grid, int slot) {
    if (grid.kind[slot] == 0) grid.surface.push_back(slot);
    grid.kind[slot] = static_cast<std::uint8_t>(grid.kind[slot] | kAux);
}

void raster(Grid& grid, const double tri[3][3], const double normal[3], bool building) {
    double tmin[3] = {tri[0][0], tri[0][1], tri[0][2]};
    double tmax[3] = {tmin[0], tmin[1], tmin[2]};
    for (int v = 1; v < 3; ++v) {
        for (int axis = 0; axis < 3; ++axis) {
            tmin[axis] = std::min(tmin[axis], tri[v][axis]);
            tmax[axis] = std::max(tmax[axis], tri[v][axis]);
        }
    }
    int lo[3], hi[3];
    for (int axis = 0; axis < 3; ++axis) {
        lo[axis] = clip_index(floor_index(tmin[axis], grid.origin[axis], grid.voxel, true), grid.dims[axis] - 1);
        hi[axis] = clip_index(floor_index(tmax[axis], grid.origin[axis], grid.voxel, false), grid.dims[axis] - 1);
    }
    for (int i = lo[0]; i <= hi[0]; ++i) {
        for (int j = lo[1]; j <= hi[1]; ++j) {
            for (int k = lo[2]; k <= hi[2]; ++k) {
                const double box_min[3] = {grid.origin[0] + i * grid.voxel, grid.origin[1] + j * grid.voxel,
                                           grid.origin[2] + k * grid.voxel};
                const double box_max[3] = {box_min[0] + grid.voxel, box_min[1] + grid.voxel, box_min[2] + grid.voxel};
                if (!tri_box_overlap(tri, box_min, box_max)) continue;
                const int slot = grid.index(i, j, k);
                if (building) add_building(grid, slot, normal);
                else add_aux(grid, slot);
            }
        }
    }
}

void add_aux_box(Grid& grid, const double a[3], const double b[3]) {
    const double quads[6][4][3] = {
        {{a[0], a[1], a[2]}, {a[0], a[1], b[2]}, {a[0], b[1], b[2]}, {a[0], b[1], a[2]}},
        {{b[0], a[1], a[2]}, {b[0], b[1], a[2]}, {b[0], b[1], b[2]}, {b[0], a[1], b[2]}},
        {{a[0], a[1], a[2]}, {b[0], a[1], a[2]}, {b[0], a[1], b[2]}, {a[0], a[1], b[2]}},
        {{a[0], b[1], a[2]}, {a[0], b[1], b[2]}, {b[0], b[1], b[2]}, {b[0], b[1], a[2]}},
        {{a[0], a[1], a[2]}, {a[0], b[1], a[2]}, {b[0], b[1], a[2]}, {b[0], a[1], a[2]}},
        {{a[0], a[1], b[2]}, {b[0], a[1], b[2]}, {b[0], b[1], b[2]}, {a[0], b[1], b[2]}},
    };
    const double outward[6][3] = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
    for (int face = 0; face < 6; ++face) {
        const double t0[3][3] = {{quads[face][0][0], quads[face][0][1], quads[face][0][2]},
                                 {quads[face][1][0], quads[face][1][1], quads[face][1][2]},
                                 {quads[face][2][0], quads[face][2][1], quads[face][2][2]}};
        const double t1[3][3] = {{quads[face][0][0], quads[face][0][1], quads[face][0][2]},
                                 {quads[face][2][0], quads[face][2][1], quads[face][2][2]},
                                 {quads[face][3][0], quads[face][3][1], quads[face][3][2]}};
        raster(grid, t0, outward[face], false);
        raster(grid, t1, outward[face], false);
    }
}

Kind surface_kind(const Grid& grid, int slot, const double direction[3]) {
    const auto found = grid.normals.find(slot);
    if (found != grid.normals.end() && !found->second.empty()) {
        bool faces = false;
        double best = -1e300;
        for (const auto& normal : found->second) {
            const double value = direction[0] * normal[0] + direction[1] * normal[1] + direction[2] * normal[2];
            if (value < -1e-6) faces = true;
            if (value > best) best = value;
        }
        const bool clear_back = best > 0.5;
        if (faces) return Kind::Front;
        if (clear_back) return Kind::Back;
        return Kind::Skip;
    }
    if ((grid.kind[slot] & kAux) != 0 && (grid.kind[slot] & kBuilding) == 0) return Kind::Aux;
    return Kind::Skip;
}

void scan_direction(Grid& grid, const std::vector<int>& ordered, int axis, int sign, int key_a, int key_b,
                    bool record_marches, std::vector<March>& marches) {
    double direction[3] = {0, 0, 0};
    direction[axis] = static_cast<double>(sign);
    std::vector<int> positions = ordered;
    if (sign < 0) std::reverse(positions.begin(), positions.end());
    bool seeking = false;
    int outer = 0;
    for (int pos : positions) {
        int cell[3];
        line_cell(axis, key_a, key_b, pos, cell);
        const int slot = grid.index(cell[0], cell[1], cell[2]);
        const Kind kind = surface_kind(grid, slot, direction);
        if (!seeking) {
            if (kind == Kind::Front) {
                seeking = true;
                outer = pos;
            }
            continue;
        }
        if (kind == Kind::Back || kind == Kind::Aux) {
            const int step = pos > outer ? 1 : -1;
            March march;
            if (record_marches) {
                march.start[0] = march.start[1] = march.start[2] = 0;
                line_cell(axis, key_a, key_b, outer, march.start);
                march.hit[0] = cell[0];
                march.hit[1] = cell[1];
                march.hit[2] = cell[2];
                march.direction[0] = direction[0];
                march.direction[1] = direction[1];
                march.direction[2] = direction[2];
                march.filled = kind == Kind::Back;
            }
            for (int cursor = outer + step; cursor != pos; cursor += step) {
                int between[3];
                line_cell(axis, key_a, key_b, cursor, between);
                if (kind == Kind::Back) grid.occupied[grid.index(between[0], between[1], between[2])] = 1;
                if (record_marches) march.between.push_back({between[0], between[1], between[2]});
            }
            if (record_marches) marches.push_back(std::move(march));
            seeking = false;
        } else if (kind == Kind::Front) {
            outer = pos;
        }
    }
}

int count_double_sided(const Grid& grid) {
    int count = 0;
    for (const auto& item : grid.normals) {
        const auto& list = item.second;
        bool sided = false;
        for (size_t a = 0; a < list.size() && !sided; ++a) {
            for (size_t b = a + 1; b < list.size(); ++b) {
                const double value = list[a][0] * list[b][0] + list[a][1] * list[b][1] + list[a][2] * list[b][2];
                if (value < -0.5) {
                    sided = true;
                    break;
                }
            }
        }
        if (sided) ++count;
    }
    return count;
}

}  // namespace

bool tri_box_overlap(const double tri[3][3], const double box_min[3], const double box_max[3]) {
    const double center[3] = {(box_min[0] + box_max[0]) * 0.5, (box_min[1] + box_max[1]) * 0.5,
                              (box_min[2] + box_max[2]) * 0.5};
    const double half[3] = {(box_max[0] - box_min[0]) * 0.5, (box_max[1] - box_min[1]) * 0.5,
                            (box_max[2] - box_min[2]) * 0.5};
    double v0[3], v1[3], v2[3];
    for (int axis = 0; axis < 3; ++axis) {
        v0[axis] = tri[0][axis] - center[axis];
        v1[axis] = tri[1][axis] - center[axis];
        v2[axis] = tri[2][axis] - center[axis];
    }
    const double e0[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    const double e1[3] = {v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]};
    const double e2[3] = {v0[0] - v2[0], v0[1] - v2[1], v0[2] - v2[2]};
    if (separated_by_edges(v0, v1, v2, e0, e1, e2, half)) return false;
    if (separated_by_aabb(v0, v1, v2, half)) return false;
    return plane_overlaps(e0, e1, v0, half);
}

FillResult fill_occupancy(const std::vector<Triangle>& triangles, double voxel_size, bool record_marches) {
    if (triangles.empty()) throw std::runtime_error("没有三角形");
    if (!(voxel_size > 0.0)) throw std::runtime_error("体素边长必须为正");

    FillResult result;
    result.voxel = voxel_size;
    for (int axis = 0; axis < 3; ++axis) {
        result.aabb_min[axis] = triangles[0].v[0][axis];
        result.aabb_max[axis] = triangles[0].v[0][axis];
    }
    for (const Triangle& tri : triangles) {
        for (int v = 0; v < 3; ++v) {
            for (int axis = 0; axis < 3; ++axis) {
                result.aabb_min[axis] = std::min(result.aabb_min[axis], tri.v[v][axis]);
                result.aabb_max[axis] = std::max(result.aabb_max[axis], tri.v[v][axis]);
            }
        }
    }
    long long cells = 1;
    for (int axis = 0; axis < 3; ++axis) {
        // Cell i covers [i, i+1) * voxel_size in world meters, like a pixel on a shared lattice.
        const int index_min = static_cast<int>(std::floor(result.aabb_min[axis] / voxel_size + 1e-9));
        int index_max = static_cast<int>(std::floor(result.aabb_max[axis] / voxel_size - 1e-9));
        if (index_max < index_min) index_max = index_min;
        result.origin[axis] = index_min * voxel_size;
        result.dims[axis] = index_max - index_min + 1;
        cells *= result.dims[axis];
    }
    if (cells <= 0 || cells > 400000000LL) throw std::runtime_error("格子太大");

    Grid grid;
    for (int axis = 0; axis < 3; ++axis) {
        grid.dims[axis] = result.dims[axis];
        grid.origin[axis] = result.origin[axis];
    }
    grid.voxel = voxel_size;
    grid.kind.assign(static_cast<size_t>(cells), 0);
    grid.occupied.assign(static_cast<size_t>(cells), 0);
    grid.normals.rehash(1 << 18);
    grid.surface.reserve(200000);

    const auto raster_start = std::chrono::steady_clock::now();
    for (const Triangle& tri : triangles) raster(grid, tri.v, tri.n, true);
    const double building_raster_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - raster_start).count();
    std::cout << "STAGE 建筑表面 " << building_raster_seconds << "s\n" << std::flush;
    const auto aux_start = std::chrono::steady_clock::now();
    add_aux_box(grid, result.aabb_min, result.aabb_max);
    const double aux_raster_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - aux_start).count();
    std::cout << "STAGE 辅助面 " << aux_raster_seconds << "s\n" << std::flush;
    result.raster_seconds = building_raster_seconds + aux_raster_seconds;
    std::cout << "STAGE 表面格扫描 " << result.raster_seconds << "s\n" << std::flush;

    const auto scan_start = std::chrono::steady_clock::now();
    for (int axis = 0; axis < 3; ++axis) {
        const auto axis_start = std::chrono::steady_clock::now();
        std::unordered_map<std::uint64_t, std::vector<int>> lines;
        lines.reserve(grid.surface.size());
        for (int slot : grid.surface) {
            int i, j, k;
            decode(grid.dims, slot, i, j, k);
            std::uint64_t key = 0;
            int pos = 0;
            if (axis == 0) {
                key = (static_cast<std::uint64_t>(j) << 32) | static_cast<std::uint32_t>(k);
                pos = i;
            } else if (axis == 1) {
                key = (static_cast<std::uint64_t>(i) << 32) | static_cast<std::uint32_t>(k);
                pos = j;
            } else {
                key = (static_cast<std::uint64_t>(i) << 32) | static_cast<std::uint32_t>(j);
                pos = k;
            }
            lines[key].push_back(pos);
        }
        for (auto& item : lines) {
            auto& indices = item.second;
            std::sort(indices.begin(), indices.end());
            indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
            const int key_a = static_cast<int>(item.first >> 32);
            const int key_b = static_cast<int>(item.first & 0xffffffffu);
            scan_direction(grid, indices, axis, 1, key_a, key_b, record_marches, result.marches);
            scan_direction(grid, indices, axis, -1, key_a, key_b, record_marches, result.marches);
        }
        result.axis_seconds[axis] =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - axis_start).count();
        const char name = static_cast<char>('x' + axis);
        std::cout << "STAGE 扫描线 " << name << " " << result.axis_seconds[axis] << "s\n" << std::flush;
    }
    result.scan_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - scan_start).count();
    std::cout << "STAGE 扫描线填充 " << result.scan_seconds << "s\n" << std::flush;

    result.double_sided = count_double_sided(grid);
    result.kind.swap(grid.kind);
    result.occupied.swap(grid.occupied);
    result.normals.swap(grid.normals);
    for (size_t slot = 0; slot < result.kind.size(); ++slot) {
        const auto flag = result.kind[slot];
        if ((flag & kBuilding) != 0) ++result.surface_count;
        if ((flag & kAux) != 0) ++result.aux_surface_count;
        if ((flag & kAux) != 0 && (flag & kBuilding) == 0) ++result.aux_only_count;
        if (result.occupied[slot] != 0) ++result.occupied_count;
    }
    return result;
}

}  // namespace beidou
