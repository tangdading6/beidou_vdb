#pragma once

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace beidou {

struct Triangle {
    double v[3][3];
    double n[3];
};

struct March {
    int start[3]{};
    int hit[3]{};
    double direction[3]{};
    std::vector<std::array<int, 3>> between;
    bool filled = false;
};

struct FillResult {
    double origin[3]{};
    double aabb_min[3]{};
    double aabb_max[3]{};
    double voxel = 1.0;
    int dims[3]{};
    std::vector<std::uint8_t> kind;
    std::vector<std::uint8_t> occupied;
    std::unordered_map<int, std::vector<std::array<double, 3>>> normals;
    std::vector<March> marches;
    int surface_count = 0;
    int occupied_count = 0;
    int aux_only_count = 0;
    int aux_surface_count = 0;
    int double_sided = 0;
    double raster_seconds = 0;
    double scan_seconds = 0;
    double axis_seconds[3]{};

    int slot(int i, int j, int k) const { return (i * dims[1] + j) * dims[2] + k; }
    bool building(int i, int j, int k) const { return (kind[slot(i, j, k)] & 2) != 0; }
    bool aux_surface(int i, int j, int k) const { return (kind[slot(i, j, k)] & 1) != 0; }
    bool aux_only(int i, int j, int k) const {
        const auto flag = kind[slot(i, j, k)];
        return (flag & 1) != 0 && (flag & 2) == 0;
    }
    bool occ(int i, int j, int k) const { return occupied[slot(i, j, k)] != 0; }
};

bool tri_box_overlap(const double tri[3][3], const double box_min[3], const double box_max[3]);

FillResult fill_occupancy(const std::vector<Triangle>& triangles, double voxel_size, bool record_marches);

}  // namespace beidou
