#include "export.hpp"
#include "gltf.hpp"
#include "scanfill.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr double kWorldMin[3] = {10, 0, 20};
constexpr double kWorldMax[3] = {14, 3, 25};

struct ShellCase {
    const char* name;
    const char* filename;
    const char* drop;
    int axis;
    int side;
    int expect_surface;
    int expect_occupied;
};

const ShellCase kShells[] = {
    {"缺底面", "shell_missing_floor.gltf", "-Y", 1, -1, 48, 54},
    {"缺一面侧墙", "shell_missing_wall.gltf", "+X", 0, 1, 51, 57},
};

const char* kFaces[] = {"-X", "+X", "-Y", "+Y", "-Z", "+Z"};

fs::path executable_dir() {
    std::error_code error;
    const fs::path exe = fs::read_symlink("/proc/self/exe", error);
    if (error) return fs::current_path();
    return exe.parent_path();
}

fs::path beidou_root() { return executable_dir().parent_path(); }

std::string base64_encode(const std::vector<unsigned char>& data) {
    static const char* table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    size_t i = 0;
    while (i + 3 <= data.size()) {
        const unsigned value = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
        out.push_back(table[(value >> 18) & 63]);
        out.push_back(table[(value >> 12) & 63]);
        out.push_back(table[(value >> 6) & 63]);
        out.push_back(table[value & 63]);
        i += 3;
    }
    if (i < data.size()) {
        unsigned value = data[i] << 16;
        if (i + 1 < data.size()) value |= data[i + 1] << 8;
        out.push_back(table[(value >> 18) & 63]);
        out.push_back(table[(value >> 12) & 63]);
        if (i + 1 < data.size()) out.push_back(table[(value >> 6) & 63]);
        else out.push_back('=');
        out.push_back('=');
    }
    return out;
}

void append_f32(std::vector<unsigned char>& bytes, float value) {
    unsigned char raw[4];
    std::memcpy(raw, &value, 4);
    bytes.insert(bytes.end(), raw, raw + 4);
}

void append_u16(std::vector<unsigned char>& bytes, std::uint16_t value) {
    unsigned char raw[2];
    std::memcpy(raw, &value, 2);
    bytes.insert(bytes.end(), raw, raw + 2);
}

bool write_unit_gltf(const fs::path& path, const char* transform_json, bool indexed) {
    std::vector<unsigned char> blob;
    const float position[9] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
    for (float value : position) append_f32(blob, value);
    if (indexed) {
        append_u16(blob, 0);
        append_u16(blob, 1);
        append_u16(blob, 2);
    }
    std::ostringstream json;
    json << "{\"asset\":{\"version\":\"2.0\"},\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[{\"mesh\":0,";
    json << transform_json;
    json << "}],\"meshes\":[{\"primitives\":[{\"attributes\":{\"POSITION\":0},";
    if (indexed) json << "\"indices\":1,";
    json << "\"mode\":4}]}],\"accessors\":[{\"bufferView\":0,\"componentType\":5126,\"count\":3,\"type\":\"VEC3\"}";
    if (indexed) {
        json << ",{\"bufferView\":1,\"componentType\":5123,\"count\":3,\"type\":\"SCALAR\"}";
    }
    json << "],\"bufferViews\":[{\"buffer\":0,\"byteOffset\":0,\"byteLength\":36}";
    if (indexed) json << ",{\"buffer\":0,\"byteOffset\":36,\"byteLength\":6}";
    json << "],\"buffers\":[{\"byteLength\":" << blob.size()
         << ",\"uri\":\"data:application/octet-stream;base64," << base64_encode(blob) << "\"}]}";
    std::ofstream out(path);
    if (!out) return false;
    out << json.str();
    return true;
}

bool near_normal(const double normal[3], double x, double y, double z) {
    return std::fabs(normal[0] - x) < 1e-6 && std::fabs(normal[1] - y) < 1e-6 && std::fabs(normal[2] - z) < 1e-6;
}

bool self_check() {
    const double box_min[3] = {0, 0, 0};
    const double box_max[3] = {1, 1, 1};
    const double on_face[3][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    const double above[3][3] = {{0, 0, 1}, {1, 0, 1}, {0, 1, 1}};
    const double outside[3][3] = {{0, 0, 2}, {1, 0, 2}, {0, 1, 2}};
    const double cutting[3][3] = {{0, 0, 0.5}, {1, 0, 0.5}, {0, 1, 0.5}};
    if (!beidou::tri_box_overlap(on_face, box_min, box_max) || !beidou::tri_box_overlap(above, box_min, box_max) ||
        beidou::tri_box_overlap(outside, box_min, box_max) || !beidou::tri_box_overlap(cutting, box_min, box_max)) {
        std::cout << "自检：失败（三角形与方盒相交）\n";
        return false;
    }

    const double half = std::sin(3.14159265358979323846 / 4.0);
    std::ostringstream yaw;
    yaw.setf(std::ios::fmtflags(0), std::ios::floatfield);
    yaw.precision(17);
    yaw << "\"rotation\":[0," << half << ",0," << half << "]";
    const fs::path yaw_path = "/tmp/beidou-yaw.gltf";
    const fs::path mirror_path = "/tmp/beidou-mirror.gltf";
    if (!write_unit_gltf(yaw_path, yaw.str().c_str(), true) ||
        !write_unit_gltf(mirror_path, "\"scale\":[-1,1,1]", false)) {
        std::cout << "自检：失败（写临时 glTF）\n";
        return false;
    }
    const auto yaw_tris = beidou::load_gltf_triangles(yaw_path.string());
    const auto mirror_tris = beidou::load_gltf_triangles(mirror_path.string());
    if (yaw_tris.size() != 1 || !near_normal(yaw_tris[0].n, 1, 0, 0) || mirror_tris.size() != 1 ||
        !near_normal(mirror_tris[0].n, 0, 0, 1)) {
        std::cout << "自检：失败（节点旋转或镜像法线）\n";
        return false;
    }
    std::cout << "自检：通过\n";
    return true;
}

void face_normal(const char* name, double out[3]) {
    out[0] = out[1] = out[2] = 0;
    const double sign = name[0] == '+' ? 1.0 : -1.0;
    if (name[1] == 'X') out[0] = sign;
    else if (name[1] == 'Y') out[1] = sign;
    else out[2] = sign;
}

int face_axis(const char* name) { return name[1] == 'X' ? 0 : name[1] == 'Y' ? 1 : 2; }

std::vector<std::array<int, 3>> face_interior(const int dims[3], const char* name) {
    const int axis = face_axis(name);
    const int side = name[0] == '-' ? -1 : 1;
    const int bound = side < 0 ? 0 : dims[axis] - 1;
    std::vector<std::array<int, 3>> found;
    for (int i = 0; i < dims[0]; ++i) {
        for (int j = 0; j < dims[1]; ++j) {
            for (int k = 0; k < dims[2]; ++k) {
                const int ijk[3] = {i, j, k};
                if (ijk[axis] != bound) continue;
                bool interior = true;
                for (int other = 0; other < 3; ++other) {
                    if (other == axis) continue;
                    if (!(ijk[other] > 0 && ijk[other] < dims[other] - 1)) interior = false;
                }
                if (interior) found.push_back({i, j, k});
            }
        }
    }
    return found;
}

bool shell_matches(const beidou::FillResult& result, const char* drop) {
    for (const char* name : kFaces) {
        const auto voxels = face_interior(result.dims, name);
        if (voxels.empty()) return false;
        if (std::strcmp(name, drop) == 0) {
            for (const auto& voxel : voxels) {
                if (result.building(voxel[0], voxel[1], voxel[2])) return false;
                if (!result.aux_surface(voxel[0], voxel[1], voxel[2])) return false;
            }
            continue;
        }
        double expected[3];
        face_normal(name, expected);
        for (const auto& voxel : voxels) {
            if (!result.building(voxel[0], voxel[1], voxel[2])) return false;
            const auto found = result.normals.find(result.slot(voxel[0], voxel[1], voxel[2]));
            if (found == result.normals.end()) return false;
            bool aligned = false;
            for (const auto& normal : found->second) {
                const double value = normal[0] * expected[0] + normal[1] * expected[1] + normal[2] * expected[2];
                if (value > 0.99) aligned = true;
            }
            if (!aligned) return false;
        }
    }
    return true;
}

void print_slice(const beidou::FillResult& result) {
    std::cout << "水平切片（x 从左到右，z 从上到下，█ 占用，· 空）：\n";
    const int cells_x = result.dims[0];
    for (int j = 0; j < result.dims[1]; ++j) {
        bool solid = true;
        std::vector<std::string> rows;
        for (int k = 0; k < result.dims[2]; ++k) {
            std::string row;
            for (int i = 0; i < cells_x; ++i) {
                const bool filled = result.occ(i, j, k);
                row += filled ? "█" : "·";
                if (!filled) solid = false;
            }
            rows.push_back(row);
        }
        if (solid) {
            std::cout << "  y=" << j << ": 整层实心 " << result.dims[0] << "×" << result.dims[2] << "\n";
        } else {
            std::cout << "  y=" << j << ":\n";
            for (const auto& row : rows) std::cout << "    " << row << "\n";
        }
    }
}

const char* yes(bool flag) { return flag ? "是" : "否"; }

bool run_shell(const ShellCase& shell) {
    const fs::path path = beidou_root() / "cases" / shell.filename;
    const auto triangles = beidou::load_gltf_triangles(path.string());
    const auto result = beidou::fill_occupancy(triangles, 1.0, true);
    const int cells = result.dims[0] * result.dims[1] * result.dims[2];

    bool transform_ok = true;
    for (int axis = 0; axis < 3; ++axis) {
        if (std::fabs(result.aabb_min[axis] - kWorldMin[axis]) > 1e-5) transform_ok = false;
        if (std::fabs(result.aabb_max[axis] - kWorldMax[axis]) > 1e-5) transform_ok = false;
    }
    const double center[3] = {(result.aabb_min[0] + result.aabb_max[0]) * 0.5,
                              (result.aabb_min[1] + result.aabb_max[1]) * 0.5,
                              (result.aabb_min[2] + result.aabb_max[2]) * 0.5};
    bool outward = true;
    for (const auto& tri : triangles) {
        const double mean[3] = {(tri.v[0][0] + tri.v[1][0] + tri.v[2][0]) / 3.0,
                                (tri.v[0][1] + tri.v[1][1] + tri.v[2][1]) / 3.0,
                                (tri.v[0][2] + tri.v[1][2] + tri.v[2][2]) / 3.0};
        const double value = tri.n[0] * (mean[0] - center[0]) + tri.n[1] * (mean[1] - center[1]) +
                             tri.n[2] * (mean[2] - center[2]);
        if (!(value > 0.0)) outward = false;
    }

    int holes = 0;
    int leak_centers = 0;
    int aux_off_face = 0;
    const int bound = shell.side < 0 ? 0 : result.dims[shell.axis] - 1;
    for (int i = 0; i < result.dims[0]; ++i) {
        for (int j = 0; j < result.dims[1]; ++j) {
            for (int k = 0; k < result.dims[2]; ++k) {
                const int ijk[3] = {i, j, k};
                const bool aux_only = result.aux_only(i, j, k);
                if (!result.occ(i, j, k) && !aux_only) ++holes;
                if (aux_only && ijk[shell.axis] != bound) ++aux_off_face;
                if (!result.occ(i, j, k)) continue;
                bool leaked = false;
                for (int axis = 0; axis < 3; ++axis) {
                    const double coord = result.origin[axis] + (ijk[axis] + 0.5) * result.voxel;
                    if (coord < result.aabb_min[axis] - 1e-6 || coord > result.aabb_max[axis] + 1e-6) leaked = true;
                }
                if (leaked) ++leak_centers;
            }
        }
    }

    int gap_hits = 0;
    bool between = false;
    for (const auto& march : result.marches) {
        if (!march.between.empty()) between = true;
        const int hit_slot_ok = result.aux_only(march.hit[0], march.hit[1], march.hit[2]);
        if (hit_slot_ok && march.hit[shell.axis] == bound && march.direction[shell.axis] * shell.side > 0.5) {
            ++gap_hits;
        }
    }
    const bool shell_ok = shell_matches(result, shell.drop);
    const bool interior_filled = holes == 0 && between && shell_ok;
    const bool stopped = aux_off_face == 0 && result.aux_only_count > 0 && gap_hits > 0;
    const int leaks = leak_centers;
    const double total = result.raster_seconds + result.scan_seconds;
    const bool counts_ok = result.surface_count == shell.expect_surface && result.occupied_count == shell.expect_occupied &&
                           cells == 60 && triangles.size() == 10;
    const bool pass = transform_ok && outward && interior_filled && stopped && leaks == 0 && counts_ok;

    std::cout << "\n## " << shell.name << "（" << shell.filename << "，缺 " << shell.drop << "）\n";
    std::cout << "三角形 " << triangles.size() << " 个。格子 " << result.dims[0] << "×" << result.dims[1] << "×"
              << result.dims[2] << "。\n";
    std::cout << "表面体素 " << result.surface_count << "\n";
    std::cout << "占用体素 " << result.occupied_count << "/" << cells << "\n";
    std::cout << "双面墙 " << result.double_sided << "\n";
    std::cout << "总时间 " << total << " s\n";
    std::cout << "节点变换后的包围盒与壳体世界坐标一致：" << yes(transform_ok) << "。正面法线朝外：" << yes(outward)
              << "。\n";
    std::cout << "内部是否填上：" << yes(interior_filled) << "。\n";
    std::cout << "缺口是否停在包围盒：" << yes(stopped) << "。朝缺口打到辅助面的扫描 " << gap_hits << " 条。\n";
    std::cout << "壳外渗漏：" << (leaks == 0 ? "无" : "有") << "。\n";
    std::cout << "结果：" << (pass ? "通过" : "失败") << "\n";
    if (!pass) {
        std::cout << "空体素 " << holes << "，盒外中心 " << leak_centers << "，壳体表面一致：" << yes(shell_ok)
                  << "，数量一致：" << yes(counts_ok) << "。\n";
    }
    print_slice(result);
    return pass;
}

fs::path shenyang_path(int argc, char** argv) {
    if (argc > 1) return argv[1];
    const fs::path bundled = beidou_root().parent_path() / "gltf" / "3dbuilding_ln_sy_hn_gwzx_hycgwzx" /
                             "3dbuilding_ln_sy_hn_gwzx_hycgwzx.gltf";
    if (fs::exists(bundled)) return bundled;
    const fs::path from_cwd = fs::current_path() / "gltf" / "3dbuilding_ln_sy_hn_gwzx_hycgwzx" /
                              "3dbuilding_ln_sy_hn_gwzx_hycgwzx.gltf";
    return from_cwd;
}

bool run_shenyang(const fs::path& path) {
    std::cout << "\n## 沈阳楼（" << path.filename().string() << "）\n";
    if (!fs::exists(path)) {
        std::cout << "结果：失败（找不到 " << path.string() << "）\n";
        return false;
    }
    const auto load_start = std::chrono::steady_clock::now();
    const auto triangles = beidou::load_gltf_triangles(path.string());
    const double load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count();
    const auto result = beidou::fill_occupancy(triangles, 1.0, false);
    const double total = result.raster_seconds + result.scan_seconds;
    const bool pass = triangles.size() == 5651 && total < 63.0;
    std::cout << "三角形 " << triangles.size() << " 个。格子 " << result.dims[0] << "×" << result.dims[1] << "×"
              << result.dims[2] << "。\n";
    std::cout << "表面体素 " << result.surface_count << "\n";
    std::cout << "占用体素 " << result.occupied_count << "\n";
    std::cout << "双面墙 " << result.double_sided << "\n";
    std::cout << "读取 " << load_seconds << " s\n";
    std::cout << "总时间 " << total << " s\n";
    const fs::path exported = beidou_root() / "figures" / "native" / "shenyang-hycgwzx-occupancy-1m-scanline.gltf";
    const fs::path preview = beidou_root() / "figures" / "native" / "shenyang-hycgwzx-occupancy-1m-scanline-preview.gltf";
    const auto cube_export_start = std::chrono::steady_clock::now();
    beidou::export_occupancy_gltf(result, exported.string(), true, false);
    const double cube_export_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - cube_export_start).count();
    const auto preview_export_start = std::chrono::steady_clock::now();
    beidou::export_merged_boxes(result, preview.string(), false);
    const fs::path surface = beidou_root() / "figures" / "native" / "shenyang-hycgwzx-surface-1m.gltf";
    const fs::path surface_preview = beidou_root() / "figures" / "native" / "shenyang-hycgwzx-surface-1m-preview.gltf";
    beidou::export_occupancy_gltf(result, surface.string(), false, false);
    beidou::export_merged_boxes(result, surface_preview.string(), true);
    std::cout << "表面可预览 " << surface_preview.string() << "\n";
    const double preview_export_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - preview_export_start).count();
    std::cout << "写出 1 米方块 " << cube_export_seconds << " s\n";
    std::cout << "写出预览 " << preview_export_seconds << " s\n";
    std::cout << "已写出 " << exported.string() << "\n";
    std::cout << "可预览 " << preview.string() << "\n";
    std::cout << "结果：" << (pass ? "通过" : "失败") << "\n";
    return pass;
}

}  // namespace

int main(int argc, char** argv) {
    std::cout.setf(std::ios::fixed);
    std::cout.precision(3);
    try {
        bool ok = self_check();
        for (const ShellCase& shell : kShells) ok = run_shell(shell) && ok;
        ok = run_shenyang(shenyang_path(argc, argv)) && ok;
        return ok ? 0 : 1;
    } catch (const std::exception& ex) {
        std::cout << "结果：失败（" << ex.what() << "）\n";
        return 1;
    }
}
