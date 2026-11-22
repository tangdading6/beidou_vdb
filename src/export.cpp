#include "export.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace beidou {
namespace {

struct Batch {
    const char* name;
    float color[4];
    std::vector<float> xyz;
    float mn[3];
    float mx[3];
    bool used = false;

    void add(float x, float y, float z) {
        const float point[3] = {x, y, z};
        if (!used) {
            for (int axis = 0; axis < 3; ++axis) mn[axis] = mx[axis] = point[axis];
            used = true;
        } else {
            for (int axis = 0; axis < 3; ++axis) {
                mn[axis] = std::min(mn[axis], point[axis]);
                mx[axis] = std::max(mx[axis], point[axis]);
            }
        }
        xyz.push_back(x);
        xyz.push_back(y);
        xyz.push_back(z);
    }

    int count() const { return static_cast<int>(xyz.size() / 3); }
};

bool has_opposite_normals(const std::vector<std::array<double, 3>>& normals) {
    for (size_t a = 0; a < normals.size(); ++a) {
        for (size_t b = a + 1; b < normals.size(); ++b) {
            const double dot = normals[a][0] * normals[b][0] + normals[a][1] * normals[b][1] +
                               normals[a][2] * normals[b][2];
            if (dot < -0.5) return true;
        }
    }
    return false;
}

int orientation_bucket(const std::vector<std::array<double, 3>>& normals) {
    int best_axis = 0;
    double best = -1.0;
    double sign = 1.0;
    for (const auto& normal : normals) {
        for (int axis = 0; axis < 3; ++axis) {
            const double magnitude = std::fabs(normal[axis]);
            if (magnitude > best) {
                best = magnitude;
                best_axis = axis;
                sign = normal[axis];
            }
        }
    }
    return sign >= 0.0 ? best_axis : best_axis + 3;
}

void write_f32(std::ostream& out, const float* data, size_t count) {
    out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count * sizeof(float)));
}

void write_json_number(std::ostream& out, double value) {
    out.setf(std::ios::fmtflags(0), std::ios::floatfield);
    out.precision(9);
    out << value;
}

}  // namespace

void export_occupancy_gltf(const FillResult& result, const std::string& gltf_path, bool include_interior,
                           bool split_colors) {
    Batch batches[8] = {
        {"interior", {0.20f, 0.62f, 0.55f, 1.f}, {}, {}, {}, false},
        {"surface+X", {0.86f, 0.32f, 0.28f, 1.f}, {}, {}, {}, false},
        {"surface+Y", {0.90f, 0.74f, 0.28f, 1.f}, {}, {}, {}, false},
        {"surface+Z", {0.30f, 0.48f, 0.86f, 1.f}, {}, {}, {}, false},
        {"surface-X", {0.45f, 0.16f, 0.16f, 1.f}, {}, {}, {}, false},
        {"surface-Y", {0.42f, 0.30f, 0.16f, 1.f}, {}, {}, {}, false},
        {"surface-Z", {0.16f, 0.22f, 0.48f, 1.f}, {}, {}, {}, false},
        {"double-sided", {0.95f, 0.48f, 0.12f, 1.f}, {}, {}, {}, false},
    };
    if (!split_colors) {
        batches[0].name = include_interior ? "occupied" : "surface";
        batches[0].color[0] = include_interior ? 0.20f : 0.78f;
        batches[0].color[1] = include_interior ? 0.53f : 0.74f;
        batches[0].color[2] = include_interior ? 0.46f : 0.66f;
        batches[0].color[3] = 1.f;
    }
    for (auto& batch : batches) batch.xyz.reserve(4096);

    for (int i = 0; i < result.dims[0]; ++i) {
        for (int j = 0; j < result.dims[1]; ++j) {
            for (int k = 0; k < result.dims[2]; ++k) {
                if (!result.occ(i, j, k)) continue;
                const float x = static_cast<float>(result.origin[0] + i * result.voxel);
                const float y = static_cast<float>(result.origin[1] + j * result.voxel);
                const float z = static_cast<float>(result.origin[2] + k * result.voxel);
                int batch = 0;
                if (!include_interior && !result.building(i, j, k)) continue;
                if (split_colors && result.building(i, j, k)) {
                    const auto found = result.normals.find(result.slot(i, j, k));
                    if (found != result.normals.end() && has_opposite_normals(found->second)) batch = 7;
                    else if (found != result.normals.end()) batch = 1 + orientation_bucket(found->second);
                    else batch = 1;
                }
                batches[batch].add(x, y, z);
            }
        }
    }

    int occupied = 0;
    int surface = 0;
    for (int index = 0; index < 8; ++index) {
        occupied += batches[index].count();
        if (index > 0) surface += batches[index].count();
    }
    const int expected = include_interior ? result.occupied_count : result.surface_count;
    if (occupied != expected || (split_colors && surface != result.surface_count)) {
        throw std::runtime_error("导出分类数量与填充结果不一致");
    }

    const std::filesystem::path json_path(gltf_path);
    std::filesystem::create_directories(json_path.parent_path());
    const std::filesystem::path bin_path = json_path.parent_path() / (json_path.stem().string() + ".bin");

    const float cube[] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
    const std::uint16_t indices[] = {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
                                     3, 6, 2, 3, 7, 6, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5};

    std::ofstream bin(bin_path, std::ios::binary);
    if (!bin) throw std::runtime_error("无法写入 " + bin_path.string());
    write_f32(bin, cube, 24);
    bin.write(reinterpret_cast<const char*>(indices), sizeof(indices));
    int translation_bytes = 0;
    for (const auto& batch : batches) {
        if (batch.xyz.empty()) continue;
        write_f32(bin, batch.xyz.data(), batch.xyz.size());
        translation_bytes += static_cast<int>(batch.xyz.size() * sizeof(float));
    }
    if (!bin) throw std::runtime_error("写入实例坐标失败");
    bin.close();

    std::ofstream json(json_path);
    if (!json) throw std::runtime_error("无法写入 " + json_path.string());
    json << "{\n";
    json << "  \"asset\": {\"version\": \"2.0\", \"generator\": \"beidou_vdb native scanline\"},\n";
    json << "  \"extensionsUsed\": [\"EXT_mesh_gpu_instancing\"],\n";
    json << "  \"extensionsRequired\": [\"EXT_mesh_gpu_instancing\"],\n";
    json << "  \"scene\": 0,\n";
    json << "  \"scenes\": [{\"nodes\": [";
    int emitted = 0;
    for (const auto& batch : batches) {
        if (batch.xyz.empty()) continue;
        if (emitted) json << ", ";
        json << emitted;
        ++emitted;
    }
    json << "]}],\n  \"nodes\": [\n";
    int node = 0;
    int accessor = 2;
    for (const auto& batch : batches) {
        if (batch.xyz.empty()) continue;
        if (node) json << ",\n";
        json << "    {\"name\": \"" << batch.name << "\", \"mesh\": " << node
             << ", \"extensions\": {\"EXT_mesh_gpu_instancing\": {\"attributes\": {\"TRANSLATION\": " << accessor
             << "}}}}";
        ++node;
        ++accessor;
    }
    json << "\n  ],\n  \"materials\": [\n";
    node = 0;
    for (const auto& batch : batches) {
        if (batch.xyz.empty()) continue;
        if (node) json << ",\n";
        json << "    {\"name\": \"" << batch.name
             << "\", \"doubleSided\": true, \"pbrMetallicRoughness\": {\"baseColorFactor\": [";
        for (int channel = 0; channel < 4; ++channel) {
            if (channel) json << ", ";
            write_json_number(json, batch.color[channel]);
        }
        json << "], \"metallicFactor\": 0.0, \"roughnessFactor\": 1.0}}";
        ++node;
    }
    json << "\n  ],\n  \"meshes\": [\n";
    node = 0;
    for (const auto& batch : batches) {
        if (batch.xyz.empty()) continue;
        if (node) json << ",\n";
        json << "    {\"name\": \"" << batch.name
             << "\", \"primitives\": [{\"attributes\": {\"POSITION\": 0}, \"indices\": 1, \"material\": " << node
             << ", \"mode\": 4}]}";
        ++node;
    }
    json << "\n  ],\n  \"accessors\": [\n";
    json << "    {\"bufferView\": 0, \"componentType\": 5126, \"count\": 8, \"type\": \"VEC3\", \"min\": [0, 0, 0], "
            "\"max\": [1, 1, 1]},\n";
    json << "    {\"bufferView\": 1, \"componentType\": 5123, \"count\": 36, \"type\": \"SCALAR\"}";
    int view = 2;
    int byte_offset = 168;
    for (const auto& batch : batches) {
        if (batch.xyz.empty()) continue;
        json << ",\n    {\"bufferView\": " << view << ", \"componentType\": 5126, \"count\": " << batch.count()
             << ", \"type\": \"VEC3\", \"min\": [";
        write_json_number(json, batch.mn[0]);
        json << ", ";
        write_json_number(json, batch.mn[1]);
        json << ", ";
        write_json_number(json, batch.mn[2]);
        json << "], \"max\": [";
        write_json_number(json, batch.mx[0]);
        json << ", ";
        write_json_number(json, batch.mx[1]);
        json << ", ";
        write_json_number(json, batch.mx[2]);
        json << "]}";
        ++view;
        byte_offset += static_cast<int>(batch.xyz.size() * sizeof(float));
    }
    json << "\n  ],\n  \"bufferViews\": [\n";
    json << "    {\"buffer\": 0, \"byteOffset\": 0, \"byteLength\": 96, \"target\": 34962},\n";
    json << "    {\"buffer\": 0, \"byteOffset\": 96, \"byteLength\": 72, \"target\": 34963}";
    view = 2;
    byte_offset = 168;
    for (const auto& batch : batches) {
        if (batch.xyz.empty()) continue;
        const int bytes = static_cast<int>(batch.xyz.size() * sizeof(float));
        json << ",\n    {\"buffer\": 0, \"byteOffset\": " << byte_offset << ", \"byteLength\": " << bytes
             << ", \"target\": 34962}";
        byte_offset += bytes;
        ++view;
    }
    const int bin_bytes = 168 + translation_bytes;
    json << "\n  ],\n  \"buffers\": [{\"uri\": \"" << bin_path.filename().string() << "\", \"byteLength\": " << bin_bytes
         << "}],\n";
    const int surface_reported = split_colors ? surface : result.surface_count;
    const int double_reported = split_colors ? batches[7].count() : result.double_sided;
    json << "  \"extras\": {\"source\": \"3dbuilding_ln_sy_hn_gwzx_hycgwzx.gltf\", \"voxel_size_m\": 1.0, "
            "\"occupied\": "
         << occupied << ", \"building_surface\": " << surface_reported << ", \"double_sided\": " << double_reported
         << ", \"interior\": " << (occupied - surface_reported)
         << ", \"representation\": \""
         << (split_colors ? "instanced 1 m cubes colored by orientation"
                          : "instanced 1 m cubes, one material")
         << "\"}\n}\n";
    if (!json) throw std::runtime_error("写入 glTF 失败");
}

namespace {

struct Face {
    int layer;
    int u;
    int v;
};

struct MeshPart {
    const char* name;
    float color[4];
    std::vector<float> position;
    std::vector<float> normal;
    std::vector<std::uint32_t> indices;
    float mn[3]{1e30f, 1e30f, 1e30f};
    float mx[3]{-1e30f, -1e30f, -1e30f};

    void vertex(float x, float y, float z, float nx, float ny, float nz) {
        const float point[3] = {x, y, z};
        const float nrm[3] = {nx, ny, nz};
        for (int axis = 0; axis < 3; ++axis) {
            mn[axis] = std::min(mn[axis], point[axis]);
            mx[axis] = std::max(mx[axis], point[axis]);
        }
        position.insert(position.end(), point, point + 3);
        normal.insert(normal.end(), nrm, nrm + 3);
    }
};

void dominant_face(const std::array<double, 3>& normal, int& axis, int& positive) {
    axis = 0;
    double best = std::fabs(normal[0]);
    for (int next = 1; next < 3; ++next) {
        if (std::fabs(normal[next]) > best) {
            best = std::fabs(normal[next]);
            axis = next;
        }
    }
    positive = normal[axis] >= 0.0 ? 1 : 0;
}

void add_rectangle(MeshPart& part, int axis, int positive, float fixed, float u0, float u1, float v0, float v1) {
    const int u_axis = (axis + 1) % 3;
    const int v_axis = (axis + 2) % 3;
    float corner[4][3] = {};
    const float u_value[4] = {u0, u1, u1, u0};
    const float v_value[4] = {v0, v0, v1, v1};
    for (int index = 0; index < 4; ++index) {
        corner[index][axis] = fixed;
        corner[index][u_axis] = u_value[index];
        corner[index][v_axis] = v_value[index];
    }
    float nrm[3] = {};
    nrm[axis] = positive ? 1.f : -1.f;
    const std::uint32_t base = static_cast<std::uint32_t>(part.position.size() / 3);
    for (int index = 0; index < 4; ++index) part.vertex(corner[index][0], corner[index][1], corner[index][2], nrm[0], nrm[1], nrm[2]);
    if (positive) {
        const std::uint32_t tri[6] = {0, 1, 2, 0, 2, 3};
        for (std::uint32_t index : tri) part.indices.push_back(base + index);
    } else {
        const std::uint32_t tri[6] = {0, 2, 1, 0, 3, 2};
        for (std::uint32_t index : tri) part.indices.push_back(base + index);
    }
}

}  // namespace

void export_surface_mesh(const FillResult& result, const std::string& gltf_path) {
    MeshPart parts[7] = {
        {"surface+X", {0.86f, 0.32f, 0.28f, 1.f}, {}, {}, {}},
        {"surface+Y", {0.90f, 0.74f, 0.28f, 1.f}, {}, {}, {}},
        {"surface+Z", {0.30f, 0.48f, 0.86f, 1.f}, {}, {}, {}},
        {"surface-X", {0.45f, 0.16f, 0.16f, 1.f}, {}, {}, {}},
        {"surface-Y", {0.42f, 0.30f, 0.16f, 1.f}, {}, {}, {}},
        {"surface-Z", {0.16f, 0.22f, 0.48f, 1.f}, {}, {}, {}},
        {"double-sided", {0.95f, 0.48f, 0.12f, 1.f}, {}, {}, {}},
    };
    std::vector<Face> faces[7][3][2];
    for (int i = 0; i < result.dims[0]; ++i) {
        for (int j = 0; j < result.dims[1]; ++j) {
            for (int k = 0; k < result.dims[2]; ++k) {
                if (!result.building(i, j, k)) continue;
                const auto found = result.normals.find(result.slot(i, j, k));
                if (found == result.normals.end() || found->second.empty()) continue;
                const int ijk[3] = {i, j, k};
                const bool doubled = has_opposite_normals(found->second);
                int marked = 0;
                const auto mark = [&](int material, int axis, int positive) {
                    const int bit = 1 << (axis + (positive ? 0 : 3));
                    if (marked & bit) return;
                    marked |= bit;
                    const int u_axis = (axis + 1) % 3;
                    const int v_axis = (axis + 2) % 3;
                    faces[material][axis][positive].push_back(Face{ijk[axis], ijk[u_axis], ijk[v_axis]});
                };
                if (doubled) {
                    for (const auto& normal : found->second) {
                        int axis = 0;
                        int positive = 1;
                        dominant_face(normal, axis, positive);
                        mark(6, axis, positive);
                    }
                } else {
                    const int bucket = orientation_bucket(found->second);
                    const int axis = bucket < 3 ? bucket : bucket - 3;
                    const int positive = bucket < 3 ? 1 : 0;
                    mark(bucket, axis, positive);
                }
            }
        }
    }

    int rectangles = 0;
    for (int material = 0; material < 7; ++material) {
        for (int axis = 0; axis < 3; ++axis) {
            const int u_axis = (axis + 1) % 3;
            const int v_axis = (axis + 2) % 3;
            const int nu = result.dims[u_axis];
            const int nv = result.dims[v_axis];
            for (int positive = 0; positive < 2; ++positive) {
                auto& list = faces[material][axis][positive];
                if (list.empty()) continue;
                std::sort(list.begin(), list.end(), [](const Face& a, const Face& b) {
                    if (a.layer != b.layer) return a.layer < b.layer;
                    if (a.u != b.u) return a.u < b.u;
                    return a.v < b.v;
                });
                std::vector<char> mask(static_cast<size_t>(nu) * static_cast<size_t>(nv), 0);
                size_t cursor = 0;
                while (cursor < list.size()) {
                    const int layer = list[cursor].layer;
                    std::fill(mask.begin(), mask.end(), 0);
                    while (cursor < list.size() && list[cursor].layer == layer) {
                        mask[static_cast<size_t>(list[cursor].u) * static_cast<size_t>(nv) +
                             static_cast<size_t>(list[cursor].v)] = 1;
                        ++cursor;
                    }
                    const float fixed = static_cast<float>(result.origin[axis] + (layer + positive) * result.voxel);
                    for (int u = 0; u < nu; ++u) {
                        for (int v = 0; v < nv;) {
                            if (!mask[static_cast<size_t>(u) * static_cast<size_t>(nv) + static_cast<size_t>(v)]) {
                                ++v;
                                continue;
                            }
                            int width = 1;
                            while (v + width < nv &&
                                   mask[static_cast<size_t>(u) * static_cast<size_t>(nv) + static_cast<size_t>(v + width)]) {
                                ++width;
                            }
                            int height = 1;
                            while (u + height < nu) {
                                bool row = true;
                                for (int step = 0; step < width; ++step) {
                                    if (!mask[static_cast<size_t>(u + height) * static_cast<size_t>(nv) +
                                              static_cast<size_t>(v + step)]) {
                                        row = false;
                                        break;
                                    }
                                }
                                if (!row) break;
                                ++height;
                            }
                            for (int uu = u; uu < u + height; ++uu) {
                                for (int step = 0; step < width; ++step) {
                                    mask[static_cast<size_t>(uu) * static_cast<size_t>(nv) + static_cast<size_t>(v + step)] = 0;
                                }
                            }
                            const float u0 = static_cast<float>(result.origin[u_axis] + u * result.voxel);
                            const float u1 = static_cast<float>(result.origin[u_axis] + (u + height) * result.voxel);
                            const float v0 = static_cast<float>(result.origin[v_axis] + v * result.voxel);
                            const float v1 = static_cast<float>(result.origin[v_axis] + (v + width) * result.voxel);
                            add_rectangle(parts[material], axis, positive, fixed, u0, u1, v0, v1);
                            ++rectangles;
                            v += width;
                        }
                    }
                }
            }
        }
    }

    const std::filesystem::path json_path(gltf_path);
    std::filesystem::create_directories(json_path.parent_path());
    const std::filesystem::path bin_path = json_path.parent_path() / (json_path.stem().string() + ".bin");
    std::ofstream bin(bin_path, std::ios::binary);
    if (!bin) throw std::runtime_error("无法写入 " + bin_path.string());
    struct Span {
        int position = 0;
        int normal = 0;
        int index = 0;
        int position_bytes = 0;
        int normal_bytes = 0;
        int index_bytes = 0;
    };
    std::vector<Span> spans;
    int offset = 0;
    int triangles = 0;
    for (const auto& part : parts) {
        if (part.indices.empty()) continue;
        Span span;
        span.position = offset;
        span.position_bytes = static_cast<int>(part.position.size() * sizeof(float));
        write_f32(bin, part.position.data(), part.position.size());
        offset += span.position_bytes;
        span.normal = offset;
        span.normal_bytes = static_cast<int>(part.normal.size() * sizeof(float));
        write_f32(bin, part.normal.data(), part.normal.size());
        offset += span.normal_bytes;
        span.index = offset;
        span.index_bytes = static_cast<int>(part.indices.size() * sizeof(std::uint32_t));
        bin.write(reinterpret_cast<const char*>(part.indices.data()), span.index_bytes);
        offset += span.index_bytes;
        triangles += static_cast<int>(part.indices.size() / 3);
        spans.push_back(span);
    }
    if (!bin) throw std::runtime_error("写入表面网格失败");
    bin.close();

    std::ofstream json(json_path);
    if (!json) throw std::runtime_error("无法写入 " + json_path.string());
    json << "{\n  \"asset\": {\"version\": \"2.0\", \"generator\": \"beidou_vdb native scanline\"},\n";
    json << "  \"scene\": 0,\n  \"scenes\": [{\"nodes\": [0]}],\n  \"nodes\": [{\"name\": \"surface\", \"mesh\": 0}],\n";
    json << "  \"materials\": [\n";
    int emitted = 0;
    for (const auto& part : parts) {
        if (part.indices.empty()) continue;
        if (emitted) json << ",\n";
        json << "    {\"name\": \"" << part.name << "\", \"doubleSided\": true, \"pbrMetallicRoughness\": {\"baseColorFactor\": [";
        for (int channel = 0; channel < 4; ++channel) {
            if (channel) json << ", ";
            write_json_number(json, part.color[channel]);
        }
        json << "], \"metallicFactor\": 0.0, \"roughnessFactor\": 1.0}}";
        ++emitted;
    }
    json << "\n  ],\n  \"meshes\": [{\"name\": \"merged-surface\", \"primitives\": [\n";
    emitted = 0;
    int accessor = 0;
    for (const auto& part : parts) {
        if (part.indices.empty()) continue;
        if (emitted) json << ",\n";
        json << "    {\"attributes\": {\"POSITION\": " << accessor << ", \"NORMAL\": " << accessor + 1
             << "}, \"indices\": " << accessor + 2 << ", \"material\": " << emitted << ", \"mode\": 4}";
        accessor += 3;
        ++emitted;
    }
    json << "\n  ]}],\n  \"accessors\": [\n";
    emitted = 0;
    int span_index = 0;
    for (const auto& part : parts) {
        if (part.indices.empty()) continue;
        const Span& span = spans[static_cast<size_t>(span_index)];
        if (emitted) json << ",\n";
        json << "    {\"bufferView\": " << emitted << ", \"componentType\": 5126, \"count\": " << part.position.size() / 3
             << ", \"type\": \"VEC3\", \"min\": [";
        write_json_number(json, part.mn[0]);
        json << ", ";
        write_json_number(json, part.mn[1]);
        json << ", ";
        write_json_number(json, part.mn[2]);
        json << "], \"max\": [";
        write_json_number(json, part.mx[0]);
        json << ", ";
        write_json_number(json, part.mx[1]);
        json << ", ";
        write_json_number(json, part.mx[2]);
        json << "]},\n    {\"bufferView\": " << emitted + 1 << ", \"componentType\": 5126, \"count\": "
             << part.normal.size() / 3 << ", \"type\": \"VEC3\"},\n    {\"bufferView\": " << emitted + 2
             << ", \"componentType\": 5125, \"count\": " << part.indices.size() << ", \"type\": \"SCALAR\"}";
        emitted += 3;
        ++span_index;
        (void)span;
    }
    json << "\n  ],\n  \"bufferViews\": [\n";
    emitted = 0;
    span_index = 0;
    for (const auto& part : parts) {
        if (part.indices.empty()) continue;
        const Span& span = spans[static_cast<size_t>(span_index++)];
        if (emitted) json << ",\n";
        json << "    {\"buffer\": 0, \"byteOffset\": " << span.position << ", \"byteLength\": " << span.position_bytes
             << ", \"target\": 34962},\n    {\"buffer\": 0, \"byteOffset\": " << span.normal
             << ", \"byteLength\": " << span.normal_bytes
             << ", \"target\": 34962},\n    {\"buffer\": 0, \"byteOffset\": " << span.index
             << ", \"byteLength\": " << span.index_bytes << ", \"target\": 34963}";
        ++emitted;
    }
    json << "\n  ],\n  \"buffers\": [{\"uri\": \"" << bin_path.filename().string() << "\", \"byteLength\": " << offset
         << "}],\n";
    json << "  \"extras\": {\"rectangles\": " << rectangles << ", \"triangles\": " << triangles
         << ", \"building_surface\": " << result.surface_count
         << ", \"representation\": \"greedy-merged surface quads, no instancing and no interior fill\"}\n}\n";
    if (!json) throw std::runtime_error("写入 glTF 失败");
    std::cout << "表面网格 矩形 " << rectangles << " 三角形 " << triangles << "\n";
}

void export_merged_boxes(const FillResult& result, const std::string& gltf_path, bool surface_only) {
    const int dx = result.dims[0];
    const int dy = result.dims[1];
    const int dz = result.dims[2];
    std::vector<std::uint8_t> left = result.occupied;
    if (surface_only) {
        for (int i = 0; i < dx; ++i) {
            for (int j = 0; j < dy; ++j) {
                for (int k = 0; k < dz; ++k) {
                    if (!result.building(i, j, k)) left[(i * dy + j) * dz + k] = 0;
                }
            }
        }
    }
    const auto index = [dy, dz](int i, int j, int k) { return (i * dy + j) * dz + k; };
    struct Box {
        float translation[3];
        float scale[3];
    };
    std::vector<Box> boxes;
    boxes.reserve(65536);
    int covered = 0;
    for (int k = 0; k < dz; ++k) {
        for (int j = 0; j < dy; ++j) {
            for (int i = 0; i < dx; ++i) {
                if (left[index(i, j, k)] == 0) continue;
                int width = 1;
                while (i + width < dx && left[index(i + width, j, k)] != 0) ++width;
                int height = 1;
                while (j + height < dy) {
                    bool row = true;
                    for (int step = 0; step < width; ++step) {
                        if (left[index(i + step, j + height, k)] == 0) {
                            row = false;
                            break;
                        }
                    }
                    if (!row) break;
                    ++height;
                }
                int depth = 1;
                while (k + depth < dz) {
                    bool slab = true;
                    for (int yy = 0; yy < height && slab; ++yy) {
                        for (int step = 0; step < width; ++step) {
                            if (left[index(i + step, j + yy, k + depth)] == 0) {
                                slab = false;
                                break;
                            }
                        }
                    }
                    if (!slab) break;
                    ++depth;
                }
                for (int zz = 0; zz < depth; ++zz) {
                    for (int yy = 0; yy < height; ++yy) {
                        for (int step = 0; step < width; ++step) left[index(i + step, j + yy, k + zz)] = 0;
                    }
                }
                Box box{};
                box.translation[0] = static_cast<float>(result.origin[0] + i * result.voxel);
                box.translation[1] = static_cast<float>(result.origin[1] + j * result.voxel);
                box.translation[2] = static_cast<float>(result.origin[2] + k * result.voxel);
                box.scale[0] = static_cast<float>(width * result.voxel);
                box.scale[1] = static_cast<float>(height * result.voxel);
                box.scale[2] = static_cast<float>(depth * result.voxel);
                boxes.push_back(box);
                covered += width * height * depth;
            }
        }
    }
    const int expected = surface_only ? result.surface_count : result.occupied_count;
    if (covered != expected) throw std::runtime_error("合并方块没有覆盖全部目标体素");

    float translation_min[3] = {1e30f, 1e30f, 1e30f};
    float translation_max[3] = {-1e30f, -1e30f, -1e30f};
    float scale_min[3] = {1e30f, 1e30f, 1e30f};
    float scale_max[3] = {-1e30f, -1e30f, -1e30f};
    std::vector<float> translations(boxes.size() * 3);
    std::vector<float> scales(boxes.size() * 3);
    for (size_t box_index = 0; box_index < boxes.size(); ++box_index) {
        const Box& box = boxes[box_index];
        for (int axis = 0; axis < 3; ++axis) {
            translations[box_index * 3 + static_cast<size_t>(axis)] = box.translation[axis];
            scales[box_index * 3 + static_cast<size_t>(axis)] = box.scale[axis];
            translation_min[axis] = std::min(translation_min[axis], box.translation[axis]);
            translation_max[axis] = std::max(translation_max[axis], box.translation[axis]);
            scale_min[axis] = std::min(scale_min[axis], box.scale[axis]);
            scale_max[axis] = std::max(scale_max[axis], box.scale[axis]);
        }
    }

    const std::filesystem::path json_path(gltf_path);
    std::filesystem::create_directories(json_path.parent_path());
    const std::filesystem::path bin_path = json_path.parent_path() / (json_path.stem().string() + ".bin");
    const float cube[] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
    const std::uint16_t indices[] = {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
                                     3, 6, 2, 3, 7, 6, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5};
    const int translation_bytes = static_cast<int>(translations.size() * sizeof(float));
    const int scale_bytes = static_cast<int>(scales.size() * sizeof(float));
    std::ofstream bin(bin_path, std::ios::binary);
    if (!bin) throw std::runtime_error("无法写入 " + bin_path.string());
    write_f32(bin, cube, 24);
    bin.write(reinterpret_cast<const char*>(indices), sizeof(indices));
    write_f32(bin, translations.data(), translations.size());
    write_f32(bin, scales.data(), scales.size());
    if (!bin) throw std::runtime_error("写入合并方块失败");
    bin.close();

    const int count = static_cast<int>(boxes.size());
    const int triangles = count * 12;
    std::ofstream json(json_path);
    if (!json) throw std::runtime_error("无法写入 " + json_path.string());
    json << "{\n  \"asset\": {\"version\": \"2.0\", \"generator\": \"beidou_vdb native scanline\"},\n";
    json << "  \"extensionsUsed\": [\"EXT_mesh_gpu_instancing\"],\n";
    json << "  \"extensionsRequired\": [\"EXT_mesh_gpu_instancing\"],\n";
    json << "  \"scene\": 0,\n  \"scenes\": [{\"nodes\": [0]}],\n";
    json << "  \"nodes\": [{\"name\": \"occupied\", \"mesh\": 0, \"extensions\": {\"EXT_mesh_gpu_instancing\": "
            "{\"attributes\": {\"TRANSLATION\": 2, \"SCALE\": 3}}}}],\n";
    json << "  \"materials\": [{\"name\": \"" << (surface_only ? "surface" : "occupied")
         << "\", \"doubleSided\": true, \"pbrMetallicRoughness\": {\"baseColorFactor\": ["
         << (surface_only ? "0.78, 0.74, 0.66, 1" : "0.2, 0.53, 0.46, 1")
         << "], \"metallicFactor\": 0, \"roughnessFactor\": 1}}],\n";
    json << "  \"meshes\": [{\"name\": \"box\", \"primitives\": [{\"attributes\": {\"POSITION\": 0}, \"indices\": 1, "
            "\"material\": 0, \"mode\": 4}]}],\n";
    json << "  \"accessors\": [\n";
    json << "    {\"bufferView\": 0, \"componentType\": 5126, \"count\": 8, \"type\": \"VEC3\", \"min\": [0, 0, 0], "
            "\"max\": [1, 1, 1]},\n";
    json << "    {\"bufferView\": 1, \"componentType\": 5123, \"count\": 36, \"type\": \"SCALAR\"},\n";
    json << "    {\"bufferView\": 2, \"componentType\": 5126, \"count\": " << count << ", \"type\": \"VEC3\", \"min\": [";
    for (int axis = 0; axis < 3; ++axis) {
        if (axis) json << ", ";
        write_json_number(json, translation_min[axis]);
    }
    json << "], \"max\": [";
    for (int axis = 0; axis < 3; ++axis) {
        if (axis) json << ", ";
        write_json_number(json, translation_max[axis]);
    }
    json << "]},\n    {\"bufferView\": 3, \"componentType\": 5126, \"count\": " << count
         << ", \"type\": \"VEC3\", \"min\": [";
    for (int axis = 0; axis < 3; ++axis) {
        if (axis) json << ", ";
        write_json_number(json, scale_min[axis]);
    }
    json << "], \"max\": [";
    for (int axis = 0; axis < 3; ++axis) {
        if (axis) json << ", ";
        write_json_number(json, scale_max[axis]);
    }
    json << "]}\n  ],\n  \"bufferViews\": [\n";
    json << "    {\"buffer\": 0, \"byteOffset\": 0, \"byteLength\": 96, \"target\": 34962},\n";
    json << "    {\"buffer\": 0, \"byteOffset\": 96, \"byteLength\": 72, \"target\": 34963},\n";
    json << "    {\"buffer\": 0, \"byteOffset\": 168, \"byteLength\": " << translation_bytes << ", \"target\": 34962},\n";
    json << "    {\"buffer\": 0, \"byteOffset\": " << (168 + translation_bytes) << ", \"byteLength\": " << scale_bytes
         << ", \"target\": 34962}\n  ],\n";
    json << "  \"buffers\": [{\"uri\": \"" << bin_path.filename().string() << "\", \"byteLength\": "
         << (168 + translation_bytes + scale_bytes) << "}],\n";
    json << "  \"extras\": {\"voxels\": " << expected << ", \"boxes\": " << count
         << ", \"triangles\": " << triangles
         << ", \"representation\": \"one instanced cube per solid rectangular block; adjacent occupied voxels are "
            "merged\"}\n}\n";
    if (!json) throw std::runtime_error("写入 glTF 失败");
    std::cout << "合并方块 " << count << " 个，三角形 " << triangles << "\n";
}

}  // namespace beidou
