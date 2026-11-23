#pragma once

#include "scanfill.hpp"

#include <string>

namespace beidou {

void export_occupancy_gltf(const FillResult& result, const std::string& gltf_path, bool include_interior,
                           bool split_colors);

void export_merged_boxes(const FillResult& result, const std::string& gltf_path, bool surface_only);

void export_surface_mesh(const FillResult& result, const std::string& gltf_path);

}  // namespace beidou
