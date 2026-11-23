#pragma once

#include "scanfill.hpp"

#include <string>
#include <vector>

namespace beidou {

std::vector<Triangle> load_gltf_triangles(const std::string& path);

}  // namespace beidou
