#include "gltf.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace beidou {
namespace {

struct Json {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool b = false;
    double n = 0;
    std::string s;
    std::vector<Json> a;
    std::vector<std::pair<std::string, Json>> o;

    const Json* find(const std::string& key) const {
        for (const auto& item : o) {
            if (item.first == key) return &item.second;
        }
        return nullptr;
    }
    const Json& at(const std::string& key) const {
        const Json* value = find(key);
        if (value == nullptr) throw std::runtime_error("glTF 缺少字段 " + key);
        return *value;
    }
    const Json& operator[](size_t index) const {
        if (index >= a.size()) throw std::runtime_error("glTF 数组越界");
        return a[index];
    }
    int as_int() const {
        if (type != Type::Number) throw std::runtime_error("glTF 字段不是数字");
        return static_cast<int>(std::llround(n));
    }
    double as_num() const {
        if (type != Type::Number) throw std::runtime_error("glTF 字段不是数字");
        return n;
    }
    const std::string& as_str() const {
        if (type != Type::String) throw std::runtime_error("glTF 字段不是字符串");
        return s;
    }
};

class Parser {
public:
    explicit Parser(const std::string& text) : text_(text) {}

    Json parse() {
        skip();
        Json value = parse_value();
        skip();
        if (i_ != text_.size()) throw std::runtime_error("glTF JSON 末尾有多余内容");
        return value;
    }

private:
    const std::string& text_;
    size_t i_ = 0;

    void skip() {
        while (i_ < text_.size()) {
            const char c = text_[i_];
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
            ++i_;
        }
    }

    char peek() const {
        if (i_ >= text_.size()) throw std::runtime_error("glTF JSON 意外结束");
        return text_[i_];
    }

    Json parse_value() {
        skip();
        const char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return parse_string();
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        return parse_number();
    }

    Json parse_object() {
        ++i_;
        Json value;
        value.type = Json::Type::Object;
        skip();
        if (peek() == '}') {
            ++i_;
            return value;
        }
        while (true) {
            skip();
            Json key = parse_string();
            skip();
            if (peek() != ':') throw std::runtime_error("glTF JSON 对象缺少冒号");
            ++i_;
            Json child = parse_value();
            value.o.emplace_back(std::move(key.s), std::move(child));
            skip();
            const char c = peek();
            ++i_;
            if (c == '}') break;
            if (c != ',') throw std::runtime_error("glTF JSON 对象格式错误");
        }
        return value;
    }

    Json parse_array() {
        ++i_;
        Json value;
        value.type = Json::Type::Array;
        skip();
        if (peek() == ']') {
            ++i_;
            return value;
        }
        while (true) {
            value.a.push_back(parse_value());
            skip();
            const char c = peek();
            ++i_;
            if (c == ']') break;
            if (c != ',') throw std::runtime_error("glTF JSON 数组格式错误");
        }
        return value;
    }

    Json parse_string() {
        if (peek() != '"') throw std::runtime_error("glTF JSON 缺少字符串");
        ++i_;
        Json value;
        value.type = Json::Type::String;
        while (i_ < text_.size()) {
            char c = text_[i_++];
            if (c == '"') return value;
            if (c != '\\') {
                value.s.push_back(c);
                continue;
            }
            if (i_ >= text_.size()) throw std::runtime_error("glTF JSON 字符串转义中断");
            const char e = text_[i_++];
            switch (e) {
                case '"':
                case '\\':
                case '/':
                    value.s.push_back(e);
                    break;
                case 'b':
                    value.s.push_back('\b');
                    break;
                case 'f':
                    value.s.push_back('\f');
                    break;
                case 'n':
                    value.s.push_back('\n');
                    break;
                case 'r':
                    value.s.push_back('\r');
                    break;
                case 't':
                    value.s.push_back('\t');
                    break;
                case 'u': {
                    if (i_ + 4 > text_.size()) throw std::runtime_error("glTF JSON \\u 中断");
                    int code = 0;
                    for (int k = 0; k < 4; ++k) {
                        const char h = text_[i_++];
                        code <<= 4;
                        if (h >= '0' && h <= '9') code += h - '0';
                        else if (h >= 'a' && h <= 'f') code += h - 'a' + 10;
                        else if (h >= 'A' && h <= 'F') code += h - 'A' + 10;
                        else throw std::runtime_error("glTF JSON \\u 非法");
                    }
                    if (code < 0x80) value.s.push_back(static_cast<char>(code));
                    else if (code < 0x800) {
                        value.s.push_back(static_cast<char>(0xc0 | (code >> 6)));
                        value.s.push_back(static_cast<char>(0x80 | (code & 0x3f)));
                    } else {
                        value.s.push_back(static_cast<char>(0xe0 | (code >> 12)));
                        value.s.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
                        value.s.push_back(static_cast<char>(0x80 | (code & 0x3f)));
                    }
                    break;
                }
                default:
                    throw std::runtime_error("glTF JSON 未知转义");
            }
        }
        throw std::runtime_error("glTF JSON 字符串未结束");
    }

    Json parse_bool() {
        Json value;
        value.type = Json::Type::Bool;
        if (text_.compare(i_, 4, "true") == 0) {
            value.b = true;
            i_ += 4;
            return value;
        }
        if (text_.compare(i_, 5, "false") == 0) {
            value.b = false;
            i_ += 5;
            return value;
        }
        throw std::runtime_error("glTF JSON 布尔值非法");
    }

    Json parse_null() {
        if (text_.compare(i_, 4, "null") != 0) throw std::runtime_error("glTF JSON 空值非法");
        i_ += 4;
        return Json{};
    }

    Json parse_number() {
        const char* start = text_.c_str() + i_;
        char* end = nullptr;
        Json value;
        value.type = Json::Type::Number;
        value.n = std::strtod(start, &end);
        if (end == start) throw std::runtime_error("glTF JSON 数字非法");
        i_ += static_cast<size_t>(end - start);
        return value;
    }
};

int b64_value(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

std::vector<unsigned char> base64_decode(const std::string& text) {
    std::string filtered;
    filtered.reserve(text.size());
    for (char c : text) {
        if (c == '=' || c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
        filtered.push_back(c);
    }
    if (filtered.size() % 4 == 1) throw std::runtime_error("base64 长度非法");
    std::vector<unsigned char> out;
    out.reserve(filtered.size() / 4 * 3);
    size_t i = 0;
    while (i + 4 <= filtered.size()) {
        const int a = b64_value(filtered[i]);
        const int b = b64_value(filtered[i + 1]);
        const int c = b64_value(filtered[i + 2]);
        const int d = b64_value(filtered[i + 3]);
        if (a < 0 || b < 0 || c < 0 || d < 0) throw std::runtime_error("base64 字符非法");
        out.push_back(static_cast<unsigned char>((a << 2) | (b >> 4)));
        out.push_back(static_cast<unsigned char>((b << 4) | (c >> 2)));
        out.push_back(static_cast<unsigned char>((c << 6) | d));
        i += 4;
    }
    if (i + 2 == filtered.size()) {
        const int a = b64_value(filtered[i]);
        const int b = b64_value(filtered[i + 1]);
        if (a < 0 || b < 0) throw std::runtime_error("base64 字符非法");
        out.push_back(static_cast<unsigned char>((a << 2) | (b >> 4)));
    } else if (i + 3 == filtered.size()) {
        const int a = b64_value(filtered[i]);
        const int b = b64_value(filtered[i + 1]);
        const int c = b64_value(filtered[i + 2]);
        if (a < 0 || b < 0 || c < 0) throw std::runtime_error("base64 字符非法");
        out.push_back(static_cast<unsigned char>((a << 2) | (b >> 4)));
        out.push_back(static_cast<unsigned char>((b << 4) | (c >> 2)));
    }
    return out;
}

std::vector<unsigned char> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("无法读取 " + path);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    const std::string bytes = buffer.str();
    return std::vector<unsigned char>(bytes.begin(), bytes.end());
}

std::vector<std::vector<unsigned char>> buffer_bytes(const Json& gltf, const std::string& directory) {
    std::vector<std::vector<unsigned char>> blobs;
    const Json* buffers = gltf.find("buffers");
    if (buffers == nullptr) return blobs;
    for (const Json& buf : buffers->a) {
        const Json* uri = buf.find("uri");
        if (uri == nullptr) throw std::runtime_error("不支持 GLB 内嵌 buffer");
        const std::string& text = uri->as_str();
        std::vector<unsigned char> data;
        if (text.rfind("data:", 0) == 0) {
            const auto comma = text.find(',');
            if (comma == std::string::npos) throw std::runtime_error("data URI 缺少逗号");
            data = base64_decode(text.substr(comma + 1));
        } else {
            data = read_file(directory + "/" + text);
        }
        const Json* expected = buf.find("byteLength");
        if (expected != nullptr && data.size() < static_cast<size_t>(expected->as_int())) {
            throw std::runtime_error("buffer 短于 byteLength");
        }
        blobs.push_back(std::move(data));
    }
    return blobs;
}

int component_size(int component_type) {
    switch (component_type) {
        case 5120:
        case 5121:
            return 1;
        case 5122:
        case 5123:
            return 2;
        case 5125:
        case 5126:
            return 4;
        default:
            throw std::runtime_error("不支持的 componentType");
    }
}

int component_count(const std::string& type) {
    if (type == "SCALAR") return 1;
    if (type == "VEC2") return 2;
    if (type == "VEC3") return 3;
    if (type == "VEC4") return 4;
    if (type == "MAT4") return 16;
    throw std::runtime_error("不支持的 accessor 类型 " + type);
}

double read_component(const unsigned char* ptr, int component_type) {
    switch (component_type) {
        case 5120: {
            std::int8_t value;
            std::memcpy(&value, ptr, 1);
            return value;
        }
        case 5121:
            return ptr[0];
        case 5122: {
            std::int16_t value;
            std::memcpy(&value, ptr, 2);
            return value;
        }
        case 5123: {
            std::uint16_t value;
            std::memcpy(&value, ptr, 2);
            return value;
        }
        case 5125: {
            std::uint32_t value;
            std::memcpy(&value, ptr, 4);
            return static_cast<double>(value);
        }
        case 5126: {
            float value;
            std::memcpy(&value, ptr, 4);
            return static_cast<double>(value);
        }
        default:
            throw std::runtime_error("不支持的 componentType");
    }
}

std::vector<double> read_accessor(const Json& gltf, const std::vector<std::vector<unsigned char>>& blobs, int index) {
    const Json& accessor = gltf.at("accessors")[static_cast<size_t>(index)];
    if (accessor.find("bufferView") == nullptr) throw std::runtime_error("accessor 没有 bufferView");
    const Json& view = gltf.at("bufferViews")[static_cast<size_t>(accessor.at("bufferView").as_int())];
    const int component_type = accessor.at("componentType").as_int();
    const int count = accessor.at("count").as_int();
    const int ncomp = component_count(accessor.at("type").as_str());
    const int comp = component_size(component_type);
    const int tightly = comp * ncomp;
    const int stride = view.find("byteStride") != nullptr ? view.at("byteStride").as_int() : tightly;
    const int buffer_index = view.find("buffer") != nullptr ? view.at("buffer").as_int() : 0;
    if (buffer_index < 0 || static_cast<size_t>(buffer_index) >= blobs.size()) {
        throw std::runtime_error("buffer 下标越界");
    }
    const auto& buffer = blobs[static_cast<size_t>(buffer_index)];
    const int view_offset = view.find("byteOffset") != nullptr ? view.at("byteOffset").as_int() : 0;
    const int acc_offset = accessor.find("byteOffset") != nullptr ? accessor.at("byteOffset").as_int() : 0;
    const size_t start = static_cast<size_t>(view_offset + acc_offset);
    std::vector<double> out(static_cast<size_t>(count) * static_cast<size_t>(ncomp));
    for (int i = 0; i < count; ++i) {
        const size_t row = start + static_cast<size_t>(i) * static_cast<size_t>(stride);
        if (row + static_cast<size_t>(tightly) > buffer.size()) throw std::runtime_error("accessor 超出 buffer");
        for (int c = 0; c < ncomp; ++c) {
            out[static_cast<size_t>(i) * static_cast<size_t>(ncomp) + static_cast<size_t>(c)] =
                read_component(buffer.data() + row + static_cast<size_t>(c * comp), component_type);
        }
    }
    return out;
}

void quat_matrix(double x, double y, double z, double w, double m[3][3]) {
    const double norm = x * x + y * y + z * z + w * w;
    if (norm < 1e-20) {
        m[0][0] = m[1][1] = m[2][2] = 1;
        m[0][1] = m[0][2] = m[1][0] = m[1][2] = m[2][0] = m[2][1] = 0;
        return;
    }
    const double s = 2.0 / norm;
    const double xx = x * x * s, yy = y * y * s, zz = z * z * s;
    const double xy = x * y * s, xz = x * z * s, yz = y * z * s;
    const double wx = w * x * s, wy = w * y * s, wz = w * z * s;
    m[0][0] = 1.0 - (yy + zz);
    m[0][1] = xy - wz;
    m[0][2] = xz + wy;
    m[1][0] = xy + wz;
    m[1][1] = 1.0 - (xx + zz);
    m[1][2] = yz - wx;
    m[2][0] = xz - wy;
    m[2][1] = yz + wx;
    m[2][2] = 1.0 - (xx + yy);
}

struct Mat4 {
    double m[4][4]{};
};

Mat4 identity() {
    Mat4 out;
    out.m[0][0] = out.m[1][1] = out.m[2][2] = out.m[3][3] = 1;
    return out;
}

Mat4 multiply(const Mat4& a, const Mat4& b) {
    Mat4 out;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            out.m[r][c] = a.m[r][0] * b.m[0][c] + a.m[r][1] * b.m[1][c] + a.m[r][2] * b.m[2][c] +
                          a.m[r][3] * b.m[3][c];
        }
    }
    return out;
}

Mat4 local_matrix(const Json& node) {
    if (const Json* matrix = node.find("matrix")) {
        if (matrix->a.size() != 16) throw std::runtime_error("节点 matrix 不是 16 个数");
        Mat4 out;
        for (int e = 0; e < 16; ++e) out.m[e % 4][e / 4] = matrix->a[static_cast<size_t>(e)].as_num();
        return out;
    }
    double translation[3] = {0, 0, 0};
    double rotation[4] = {0, 0, 0, 1};
    double scale[3] = {1, 1, 1};
    if (const Json* value = node.find("translation")) {
        for (int i = 0; i < 3; ++i) translation[i] = value->a[static_cast<size_t>(i)].as_num();
    }
    if (const Json* value = node.find("rotation")) {
        for (int i = 0; i < 4; ++i) rotation[i] = value->a[static_cast<size_t>(i)].as_num();
    }
    if (const Json* value = node.find("scale")) {
        for (int i = 0; i < 3; ++i) scale[i] = value->a[static_cast<size_t>(i)].as_num();
    }
    double r[3][3];
    quat_matrix(rotation[0], rotation[1], rotation[2], rotation[3], r);
    Mat4 out = identity();
    for (int col = 0; col < 3; ++col) {
        for (int row = 0; row < 3; ++row) out.m[row][col] = r[row][col] * scale[col];
    }
    out.m[0][3] = translation[0];
    out.m[1][3] = translation[1];
    out.m[2][3] = translation[2];
    return out;
}

double det3(const Mat4& m) {
    const double a = m.m[0][0], b = m.m[0][1], c = m.m[0][2];
    const double d = m.m[1][0], e = m.m[1][1], f = m.m[1][2];
    const double g = m.m[2][0], h = m.m[2][1], i = m.m[2][2];
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

void transform_point(const Mat4& m, const double in[3], double out[3]) {
    out[0] = m.m[0][0] * in[0] + m.m[0][1] * in[1] + m.m[0][2] * in[2] + m.m[0][3];
    out[1] = m.m[1][0] * in[0] + m.m[1][1] * in[1] + m.m[1][2] * in[2] + m.m[1][3];
    out[2] = m.m[2][0] * in[0] + m.m[2][1] * in[1] + m.m[2][2] * in[2] + m.m[2][3];
}

void append_mesh(const Json& gltf, const std::vector<std::vector<unsigned char>>& blobs, int mesh_index,
                 const Mat4& world, std::vector<Triangle>& triangles) {
    const bool flip = det3(world) < 0.0;
    const Json& mesh = gltf.at("meshes")[static_cast<size_t>(mesh_index)];
    const Json* primitives = mesh.find("primitives");
    if (primitives == nullptr) return;
    for (const Json& prim : primitives->a) {
        const int mode = prim.find("mode") != nullptr ? prim.at("mode").as_int() : 4;
        if (mode != 4) continue;
        const Json* attributes = prim.find("attributes");
        if (attributes == nullptr || attributes->find("POSITION") == nullptr) continue;
        const std::vector<double> local = read_accessor(gltf, blobs, attributes->at("POSITION").as_int());
        const size_t vertex_count = local.size() / 3;
        std::vector<double> world_pos(local.size());
        for (size_t v = 0; v < vertex_count; ++v) {
            double in[3] = {local[v * 3], local[v * 3 + 1], local[v * 3 + 2]};
            double out[3];
            transform_point(world, in, out);
            world_pos[v * 3] = out[0];
            world_pos[v * 3 + 1] = out[1];
            world_pos[v * 3 + 2] = out[2];
        }
        std::vector<double> indices;
        if (prim.find("indices") != nullptr) {
            indices = read_accessor(gltf, blobs, prim.at("indices").as_int());
        } else {
            indices.resize(vertex_count);
            for (size_t v = 0; v < vertex_count; ++v) indices[v] = static_cast<double>(v);
        }
        if (indices.size() % 3 != 0) throw std::runtime_error("三角形索引个数不是 3 的倍数");
        for (size_t t = 0; t < indices.size(); t += 3) {
            const int corner[3] = {static_cast<int>(std::llround(indices[t])),
                                   static_cast<int>(std::llround(indices[t + 1])),
                                   static_cast<int>(std::llround(indices[t + 2]))};
            Triangle tri{};
            for (int c = 0; c < 3; ++c) {
                if (corner[c] < 0 || static_cast<size_t>(corner[c]) >= vertex_count) {
                    throw std::runtime_error("三角形索引越界");
                }
                for (int axis = 0; axis < 3; ++axis) {
                    tri.v[c][axis] = world_pos[static_cast<size_t>(corner[c]) * 3 + static_cast<size_t>(axis)];
                }
            }
            const double e1[3] = {tri.v[1][0] - tri.v[0][0], tri.v[1][1] - tri.v[0][1], tri.v[1][2] - tri.v[0][2]};
            const double e2[3] = {tri.v[2][0] - tri.v[0][0], tri.v[2][1] - tri.v[0][1], tri.v[2][2] - tri.v[0][2]};
            double normal[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                                e1[0] * e2[1] - e1[1] * e2[0]};
            const double len = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
            if (len < 1e-12) continue;
            const double sign = flip ? -1.0 : 1.0;
            tri.n[0] = sign * normal[0] / len;
            tri.n[1] = sign * normal[1] / len;
            tri.n[2] = sign * normal[2] / len;
            triangles.push_back(tri);
        }
    }
}

std::string directory_of(const std::string& path) {
    const auto slash = path.find_last_of('/');
    if (slash == std::string::npos) return ".";
    if (slash == 0) return "/";
    return path.substr(0, slash);
}

}  // namespace

std::vector<Triangle> load_gltf_triangles(const std::string& path) {
    const std::vector<unsigned char> raw = read_file(path);
    if (raw.size() >= 4 && raw[0] == 'g' && raw[1] == 'l' && raw[2] == 'T' && raw[3] == 'F') {
        throw std::runtime_error(path + " 是二进制 glTF，读取器只接受 JSON .gltf");
    }
    const std::string text(raw.begin(), raw.end());
    Parser parser(text);
    const Json gltf = parser.parse();
    const auto blobs = buffer_bytes(gltf, directory_of(path));
    const Json* nodes = gltf.find("nodes");
    if (nodes == nullptr) throw std::runtime_error(path + " 没有节点");

    std::vector<int> roots;
    const Json* scenes = gltf.find("scenes");
    if (scenes == nullptr || scenes->a.empty()) {
        for (size_t i = 0; i < nodes->a.size(); ++i) roots.push_back(static_cast<int>(i));
    } else {
        int scene_index = 0;
        if (const Json* scene = gltf.find("scene")) scene_index = scene->as_int();
        const Json& scene = scenes->a.at(static_cast<size_t>(scene_index));
        if (const Json* scene_nodes = scene.find("nodes")) {
            for (const Json& node : scene_nodes->a) roots.push_back(node.as_int());
        }
    }

    std::vector<Triangle> triangles;
    std::vector<char> seen(nodes->a.size(), 0);
    const auto visit = [&](auto&& self, int index, const Mat4& parent) -> void {
        if (index < 0 || static_cast<size_t>(index) >= nodes->a.size()) throw std::runtime_error("节点下标越界");
        if (seen[static_cast<size_t>(index)]) return;
        seen[static_cast<size_t>(index)] = 1;
        const Json& node = nodes->a[static_cast<size_t>(index)];
        const Mat4 world = multiply(parent, local_matrix(node));
        if (const Json* mesh = node.find("mesh")) append_mesh(gltf, blobs, mesh->as_int(), world, triangles);
        if (const Json* children = node.find("children")) {
            for (const Json& child : children->a) self(self, child.as_int(), world);
        }
    };
    for (int root : roots) visit(visit, root, identity());
    if (triangles.empty()) throw std::runtime_error(path + " 没有三角形");
    return triangles;
}

}  // namespace beidou
