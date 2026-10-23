// genie/sha256.hpp — SHA-256 摘要
//
// 用于模型包完整性校验：模型文件被替换是最常见的部署事故，
// 而 ONNX 加载失败时的报错信息通常看不出根因。
//
// 自行实现是为了避免为一个摘要算法引入 OpenSSL 之类的依赖。

#ifndef GENIE_SHA256_HPP
#define GENIE_SHA256_HPP

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace genie {

/// 32 字节 SHA-256 摘要。
using Sha256Digest = std::array<std::uint8_t, 32>;

/// 计算 SHA-256。
[[nodiscard]] Sha256Digest sha256(const std::vector<std::uint8_t>& data);
[[nodiscard]] Sha256Digest sha256(std::string_view data);

/// 计算 SHA-256 并返回小写十六进制字符串。
[[nodiscard]] std::string sha256_hex(std::string_view data);

/// 计算文件内容的 SHA-256。
///
/// @throws genie::Error IoError —— 文件不存在或读取失败
[[nodiscard]] Sha256Digest sha256_file(const std::string& path);

/// 把摘要渲染为小写十六进制。
[[nodiscard]] std::string to_hex(const Sha256Digest& digest);

}  // namespace genie

#endif  // GENIE_SHA256_HPP
