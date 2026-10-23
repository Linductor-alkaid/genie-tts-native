// include/genie/symbols.hpp
//
// 音素符号表。
//
// 这是整个项目**最不能出错**的一张表：模型输入的 text_seq 是符号 ID 序列，
// 只要这里的顺序错一位，模型收到的就是完全不同的音素序列 —— 而且不会报错，
// 只会合成出乱码。
//
// 因此本表不是"抄"上游的列表，而是**重新实现上游的生成规则**
// （上游 `src/genie_tts/G2P/SymbolsV2.py` 的 `create_master_symbol_list()`）。
// 规则是确定的，得到的 732 个符号必然一致；`tests/unit/test_symbols.cpp`
// 用计数 + 抽样 + 全表摘要把它钉死。
//
// 上游的生成规则（逐条对应）：
//   1. main_symbols = { "_" } ∪ 声母 ∪ (基础韵母 × 声调 1..5) ∪ 日语音素 ∪ 标点 ∪ ARPAbet
//   2. master_list  = sorted(main_symbols)                ← 按码位升序
//   3. master_list += "[", "]"
//   4. master_list += sorted(韩语音素字符集)
//   5. master_list += sorted(粤语音素集合)
//
// 注意：Python 的 sorted() 按 Unicode 码位排序，本实现按 UTF-8 字节序排序。
// 对本表的全部符号两者等价（唯一的多字节字符是 "…"(U+2026)，它排在所有
// ASCII 之后，也排在所有韩文/粤文符号之前 —— 与码位序一致）。

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace genie::symbols {

/// 完整符号表（V2）。顺序即符号 ID 的顺序，不可变更。
[[nodiscard]] const std::vector<std::string>& master_v2();

/// 符号 → ID。表里没有则返回 -1。
[[nodiscard]] std::int64_t id_of(std::string_view symbol) noexcept;

/// ID → 符号。越界返回空字符串。
[[nodiscard]] std::string_view name_of(std::int64_t id) noexcept;

/// 符号 → ID，表里没有则返回空。
[[nodiscard]] std::optional<std::int64_t> find(std::string_view symbol) noexcept;

/// 符号是否在表里。
[[nodiscard]] bool contains(std::string_view symbol) noexcept;

/// 符号总数。当前为 732。
[[nodiscard]] std::size_t size() noexcept;

}  // namespace genie::symbols
