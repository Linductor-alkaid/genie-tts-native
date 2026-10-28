// src/text/jieba.cpp —— jieba 0.42.1 分词算法的 C++ 直译。
//
// 参照 jieba/__init__.py（get_DAG / calc / __cut_DAG）与
// finalseg/__init__.py（viterbi / __cut），MIT。与上游逐行对应的点
// 都写在注释里；两个最容易走样的地方：
//   1. calc 的平局语义 —— Python `max((v, x) for x in ...)` 在值相等时
//      取**更大的 x**（升序遍历 + 大于等于替换）；
//   2. viterbi 的 PrevStatus 列表顺序 —— 平局取列表**最后一个**状态。
// oracle（tools/g2p/verify_jieba.py）以 jieba Python 为基准逐条验证。

#include "text/jieba.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <utility>

#include "genie/errors.hpp"
#include "genie/utf8.hpp"

namespace genie::text::jieba {
namespace {

constexpr std::uint32_t kVersion = 1;
// 上游 finalseg.MIN_FLOAT：缺失概率的哨兵值。
constexpr double kMinFloat = -3.14e100;
// 状态下标（与转换器的写出顺序一致）：B M E S。
constexpr std::size_t kB = 0, kM = 1, kE = 2, kS = 3;

struct Reader {
    const std::vector<unsigned char>& data;
    std::size_t pos = 0;

    void take(void* out, std::size_t n) {
        if (pos + n > data.size()) {
            throw_error(ErrorCode::InvalidArgument, "jieba container truncated");
        }
        std::memcpy(out, data.data() + pos, n);
        pos += n;
    }
    std::uint32_t u32() {
        std::uint32_t v = 0;
        take(&v, sizeof(v));
        return v;
    }
    std::uint64_t u64() {
        std::uint64_t v = 0;
        take(&v, sizeof(v));
        return v;
    }
    double f64() {
        double v = 0;
        take(&v, sizeof(v));
        return v;
    }
    std::uint8_t u8() {
        std::uint8_t v = 0;
        take(&v, sizeof(v));
        return v;
    }
    std::string sized_string() {
        const auto len = u8();
        std::string s(len, '\0');
        take(s.data(), len);
        return s;
    }
};

[[nodiscard]] std::vector<std::string> decode_chars(std::string_view text) {
    std::vector<std::string> chars;
    std::size_t offset = 0;
    utf8::Char ch;
    while (utf8::decode_next(text, offset, ch)) {
        chars.emplace_back(text.substr(ch.offset, ch.length));
    }
    return chars;
}

/// finalseg.viterbi 的直译：obs 为逐字序列，返回每个字的 BMES 标注。
[[nodiscard]] std::vector<std::size_t> viterbi(
    const std::vector<std::string>& obs, const std::array<double, 4>& start,
    const std::array<std::array<double, 4>, 4>& trans,
    const std::unordered_map<std::string, std::array<double, 4>>& emit) {
    const std::size_t n = obs.size();
    // PrevStatus（按下标 B M E S 排列；每行保持 Python 字面量的列表
    // 顺序 —— 平局取最后一个）：
    //   B: E S | M: M B | E: B M | S: S E
    constexpr std::size_t kPrevSize[4] = {2, 2, 2, 2};
    constexpr std::size_t kPrev[4][2] = {{kE, kS}, {kM, kB}, {kB, kM}, {kS, kE}};

    std::vector<std::array<double, 4>> v(n);
    std::vector<std::array<std::size_t, 4>> back(n);
    for (std::size_t y = 0; y < 4; ++y) {
        const auto it = emit.find(obs[0]);
        const double em = it != emit.end() ? (*it).second[y] : kMinFloat;
        v[0][y] = start[y] + em;
        back[0][y] = 0;
    }
    for (std::size_t t = 1; t < n; ++t) {
        const auto it = emit.find(obs[t]);
        for (std::size_t y = 0; y < 4; ++y) {
            const double em = it != emit.end() ? (*it).second[y] : kMinFloat;
            // 平局取后遍历的状态（与 Python 元组 max 一致）。
            double best = 0;
            std::size_t best_prev = 0;
            for (std::size_t p = 0; p < kPrevSize[y]; ++p) {
                const std::size_t y0 = kPrev[y][p];
                const double score = v[t - 1][y0] + trans[y0][y] + em;
                if (p == 0 || score >= best) {
                    best = score;
                    best_prev = y0;
                }
            }
            v[t][y] = best;
            back[t][y] = best_prev;
        }
    }
    // 终态只能在 E / S（平局取 S —— 'ES' 的最后一个）。
    std::size_t state = v[n - 1][kS] >= v[n - 1][kE] ? kS : kE;
    std::vector<std::size_t> path(n, kB);
    for (std::size_t t = n; t-- > 0;) {
        path[t] = state;
        state = back[t][state];
    }
    return path;
}

}  // namespace

Tokenizer Tokenizer::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw_error(ErrorCode::NotFound, "jieba model not found: " + path);
    }
    std::vector<unsigned char> data((std::istreambuf_iterator<char>(file)),
                                    std::istreambuf_iterator<char>());
    Reader r{data};
    char magic[4] = {};
    r.take(magic, 4);
    if (std::string_view(magic, 4) != "JIEB") {
        throw_error(ErrorCode::InvalidArgument,
                    "jieba container has wrong magic (not converted by convert_jieba.py?)");
    }
    if (r.u32() != kVersion) {
        throw_error(ErrorCode::InvalidArgument, "jieba container version mismatch");
    }

    Tokenizer tk;
    tk.total_ = r.u64();
    const auto word_count = r.u32();
    tk.dict_.reserve(word_count);
    for (std::uint32_t i = 0; i < word_count; ++i) {
        auto word = r.sized_string();
        const auto freq = r.u32();
        auto tag = r.sized_string();
        tk.dict_.emplace(std::move(word), DictEntry{freq, std::move(tag)});
    }
    for (auto& s : tk.start_) {
        s = r.f64();
    }
    for (auto& row : tk.trans_) {
        for (auto& v : row) {
            v = r.f64();
        }
    }
    const auto emit_count = r.u32();
    tk.emit_.reserve(emit_count);
    for (std::uint32_t i = 0; i < emit_count; ++i) {
        auto ch = r.sized_string();
        std::array<double, 4> probs{};
        for (auto& p : probs) {
            p = r.f64();
        }
        tk.emit_.emplace(std::move(ch), probs);
    }
    if (r.pos != data.size()) {
        throw_error(ErrorCode::InvalidArgument, "jieba container has trailing bytes");
    }
    return tk;
}

std::vector<Word> Tokenizer::cut(const std::string_view han_text) const {
    const auto chars = decode_chars(han_text);
    const std::size_t n = chars.size();
    if (n == 0) {
        return {};
    }

    // ---- get_DAG：从每个位置向前扫描词典（jieba 只查**完整词**成员，
    //      不建前缀表 —— 首个非词前缀即停，这是 jieba 的真实行为）。----
    std::vector<std::vector<std::size_t>> dag(n);
    for (std::size_t k = 0; k < n; ++k) {
        std::string frag = chars[k];
        std::size_t i = k;
        while (i < n && dict_.count(frag) != 0) {
            if (dict_.at(frag).freq != 0) {
                dag[k].push_back(i);
            }
            ++i;
            if (i < n) {
                frag += chars[i];
            }
        }
        if (dag[k].empty()) {
            dag[k].push_back(k);
        }
    }

    // ---- calc：逆序 DP。route 值 = log(freq or 1) - log(total) + 后继。----
    const double log_total = std::log(static_cast<double>(total_));
    std::vector<std::pair<double, std::size_t>> route(n + 1, {0.0, 0});
    for (std::size_t idx = n; idx-- > 0;) {
        double best_value = 0;
        std::size_t best_x = 0;
        bool first = true;
        for (const std::size_t x : dag[idx]) {
            std::string word;
            for (std::size_t k = idx; k <= x; ++k) {
                word += chars[k];
            }
            const auto it = dict_.find(word);
            const double freq = it != dict_.end() && it->second.freq != 0
                                    ? static_cast<double>(it->second.freq)
                                    : 1.0;
            const double value = std::log(freq) - log_total + route[x + 1].first;
            // 平局取更大的 x：升序遍历 + 大于等于（Python 元组 max 语义）。
            if (first || value >= best_value) {
                best_value = value;
                best_x = x;
                first = false;
            }
        }
        route[idx] = {best_value, best_x};
    }

    // ---- __cut_DAG：沿 route 走，单字缓存块按词典存在性走 HMM 或逐字。----
    std::vector<Word> words;
    // 上游 __cut_DAG 的缓存块产出：
    //   单字 → 直接产出；
    //   多字且**不是**完整词典词（`not self.FREQ.get(buf)`）→ HMM 细分；
    //   多字且是完整词典词 → 逐字产出（jieba 原版怪癖，原样保留）。
    const auto flush_buf = [&](std::vector<std::string>& buf) {
        if (buf.empty()) {
            return;
        }
        const auto tag_of = [&](const std::string& ch) {
            const auto it = dict_.find(ch);
            return it != dict_.end() ? it->second.tag : std::string{};
        };
        if (buf.size() == 1) {
            words.push_back(Word{buf.front(), tag_of(buf.front())});
            buf.clear();
            return;
        }
        std::string joined;
        for (const auto& c : buf) {
            joined += c;
        }
        // 上游 `not self.FREQ.get(buf)`：缺失**或 freq=0**（前缀片段）都
        // 算"不是完整词"，走 HMM。
        const auto entry = dict_.find(joined);
        if (entry == dict_.end() || entry->second.freq == 0) {
            const auto tags = viterbi(buf, start_, trans_, emit_);
            std::size_t begin = 0;
            for (std::size_t i = 0; i < buf.size(); ++i) {
                if (tags[i] == kE) {
                    std::string word;
                    for (std::size_t k = begin; k <= i; ++k) {
                        word += buf[k];
                    }
                    words.push_back(Word{std::move(word), {}});
                    begin = i + 1;
                } else if (tags[i] == kS) {
                    words.push_back(Word{buf[i], tag_of(buf[i])});
                    begin = i + 1;
                }
            }
            if (begin < buf.size()) {
                std::string word;
                for (std::size_t k = begin; k < buf.size(); ++k) {
                    word += buf[k];
                }
                words.push_back(Word{std::move(word), {}});
            }
        } else {
            for (const auto& c : buf) {
                words.push_back(Word{c, tag_of(c)});
            }
        }
        buf.clear();
    };

    std::vector<std::string> buf;
    std::size_t x = 0;
    while (x < n) {
        const std::size_t y = route[x].second + 1;
        if (y - x == 1) {
            buf.push_back(chars[x]);
        } else {
            flush_buf(buf);
            std::string word;
            for (std::size_t k = x; k < y; ++k) {
                word += chars[k];
            }
            const auto it = dict_.find(word);
            words.push_back(Word{std::move(word), it != dict_.end() ? it->second.tag : ""});
        }
        x = y;
    }
    flush_buf(buf);
    return words;
}

}  // namespace genie::text::jieba
