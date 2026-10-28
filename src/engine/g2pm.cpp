// src/engine/g2pm.cpp — g2pM 前向计算的 C++ 复刻。
//
// 参照 mozillazg/pypinyin-g2pM 的 g2pM.py（Apache-2.0）逐行移植：
// reverse_sequence / fw_lstm_cell / bw_lstm_cell / fc_layer / predict /
// __call__(char_split=True)。与上游有意的差异都写在对应位置。

#include "engine/g2pm.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <unordered_map>
#include <utility>

#include "genie/errors.hpp"
#include "genie/utf8.hpp"

namespace genie::g2pm {
namespace {

constexpr std::uint32_t kVersion = 2;  // v2: 附带词级读音修正表（polyphonic.pickle）
// 模型结构是训练时定死的（embed 64 / hidden 32），载入时校验。
constexpr std::size_t kEmbedDim = 64;
constexpr std::size_t kHidden = 32;
constexpr std::size_t kConcatDim = 2 * kHidden;  // 双向拼接后的隐状态维
constexpr std::size_t kGates = 4 * kHidden;      // i, f, g, o
// 上游 __call__ 里的特殊 token id（char2idx.pkl 固定值）。
constexpr std::uint32_t kBosId = 2;  // "시"
constexpr std::uint32_t kEosId = 3;  // "끝"
constexpr std::uint32_t kUnkId = 1;  // "<UNK>"

struct Reader {
    const std::vector<unsigned char>& data;
    std::size_t pos = 0;

    void take(void* out, std::size_t n) {
        if (pos + n > data.size()) {
            throw_error(ErrorCode::InvalidArgument, "g2pm container truncated");
        }
        std::memcpy(out, data.data() + pos, n);
        pos += n;
    }
    std::uint32_t u32() {
        std::uint32_t v = 0;
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
    std::vector<double> f32_as_f64(std::size_t count) {
        std::vector<float> raw(count);
        take(raw.data(), count * sizeof(float));
        return {raw.begin(), raw.end()};
    }
};

[[nodiscard]] double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

/// dot(x, W 的第 g 行) —— 对齐上游 `np.matmul(x, W.T)` 的逐元素乘加。
[[nodiscard]] double dot_row(const std::vector<double>& x, const std::vector<double>& w,
                             std::size_t in_dim, std::size_t g) {
    const double* row = w.data() + g * in_dim;
    return std::inner_product(x.begin(), x.end(), row, 0.0);
}

/// 单步 LSTM cell：上游 fw_lstm_cell / bw_lstm_cell 的共用体。
///
/// 权重按 [i f | g o] 两段拼装（上游把 4 组门拆成 if/go 两半再各分二），
/// 与 PyTorch 的 i,f,g,o 顺序一致。
void lstm_step(const std::vector<double>& x, const std::vector<double>& prev_h,
               const std::vector<double>& prev_c, const std::vector<double>& w_ih,
               const std::vector<double>& w_hh, const std::vector<double>& b_ih,
               const std::vector<double>& b_hh, std::vector<double>& h, std::vector<double>& c) {
    std::vector<double> ifgo_ih(kGates), ifgo_hh(kGates);
    for (std::size_t g = 0; g < kGates; ++g) {
        ifgo_ih[g] = dot_row(x, w_ih, kEmbedDim, g) + b_ih[g];
        ifgo_hh[g] = dot_row(prev_h, w_hh, kHidden, g) + b_hh[g];
    }

    h.resize(kHidden);
    c.resize(kHidden);
    for (std::size_t k = 0; k < kHidden; ++k) {
        const double i = sigmoid(ifgo_ih[k] + ifgo_hh[k]);
        const double f = sigmoid(ifgo_ih[kHidden + k] + ifgo_hh[kHidden + k]);
        const double g = std::tanh(ifgo_ih[2 * kHidden + k] + ifgo_hh[2 * kHidden + k]);
        const double o = sigmoid(ifgo_ih[3 * kHidden + k] + ifgo_hh[3 * kHidden + k]);
        c[k] = f * prev_c[k] + i * g;
        h[k] = o * std::tanh(c[k]);
    }
}

}  // namespace

Model Model::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw_error(ErrorCode::NotFound, "g2pm model not found: " + path);
    }
    std::vector<unsigned char> data((std::istreambuf_iterator<char>(file)),
                                    std::istreambuf_iterator<char>());

    Reader r{data};
    char magic[4] = {};
    r.take(magic, 4);
    if (std::string_view(magic, 4) != "G2PM") {
        throw_error(ErrorCode::InvalidArgument,
                    "g2pm container has wrong magic (not converted by convert_g2pm.py?)");
    }
    if (r.u32() != kVersion) {
        throw_error(ErrorCode::InvalidArgument, "g2pm container version mismatch");
    }

    Model model;
    model.vocab_size_ = r.u32();
    model.class_count_ = r.u32();
    const auto cedict_count = r.u32();

    const auto weights = r.f32_as_f64(static_cast<std::size_t>(model.vocab_size_) * kEmbedDim +
                                      2 * (kGates * kEmbedDim + kGates * kHidden + 2 * kGates) +
                                      kHidden * kConcatDim + kHidden +
                                      static_cast<std::size_t>(model.class_count_) * (kHidden + 1));
    // 按转换器写出的固定顺序切开。
    auto cursor = weights.begin();
    auto cut = [&](std::size_t n) {
        std::vector<double> out(cursor, cursor + static_cast<std::ptrdiff_t>(n));
        cursor += static_cast<std::ptrdiff_t>(n);
        return out;
    };
    model.embeddings_ = cut(static_cast<std::size_t>(model.vocab_size_) * kEmbedDim);
    model.w_ih_ = cut(kGates * kEmbedDim);
    model.w_hh_ = cut(kGates * kHidden);
    model.b_ih_ = cut(kGates);
    model.b_hh_ = cut(kGates);
    model.w_ih_rev_ = cut(kGates * kEmbedDim);
    model.w_hh_rev_ = cut(kGates * kHidden);
    model.b_ih_rev_ = cut(kGates);
    model.b_hh_rev_ = cut(kGates);
    model.fc0_w_ = cut(kHidden * kConcatDim);
    model.fc0_b_ = cut(kHidden);
    model.fc1_w_ = cut(static_cast<std::size_t>(model.class_count_) * kHidden);
    model.fc1_b_ = cut(model.class_count_);
    if (cursor != weights.end()) {
        throw_error(ErrorCode::InvalidArgument, "g2pm weight section size mismatch");
    }

    model.char_ids_.reserve(model.vocab_size_);
    for (std::uint32_t i = 0; i < model.vocab_size_; ++i) {
        auto ch = r.sized_string();
        const auto id = r.u32();
        model.char_ids_.emplace_back(std::move(ch), id);
    }
    model.idx_class_.reserve(model.class_count_);
    for (std::uint32_t i = 0; i < model.class_count_; ++i) {
        model.idx_class_.push_back(r.sized_string());
    }
    model.cedict_.reserve(cedict_count);
    for (std::uint32_t i = 0; i < cedict_count; ++i) {
        auto ch = r.sized_string();
        const auto n = r.u32();
        std::vector<std::string> prons;
        prons.reserve(n);
        for (std::uint32_t k = 0; k < n; ++k) {
            prons.push_back(r.sized_string());
        }
        model.cedict_.emplace_back(std::move(ch), std::move(prons));
    }

    // 词级读音修正表（v2）。转换器保证「值长度 = 词字数」，这里照查。
    const auto poly_count = r.u32();
    model.poly_entries_.reserve(poly_count);
    for (std::uint32_t i = 0; i < poly_count; ++i) {
        auto word = r.sized_string();
        const auto n = r.u32();
        std::vector<std::string> prons;
        prons.reserve(n);
        for (std::uint32_t k = 0; k < n; ++k) {
            prons.push_back(r.sized_string());
        }
        // word.size() 是字节数；表的对齐按**字符数**（转换器已校验，
        // 这里按 UTF-8 字符数复核）。
        const auto char_count =
            static_cast<std::size_t>(std::count_if(word.begin(), word.end(), [](char c) {
                return (static_cast<unsigned char>(c) & 0xC0) != 0x80;
            }));
        if (prons.size() != char_count) {
            throw_error(ErrorCode::InvalidArgument,
                        "g2pm polyphonic entry length mismatch: " + word);
        }
        model.poly_max_len_ = std::max(model.poly_max_len_, char_count);
        model.poly_entries_.emplace_back(std::move(word), std::move(prons));
    }
    if (r.pos != data.size()) {
        throw_error(ErrorCode::InvalidArgument, "g2pm container has trailing bytes");
    }

    // 词典排序后二分查找（段长有限，但排序让最坏情况也对数）。
    std::sort(model.char_ids_.begin(), model.char_ids_.end());
    std::sort(model.cedict_.begin(), model.cedict_.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    model.poly_lookup_.reserve(model.poly_entries_.size());
    for (const auto& [word, prons] : model.poly_entries_) {
        model.poly_lookup_.emplace(word, &prons);
    }
    return model;
}

namespace {

/// utf-8 字符串的按字符二分辅助：把单字读进来查词典。
template <typename Map>
const typename Map::value_type* find_char(const Map& sorted, std::string_view ch) {
    const auto it =
        std::lower_bound(sorted.begin(), sorted.end(), ch,
                         [](const auto& entry, std::string_view key) { return entry.first < key; });
    if (it != sorted.end() && it->first == ch) {
        return &*it;
    }
    return nullptr;
}

}  // namespace

std::vector<std::string> Model::pinyins_of(std::string_view segment) const {
    std::vector<std::string> chars;
    std::vector<std::uint32_t> ids;
    std::vector<std::size_t> poly_positions;
    std::vector<std::string> result;

    std::size_t offset = 0;
    utf8::Char ch;
    while (utf8::decode_next(segment, offset, ch)) {
        std::string piece(segment.substr(ch.offset, ch.length));
        chars.push_back(piece);
        const auto* entry = find_char(char_ids_, piece);
        ids.push_back(entry != nullptr ? entry->second : kUnkId);

        const auto* pron = find_char(cedict_, piece);
        if (pron == nullptr) {
            // 上游：不在 cedict 的字（生僻字、标点）原样透传。
            result.push_back(piece);
        } else if (pron->second.size() == 1) {
            result.push_back(pron->second.front());
        } else {
            poly_positions.push_back(chars.size() - 1);
            result.emplace_back();
        }
    }
    if (poly_positions.empty()) {
        return result;
    }

    // 上游 predict：句首尾插 BOS/EOS，多音字下标随之 +1。
    ids.insert(ids.begin(), kBosId);
    ids.push_back(kEosId);
    const std::size_t n = ids.size();

    // 前向 LSTM（原序）。
    std::vector<std::vector<double>> fw_h(n, std::vector<double>(kHidden, 0.0));
    std::vector<double> h(kHidden, 0.0), c(kHidden, 0.0);
    for (std::size_t t = 0; t < n; ++t) {
        const double* emb = embeddings_.data() + static_cast<std::size_t>(ids[t]) * kEmbedDim;
        std::vector<double> x(emb, emb + kEmbedDim);
        lstm_step(x, h, c, w_ih_, w_hh_, b_ih_, b_hh_, h, c);
        fw_h[t] = h;
    }
    // 后向 LSTM（反序输入，输出再翻回来）。
    std::vector<std::vector<double>> bw_h(n, std::vector<double>(kHidden, 0.0));
    std::fill(h.begin(), h.end(), 0.0);
    std::fill(c.begin(), c.end(), 0.0);
    for (std::size_t step = 0; step < n; ++step) {
        const std::size_t t = n - 1 - step;
        const double* emb = embeddings_.data() + static_cast<std::size_t>(ids[t]) * kEmbedDim;
        std::vector<double> x(emb, emb + kEmbedDim);
        lstm_step(x, h, c, w_ih_rev_, w_hh_rev_, b_ih_rev_, b_hh_rev_, h, c);
        bw_h[t] = h;
    }

    // 拼接双向隐状态 → FC(relu) → FC → argmax（上游 fc_layer / predict）。
    for (const auto raw_pos : poly_positions) {
        const std::size_t t = raw_pos + 1;  // BOS 占位
        std::vector<double> hidden(2 * kHidden);
        for (std::size_t k = 0; k < kHidden; ++k) {
            hidden[k] = fw_h[t][k];
            hidden[kHidden + k] = bw_h[t][k];
        }
        std::vector<double> fc0(kHidden);
        double best = 0.0;
        std::size_t best_class = 0;
        for (std::size_t k = 0; k < kHidden; ++k) {
            fc0[k] = std::max(0.0, dot_row(hidden, fc0_w_, kConcatDim, k) + fc0_b_[k]);
        }
        for (std::uint32_t k = 0; k < class_count_; ++k) {
            const double logit = dot_row(fc0, fc1_w_, kHidden, k) + fc1_b_[k];
            if (k == 0 || logit > best) {
                best = logit;
                best_class = k;
            }
        }
        result[raw_pos] = idx_class_[best_class];
    }

    return result;
}

void Model::correct_pronunciation(const std::vector<std::string>& chars,
                                  std::vector<std::string>& pinyins) const {
    if (pinyins.size() != chars.size() || poly_entries_.empty()) {
        return;
    }
    // 上游 correct_pronunciation 的近似：整词命中整体替换（无 jieba，
    // 用词表最长匹配近似词边界），未命中的字走单字兜底。
    std::size_t i = 0;
    while (i < chars.size()) {
        const std::size_t max_len = std::min(poly_max_len_, chars.size() - i);
        bool matched = false;
        for (std::size_t len = max_len; len >= 2; --len) {
            std::string key;
            for (std::size_t k = i; k < i + len; ++k) {
                key += chars[k];
            }
            if (const auto it = poly_lookup_.find(key); it != poly_lookup_.end()) {
                const auto& prons = *it->second;
                for (std::size_t k = 0; k < len; ++k) {
                    pinyins[i + k] = prons[k];
                }
                i += len;
                matched = true;
                break;
            }
        }
        if (!matched) {
            const auto it = poly_lookup_.find(chars[i]);
            if (it != poly_lookup_.end() && !it->second->empty()) {
                pinyins[i] = it->second->front();
            }
            ++i;
        }
    }
}

}  // namespace genie::g2pm
