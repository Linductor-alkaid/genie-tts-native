// src/text/english/seq2seq.cpp —— 英文 G2P 的 seq2seq GRU 兜底。
//
// 上游 `G2P/English/EnglishG2P.py` 的 `CleanG2p` 在 CMU 词典查不到词时，
// 用一个小的 GRU seq2seq 猜发音：
//
//   - 编码器：字符 embedding → GRU（输入 256 维、隐层 256 维）
//   - 解码器：自回归，最多 20 步，每步一个音素
//
// 权重来自 g2p_en 的 `checkpoint20.npz`（MIT，见
// third_party/english/LICENSE-g2p_en），由
// tools/deploy/convert_english_g2p.py 转成裸二进制
// third_party/english/english_g2p.bin。
//
// 权重不在时 `predict()` 返回空 —— 上游这时会抛异常（模型文件缺失），
// 本实现降级为"这个词没有发音"，让整句合成继续。

#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "genie/errors.hpp"
#include "text/english/english.hpp"

namespace genie::text::english {
namespace {

constexpr char kMagic[] = "GENIEG2P";
constexpr std::uint32_t kVersion = 1;

/// 词表（上游 `CleanG2p._setup_nn_components`）。
const std::vector<std::string>& graphemes() {
    static const std::vector<std::string> kGraphemes = {
        "<pad>", "<unk>", "</s>", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
        "m",     "n",     "o",    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"};
    return kGraphemes;
}

const std::vector<std::string>& phoneme_table() {
    static const std::vector<std::string> kPhonemes = {
        "<pad>", "<unk>", "<s>", "</s>", "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0",
        "AH1",   "AH2",   "AO0", "AO1",  "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2",
        "B",     "CH",    "D",   "DH",   "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0",
        "EY1",   "EY2",   "F",   "G",    "HH",  "IH0", "IH1", "IH2", "IY0", "IY1", "IY2",
        "JH",    "K",     "L",   "M",    "N",   "NG",  "OW0", "OW1", "OW2", "OY0", "OY1",
        "OY2",   "P",     "R",   "S",    "SH",  "T",   "TH",  "UH0", "UH1", "UH2", "UW",
        "UW0",   "UW1",   "UW2", "V",    "W",   "Y",   "Z",   "ZH"};
    return kPhonemes;
}

/// 上游 `_grucell` 的一个 cell。
///
/// @param x        输入向量（256）
/// @param h        隐状态（256），会被就地更新
/// @param w_ih     (768, 256)
/// @param w_hh     (768, 256)
/// @param b_ih     (768)
/// @param b_hh     (768)
void gru_cell(const float* x, std::vector<float>& h, const float* w_ih, const float* w_hh,
              const float* b_ih, const float* b_hh) {
    constexpr std::size_t kDim = 256;
    constexpr std::size_t kRows = 768;
    constexpr std::size_t kReset = kRows * 2 / 3;  // 512：rz 的长度

    std::vector<float> rzn_ih(kRows);
    std::vector<float> rzn_hh(kRows);
    for (std::size_t i = 0; i < kRows; ++i) {
        float sum_ih = b_ih[i];
        float sum_hh = b_hh[i];
        const float* row_ih = w_ih + i * kDim;
        const float* row_hh = w_hh + i * kDim;
        for (std::size_t j = 0; j < kDim; ++j) {
            sum_ih += x[j] * row_ih[j];
            sum_hh += h[j] * row_hh[j];
        }
        rzn_ih[i] = sum_ih;
        rzn_hh[i] = sum_hh;
    }

    std::vector<float> next(kDim);
    for (std::size_t i = 0; i < kDim; ++i) {
        // rz = sigmoid(rz_ih + rz_hh)；r = rz[:256]，z = rz[256:512]
        const float gate = 1.0f / (1.0f + std::exp(-(rzn_ih[i] + rzn_hh[i])));
        const float gate_z = 1.0f / (1.0f + std::exp(-(rzn_ih[kDim + i] + rzn_hh[kDim + i])));
        const float n_ih = rzn_ih[kReset + i];
        const float n_hh = rzn_hh[kReset + i];
        const float n = std::tanh(n_ih + gate * n_hh);
        next[i] = (1.0f - gate_z) * n + gate_z * h[i];
    }
    h.swap(next);
}

}  // namespace

bool Seq2SeqG2p::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    std::vector<unsigned char> data((std::istreambuf_iterator<char>(file)),
                                    std::istreambuf_iterator<char>());
    if (data.size() < sizeof(kMagic) - 1) {
        return false;
    }
    if (std::memcmp(data.data(), kMagic, sizeof(kMagic) - 1) != 0) {
        return false;
    }
    std::size_t pos = sizeof(kMagic) - 1;
    auto read_u32 = [&data, &pos]() -> std::uint32_t {
        if (pos + 4 > data.size()) {
            return 0;
        }
        std::uint32_t value = 0;
        std::memcpy(&value, data.data() + pos, 4);
        pos += 4;
        return value;
    };
    auto read_f32_array = [&](std::size_t count) {
        std::vector<float> values;
        if (pos + count * 4 > data.size()) {
            return values;
        }
        values.resize(count);
        std::memcpy(values.data(), data.data() + pos, count * 4);
        pos += count * 4;
        return values;
    };

    const std::uint32_t version = read_u32();
    if (version != kVersion) {
        return false;
    }
    const std::uint32_t phoneme_count = read_u32();
    if (phoneme_count != phoneme_table().size()) {
        return false;
    }
    for (std::uint32_t i = 0; i < phoneme_count; ++i) {
        const std::uint32_t length = read_u32();
        if (pos + length > data.size()) {
            return false;
        }
        pos += length;
    }

    GruWeights weights;
    // 容器里每个数组前面有 rows / cols 两个 u32 —— 必须读掉，
    // 否则后面所有数组都会错位 8 字节。
    auto read_dims = [&](std::uint32_t& rows, std::uint32_t& cols) {
        rows = read_u32();
        cols = read_u32();
    };
    auto load_matrix = [&](std::vector<float>& out, std::uint32_t rows, std::uint32_t cols) {
        std::uint32_t actual_rows = 0;
        std::uint32_t actual_cols = 0;
        read_dims(actual_rows, actual_cols);
        if (actual_rows != rows || actual_cols != cols) {
            return false;
        }
        out = read_f32_array(static_cast<std::size_t>(rows) * cols);
        return out.size() == static_cast<std::size_t>(rows) * cols;
    };
    const std::uint32_t dim = 256;
    const std::uint32_t rows = 768;
    if (!load_matrix(weights.enc_emb, 29, dim)) {
        return false;
    }
    if (!load_matrix(weights.enc_w_ih, rows, dim) || !load_matrix(weights.enc_w_hh, rows, dim)) {
        return false;
    }
    {
        std::uint32_t r = 0;
        std::uint32_t c = 0;
        read_dims(r, c);
        if (r != 1 || c != rows) {
            return false;
        }
        weights.enc_b_ih = read_f32_array(rows);
        read_dims(r, c);
        if (r != 1 || c != rows) {
            return false;
        }
        weights.enc_b_hh = read_f32_array(rows);
    }
    if (weights.enc_b_ih.size() != rows || weights.enc_b_hh.size() != rows) {
        return false;
    }
    if (!load_matrix(weights.dec_emb, phoneme_count, dim)) {
        return false;
    }
    if (!load_matrix(weights.dec_w_ih, rows, dim) || !load_matrix(weights.dec_w_hh, rows, dim)) {
        return false;
    }
    {
        std::uint32_t r = 0;
        std::uint32_t c = 0;
        read_dims(r, c);
        if (r != 1 || c != rows) {
            return false;
        }
        weights.dec_b_ih = read_f32_array(rows);
        read_dims(r, c);
        if (r != 1 || c != rows) {
            return false;
        }
        weights.dec_b_hh = read_f32_array(rows);
    }
    if (weights.dec_b_ih.size() != rows || weights.dec_b_hh.size() != rows) {
        return false;
    }
    if (!load_matrix(weights.fc_w, phoneme_count, dim)) {
        return false;
    }
    {
        std::uint32_t r = 0;
        std::uint32_t c = 0;
        read_dims(r, c);
        if (r != 1 || c != phoneme_count) {
            return false;
        }
        weights.fc_b = read_f32_array(phoneme_count);
    }
    if (weights.fc_b.size() != phoneme_count) {
        return false;
    }

    encoder_ = std::move(weights);
    phonemes_ = phoneme_table();
    return true;
}

std::vector<std::string> Seq2SeqG2p::predict(std::string_view word) const {
    if (encoder_.enc_emb.empty() || encoder_.fc_w.empty()) {
        return {};
    }
    constexpr std::size_t kDim = 256;

    // 编码：`list(word.lower()) + ["</s>"]`
    std::string lower(word);
    for (char& c : lower) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    std::vector<std::uint32_t> ids;
    ids.reserve(lower.size() + 1);
    for (const char c : lower) {
        std::uint32_t id = 1;  // "<unk>"
        for (std::size_t g = 0; g < graphemes().size(); ++g) {
            if (graphemes()[g].size() == 1 && graphemes()[g][0] == c) {
                id = static_cast<std::uint32_t>(g);
                break;
            }
        }
        ids.push_back(id);
    }
    ids.push_back(2);  // "</s>"

    // 编码器前向
    std::vector<float> h(kDim, 0.0f);
    std::vector<float> x(kDim);
    for (const std::uint32_t id : ids) {
        std::copy(encoder_.enc_emb.begin() + static_cast<std::ptrdiff_t>(id) * kDim,
                  encoder_.enc_emb.begin() + static_cast<std::ptrdiff_t>(id + 1) * kDim, x.begin());
        gru_cell(x.data(), h, encoder_.enc_w_ih.data(), encoder_.enc_w_hh.data(),
                 encoder_.enc_b_ih.data(), encoder_.enc_b_hh.data());
    }

    // 解码器：最多 20 步
    std::vector<std::string> result;
    const std::uint32_t start = 2;  // "<s>"
    std::copy(encoder_.dec_emb.begin() + static_cast<std::ptrdiff_t>(start) * kDim,
              encoder_.dec_emb.begin() + static_cast<std::ptrdiff_t>(start + 1) * kDim, x.begin());
    for (int step = 0; step < 20; ++step) {
        gru_cell(x.data(), h, encoder_.dec_w_ih.data(), encoder_.dec_w_hh.data(),
                 encoder_.dec_b_ih.data(), encoder_.dec_b_hh.data());
        std::size_t best = 0;
        float best_value = 0.0f;
        for (std::size_t p = 0; p < phonemes_.size(); ++p) {
            const float* row = encoder_.fc_w.data() + p * kDim;
            float logit = encoder_.fc_b[p];
            for (std::size_t j = 0; j < kDim; ++j) {
                logit += h[j] * row[j];
            }
            if (p == 0 || logit > best_value) {
                best_value = logit;
                best = p;
            }
        }
        if (best == 3) {  // "</s>"
            break;
        }
        result.push_back(phonemes_[best]);
        const auto row = static_cast<std::size_t>(best) * kDim;
        std::copy(encoder_.dec_emb.begin() + static_cast<std::ptrdiff_t>(row),
                  encoder_.dec_emb.begin() + static_cast<std::ptrdiff_t>(row + kDim), x.begin());
    }
    return result;
}

}  // namespace genie::text::english
