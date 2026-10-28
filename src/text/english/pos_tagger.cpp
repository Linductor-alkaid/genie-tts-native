// src/text/english/pos_tagger.cpp —— nltk averaged perceptron 词性标注的移植。
//
// 上游（Genie-TTS）用 `nltk.pos_tag(corrected_words)` 给英文句子标词性。
// 它的唯一用途是给 4 个同形异音词挑读音（read / complex / lead /
// presents），但标签是**整句**算出来的 —— 前一个词的词性会作为特征
// 影响当前词，所以不能只做局部判断。
//
// nltk 的实现（`nltk/tag/perceptron.py`）：
//   - `tagdict` 里有这个词就直接用（1,549 个高频词）
//   - 否则抽 13 个特征，与权重做点积，取分数最高的标签
//   - 平局时取**字典序最大**的标签（`max(classes, key=(score, label))`）
//
// 权重来自 `averaged_perceptron_tagger_eng`（Apache-2.0），由
// tools/deploy/convert_pos_tagger.py 打包成 third_party/english/pos_tagger.bin。

#include <algorithm>
#include <cstring>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include "genie/errors.hpp"
#include "text/english/english.hpp"

namespace genie::text::english {
namespace {

constexpr char kMagic[] = "GENIEPOS";
constexpr std::uint32_t kVersion = 1;

struct Reader {
    const std::vector<unsigned char>& data;
    std::size_t pos = 0;

    void take(void* out, std::size_t n) {
        if (pos + n > data.size()) {
            throw_error(ErrorCode::InvalidArgument, "pos tagger container truncated");
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
    float f32() {
        float v = 0;
        take(&v, sizeof(v));
        return v;
    }
    std::string sized_string() {
        const auto len = u32();
        std::string s(len, '\0');
        take(s.data(), len);
        return s;
    }
};

/// nltk 的 `PerceptronTagger.normalize`。
[[nodiscard]] std::string normalize_tag_word(std::string_view word) {
    if (word.find('-') != std::string_view::npos && !word.empty() && word.front() != '-') {
        return "!HYPHEN";
    }
    const bool all_digits = !word.empty() && std::all_of(word.begin(), word.end(), [](char c) {
        return c >= '0' && c <= '9';
    });
    if (all_digits && word.size() == 4) {
        return "!YEAR";
    }
    if (!word.empty() && word[0] >= '0' && word[0] <= '9') {
        return "!DIGITS";
    }
    std::string out(word);
    for (char& c : out) {
        if (c >= 'A' && c <= 'Z') {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return out;
}

/// 取最后 3 个字符（Python 的 `word[-3:]`）。
[[nodiscard]] std::string suffix3(std::string_view word) {
    if (word.size() <= 3) {
        return std::string(word);
    }
    return std::string(word.substr(word.size() - 3));
}

}  // namespace

PosTagger PosTagger::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw_error(ErrorCode::InvalidArgument, "pos tagger container not found: " + path);
    }
    std::vector<unsigned char> data((std::istreambuf_iterator<char>(file)),
                                    std::istreambuf_iterator<char>());
    if (data.size() < sizeof(kMagic) - 1) {
        throw_error(ErrorCode::InvalidArgument, "pos tagger container too small");
    }
    if (std::memcmp(data.data(), kMagic, sizeof(kMagic) - 1) != 0) {
        throw_error(ErrorCode::InvalidArgument, "pos tagger container magic mismatch");
    }

    PosTagger tagger;
    Reader reader{data, sizeof(kMagic) - 1};
    const std::uint32_t version = reader.u32();
    if (version != kVersion) {
        throw_error(ErrorCode::InvalidArgument, "pos tagger container version mismatch");
    }
    const std::uint32_t class_count = reader.u32();
    tagger.classes_.reserve(class_count);
    for (std::uint32_t i = 0; i < class_count; ++i) {
        tagger.classes_.push_back(reader.sized_string());
    }

    const std::uint32_t tagdict_count = reader.u32();
    tagger.tagdict_.reserve(tagdict_count);
    for (std::uint32_t i = 0; i < tagdict_count; ++i) {
        std::string word = reader.sized_string();
        const std::uint8_t label = reader.u8();
        tagger.tagdict_.emplace_back(std::move(word), label);
    }
    std::sort(tagger.tagdict_.begin(), tagger.tagdict_.end());

    const std::uint32_t feature_count = reader.u32();
    tagger.features_.reserve(feature_count);
    for (std::uint32_t i = 0; i < feature_count; ++i) {
        std::string feature = reader.sized_string();
        const std::uint32_t weight_count = reader.u32();
        FeatureWeights weights;
        weights.weights.reserve(weight_count);
        for (std::uint32_t k = 0; k < weight_count; ++k) {
            const std::uint8_t label = reader.u8();
            const float weight = reader.f32();
            weights.weights.emplace_back(label, weight);
        }
        tagger.features_.emplace_back(std::move(feature), std::move(weights));
    }
    std::sort(tagger.features_.begin(), tagger.features_.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    return tagger;
}

std::string PosTagger::normalize_word(std::string_view word) const {
    return normalize_tag_word(word);
}

const std::pair<std::string, std::uint8_t>* PosTagger::find_tagdict(std::string_view word) const {
    std::size_t low = 0;
    std::size_t high = tagdict_.size();
    while (low < high) {
        const std::size_t mid = low + (high - low) / 2;
        if (tagdict_[mid].first < word) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    if (low >= tagdict_.size() || tagdict_[low].first != word) {
        return nullptr;
    }
    return &tagdict_[low];
}

const PosTagger::FeatureWeights* PosTagger::find_feature(std::string_view feature) const {
    // 手工二分：`std::lower_bound` 的比较器要同时接受 (元素, 元素) 和
    // (元素, 值) 两种调用，string_view 与 string 混用会推导不出来。
    std::size_t low = 0;
    std::size_t high = features_.size();
    while (low < high) {
        const std::size_t mid = low + (high - low) / 2;
        if (features_[mid].first < feature) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    if (low >= features_.size() || features_[low].first != feature) {
        return nullptr;
    }
    return &features_[low].second;
}

std::vector<std::string> PosTagger::tag(const std::vector<std::string>& words) const {
    std::vector<std::string> tags;
    tags.reserve(words.size());
    if (classes_.empty()) {
        tags.assign(words.size(), std::string());
        return tags;
    }

    // nltk 的 context：START 两个 + 归一化后的词 + END 两个
    std::vector<std::string> context;
    context.reserve(words.size() + 4);
    context.emplace_back("-START-");
    context.emplace_back("-START2-");
    for (const auto& word : words) {
        context.push_back(normalize_word(word));
    }
    context.emplace_back("-END-");
    context.emplace_back("-END2-");

    std::string prev = "-START-";
    std::string prev2 = "-START2-";
    for (std::size_t i = 0; i < words.size(); ++i) {
        // tagdict 命中就直接用
        // tagdict 命中就直接用（手工二分：`std::lower_bound` 的比较器要同时
        // 接受 (元素, 元素) 与 (元素, 值) 两种调用，混用会推导不出来）。
        const auto dict_it = find_tagdict(words[i]);
        if (dict_it != nullptr) {
            tags.push_back(classes_[dict_it->second]);
            prev2 = prev;
            prev = tags.back();
            continue;
        }

        const std::size_t j = i + 2;  // len(START)
        const std::string& word = words[i];
        std::vector<std::string> features;
        features.reserve(13);
        features.push_back("bias");
        features.push_back("i suffix " + suffix3(word));
        features.push_back(std::string("i pref1 ") + (word.empty() ? "" : word.substr(0, 1)));
        features.push_back("i-1 tag " + prev);
        features.push_back("i-2 tag " + prev2);
        features.push_back("i tag+i-2 tag " + prev + " " + prev2);
        features.push_back("i word " + context[j]);
        features.push_back("i-1 tag+i word " + prev + " " + context[j]);
        features.push_back("i-1 word " + context[j - 1]);
        features.push_back("i-1 suffix " + suffix3(context[j - 1]));
        features.push_back("i-2 word " + context[j - 2]);
        features.push_back("i+1 word " + context[j + 1]);
        features.push_back("i+1 suffix " + suffix3(context[j + 1]));
        features.push_back("i+2 word " + context[j + 2]);

        std::vector<float> scores(classes_.size(), 0.0f);
        for (const auto& feature : features) {
            const FeatureWeights* weights = find_feature(feature);
            if (weights == nullptr) {
                continue;
            }
            for (const auto& entry : weights->weights) {
                if (entry.first < scores.size()) {
                    scores[entry.first] += entry.second;
                }
            }
        }

        // `max(self.classes, key=lambda label: (scores[label], label))`
        // → 分数最高；平局取字典序最大的标签
        std::size_t best = 0;
        for (std::size_t k = 1; k < scores.size(); ++k) {
            if (scores[k] > scores[best] ||
                (scores[k] == scores[best] && classes_[k] > classes_[best])) {
                best = k;
            }
        }
        tags.push_back(classes_[best]);
        prev2 = prev;
        prev = tags.back();
    }
    return tags;
}

}  // namespace genie::text::english
