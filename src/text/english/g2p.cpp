// src/text/english/g2p.cpp —— 英文 G2P 主流程。
//
// 移植自上游 `G2P/English/EnglishG2P.py` 的 `CleanG2p.__call__` 与 `g2p()`：
//
//   1. 原始文本分词（`word_tokenize(text)`）
//   2. 规范化（`text_normalize(text)`）
//   3. 规范化后的文本再分词（`word_tokenize(normalized_text)`）
//   4. 两个词表对齐（上游有个 "I ... the first" 的特判）
//   5. 词性标注（只为 4 个同形异音词服务）
//   6. 逐词查发音：
//        - 不含小写字母 → 原样当音素（标点）
//        - 同形异音词 → 按词性挑读音
//        - 其余 → CMU 词典 → 's → 连字符 → 分词 → GRU 兜底
//   7. 过滤：丢掉空格、`<pad>`、`UW`、`</s>`、`<s>`；`<unk>` → `UNK`；
//      只保留符号表里有的音素；`'` → `-`
//
// 上游的符号表里 ARPAbet 音素是**大写**的（"AH0"、"IY1"…），而规范化
// 把文本转成了小写 —— 词典查出来的音素不受影响（词典里就是大写）。

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

#include "genie/symbols.hpp"
#include "text/english/english.hpp"

namespace genie::text::english {
namespace {

/// 上游 `CleanG2p._setup_homographs`：4 个同形异音词。
struct Homograph {
    std::string_view word;
    /// 词性以这个前缀开头时用第一个读音。
    std::string_view pos_prefix;
    std::vector<std::string> pron1;
    std::vector<std::string> pron2;
};

const std::vector<Homograph>& homographs() {
    static const std::vector<Homograph> kHomographs = {
        {"read", "VBD", {"R", "EH1", "D"}, {"R", "IY1", "D"}},
        {"complex",
         "JJ",
         {"K", "AH0", "M", "P", "L", "EH1", "K", "S"},
         {"K", "AA1", "M", "P", "L", "EH0", "K", "S"}},
        {"lead", "NN", {"L", "IY1", "D"}, {"L", "EH1", "D"}},
        {"presents",
         "VBZ",
         {"P", "R", "IY0", "Z", "EH1", "N", "T", "S"},
         {"P", "R", "EH1", "Z", "AH0", "N", "T", "S"}},
    };
    return kHomographs;
}

/// 上游 `_query_word` 里的 's 处理。
[[nodiscard]] std::vector<std::string> possessive_of(std::vector<std::string> base) {
    if (base.empty()) {
        return {};
    }
    const std::string& last = base.back();
    static const std::vector<std::string> kSibilants = {"S", "Z", "SH", "ZH", "CH", "JH"};
    static const std::vector<std::string> kVoiceless = {"P", "T", "K", "F", "TH"};
    if (std::find(kSibilants.begin(), kSibilants.end(), last) != kSibilants.end()) {
        base.push_back("AH0");
        base.push_back("Z");
        return base;
    }
    if (std::find(kVoiceless.begin(), kVoiceless.end(), last) != kVoiceless.end()) {
        base.push_back("S");
        return base;
    }
    base.push_back("Z");
    return base;
}

class EnglishG2p {
public:
    EnglishG2p(const CmuDict& dict, const Seq2SeqG2p& seq2seq, const WordSegmenter& segmenter,
               const PosTagger* tagger)
        : dict_(dict), seq2seq_(seq2seq), segmenter_(segmenter), tagger_(tagger) {}

    [[nodiscard]] std::vector<std::string> run(std::string_view text) const;

private:
    [[nodiscard]] std::vector<std::string> query_word(std::string_view word) const;

    const CmuDict& dict_;
    const Seq2SeqG2p& seq2seq_;
    const WordSegmenter& segmenter_;
    const PosTagger* tagger_;
};

std::vector<std::string> EnglishG2p::query_word(std::string_view word) const {
    // 上游的词是**小写**的（规范化阶段已经 lower 过了）。
    std::string lower(word);
    for (char& c : lower) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    const auto found = dict_.lookup(lower);
    if (!found.empty()) {
        return found;
    }

    // 's 结尾
    if (lower.size() > 2 && lower.compare(lower.size() - 2, 2, "'s") == 0) {
        std::vector<std::string> base = query_word(lower.substr(0, lower.size() - 2));
        if (!base.empty()) {
            return possessive_of(std::move(base));
        }
    }

    // 连字符
    if (lower.find('-') != std::string::npos && lower.size() > 1) {
        std::vector<std::string> parts;
        std::size_t start = 0;
        while (start <= lower.size()) {
            const std::size_t dash = lower.find('-', start);
            const std::size_t end = dash == std::string::npos ? lower.size() : dash;
            if (end > start) {
                parts.push_back(lower.substr(start, end - start));
            }
            if (dash == std::string::npos) {
                break;
            }
            start = dash + 1;
        }
        if (parts.size() > 1) {
            std::vector<std::string> result;
            for (const auto& part : parts) {
                const auto phonemes = query_word(part);
                result.insert(result.end(), phonemes.begin(), phonemes.end());
            }
            if (!result.empty()) {
                return result;
            }
        }
    }

    // ngram 分词后再查
    if (segmenter_.available()) {
        const auto segments = segmenter_.segment(lower);
        if (segments.size() > 1) {
            std::string joined;
            for (const auto& segment : segments) {
                joined += segment;
            }
            if (joined == lower) {
                std::vector<std::string> result;
                for (const auto& segment : segments) {
                    const auto phonemes = query_word(segment);
                    result.insert(result.end(), phonemes.begin(), phonemes.end());
                }
                if (!result.empty()) {
                    return result;
                }
            }
        }
    }

    // GRU 兜底
    return seq2seq_.predict(lower);
}

std::vector<std::string> EnglishG2p::run(std::string_view text) const {
    const auto original_words = tokenize(text);
    const std::string normalized_text = text_normalize(text);
    const auto normalized_words = tokenize(normalized_text);

    // 上游的两词表对齐：只有 "I" + "the first" 这一个特判。
    std::vector<std::string> corrected;
    corrected.reserve(normalized_words.size());
    std::size_t original_index = 0;
    std::size_t normalized_index = 0;
    while (original_index < original_words.size() && normalized_index < normalized_words.size()) {
        bool special = false;
        if (original_words[original_index].text == "I" &&
            normalized_index + 1 < normalized_words.size() &&
            normalized_words[normalized_index].text == "the" &&
            normalized_words[normalized_index + 1].text == "first") {
            special = true;
        }
        if (special) {
            corrected.push_back("I");
            ++original_index;
            normalized_index += 2;
        } else {
            corrected.push_back(normalized_words[normalized_index].text);
            ++original_index;
            ++normalized_index;
        }
    }
    while (normalized_index < normalized_words.size()) {
        corrected.push_back(normalized_words[normalized_index].text);
        ++normalized_index;
    }
    if (corrected.empty()) {
        return {};
    }

    // 词性标注（只为同形异音词）
    std::vector<std::string> tags;
    if (tagger_ != nullptr && tagger_->available()) {
        tags = tagger_->tag(corrected);
    }

    std::vector<std::string> phonemes;
    for (std::size_t i = 0; i < corrected.size(); ++i) {
        const std::string& word = corrected[i];
        std::vector<std::string> pron;
        // 上游：`re.search("[a-z]", word) is None` → 原样当音素
        const bool has_lowercase =
            std::any_of(word.begin(), word.end(), [](char c) { return c >= 'a' && c <= 'z'; });
        if (!has_lowercase) {
            pron.push_back(word);
        } else {
            const Homograph* homograph = nullptr;
            for (const auto& candidate : homographs()) {
                if (candidate.word == word) {
                    homograph = &candidate;
                    break;
                }
            }
            if (homograph != nullptr) {
                const std::string tag = i < tags.size() ? tags[i] : std::string();
                const bool first =
                    tag.compare(0, homograph->pos_prefix.size(), homograph->pos_prefix) == 0;
                pron = first ? homograph->pron1 : homograph->pron2;
            } else {
                pron = query_word(word);
            }
        }
        phonemes.insert(phonemes.end(), pron.begin(), pron.end());
    }
    return phonemes;
}

}  // namespace

std::vector<std::string> phonemes(std::string_view text, const CmuDict& dict,
                                  const Seq2SeqG2p& seq2seq, const WordSegmenter& segmenter,
                                  const PosTagger* tagger) {
    const EnglishG2p g2p(dict, seq2seq, segmenter, tagger);
    std::vector<std::string> raw = g2p.run(text);

    // 上游 `g2p()` 的过滤：
    //   - 丢掉 " "、`<pad>`、`UW`、`</s>`、`<s>`
    //   - `<unk>` → `UNK`
    //   - `replace_phs`：只保留符号表里有的；`'` → `-`
    static const std::vector<std::string> kUndesired = {" ", "<pad>", "UW", "</s>", "<s>"};
    std::vector<std::string> result;
    result.reserve(raw.size());
    for (auto& phoneme : raw) {
        if (std::find(kUndesired.begin(), kUndesired.end(), phoneme) != kUndesired.end()) {
            continue;
        }
        if (phoneme == "<unk>") {
            phoneme = "UNK";
        }
        if (phoneme == "'") {
            phoneme = "-";
        }
        if (!symbols::contains(phoneme)) {
            continue;
        }
        result.push_back(std::move(phoneme));
    }
    return result;
}

}  // namespace genie::text::english
