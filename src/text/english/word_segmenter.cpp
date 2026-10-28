// src/text/english/word_segmenter.cpp —— ngram Viterbi 分词的移植。
//
// 上游 `G2P/English/WordSegment.py` 的 `WordSegmenter`：当一个词在 CMU
// 词典里查不到时，先用 ngram 词典把它切成已知的词，再逐个查发音。
//
// 算法（上游 `isegment`）：
//   - `_clean`：转小写，只留 [a-z0-9]
//   - `search`：对每个前缀（长度 1..24）算对数概率，取**最大**的一条路径
//   - 打分：`score(word, previous)` —— unigram 频次 / 总频次；
//     有 bigram 且前词在 unigram 里时用 bigram 概率
//   - 长文本按 250 字符切块，块间保留最后 5 个词做上下文
//
// 数据（unigrams.txt / bigrams.txt）来自上游的 English_G2P 目录，
// 体积很大，不随仓库发布 —— 没有数据时这个回退不可用。

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "text/english/english.hpp"

namespace genie::text::english {
namespace {

/// 上游 `WordSegmenter.TOTAL`
constexpr double kTotal = 1024908267229.0;
/// 上游 `WordSegmenter.LIMIT`
constexpr std::size_t kLimit = 24;

[[nodiscard]] bool is_alphabet(char c) noexcept {
    return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
}

/// 上游 `_clean`：转小写 + 只留 [a-z0-9]
[[nodiscard]] std::string clean(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const char c : text) {
        if (c >= 'A' && c <= 'Z') {
            out.push_back(static_cast<char>(c - 'A' + 'a'));
            continue;
        }
        if (is_alphabet(c)) {
            out.push_back(c);
        }
    }
    return out;
}

struct ParseResult {
    std::vector<std::pair<std::string, double>> entries;
};

[[nodiscard]] bool parse_counts(const std::string& path, ParseResult& out) {
    std::ifstream file(path);
    if (!file) {
        return false;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const std::size_t tab = line.find('\t');
        if (tab == std::string::npos || tab == 0) {
            continue;
        }
        const std::string word = line.substr(0, tab);
        const std::string number = line.substr(tab + 1);
        if (word.empty() || number.empty()) {
            continue;
        }
        try {
            out.entries.emplace_back(word, std::stod(number));
        } catch (const std::exception&) {
            continue;
        }
    }
    return true;
}

/// 手工二分查一个词的频次（`std::lower_bound` 的比较器在混用
/// string / string_view 时推导不出来）。
[[nodiscard]] const double* find_count(const std::vector<std::pair<std::string, double>>& table,
                                       std::string_view word) {
    std::size_t low = 0;
    std::size_t high = table.size();
    while (low < high) {
        const std::size_t mid = low + (high - low) / 2;
        if (table[mid].first < word) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    if (low >= table.size() || table[low].first != word) {
        return nullptr;
    }
    return &table[low].second;
}

}  // namespace

bool WordSegmenter::load(const std::string& directory) {
    ParseResult unigrams;
    ParseResult bigrams;
    if (!parse_counts(directory + "/unigrams.txt", unigrams)) {
        return false;
    }
    if (!parse_counts(directory + "/bigrams.txt", bigrams)) {
        return false;
    }
    std::sort(unigrams.entries.begin(), unigrams.entries.end());
    std::sort(bigrams.entries.begin(), bigrams.entries.end());
    unigrams_ = std::move(unigrams.entries);
    bigrams_ = std::move(bigrams.entries);
    total_ = kTotal;
    return true;
}

double WordSegmenter::score(std::string_view word, std::string_view previous) const {
    if (previous.empty()) {
        const double* count = find_count(unigrams_, word);
        if (count != nullptr) {
            return *count / total_;
        }
        return 10.0 / (total_ * std::pow(10.0, static_cast<double>(word.size())));
    }
    std::string bigram(previous);
    bigram.push_back(' ');
    bigram.append(word);
    const double* bigram_count = find_count(bigrams_, bigram);
    const double* prev_count = find_count(unigrams_, previous);
    if (bigram_count != nullptr && prev_count != nullptr) {
        return *bigram_count / total_ / score(previous, "");
    }
    return score(word, "");
}

std::vector<std::string> WordSegmenter::segment(std::string_view word) const {
    std::vector<std::string> result;
    if (!available()) {
        return result;
    }
    const std::string text = clean(word);
    if (text.empty()) {
        return result;
    }

    // 上游用 (suffix, prefix) 做记忆化
    std::map<std::pair<std::string, std::string>, std::pair<double, std::vector<std::string>>> memo;

    // 递归搜索（上游用 Python 递归，靠 250 字符分块规避递归深度限制）
    struct Search {
        const WordSegmenter& owner;
        std::map<std::pair<std::string, std::string>, std::pair<double, std::vector<std::string>>>&
            memo;

        std::pair<double, std::vector<std::string>> run(const std::string& text,
                                                        const std::string& previous) {
            if (text.empty()) {
                return {0.0, {}};
            }
            const auto key = std::make_pair(text, previous);
            const auto cached = memo.find(key);
            if (cached != memo.end()) {
                return cached->second;
            }
            bool has_best = false;
            double best_score = 0.0;
            std::vector<std::string> best_words;
            const std::size_t limit = std::min(text.size(), kLimit);
            for (std::size_t pos = 1; pos <= limit; ++pos) {
                const std::string prefix = text.substr(0, pos);
                const std::string suffix = text.substr(pos);
                const double prefix_score = std::log10(owner.score(prefix, previous));
                const auto suffix_result = run(suffix, prefix);
                const double total = prefix_score + suffix_result.first;
                std::vector<std::string> words;
                words.push_back(prefix);
                words.insert(words.end(), suffix_result.second.begin(), suffix_result.second.end());
                // Python 的 `max()` 取**第一个**最大值；元组比较先比分数，
                // 分数相同再比词表（字典序）。
                if (!has_best || total > best_score ||
                    (total == best_score && words > best_words)) {
                    has_best = true;
                    best_score = total;
                    best_words = std::move(words);
                }
            }
            memo[key] = {best_score, best_words};
            return {best_score, best_words};
        }
    };

    Search search{*this, memo};
    result = search.run(text, "<s>").second;
    return result;
}

}  // namespace genie::text::english
