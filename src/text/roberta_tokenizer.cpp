// src/text/roberta_tokenizer.cpp — BERT WordPiece 分词器的 C++ 复刻
//
// 对齐 HuggingFace tokenizers 的 BertNormalizer(clean_text=true,
// handle_chinese_chars=true, strip_accents=null, lowercase=true) →
// BertPreTokenizer → WordPiece([UNK], ##, max_chars=100) → [CLS]...[SEP]。
//
// 与上游的差异（记录在 docs/upstream-audit.md §14）：
//   - strip_accents 不做完整 NFD 分解，只覆盖 Latin-1/Latin-Extended-A
//     的常用预组合字符 —— TTS 中文输入里带变音符的拉丁文极罕见。
//   - 标点集合用 ASCII ispunct + 常用 CJK 标点；经文本规范化后
//     CJK 标点大多已被映射成 ASCII，这个集合只是兜底。
//   - 词表查找是线性扫描（vocab 21k 条，句子长度 ≤ 数百字符），
//     每次编码毫秒级，不值得上哈希表。

#include "text/roberta_tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <utility>

#include "genie/errors.hpp"
#include "genie/utf8.hpp"

namespace genie::text {
namespace {

using utf8::Char;

/// BertNormalizer 的 clean_text：控制字符清除，\t\n\r 变空格。
[[nodiscard]] std::string clean_text(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t offset = 0;
    Char ch;
    while (utf8::decode_next(text, offset, ch)) {
        const auto cp = ch.codepoint;
        if (cp == 0) {
            continue;  // NUL：直接丢弃
        }
        if (cp == '\t' || cp == '\n' || cp == '\r') {
            out.push_back(' ');
            continue;
        }
        if (cp >= 32 && cp != 127) {
            out.append(text.substr(ch.offset, ch.length));
        }
    }
    return out;
}

/// tokenizers 的 CJK 判定（BasicTokenizer._tokenize_chinese_chars 的区段表）。
[[nodiscard]] bool is_cjk_codepoint(std::uint32_t cp) noexcept {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

/// 给 CJK 字符两侧补空格（handle_chinese_chars）。
[[nodiscard]] std::string pad_cjk(std::string_view text) {
    std::string out;
    out.reserve(text.size() * 3);
    std::size_t offset = 0;
    Char ch;
    while (utf8::decode_next(text, offset, ch)) {
        if (is_cjk_codepoint(ch.codepoint)) {
            out.push_back(' ');
            out.append(text.substr(ch.offset, ch.length));
            out.push_back(' ');
        } else {
            out.append(text.substr(ch.offset, ch.length));
        }
    }
    return out;
}

/// strip_accents + lowercase 的覆盖范围：常用预组合拉丁字母 → 小写基字母。
/// 表外的码位原样保留（审计文档记录的已知差异）。
struct AccentEntry {
    std::uint32_t composed;
    char base;
};
constexpr std::array<AccentEntry, 61> kAccentTable{{
    {0x00C0, 'a'}, {0x00C1, 'a'}, {0x00C2, 'a'}, {0x00C3, 'a'}, {0x00C4, 'a'}, {0x00C5, 'a'},
    {0x00C7, 'c'}, {0x00C8, 'e'}, {0x00C9, 'e'}, {0x00CA, 'e'}, {0x00CB, 'e'}, {0x00CC, 'i'},
    {0x00CD, 'i'}, {0x00CE, 'i'}, {0x00CF, 'i'}, {0x00D1, 'n'}, {0x00D2, 'o'}, {0x00D3, 'o'},
    {0x00D4, 'o'}, {0x00D5, 'o'}, {0x00D6, 'o'}, {0x00D8, 'o'}, {0x00D9, 'u'}, {0x00DA, 'u'},
    {0x00DB, 'u'}, {0x00DC, 'u'}, {0x00DD, 'y'}, {0x00E0, 'a'}, {0x00E1, 'a'}, {0x00E2, 'a'},
    {0x00E3, 'a'}, {0x00E4, 'a'}, {0x00E5, 'a'}, {0x00E7, 'c'}, {0x00E8, 'e'}, {0x00E9, 'e'},
    {0x00EA, 'e'}, {0x00EB, 'e'}, {0x00EC, 'i'}, {0x00ED, 'i'}, {0x00EE, 'i'}, {0x00EF, 'i'},
    {0x00F1, 'n'}, {0x00F2, 'o'}, {0x00F3, 'o'}, {0x00F4, 'o'}, {0x00F5, 'o'}, {0x00F6, 'o'},
    {0x00F8, 'o'}, {0x00F9, 'u'}, {0x00FA, 'u'}, {0x00FB, 'u'}, {0x00FC, 'u'}, {0x00FD, 'y'},
    {0x00FF, 'y'}, {0x0100, 'a'}, {0x0101, 'a'}, {0x0112, 'e'}, {0x0113, 'e'}, {0x014C, 'o'},
    {0x014D, 'o'},
}};

[[nodiscard]] std::string strip_accents_and_lower(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t offset = 0;
    Char ch;
    while (utf8::decode_next(text, offset, ch)) {
        const auto cp = ch.codepoint;
        if (cp < 128) {
            out.push_back(static_cast<char>(std::tolower(static_cast<int>(cp))));
            continue;
        }
        const auto it = std::find_if(kAccentTable.begin(), kAccentTable.end(),
                                     [cp](const AccentEntry& e) { return e.composed == cp; });
        if (it != kAccentTable.end()) {
            out.push_back(it->base);
        } else {
            out.append(text.substr(ch.offset, ch.length));
        }
    }
    return out;
}

/// tokenizers 的 \p{Whitespace} 集合（常用区段）。
[[nodiscard]] bool is_unicode_space(std::uint32_t cp) noexcept {
    return cp == ' ' || (cp >= 0x09 && cp <= 0x0D) || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
           cp == 0x205F || cp == 0x3000;
}

/// 标点判定：ASCII ispunct + 常用 CJK 标点（规范化后的兜底集合）。
[[nodiscard]] bool is_punctuation(std::uint32_t cp) noexcept {
    if (cp < 128) {
        return std::ispunct(static_cast<int>(cp)) != 0;
    }
    return (cp >= 0x3001 && cp <= 0x3003) || (cp >= 0x3008 && cp <= 0x3011) ||
           (cp >= 0x3014 && cp <= 0x301F) || (cp >= 0xFF01 && cp <= 0xFF0F) ||
           (cp >= 0xFF1A && cp <= 0xFF20) || (cp >= 0xFF3B && cp <= 0xFF40) ||
           (cp >= 0xFF5B && cp <= 0xFF65);
}

/// BertPreTokenizer：先按空白切，再把标点逐个拆成单字符 token。
[[nodiscard]] std::vector<std::string> pretokenize(std::string_view text) {
    std::vector<std::string> words;
    std::string current;
    auto flush = [&] {
        if (!current.empty()) {
            words.push_back(std::move(current));
            current.clear();
        }
    };

    std::size_t offset = 0;
    Char ch;
    while (utf8::decode_next(text, offset, ch)) {
        if (is_unicode_space(ch.codepoint)) {
            flush();
        } else if (is_punctuation(ch.codepoint)) {
            flush();
            words.emplace_back(text.substr(ch.offset, ch.length));
        } else {
            current.append(text.substr(ch.offset, ch.length));
        }
    }
    flush();
    return words;
}

/// 回退一个 UTF-8 字符（end 停在字符首字节）。
void retreat_one_char(std::string_view word, std::size_t& end) noexcept {
    do {
        --end;
    } while (end > 0 && (static_cast<unsigned char>(word[end]) & 0xC0) == 0x80);
}

/// UTF-8 字符数。
[[nodiscard]] std::size_t char_count(std::string_view text) noexcept {
    std::size_t count = 0;
    std::size_t offset = 0;
    Char ch;
    while (utf8::decode_next(text, offset, ch)) {
        ++count;
    }
    return count;
}

}  // namespace

std::unique_ptr<RobertaTokenizer> RobertaTokenizer::load(const std::string& vocab_path) {
    std::ifstream file(vocab_path, std::ios::binary);
    if (!file) {
        throw_error(ErrorCode::NotFound, "roberta vocab not found: " + vocab_path);
    }

    auto tokenizer = std::unique_ptr<RobertaTokenizer>(new RobertaTokenizer());
    std::string line;
    while (std::getline(file, line)) {
        while (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        tokenizer->vocab_.push_back(line);
    }
    if (tokenizer->vocab_.empty()) {
        throw_error(ErrorCode::InvalidArgument, "roberta vocab is empty: " + vocab_path);
    }

    const auto find_required = [&tokenizer](std::string_view token,
                                            std::string_view name) -> std::int64_t {
        for (std::size_t i = 0; i < tokenizer->vocab_.size(); ++i) {
            if (tokenizer->vocab_[i] == token) {
                return static_cast<std::int64_t>(i);
            }
        }
        throw_error(ErrorCode::InvalidArgument,
                    "roberta vocab is missing the " + std::string(name) + " token");
        return -1;
    };
    tokenizer->cls_id_ = find_required("[CLS]", "[CLS]");
    tokenizer->sep_id_ = find_required("[SEP]", "[SEP]");
    tokenizer->unk_id_ = find_required("[UNK]", "[UNK]");
    return tokenizer;
}

RobertaTokenizer::Encoding RobertaTokenizer::encode(std::string_view text) const {
    // 1) BertNormalizer。
    std::string normalized = clean_text(text);
    normalized = pad_cjk(normalized);
    normalized = strip_accents_and_lower(normalized);

    // 2) BertPreTokenizer + 3) WordPiece。
    Encoding encoding;
    encoding.ids.push_back(cls_id_);
    encoding.tokens.emplace_back("[CLS]");

    for (const auto& word : pretokenize(normalized)) {
        // 字符数（不是字节数）超限整体进 [UNK]，与 tokenizers 一致。
        if (char_count(word) > 100) {
            encoding.ids.push_back(unk_id_);
            encoding.tokens.emplace_back("[UNK]");
            continue;
        }

        // 贪心最长匹配：匹配不到剩余部分就整词 [UNK]。
        std::vector<std::int64_t> word_ids;
        std::size_t start = 0;
        bool ok = true;
        while (start < word.size()) {
            std::size_t end = word.size();
            std::int64_t matched = -1;
            while (true) {
                std::string candidate(word.substr(start, end - start));
                if (start > 0) {
                    candidate.insert(0, "##");
                }
                for (std::size_t i = 0; i < vocab_.size(); ++i) {
                    if (vocab_[i] == candidate) {
                        matched = static_cast<std::int64_t>(i);
                        break;
                    }
                }
                if (matched >= 0) {
                    break;
                }
                if (end == start) {
                    break;
                }
                retreat_one_char(word, end);
            }
            if (matched < 0) {
                ok = false;
                break;
            }
            word_ids.push_back(matched);
            start = end;
        }

        if (ok) {
            for (const auto id : word_ids) {
                encoding.ids.push_back(id);
                encoding.tokens.push_back(vocab_[static_cast<std::size_t>(id)]);
            }
        } else {
            encoding.ids.push_back(unk_id_);
            encoding.tokens.emplace_back("[UNK]");
        }
    }

    encoding.ids.push_back(sep_id_);
    encoding.tokens.emplace_back("[SEP]");
    encoding.attention_mask.assign(encoding.ids.size(), 1);
    return encoding;
}

}  // namespace genie::text
