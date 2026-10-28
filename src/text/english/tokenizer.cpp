// src/text/english/tokenizer.cpp —— nltk TweetTokenizer 的 C++ 移植。
//
// 上游 `G2P/English/EnglishG2P.py` 用 `nltk.tokenize.TweetTokenizer`
// 分词（`word_tokenize`）。 TweetTokenizer 是一条大正则，按顺序尝试
// 若干个分支，取**第一个**匹配上的（Python 的 re 是"左优先"，不是
// "最长优先"）：
//
//   1. URL
//   2. 表情符号
//   3. HTML 标签 `<[^>\s]+>`
//   4. ASCII 箭头 `[-]+>|<[-]+`
//   5. Twitter 用户名 `@[\w_]+`
//   6. Twitter 话题标签 `#[\w_]+[\w'_-]*[\w_]+`
//   7. 邮箱
//   8. emoji 肤色修饰符 / ZWJ 组合
//   9. 国旗（区域指示符对）
//  10. 带撇号 / 连字符的词 `[^\W\d_]([^\W\d_]|['\-_])+[^\W\d_]`
//  11. 带分隔符的数字 `[+\-]?\d+[,/.:-]\d+[+\-]?`
//  12. 普通词 `[\w_]+`
//  13. 省略号 `\.(?:[^\S\r\n]*\.){1,}`
//  14. 其它非空白字符 `\S`
//
// 分词之前还有两步预处理：
//   - HTML 实体替换（`&amp;` → `&`）
//   - `HANG_RE`：4 个以上连续的同一个非字母数字字符压成 3 个
//
// 没有移植的分支：8（emoji 肤色）、9（国旗）。它们的 token 同样查不到
// 音素，拆成单字符后每个字符也查不到音素 —— 对最终音素序列没有影响。
// 差异清单见 docs/upstream-audit.md §17。

#include <array>
#include <cctype>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "genie/utf8.hpp"
#include "text/english/english.hpp"

namespace genie::text::english {
namespace {

// ---------------------------------------------------------------------------
// 字符分类
// ---------------------------------------------------------------------------

/// Python 正则的 `\w`：字母 / 数字 / 下划线。
[[nodiscard]] bool is_word_codepoint(std::uint32_t cp) noexcept {
    if (cp < 0x80) {
        return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z') || (cp >= '0' && cp <= '9') ||
               cp == '_';
    }
    if (cp >= 0xC0 && cp <= 0xFF) {
        return cp != 0xD7 && cp != 0xF7;
    }
    if (cp >= 0x100 && cp <= 0x2FF) {
        return true;
    }
    if (cp >= 0x3040 && cp <= 0xD7FF) {
        return true;
    }
    if (cp >= 0xF900 && cp <= 0xFDCF) {
        return true;
    }
    return cp >= 0x20000;
}

/// Python 正则的 `[^\W\d_]`：词字符里去掉数字和下划线 —— 即字母。
[[nodiscard]] bool is_letter_codepoint(std::uint32_t cp) noexcept {
    if (cp < 0x80) {
        return (cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z');
    }
    if (cp >= 0xC0 && cp <= 0xFF) {
        return cp != 0xD7 && cp != 0xF7;
    }
    if (cp >= 0x100 && cp <= 0x2FF) {
        return true;
    }
    if (cp >= 0x3040 && cp <= 0xD7FF) {
        return true;
    }
    if (cp >= 0xF900 && cp <= 0xFDCF) {
        return true;
    }
    return cp >= 0x20000;
}

/// Python 正则的 `\s`。
[[nodiscard]] bool is_space_codepoint(std::uint32_t cp) noexcept {
    switch (cp) {
        case ' ':
        case '\t':
        case '\n':
        case '\r':
        case '\f':
        case '\v':
        case 0x00A0:
        case 0x1680:
        case 0x2028:
        case 0x2029:
        case 0x202F:
        case 0x205F:
        case 0x3000:
            return true;
        default:
            return false;
    }
}

/// 读下一个码点。
[[nodiscard]] bool next_codepoint(std::string_view text, std::size_t& offset, std::uint32_t& cp,
                                  std::size_t& length) noexcept {
    utf8::Char ch{};
    if (!utf8::decode_next(text, offset, ch)) {
        return false;
    }
    cp = ch.codepoint;
    length = ch.length;
    return true;
}

[[nodiscard]] char ascii_lower(char c) noexcept {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}

/// 大小写不敏感的前缀匹配。
[[nodiscard]] bool match_ci(std::string_view text, std::size_t index,
                            std::string_view prefix) noexcept {
    if (index + prefix.size() > text.size()) {
        return false;
    }
    for (std::size_t i = 0; i < prefix.size(); ++i) {
        if (ascii_lower(text[index + i]) != ascii_lower(prefix[i])) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// 预处理 1：HTML 实体
// ---------------------------------------------------------------------------

/// 常用命名实体（上游用 `html.entities.name2codepoint`，共 250 条；
/// 这里只收常见的十余个，差异见 audit §17）。
[[nodiscard]] std::uint32_t named_entity_codepoint(std::string_view name) noexcept {
    struct Entry {
        std::string_view name;
        std::uint32_t cp;
    };
    static constexpr std::array<Entry, 16> kEntities = {{
        {"amp", '&'},
        {"lt", '<'},
        {"gt", '>'},
        {"quot", '"'},
        {"apos", '\''},
        {"nbsp", 0x00A0},
        {"copy", 0x00A9},
        {"reg", 0x00AE},
        {"hellip", 0x2026},
        {"mdash", 0x2014},
        {"ndash", 0x2013},
        {"lsquo", 0x2018},
        {"rsquo", 0x2019},
        {"ldquo", 0x201C},
        {"rdquo", 0x201D},
        {"middot", 0x00B7},
    }};
    for (const auto& entry : kEntities) {
        if (entry.name == name) {
            return entry.cp;
        }
    }
    return 0;
}

/// 上游 `_replace_html_entities`：`&(#?(x?))([^&;\s]+);`
/// 数字实体按十进制/十六进制解析，命名实体查表；解析不出来就**删掉**。
[[nodiscard]] std::string replace_html_entities(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size()) {
        if (text[i] != '&') {
            out.push_back(text[i]);
            ++i;
            continue;
        }
        // 找结尾的分号：中间不能有 & ; 或空白
        std::size_t j = i + 1;
        std::size_t body_start = j;
        while (j < text.size() && text[j] != ';' && text[j] != '&' &&
               !is_space_codepoint(static_cast<unsigned char>(text[j]))) {
            ++j;
        }
        if (j >= text.size() || text[j] != ';') {
            out.push_back(text[i]);
            ++i;
            continue;
        }
        const std::string_view body = text.substr(body_start, j - body_start);
        std::uint32_t cp = 0;
        bool resolved = false;
        if (!body.empty() && body.front() == '#') {
            const std::string_view digits = body.substr(1);
            const bool hex = !digits.empty() && (digits.front() == 'x' || digits.front() == 'X');
            const std::string_view number = hex ? digits.substr(1) : digits;
            if (!number.empty()) {
                char* end = nullptr;
                const std::string text_number(number);
                const long value = std::strtol(text_number.c_str(), &end, hex ? 16 : 10);
                if (end != nullptr && *end == '\0' && value >= 0 && value <= 0x10FFFF) {
                    cp = static_cast<std::uint32_t>(value);
                    resolved = true;
                }
            }
        } else {
            cp = named_entity_codepoint(body);
            resolved = cp != 0;
        }
        if (resolved) {
            out += utf8::encode(cp);
            i = j + 1;
        } else {
            // 上游 remove_illegal=True → 删掉整个实体
            i = j + 1;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// 预处理 2：HANG_RE —— 4 个以上连续的同一非字母数字字符压成 3 个
// ---------------------------------------------------------------------------

[[nodiscard]] std::string collapse_hang(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    std::size_t offset = 0;
    while (offset < text.size()) {
        const std::size_t start = offset;
        std::uint32_t cp = 0;
        std::size_t length = 0;
        if (!next_codepoint(text, offset, cp, length)) {
            break;
        }
        const bool is_alnum = is_letter_codepoint(cp) || (cp >= '0' && cp <= '9');
        if (is_alnum) {
            out.append(text.substr(start, length));
            continue;
        }
        // 统计连续相同的个数
        std::size_t count = 1;
        std::size_t scan = offset;
        while (scan < text.size()) {
            std::uint32_t cp2 = 0;
            std::size_t length2 = 0;
            if (!next_codepoint(text, scan, cp2, length2)) {
                break;
            }
            if (cp2 != cp) {
                break;
            }
            ++count;
            offset = scan;
        }
        if (count >= 4) {
            out.append(text.substr(start, length));
            out.append(text.substr(start, length));
            out.append(text.substr(start, length));
        } else {
            out.append(text.substr(start, offset - start));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// 各分支
// ---------------------------------------------------------------------------

/// URL 尾部**不允许**的多字节字符（上游的 `«»“”‘’`）。
/// 它们在 UTF-8 里是 2~3 字节 —— 不能写成字符字面量（multi-char constant）。
constexpr std::string_view kUrlTailMultiByte =
    "\xC2\xAB\xC2\xBB\xE2\x80\x9C\xE2\x80\x9D"
    "\xE2\x80\x98\xE2\x80\x99";

/// URL 尾部允许的单字符（上游的 `[^\s`!()\[\]{};:'".,<>?«»“”‘’]`）。
[[nodiscard]] bool is_url_tail_char(std::string_view text, std::size_t index) noexcept {
    const char c = text[index];
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v') {
        return false;
    }
    if (c == '`' || c == '!' || c == '(' || c == ')' || c == '[' || c == ']' || c == '{' ||
        c == '}' || c == ';' || c == ':' || c == '\'' || c == '"' || c == ',' || c == '.' ||
        c == '<' || c == '>' || c == '?') {
        return false;
    }
    if (static_cast<unsigned char>(c) >= 0x80) {
        for (std::size_t i = 0; i + 1 < kUrlTailMultiByte.size(); i += 2) {
            if (text.compare(index, 2, kUrlTailMultiByte.substr(i, 2)) == 0) {
                return false;
            }
        }
    }
    return true;
}

/// URL 路径里允许的字符（上游的 `[^\s()<>{}\[\]]`）。
[[nodiscard]] bool is_url_path_char(char c) noexcept {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v') {
        return false;
    }
    return !(c == '(' || c == ')' || c == '<' || c == '>' || c == '{' || c == '}' || c == '[' ||
             c == ']');
}

/// 裸域名形式：`[a-z0-9]+(?:[.\-][a-z0-9]+){0,126}[.](?:[a-z]{2,13})\b/?`
[[nodiscard]] std::size_t match_bare_domain(std::string_view text, std::size_t index) noexcept {
    std::size_t i = index;
    const std::size_t label_start = i;
    while (i < text.size() && ((ascii_lower(text[i]) >= 'a' && ascii_lower(text[i]) <= 'z') ||
                               (text[i] >= '0' && text[i] <= '9'))) {
        ++i;
    }
    if (i == label_start) {
        return 0;
    }
    // `(?:[.\-][a-z0-9]+){0,126}` —— 贪婪匹配，回溯交给后面的 `[.][a-z]{2,13}`
    std::size_t best = 0;
    std::size_t cursor = i;
    for (int round = 0; round <= 126; ++round) {
        if (cursor >= text.size() || (text[cursor] != '.' && text[cursor] != '-')) {
            break;
        }
        const std::size_t after_sep = cursor + 1;
        std::size_t j = after_sep;
        while (j < text.size() && ((ascii_lower(text[j]) >= 'a' && ascii_lower(text[j]) <= 'z') ||
                                   (text[j] >= '0' && text[j] <= '9'))) {
            ++j;
        }
        if (j == after_sep) {
            break;
        }
        cursor = j;
        // 试着在这里结束域名：`[.][a-z]{2,13}`
        std::size_t k = cursor;
        if (k < text.size() && text[k] == '.') {
            std::size_t tld_end = k + 1;
            while (tld_end < text.size() && ascii_lower(text[tld_end]) >= 'a' &&
                   ascii_lower(text[tld_end]) <= 'z' && (tld_end - (k + 1)) < 13) {
                ++tld_end;
            }
            if (tld_end - (k + 1) >= 2) {
                // `\b` —— TLD 后面不能还是字母数字
                const bool boundary =
                    tld_end >= text.size() ||
                    !((ascii_lower(text[tld_end]) >= 'a' && ascii_lower(text[tld_end]) <= 'z') ||
                      (text[tld_end] >= '0' && text[tld_end] <= '9'));
                if (boundary) {
                    best = tld_end;
                    if (best < text.size() && text[best] == '/') {
                        ++best;
                    }
                    break;
                }
            }
        }
    }
    return best;
}

/// 带协议的 URL。
[[nodiscard]] std::size_t match_url(std::string_view text, std::size_t index) noexcept {
    std::size_t scheme_end = 0;
    if (match_ci(text, index, "https://")) {
        scheme_end = index + 8;
    } else if (match_ci(text, index, "http://")) {
        scheme_end = index + 7;
    } else {
        return 0;
    }
    // 上游：`https?:(?:/{1,3}|[a-z0-9%])` 或 `域名/`
    std::size_t body = scheme_end;
    std::size_t slashes = 0;
    while (body < text.size() && text[body] == '/' && slashes < 3) {
        ++body;
        ++slashes;
    }
    if (slashes == 0) {
        // 域名形式：`[a-z0-9.\-]{1,255}[.](?:[a-z]{2,13})/`
        std::size_t scan = scheme_end;
        std::size_t best = 0;
        while (scan < text.size() && scan - scheme_end < 255) {
            const char c = ascii_lower(text[scan]);
            const bool ok =
                (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' || c == '-';
            if (!ok) {
                break;
            }
            ++scan;
            // 试着在这里匹配 `[.][a-z]{2,13}/`
            for (std::size_t k = scheme_end; k < scan; ++k) {
                if (text[k] != '.') {
                    continue;
                }
                std::size_t tld_end = k + 1;
                while (tld_end < text.size() && ascii_lower(text[tld_end]) >= 'a' &&
                       ascii_lower(text[tld_end]) <= 'z' && tld_end - (k + 1) < 13) {
                    ++tld_end;
                }
                if (tld_end - (k + 1) >= 2 && tld_end < text.size() && text[tld_end] == '/') {
                    best = tld_end + 1;
                    break;
                }
            }
            if (best != 0) {
                break;
            }
        }
        if (best == 0) {
            return 0;
        }
        body = best;
    }
    // 路径部分：`[^\s()<>{}\[\]]+`
    std::size_t path_end = body;
    while (path_end < text.size() && is_url_path_char(text[path_end])) {
        ++path_end;
    }
    if (path_end == body) {
        return 0;
    }
    // 尾部：路径里最后一个能当尾部的字符（且路径不能为空）
    std::size_t tail = 0;
    for (std::size_t k = path_end; k > body; --k) {
        if (is_url_tail_char(text, k - 1)) {
            tail = k;
            break;
        }
    }
    if (tail <= body) {
        return 0;
    }
    return tail;
}

/// 表情符号（上游 EMOTICONS）。
[[nodiscard]] std::size_t match_emoticon(std::string_view text, std::size_t index) noexcept {
    const auto is_eye = [](char c) { return c == ':' || c == '=' || c == '8'; };
    const auto is_nose = [](char c) { return c == '-' || c == 'o' || c == '*' || c == '\''; };
    const auto is_mouth = [](char c) {
        return c == ')' || c == ']' || c == '(' || c == '[' || c == 'd' || c == 'D' || c == 'p' ||
               c == 'P' || c == '/' || c == ':' || c == '}' || c == '{' || c == '@' || c == '|' ||
               c == '\\';
    };
    std::size_t i = index;
    if (i < text.size() && (text[i] == '<' || text[i] == '>')) {
        ++i;
    }
    if (i >= text.size() || !is_eye(text[i])) {
        return 0;
    }
    const std::size_t eyes = i;
    ++i;
    if (i < text.size() && is_nose(text[i])) {
        ++i;
    }
    if (i < text.size() && is_mouth(text[i])) {
        return i + 1;
    }
    // 另一种顺序：嘴在前、眼在后
    std::size_t j = index;
    if (j < text.size() && (text[j] == '<' || text[j] == '>')) {
        ++j;
    }
    if (j >= text.size() || !is_mouth(text[j])) {
        return 0;
    }
    ++j;
    if (j < text.size() && is_nose(text[j])) {
        ++j;
    }
    if (j < text.size() && is_eye(text[j])) {
        ++j;
        if (j < text.size() && (text[j] == '<' || text[j] == '>')) {
            ++j;
        }
        return j;
    }
    (void)eyes;
    return 0;
}

/// HTML 标签：`<[^>\s]+>`
[[nodiscard]] std::size_t match_html_tag(std::string_view text, std::size_t index) noexcept {
    if (index >= text.size() || text[index] != '<') {
        return 0;
    }
    std::size_t i = index + 1;
    while (i < text.size() && text[i] != '>' &&
           !is_space_codepoint(static_cast<unsigned char>(text[i]))) {
        ++i;
    }
    if (i < text.size() && text[i] == '>' && i > index + 1) {
        return i + 1 - index;
    }
    return 0;
}

/// ASCII 箭头：`[-]+>|<[-]+`
[[nodiscard]] std::size_t match_arrow(std::string_view text, std::size_t index) noexcept {
    if (index < text.size() && text[index] == '<') {
        std::size_t i = index + 1;
        while (i < text.size() && text[i] == '-') {
            ++i;
        }
        if (i > index + 1) {
            return i - index;
        }
        return 0;
    }
    if (index < text.size() && text[index] == '-') {
        std::size_t i = index;
        while (i < text.size() && text[i] == '-') {
            ++i;
        }
        return i - index;
    }
    return 0;
}

/// Twitter 用户名：`@[\w_]+`
[[nodiscard]] std::size_t match_handle(std::string_view text, std::size_t index) noexcept {
    if (index >= text.size() || text[index] != '@') {
        return 0;
    }
    std::size_t i = index + 1;
    while (i < text.size()) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 0x80) {
            if (!is_word_codepoint(c)) {
                break;
            }
            ++i;
            continue;
        }
        std::size_t offset = i;
        std::uint32_t cp = 0;
        std::size_t length = 0;
        if (!next_codepoint(text, offset, cp, length)) {
            break;
        }
        if (!is_word_codepoint(cp)) {
            break;
        }
        i = offset;
    }
    return i > index + 1 ? i - index : 0;
}

/// Twitter 话题标签：`#[\w_]+[\w'_-]*[\w_]+`
[[nodiscard]] std::size_t match_hashtag(std::string_view text, std::size_t index) noexcept {
    if (index >= text.size() || text[index] != '#') {
        return 0;
    }
    std::size_t i = index + 1;
    // [\w_]+
    std::size_t word_start = i;
    while (i < text.size() && is_word_codepoint(static_cast<unsigned char>(text[i]))) {
        ++i;
    }
    if (i == word_start) {
        return 0;
    }
    // [\w'_-]*
    while (i < text.size()) {
        const char c = text[i];
        if (c == '\'' || c == '_' || c == '-' || is_word_codepoint(static_cast<unsigned char>(c))) {
            ++i;
        } else {
            break;
        }
    }
    // [\w_]+
    const std::size_t tail_start = i;
    while (i < text.size() && is_word_codepoint(static_cast<unsigned char>(text[i]))) {
        ++i;
    }
    if (i == tail_start) {
        // 上游要求结尾必须有一个词字符 —— 退回第一个 [\w_]+ 的长度
        return word_start - index;
    }
    return i - index;
}

/// 邮箱：`[\w.+-]{1,64}@[\w-]{1,63}\.(?:[\w-]\.?){1,251}[\w-]`
[[nodiscard]] std::size_t match_email(std::string_view text, std::size_t index) noexcept {
    // 本地部分：贪婪取 [\w.+-]，然后回溯找 '@'
    std::size_t local_end = index;
    while (local_end < text.size() && local_end - index < 64) {
        const char c = text[local_end];
        if (is_word_codepoint(static_cast<unsigned char>(c)) || c == '.' || c == '+' || c == '-') {
            ++local_end;
        } else {
            break;
        }
    }
    std::size_t at = local_end;
    while (at > index && text[at - 1] != '@') {
        --at;
    }
    if (at == index || text[at - 1] != '@') {
        return 0;
    }
    const std::size_t at_pos = at - 1;
    // 域名：`[\w-]{1,63}\.` 然后 `(?:[\w-]\.?){1,251}` 再 `[\w-]`
    std::size_t i = at_pos + 1;
    const std::size_t label_start = i;
    while (i < text.size() && i - label_start < 63) {
        const char c = text[i];
        if (is_word_codepoint(static_cast<unsigned char>(c)) || c == '-') {
            ++i;
        } else {
            break;
        }
    }
    if (i == label_start || i >= text.size() || text[i] != '.') {
        return 0;
    }
    ++i;  // 吃掉 '.'
    std::size_t groups = 0;
    while (i < text.size() && groups < 251) {
        const char c = text[i];
        if (!is_word_codepoint(static_cast<unsigned char>(c)) && c != '-') {
            break;
        }
        ++i;
        ++groups;
        if (i < text.size() && text[i] == '.') {
            ++i;
        }
    }
    if (groups < 1) {
        return 0;
    }
    // 结尾的 `[\w-]`
    if (i >= text.size()) {
        return 0;
    }
    const char c = text[i];
    if (!is_word_codepoint(static_cast<unsigned char>(c)) && c != '-') {
        return 0;
    }
    return i + 1 - index;
}

/// 带撇号 / 连字符的词：`[^\W\d_](?:[^\W\d_]|['\-_])+[^\W\d_]`
[[nodiscard]] std::size_t match_word_with_separators(std::string_view text,
                                                     std::size_t index) noexcept {
    std::size_t offset = index;
    std::uint32_t cp = 0;
    std::size_t length = 0;
    if (!next_codepoint(text, offset, cp, length)) {
        return 0;
    }
    if (!is_letter_codepoint(cp)) {
        return 0;
    }
    std::size_t i = offset;
    std::size_t middle = 0;
    while (i < text.size()) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 0x80) {
            if (is_letter_codepoint(c) || c == '\'' || c == '-' || c == '_') {
                ++i;
                ++middle;
                continue;
            }
            break;
        }
        std::size_t scan = i;
        std::uint32_t cp2 = 0;
        std::size_t length2 = 0;
        if (!next_codepoint(text, scan, cp2, length2)) {
            break;
        }
        if (!is_letter_codepoint(cp2)) {
            break;
        }
        i = scan;
        ++middle;
    }
    if (middle < 1) {
        return 0;  // `(?:...)+` 至少一个
    }
    // 结尾必须是一个字母（回退掉中间部分的最后一个字符）
    std::size_t end = i;
    while (end > offset) {
        const unsigned char c = static_cast<unsigned char>(text[end - 1]);
        if (c < 0x80) {
            if (is_letter_codepoint(c)) {
                break;
            }
            --end;
            continue;
        }
        // 多字节：往前退到起始字节
        std::size_t back = end - 1;
        while (back > 0 && (static_cast<unsigned char>(text[back]) & 0xC0) == 0x80) {
            --back;
        }
        std::size_t scan = back;
        std::uint32_t cp2 = 0;
        std::size_t length2 = 0;
        if (next_codepoint(text, scan, cp2, length2) && is_letter_codepoint(cp2)) {
            break;
        }
        end = back;
    }
    if (end <= offset) {
        return 0;
    }
    return end - index;
}

/// 带分隔符的数字：`[+\-]?\d+[,/.:-]\d+[+\-]?`
[[nodiscard]] std::size_t match_number(std::string_view text, std::size_t index) noexcept {
    std::size_t i = index;
    if (i < text.size() && (text[i] == '+' || text[i] == '-')) {
        ++i;
    }
    const std::size_t int_start = i;
    while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
        ++i;
    }
    if (i == int_start) {
        return 0;
    }
    if (i >= text.size()) {
        return 0;
    }
    const char sep = text[i];
    if (sep != ',' && sep != '/' && sep != '.' && sep != ':' && sep != '-') {
        return 0;
    }
    ++i;
    const std::size_t frac_start = i;
    while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
        ++i;
    }
    if (i == frac_start) {
        return 0;
    }
    if (i < text.size() && (text[i] == '+' || text[i] == '-')) {
        ++i;
    }
    return i - index;
}

/// 普通词：`[\w_]+`
[[nodiscard]] std::size_t match_plain_word(std::string_view text, std::size_t index) noexcept {
    std::size_t i = index;
    while (i < text.size()) {
        const unsigned char c = static_cast<unsigned char>(text[i]);
        if (c < 0x80) {
            if (!is_word_codepoint(c)) {
                break;
            }
            ++i;
            continue;
        }
        std::size_t offset = i;
        std::uint32_t cp = 0;
        std::size_t length = 0;
        if (!next_codepoint(text, offset, cp, length)) {
            break;
        }
        if (!is_word_codepoint(cp)) {
            break;
        }
        i = offset;
    }
    return i - index;
}

/// 省略号：`\.(?:[^\S\r\n]*\.){1,}`
[[nodiscard]] std::size_t match_ellipsis(std::string_view text, std::size_t index) noexcept {
    if (index >= text.size() || text[index] != '.') {
        return 0;
    }
    std::size_t i = index + 1;
    int groups = 0;
    while (i < text.size()) {
        std::size_t scan = i;
        while (scan < text.size() && is_space_codepoint(static_cast<unsigned char>(text[scan])) &&
               text[scan] != '\r' && text[scan] != '\n') {
            ++scan;
        }
        if (scan >= text.size() || text[scan] != '.') {
            break;
        }
        i = scan + 1;
        ++groups;
    }
    return groups >= 1 ? i - index : 0;
}

}  // namespace

std::vector<Token> tokenize(std::string_view text) {
    const std::string prepared = collapse_hang(replace_html_entities(text));

    std::vector<Token> tokens;
    std::size_t i = 0;
    while (i < prepared.size()) {
        const unsigned char c = static_cast<unsigned char>(prepared[i]);
        if (is_space_codepoint(c)) {
            ++i;
            continue;
        }
        std::size_t length = 0;
        // 顺序与上游正则的分支顺序一致（先匹配上的分支胜出）。
        if ((length = match_url(prepared, i)) == 0) {
            length = match_bare_domain(prepared, i);
        }
        if (length == 0) {
            length = match_emoticon(prepared, i);
        }
        if (length == 0) {
            length = match_html_tag(prepared, i);
        }
        if (length == 0) {
            length = match_arrow(prepared, i);
        }
        if (length == 0) {
            length = match_handle(prepared, i);
        }
        if (length == 0) {
            length = match_hashtag(prepared, i);
        }
        if (length == 0) {
            length = match_email(prepared, i);
        }
        if (length == 0) {
            length = match_word_with_separators(prepared, i);
        }
        if (length == 0) {
            length = match_number(prepared, i);
        }
        if (length == 0) {
            length = match_plain_word(prepared, i);
        }
        if (length == 0) {
            length = match_ellipsis(prepared, i);
        }
        if (length == 0) {
            // `\S` —— 其它非空白字符（按码点取一个）
            std::size_t offset = i;
            std::uint32_t cp = 0;
            std::size_t char_length = 0;
            if (next_codepoint(prepared, offset, cp, char_length)) {
                length = char_length;
            } else {
                length = 1;
            }
        }
        tokens.push_back(Token{prepared.substr(i, length), i});
        i += length;
    }
    return tokens;
}

}  // namespace genie::text::english
