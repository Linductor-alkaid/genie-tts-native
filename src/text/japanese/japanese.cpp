// src/text/japanese/japanese.cpp —— 日语 G2P 的主流程
//
// 对应上游 `G2P/Japanese/JapaneseG2P.py` ✓。
//
// 上游用正则做标签解析，本实现按**手写扫描器**做（项目约定：不引
// std::regex ✓ —— ECMAScript 没有反向先行断言、MSVC 慢且差异多 ✓）。
// 每个扫描器都等价于上游的一条正则 ✓，注释里标了对应关系 ✓。

#include "text/japanese/japanese.hpp"

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "genie/utf8.hpp"

namespace genie::text::japanese {
namespace {

// ---------------------------------------------------------------------------
// 上游正则的等价扫描器
// ---------------------------------------------------------------------------

/// 上游 `re.search(r"-(.*?)\+", s)`：找第一个 "-" 与其后的第一个 "+"，
/// 取中间的内容 ✓（非贪婪 ✓）。
[[nodiscard]] std::optional<std::string_view> match_dash_group(std::string_view text) {
    const std::size_t dash = text.find('-');
    if (dash == std::string_view::npos) {
        return std::nullopt;
    }
    const std::size_t plus = text.find('+', dash + 1);
    if (plus == std::string_view::npos) {
        return std::nullopt;
    }
    return text.substr(dash + 1, plus - dash - 1);
}

/// 上游 `re.search(r"/A:([0-9\-]+)\+", s)` ✓。
[[nodiscard]] int match_a_value(std::string_view text) {
    const std::size_t start = text.find("/A:");
    if (start == std::string_view::npos) {
        return -50;
    }
    std::size_t i = start + 3;
    const std::size_t begin = i;
    while (i < text.size() && (text[i] == '-' || (text[i] >= '0' && text[i] <= '9'))) {
        ++i;
    }
    if (i == begin || i >= text.size() || text[i] != '+') {
        return -50;
    }
    return std::atoi(std::string(text.substr(begin, i - begin)).c_str());
}

/// 上游 `re.search(r"\+(\d+)\+", s)` ✓。
[[nodiscard]] int match_plus_digits_plus(std::string_view text) {
    std::size_t i = 0;
    while (i < text.size()) {
        if (text[i] != '+') {
            ++i;
            continue;
        }
        std::size_t j = i + 1;
        while (j < text.size() && text[j] >= '0' && text[j] <= '9') {
            ++j;
        }
        if (j > i + 1 && j < text.size() && text[j] == '+') {
            return std::atoi(std::string(text.substr(i + 1, j - i - 1)).c_str());
        }
        ++i;
    }
    return -50;
}

/// 上游 `re.search(r"\+(\d+)/", s)` ✓。
[[nodiscard]] int match_plus_digits_slash(std::string_view text) {
    std::size_t i = 0;
    while (i < text.size()) {
        if (text[i] != '+') {
            ++i;
            continue;
        }
        std::size_t j = i + 1;
        while (j < text.size() && text[j] >= '0' && text[j] <= '9') {
            ++j;
        }
        if (j > i + 1 && j < text.size() && text[j] == '/') {
            return std::atoi(std::string(text.substr(i + 1, j - i - 1)).c_str());
        }
        ++i;
    }
    return -50;
}

/// 上游 `re.search(r"/F:(\d+)_", s)` ✓。
[[nodiscard]] int match_f_value(std::string_view text) {
    const std::size_t start = text.find("/F:");
    if (start == std::string_view::npos) {
        return -50;
    }
    std::size_t i = start + 3;
    const std::size_t begin = i;
    while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
        ++i;
    }
    if (i == begin || i >= text.size() || text[i] != '_') {
        return -50;
    }
    return std::atoi(std::string(text.substr(begin, i - begin)).c_str());
}

/// 上游 `re.search(r"!(\d+)_", s)` ✓。
[[nodiscard]] int match_bang_value(std::string_view text) {
    const std::size_t start = text.find('!');
    if (start == std::string_view::npos) {
        return -50;
    }
    std::size_t i = start + 1;
    const std::size_t begin = i;
    while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
        ++i;
    }
    if (i == begin || i >= text.size() || text[i] != '_') {
        return -50;
    }
    return std::atoi(std::string(text.substr(begin, i - begin)).c_str());
}

// ---------------------------------------------------------------------------
// 文本处理
// ---------------------------------------------------------------------------

/// Python 的 `in` 对字符串是**子串**判断（不是"字符集合"）✓。
///
/// 上游 `if p3 in "AEIOU"` / `p3 in "aeiouAEIOUNcl"` 用的就是这个语义 ✓ ——
/// 注意空串也是子串（`"" in "AEIOU"` 为真 ✓），所以这里保留同样的行为 ✓。
[[nodiscard]] bool python_in(std::string_view needle, std::string_view haystack) noexcept {
    return haystack.find(needle) != std::string_view::npos;
}

/// Unicode 十进制数字（Nd 类）✓。
///
/// 上游的正则里写的是 `\d` —— **Python 3 的 `\d` 在 str 模式下是
/// Unicode 感知的** ✓：它匹配所有 Nd 类数字 ✓，不只是 ASCII 的 0-9 ✓。
///
/// 这条很容易踩坑：全角 '０'（U+FF10）**不在**上游显式列出的
/// `\uff11-\uff19` 里 ✓（那个区间从 １ 开始 ✓），但它**是** Nd 类数字 ✓
/// → 所以上游把 '０' 当日语字符 ✓，而只按显式区间判断的实现会把它
/// 当标点 ✓ → 切段位置不同 → 音素序列完全不同 ✓。
[[nodiscard]] bool is_decimal_digit(std::uint32_t cp) noexcept {
    static constexpr std::uint32_t kRanges[][2] = {
        {0x0030, 0x0039},    // ASCII 0-9
        {0x0660, 0x0669},    // 阿拉伯-印度
        {0x06F0, 0x06F9},    // 扩展阿拉伯-印度
        {0x07C0, 0x07C9},    // 尼德罗瓦字母
        {0x0966, 0x096F},    // 天城文
        {0x09E6, 0x09EF},    // 孟加拉文
        {0x0A66, 0x0A6F},    // 古木基文
        {0x0AE6, 0x0AEF},    // 古吉拉特文
        {0x0B66, 0x0B6F},    // 奥里亚文
        {0x0BE6, 0x0BEF},    // 泰米尔文
        {0x0C66, 0x0C6F},    // 泰卢固文
        {0x0CE6, 0x0CEF},    // 卡纳达文
        {0x0D66, 0x0D6F},    // 马拉雅拉姆文
        {0x0E50, 0x0E59},    // 泰文
        {0x0ED0, 0x0ED9},    // 老挝文
        {0x0F20, 0x0F29},    // 藏文
        {0x1040, 0x1049},    // 缅甸文
        {0x1090, 0x1099},    // 缅彝文
        {0x17E0, 0x17E9},    // 高棉文
        {0x1810, 0x1819},    // 蒙古文
        {0x1946, 0x194F},    // 楞严文
        {0x19D0, 0x19D9},    // 新泰文
        {0x1A80, 0x1A89},    // 苏禄文
        {0x1B50, 0x1B59},    // 爪哇文
        {0x1BB0, 0x1BB9},    // 孟加拉文（补充）
        {0x1C40, 0x1C49},    // .lepcha 文
        {0x1C50, 0x1C59},    // 奥尔雅文
        {0xA620, 0xA629},    // 维吾尔文
        {0xA8D0, 0xA8D9},    //  Saurashtra 文
        {0xA900, 0xA909},    // 彝文
        {0xA9D0, 0xA9D9},    // 扎莱纳加里文
        {0xA9F0, 0xA9F9},    // 爪哇文（补充）
        {0xAA50, 0xAA59},    // 查谟文
        {0xABF0, 0xABF9},    // 埃塞俄比亚文
        {0xFF10, 0xFF19},    // 全角 ０-９
        {0x1D7CE, 0x1D7FF},  // 数学数字
    };
    for (const auto& range : kRanges) {
        if (cp >= range[0] && cp <= range[1]) {
            return true;
        }
    }
    return false;
}

/// Unicode 空白（Python 的 `str.strip()` 会去掉这些 ✓）。
///
/// 全角空格 U+3000 也在里面 ✓ —— 上游的
/// `punctuation_marks[i].strip()` 因此会把全角空格整个丢掉 ✓。
[[nodiscard]] bool is_unicode_space(std::uint32_t cp) noexcept {
    if (cp == 0x20 || (cp >= 0x09 && cp <= 0x0D)) {
        return true;  // 空格 +
    }
    if (cp >= 0x1C && cp <= 0x1F) {
        return true;  // 文件/组/单元/记录分隔符
    }
    if (cp == 0x85 || cp == 0xA0) {
        return true;  // NEL / NBSP
    }
    if (cp == 0x1680) {
        return true;  // 奥格维姆空格
    }
    if (cp >= 0x2000 && cp <= 0x200A) {
        return true;  // 各种排版空格
    }
    if (cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F) {
        return true;  // 行/段分隔符 / 窄 NBSP / 中数学空格
    }
    if (cp == 0x3000) {
        return true;  // 全角空格
    }
    return false;
}

/// Python `str.strip()` 的等价实现 ✓（按**码点**去掉首尾 Unicode 空白 ✓）。
///
/// 注意不能按字节切 —— 多字节字符（如全角空格 U+3000 ✓）被切成半个
/// 就会产出非法 UTF-8 ✓（下游按字节比较时会静默错 ✓）。
[[nodiscard]] std::string strip_unicode_space(std::string_view text) {
    // 先算出每个码点的字节区间 ✓。
    std::vector<std::size_t> starts;
    std::vector<std::size_t> ends;
    std::size_t offset = 0;
    utf8::Char ch{};
    while (utf8::decode_next(text, offset, ch)) {
        starts.push_back(ch.offset);
        ends.push_back(offset);
    }
    std::size_t first = 0;
    std::size_t last = starts.size();
    while (first < last) {
        std::size_t inner = starts[first];
        utf8::Char c{};
        if (!utf8::decode_next(text, inner, c) || !is_unicode_space(c.codepoint)) {
            break;
        }
        ++first;
    }
    while (last > first) {
        std::size_t inner = starts[last - 1];
        utf8::Char c{};
        if (!utf8::decode_next(text, inner, c) || !is_unicode_space(c.codepoint)) {
            break;
        }
        --last;
    }
    if (first >= last) {
        return {};
    }
    return std::string(text.substr(starts[first], ends[last - 1] - starts[first]));
}

/// Unicode 小写映射（覆盖常用区段 ✓）。
///
/// 上游用的是 Python 的 `str.lower()` ✓ —— 它是 Unicode 感知的 ✓：
/// 全角 'Ａ' → 'ａ' ✓、希腊字母 ✓、西里尔字母 ✓ 都会变小写 ✓。
/// 只处理 ASCII 的话，全角大写字母会漏掉 ✓。
[[nodiscard]] std::uint32_t to_unicode_lower(std::uint32_t cp) noexcept {
    if (cp < 0x80) {
        return (cp >= 'A' && cp <= 'Z') ? cp + 32U : cp;
    }
    if (cp >= 0xC0 && cp <= 0xDE && cp != 0xD7) {
        return cp + 0x20U;  // Latin-1 补充
    }
    if (cp >= 0x100 && cp <= 0x137) {
        return (cp % 2 == 0) ? cp + 1U : cp;  // Latin 扩展 A
    }
    if (cp >= 0x139 && cp <= 0x148) {
        return (cp % 2 == 1) ? cp + 1U : cp;
    }
    if (cp >= 0x14A && cp <= 0x177) {
        return (cp % 2 == 0) ? cp + 1U : cp;
    }
    if (cp >= 0x178 && cp <= 0x17E) {
        return (cp % 2 == 0) ? cp + 1U : cp;
    }
    if (cp >= 0x391 && cp <= 0x3A1) {
        return cp + 0x20U;  // 希腊字母
    }
    if (cp >= 0x3A3 && cp <= 0x3AB) {
        return cp + 0x20U;
    }
    if (cp >= 0x410 && cp <= 0x42F) {
        return cp + 0x20U;  // 西里尔字母
    }
    if (cp >= 0x400 && cp <= 0x40F) {
        return cp + 0x50U;
    }
    if (cp >= 0xFF21 && cp <= 0xFF3A) {
        return cp + 0x20U;  // 全角 Ａ-Ｚ
    }
    return cp;
}

/// 上游 `_JAPANESE_CHARACTERS_RE` 的字符集合 ✓：
/// `[A-Za-z\d\u3005\u3040-\u30ff\u4e00-\u9fff\uff11-\uff19\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d]`
[[nodiscard]] bool is_japanese_character(std::uint32_t cp) noexcept {
    if (cp < 0x80) {
        return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z') || (cp >= '0' && cp <= '9');
    }
    if (is_decimal_digit(cp)) {
        return true;  // `\d` 是 Unicode 感知的 ✓
    }
    if (cp == 0x3005) {
        return true;  // 々
    }
    if (cp >= 0x3040 && cp <= 0x30FF) {
        return true;  // 平假名 / 片假名
    }
    if (cp >= 0x4E00 && cp <= 0x9FFF) {
        return true;  // 汉字
    }
    if (cp >= 0xFF11 && cp <= 0xFF19) {
        return true;  // 全角数字（显式区间 ✓ —— 其实是冗余的 ✓）
    }
    if (cp >= 0xFF21 && cp <= 0xFF3A) {
        return true;  // 全角大写
    }
    if (cp >= 0xFF41 && cp <= 0xFF5A) {
        return true;  // 全角小写
    }
    if (cp >= 0xFF66 && cp <= 0xFF9D) {
        return true;  // 半角片假名
    }
    return false;
}

/// 上游 `_CONSECUTIVE_PUNCTUATION_RE`：`([,./?!~…・])\1+` → `\1` ✓。
[[nodiscard]] bool is_consecutive_punctuation_char(std::uint32_t cp) noexcept {
    switch (cp) {
        case ',':
        case '.':
        case '/':
        case '?':
        case '!':
        case '~':
        case 0x2026:  // …
        case 0x30FB:  // ・
            return true;
        default:
            return false;
    }
}

/// 上游 `_post_replace_phoneme` 的替换表 ✓。
[[nodiscard]] std::string post_replace_phoneme(std::string_view ph) {
    if (ph == "：") {
        return ",";
    }
    if (ph == "；") {
        return ",";
    }
    if (ph == "，") {
        return ",";
    }
    if (ph == "。") {
        return ".";
    }
    if (ph == "！") {
        return "!";
    }
    if (ph == "？") {
        return "?";
    }
    if (ph == "\n") {
        return ".";
    }
    if (ph == "·") {
        return ",";
    }
    if (ph == "、") {
        return ",";
    }
    if (ph == "...") {
        return "…";
    }
    return std::string(ph);
}

}  // namespace

std::string text_normalize(std::string_view text) {
    // 全程按**码点**处理 ✓（项目铁律 ✓）：先解成码点数组 ✓，
    // 再做替换/折叠/小写 ✓，最后编码回 UTF-8 ✓。
    // （按字节做的话多字节字符会被切碎 ✓ —— 折叠标点时尤其致命 ✓。）
    std::vector<std::uint32_t> cps;
    cps.reserve(text.size());
    std::size_t offset = 0;
    utf8::Char ch{};
    while (utf8::decode_next(text, offset, ch)) {
        cps.push_back(ch.codepoint);
    }

    // 1. % / ％ → パーセント ✓（上游 `_SYMBOLS_TO_JAPANESE` ✓）
    {
        static constexpr std::uint32_t kPercent[] = {0x30D1, 0x30FC, 0x30BB, 0x30F3, 0x30C8};
        std::vector<std::uint32_t> replaced;
        replaced.reserve(cps.size() + 4);
        for (const std::uint32_t cp : cps) {
            if (cp == '%' || cp == 0xFF05) {
                for (const std::uint32_t item : kPercent) {
                    replaced.push_back(item);
                }
            } else {
                replaced.push_back(cp);
            }
        }
        cps.swap(replaced);
    }

    // 2. 折叠连续标点 ✓（上游 `([,./?!~…・])\1+` → `\1` ✓）
    {
        std::vector<std::uint32_t> folded;
        folded.reserve(cps.size());
        std::size_t i = 0;
        while (i < cps.size()) {
            const std::uint32_t current = cps[i];
            folded.push_back(current);
            ++i;
            while (i < cps.size() && cps[i] == current &&
                   is_consecutive_punctuation_char(current)) {
                ++i;
            }
        }
        cps.swap(folded);
    }

    // 3. 转小写 ✓（上游是 Python 的 str.lower() ✓ —— Unicode 感知 ✓）
    for (std::uint32_t& cp : cps) {
        cp = to_unicode_lower(cp);
    }

    std::string out;
    out.reserve(cps.size() * 2);
    for (const std::uint32_t cp : cps) {
        out += utf8::encode(cp);
    }
    return out;
}

Segmentation segment_japanese(std::string_view text) {
    Segmentation result;
    const std::string normalized = text_normalize(text);

    std::string current;
    std::size_t offset = 0;
    utf8::Char ch{};
    while (utf8::decode_next(normalized, offset, ch)) {
        if (is_japanese_character(ch.codepoint)) {
            current.append(normalized.substr(ch.offset, ch.length));
        } else {
            result.segments.push_back(std::move(current));
            current.clear();
            result.marks.emplace_back(normalized.substr(ch.offset, ch.length));
        }
    }
    result.segments.push_back(std::move(current));
    return result;
}

std::vector<std::string> labels_to_phonemes(const std::vector<std::string>& labels) {
    std::vector<std::string> phones;
    phones.reserve(labels.size() * 2);
    for (std::size_t n = 0; n < labels.size(); ++n) {
        const std::string& label = labels[n];
        const auto group = match_dash_group(label);
        if (!group.has_value()) {
            continue;
        }
        std::string p3(*group);
        if (python_in(p3, "AEIOU")) {
            for (char& c : p3) {
                if (c >= 'A' && c <= 'Z') {
                    c = static_cast<char>(c - 'A' + 'a');
                }
            }
        }
        if (p3 == "sil") {
            if (n == 0) {
                phones.emplace_back("^");
            } else if (n + 1 == labels.size()) {
                const int e3 = match_bang_value(label);
                phones.emplace_back(e3 == 1 ? "?" : "$");
            }
            continue;
        }
        if (p3 == "pau") {
            phones.emplace_back("_");
            continue;
        }
        phones.push_back(p3);

        const int a1 = match_a_value(label);
        const int a2 = match_plus_digits_plus(label);
        const int a3 = match_plus_digits_slash(label);
        const int f1 = match_f_value(label);
        const std::string next_label = n + 1 < labels.size() ? labels[n + 1] : std::string();
        const int a2_next = match_plus_digits_plus(next_label);

        if (a3 == 1 && a2_next == 1 && python_in(p3, "aeiouAEIOUNcl")) {
            phones.emplace_back("#");
        } else if (a1 == 0 && a2_next == a2 + 1 && a2 != f1) {
            phones.emplace_back("]");
        } else if (a2 == 1 && a2_next == 2) {
            phones.emplace_back("[");
        }
    }
    return phones;
}

std::vector<std::string> text_to_phonemes(
    std::string_view text, const std::vector<std::vector<std::string>>& segment_labels) {
    std::vector<std::string> phonemes;
    const Segmentation segmentation = segment_japanese(text);

    for (std::size_t i = 0; i < segmentation.segments.size(); ++i) {
        const std::string& segment = segmentation.segments[i];
        if (!segment.empty() && i < segment_labels.size()) {
            // 上游：`JapaneseG2P._pyopenjtalk_g2p_prosody(segment)[1:-1]`
            // —— 砍掉的是**音素列表**的首尾元素 ✓（不是"跳过首尾 label" ✓：
            // 末尾 label 若不是 sil，它贡献的音素要留下 ✓、只丢掉最后一个韵律符号 ✓）。
            std::vector<std::string> segment_phones = labels_to_phonemes(segment_labels[i]);
            if (segment_phones.size() <= 2) {
                segment_phones.clear();
            } else {
                segment_phones.erase(segment_phones.begin());
                segment_phones.pop_back();
            }
            phonemes.insert(phonemes.end(), segment_phones.begin(), segment_phones.end());
        }
        if (i < segmentation.marks.size()) {
            // 上游：`mark.strip()` —— Python 的 strip() 去的是 **Unicode 空白** ✓
            //（含全角空格 U+3000 ✓）—— 这里按码点判断 ✓。
            const std::string mark = strip_unicode_space(segmentation.marks[i]);
            if (!mark.empty()) {
                phonemes.push_back(mark);
            }
        }
    }

    std::vector<std::string> processed;
    processed.reserve(phonemes.size());
    for (const auto& phoneme : phonemes) {
        processed.push_back(post_replace_phoneme(phoneme));
    }
    return processed;
}

std::vector<std::string> JapaneseG2p::g2p(std::string_view text) const {
    std::vector<std::string> phonemes;
    // 上游：`if not text.strip(): return []` ✓（Python 的 strip 去 Unicode 空白 ✓）。
    if (strip_unicode_space(text).empty()) {
        return phonemes;
    }
    if (engine_ == nullptr || !engine_->loaded()) {
        return phonemes;
    }

    const Segmentation segmentation = segment_japanese(text);
    std::vector<std::vector<std::string>> segment_labels;
    segment_labels.reserve(segmentation.segments.size());
    for (const auto& segment : segmentation.segments) {
        std::vector<std::string> labels;
        if (!segment.empty()) {
            if (!engine_->labels(segment, labels)) {
                labels.clear();
            }
        }
        segment_labels.push_back(std::move(labels));
    }
    return text_to_phonemes(text, segment_labels);
}

}  // namespace genie::text::japanese
