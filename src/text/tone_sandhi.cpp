// src/text/tone_sandhi.cpp — 变调与儿化的 C++ 移植。
//
// 每个函数都注明对应的上游函数（G2P/Chinese/ToneSandhi.py 下文简写
// TS.xxx，G2P/Chinese/Erhua.py 简写 EH.xxx），逐分支对照移植。
// 上游依赖 jieba 分词 / POS 标注的判断，能照搬的照搬，不能的换成
// 默认行为并就地注明 —— 差异清单与影响评估在 docs/upstream-audit.md §9.6。

#include "text/tone_sandhi.hpp"

#include <string_view>
#include <utility>

#include "genie/utf8.hpp"
#include "text/sandhi_table.hpp"

namespace genie::text::sandhi {
namespace {

using detail::must_erhua;
using detail::must_neural_tone_words;
using detail::must_not_neural_tone_words;
using detail::not_erhua;
using detail::segmentation_lexicon;

/// 上游 TS.punc：`_yi_sandhi` 里"一"后接标点时还读一声。
/// 我们的单元是纯汉字，正常到不了这里 —— 照搬只为保真。
constexpr std::string_view kYiPunc = "：，；。？！“”‘’':,;.?!";

/// 轻声判定用的字符集，按上游各 elif 分支拆开：
constexpr std::string_view kModalSuffix = "吧呢哈啊呐噻嘛吖嗨呐哦哒额滴哩哟喽啰耶喔诶";
constexpr std::string_view kDeSuffix = "的地得";
constexpr std::string_view kAspectParticle = "了着过";
constexpr std::string_view kPluralSuffix = "们子";
constexpr std::string_view kLocativeSuffix = "上下里";
constexpr std::string_view kDirectionalSuffix = "来去";
constexpr std::string_view kDirectionalPrefix = "上下进出回过起开";
constexpr std::string_view kClassifierPrefix = "几有两半多各整每做是";

/// 韵母非空且以声调数字 tone 结尾。上游对 finals 只看末字符；
/// 空韵母（拼音表缺字）一律视为不匹配，规则会跳过它。
[[nodiscard]] bool ends_with_tone(const std::string& final_with_tone, char tone) noexcept {
    return final_with_tone.size() >= 2 && final_with_tone.back() == tone;
}

/// 上游 `finals[j] = finals[j][:-1] + "5"` —— 只换末尾声调数字。
void set_tone(std::string& final_with_tone, char tone) {
    if (!final_with_tone.empty()) {
        final_with_tone.back() = tone;
    }
}

/// ch 是否是 charset（UTF-8 串）里的一个字符。
[[nodiscard]] bool contains_char(std::string_view charset, const std::string& ch) noexcept {
    std::size_t offset = 0;
    utf8::Char current;
    while (utf8::decode_next(charset, offset, current)) {
        if (charset.substr(current.offset, current.length) == ch) {
            return true;
        }
    }
    return false;
}

/// 上游 `item.isnumeric()` 的近似：该分支只发生在"一"与数字混排的词里
/// （如"二一零"），规范化后的中文文本里只剩汉字数字。
[[nodiscard]] bool is_numeral(const std::string& ch) noexcept {
    if (ch.size() == 1 && ch.front() >= '0' && ch.front() <= '9') {
        return true;
    }
    return contains_char("零一二三四五六七八九百千万亿〇", ch);
}

[[nodiscard]] bool in_must_neural(const std::string& word) {
    return must_neural_tone_words().find(word) != must_neural_tone_words().end();
}

[[nodiscard]] bool in_must_not_neural(const std::string& word) {
    return must_not_neural_tone_words().find(word) != must_not_neural_tone_words().end();
}

[[nodiscard]] bool in_lexicon(const std::string& word) {
    return segmentation_lexicon().find(word) != segmentation_lexicon().end();
}

/// 全部韵母非空且都是三声。上游 TS._all_tone_three（含 len>0 守卫）。
[[nodiscard]] bool all_tone_three(const std::vector<std::string>& finals) noexcept {
    if (finals.empty()) {
        return false;
    }
    for (const auto& final : finals) {
        if (final.empty() || final.back() != '3') {
            return false;
        }
    }
    return true;
}

/// 上游 TS._is_reduplication：二字叠词。
[[nodiscard]] bool is_reduplication(const Unit& unit) noexcept {
    return unit.chars.size() == 2 && unit.chars[0] == unit.chars[1];
}

[[nodiscard]] Unit concat(Unit head, Unit tail) {
    head.chars.insert(head.chars.end(), tail.chars.begin(), tail.chars.end());
    head.finals.insert(head.finals.end(), tail.finals.begin(), tail.finals.end());
    return head;
}

[[nodiscard]] std::string join_range(const std::vector<std::string>& chars, const std::size_t begin,
                                     const std::size_t end) {
    std::string joined;
    for (std::size_t i = begin; i < end; ++i) {
        joined += chars[i];
    }
    return joined;
}

/// 上游 `word[-2:]` 的等价物：[begin, end) 区间的至多末两个字符。
[[nodiscard]] std::string last_two_range(const std::vector<std::string>& chars,
                                         const std::size_t begin, const std::size_t end) {
    const std::size_t first = end - begin >= 2 ? end - 2 : begin;
    return join_range(chars, first, end);
}

/// 词表里的最大字符数，分词贪心匹配的上界。词表构建一次后不变。
[[nodiscard]] std::size_t max_lexicon_char_length() {
    static const std::size_t kMax = [] {
        std::size_t max_chars = 1;
        for (const auto& word : segmentation_lexicon()) {
            std::size_t offset = 0;
            utf8::Char ch;
            const std::string_view view(word);
            std::size_t count = 0;
            while (utf8::decode_next(view, offset, ch)) {
                ++count;
            }
            if (count > max_chars) {
                max_chars = count;
            }
        }
        return max_chars;
    }();
    return kMax;
}

/// "三字词拆两半"的位置：最长的词表前缀（2 字），否则 1。
///
/// 上游 TS._split_word 用 jieba.cut_for_search 取最短子词定切点；
/// 我们没有 cut_for_search，用词表近似（影响见 audit §9.6）。
/// 注：即使有 jieba，这条路径仍然只查词表 —— 上游那一步走的是
/// jieba 的**搜索模式**切分，本移植没有实现。
[[nodiscard]] std::size_t split_position_for_three(const Unit& unit) {
    if (in_lexicon(unit.chars[0] + unit.chars[1])) {
        return 2;
    }
    return 1;
}

}  // namespace

std::string Unit::word() const {
    return join_range(chars, 0, chars.size());
}

std::string Unit::last_two() const {
    return last_two_range(chars, 0, chars.size());
}

std::vector<Unit> segment(const std::vector<std::string>& chars) {
    // 对应上游 jieba_fast.posseg.lcut 的分词职责：从当前字符起，
    // 优先取词表里最长的词；否则单字。上游是统计分词，边界不总是
    // 一致 —— 只有"词表词保持整词"是两边都成立的（audit §9.6）。
    const std::size_t max_len = max_lexicon_char_length();
    const auto& lexicon = segmentation_lexicon();
    std::vector<Unit> units;
    units.reserve(chars.size());

    std::size_t i = 0;
    while (i < chars.size()) {
        bool matched = false;
        const std::size_t limit = std::min(max_len, chars.size() - i);
        for (std::size_t length = limit; length >= 2; --length) {
            const std::string candidate = join_range(chars, i, i + length);
            if (lexicon.find(candidate) != lexicon.end()) {
                Unit unit;
                unit.chars.assign(chars.begin() + static_cast<std::ptrdiff_t>(i),
                                  chars.begin() + static_cast<std::ptrdiff_t>(i + length));
                unit.finals.resize(length);
                units.push_back(std::move(unit));
                i += length;
                matched = true;
                break;
            }
        }
        if (!matched) {
            Unit unit;
            unit.chars.push_back(chars[i]);
            unit.finals.emplace_back();
            units.push_back(std::move(unit));
            ++i;
        }
    }
    return units;
}

namespace {

/// 上游 TS._merge_bu：光杆"不"与后一个词合并（"看不懂"）。
///
/// 各 merge 辅助函数都按值收段、返回新段：变调流水线要连续过五道
/// 合并，按引用就地改会让"上一道移动走的元素"被"下一道"读到
/// （moved-from 空单元，已在这种写法上栽过一次）。
[[nodiscard]] std::vector<Unit> merge_bu(std::vector<Unit> seg) {
    std::vector<Unit> merged;
    Unit pending_bu;
    bool has_pending = false;
    for (auto& unit : seg) {
        if (has_pending) {
            // 上游 word = last_word + word；"不"的真实韵母要跟着走。
            unit = concat(std::move(pending_bu), std::move(unit));
            has_pending = false;
        }
        if (unit.word() == "不") {
            pending_bu = std::move(unit);
            has_pending = true;
            continue;
        }
        merged.push_back(std::move(unit));
    }
    if (has_pending) {
        // 上游末尾补 ("不", "d")；POS 无处使用，只保留单元。
        merged.push_back(std::move(pending_bu));
    }
    return merged;
}

/// 上游 TS._merge_yi 的 function 2：光杆"一"吞掉后一个词（"一天"）。
/// function 1（"V一V"）依赖 POS=v，未移植（audit §9.6）。
[[nodiscard]] std::vector<Unit> merge_yi(std::vector<Unit> seg) {
    std::vector<Unit> merged;
    for (auto& unit : seg) {
        if (!merged.empty() && merged.back().word() == "一") {
            merged.back() = concat(std::move(merged.back()), std::move(unit));
        } else {
            merged.push_back(std::move(unit));
        }
    }
    return merged;
}

/// 上游 TS._merge_reduplication：相邻同词合并（"星" + "星" → "星星"）。
[[nodiscard]] std::vector<Unit> merge_reduplication(std::vector<Unit> seg) {
    std::vector<Unit> merged;
    for (auto& unit : seg) {
        if (!merged.empty() && unit.word() == merged.back().word()) {
            merged.back() = concat(std::move(merged.back()), std::move(unit));
        } else {
            merged.push_back(std::move(unit));
        }
    }
    return merged;
}

/// 两个三声连读合并的公共骨架（上游 TS._merge_continuous_three_tones 与
/// _merge_continuous_three_tones_2 只在"这对相邻单元是否成变调域"的
/// 判定上不同）。matches 的实参是原序的相邻单元。
template <typename Matches>
[[nodiscard]] std::vector<Unit> merge_continuous_three_tones(std::vector<Unit> seg,
                                                             Matches matches) {
    std::vector<Unit> merged;
    std::vector<bool> merge_last(seg.size(), false);
    for (std::size_t i = 0; i < seg.size(); ++i) {
        // matches 读的是原序邻居 seg[i-1] —— 入栈必须**拷贝**，把 seg[i]
        // move 走的话，下一轮的 seg[i-1] 就是空单元了（上游没有这个
        // 问题：它预存了每词韵母，从不复用原列表）。
        if (i >= 1 && matches(seg[i - 1], seg[i]) && !merge_last[i - 1]) {
            // 上游：叠词不合并 —— 叠词要走轻声规则。
            if (!is_reduplication(seg[i - 1]) &&
                seg[i - 1].chars.size() + seg[i].chars.size() <= 3) {
                merged.back() = concat(std::move(merged.back()), seg[i]);
                merge_last[i] = true;
            } else {
                merged.push_back(seg[i]);
            }
        } else {
            merged.push_back(seg[i]);
        }
    }
    return merged;
}

/// 上游 TS._merge_er：光杆"儿"并入前一个词（"花" + "儿" → "花儿"）。
[[nodiscard]] std::vector<Unit> merge_er(std::vector<Unit> seg) {
    std::vector<Unit> merged;
    for (auto& unit : seg) {
        if (!merged.empty() && unit.word() == "儿" && merged.back().word() != "#") {
            merged.back() = concat(std::move(merged.back()), std::move(unit));
        } else {
            merged.push_back(std::move(unit));
        }
    }
    return merged;
}

}  // namespace

void pre_merge_for_modify(std::vector<Unit>& seg) {
    // 两遍三声连读合并：全三声相邻 / 前尾后首三声相邻。
    // 上游在此用 lazy_pinyin 现算韵母；我们的 finals 在进这里之前已备好。
    const auto all_three = [](const Unit& prev, const Unit& cur) {
        return all_tone_three(prev.finals) && all_tone_three(cur.finals);
    };
    const auto boundary_three = [](const Unit& prev, const Unit& cur) {
        return ends_with_tone(prev.finals.back(), '3') && ends_with_tone(cur.finals.front(), '3');
    };
    seg = merge_bu(std::move(seg));
    seg = merge_yi(std::move(seg));
    seg = merge_reduplication(std::move(seg));
    seg = merge_continuous_three_tones(std::move(seg), all_three);
    seg = merge_continuous_three_tones(std::move(seg), boundary_three);
    seg = merge_er(std::move(seg));
}

namespace {

/// 上游 TS._bu_sandhi："不"变调。
void bu_sandhi(Unit& unit) {
    const auto& chars = unit.chars;
    auto& finals = unit.finals;
    // 三字中间的"不"读轻声，如"看不懂"。
    if (chars.size() == 3 && chars[1] == "不") {
        set_tone(finals[1], '5');
        return;
    }
    for (std::size_t i = 0; i < chars.size(); ++i) {
        // 四声前的"不"读二声，如"不怕"。
        if (chars[i] == "不" && i + 1 < chars.size() && ends_with_tone(finals[i + 1], '4')) {
            set_tone(finals[i], '2');
        }
    }
}

/// 上游 TS._yi_sandhi："一"变调。
void yi_sandhi(Unit& unit) {
    const auto& chars = unit.chars;
    auto& finals = unit.finals;

    bool has_yi = false;
    bool all_others_numeric = true;
    for (const auto& ch : chars) {
        if (ch == "一") {
            has_yi = true;
        } else if (!is_numeral(ch)) {
            all_others_numeric = false;
        }
    }
    // 数字序列里的"一"不变调，如"一零零"。
    if (has_yi && all_others_numeric) {
        return;
    }
    // 叠词中间的"一"读轻声，如"看一看"。
    if (chars.size() == 3 && chars[1] == "一" && chars[0] == chars[2]) {
        set_tone(finals[1], '5');
        return;
    }
    // 序数词"第一"读一声。
    if (chars.size() >= 2 && chars[0] == "第" && chars[1] == "一") {
        set_tone(finals[1], '1');
        return;
    }
    for (std::size_t i = 0; i < chars.size(); ++i) {
        if (chars[i] != "一" || i + 1 >= chars.size()) {
            continue;
        }
        // 四声前读二声，如"一段"；非四声前读四声，如"一天"；
        // 后面是标点时还读一声（纯汉字单元到不了这里，照搬）。
        if (ends_with_tone(finals[i + 1], '4')) {
            set_tone(finals[i], '2');
        } else if (!contains_char(kYiPunc, chars[i + 1])) {
            set_tone(finals[i], '4');
        }
    }
}

/// 上游 TS._neural_sandhi：轻声。
void neural_sandhi(Unit& unit) {
    const auto& chars = unit.chars;
    auto& finals = unit.finals;
    const std::string word = unit.word();

    // 名词/动词/形容词叠词第二字读轻声，如"奶奶""试试"。
    // 上游限定 POS 首字母 ∈ {n, v, a}；无 POS 直接应用，must_not 表
    // 仍豁免（"天天"一类量词叠词会被误轻声 —— 已知偏差，audit §9.6）。
    for (std::size_t j = 1; j < chars.size(); ++j) {
        if (chars[j] == chars[j - 1] && !in_must_not_neural(word)) {
            set_tone(finals[j], '5');
        }
    }

    const std::ptrdiff_t ge_idx = [&] {
        for (std::size_t i = 0; i < chars.size(); ++i) {
            if (chars[i] == "个") {
                return static_cast<std::ptrdiff_t>(i);
            }
        }
        return static_cast<std::ptrdiff_t>(-1);
    }();
    const auto last_char_in = [&](const std::string_view set) {
        return !chars.empty() && contains_char(set, chars.back());
    };

    if (last_char_in(kModalSuffix)) {
        set_tone(finals.back(), '5');
    } else if (last_char_in(kDeSuffix)) {
        set_tone(finals.back(), '5');
    } else if (chars.size() == 1 && contains_char(kAspectParticle, chars.front())) {
        // 上游限定 POS ∈ {ul, uz, ug}（"走了"的"了"）；无 POS 直接应用。
        set_tone(finals.back(), '5');
    } else if (chars.size() > 1 && last_char_in(kPluralSuffix) && !in_must_not_neural(word)) {
        // 上游限定 POS ∈ {r, n}；无 POS 直接应用。
        set_tone(finals.back(), '5');
    } else if (chars.size() > 1 && last_char_in(kLocativeSuffix)) {
        // 上游限定 POS ∈ {s, l, f}（"桌上""地下""家里"）；无 POS 直接应用。
        set_tone(finals.back(), '5');
    } else if (chars.size() > 1 && last_char_in(kDirectionalSuffix) &&
               contains_char(kDirectionalPrefix, chars[chars.size() - 2])) {
        // 趋向补语，如"上来""下去"。
        set_tone(finals.back(), '5');
    } else if ((ge_idx >= 1 &&
                (is_numeral(chars[static_cast<std::size_t>(ge_idx) - 1]) ||
                 contains_char(kClassifierPrefix, chars[static_cast<std::size_t>(ge_idx) - 1]))) ||
               word == "个") {
        // "个"作量词读轻声，如"三个""这个"。
        set_tone(finals[static_cast<std::size_t>(ge_idx)], '5');
    } else if (in_must_neural(word) || in_must_neural(unit.last_two())) {
        set_tone(finals.back(), '5');
    }

    // 上游随后 _split_word 把词拆两半、各查一遍轻声词表。我们的分词
    // 已让词表词成独立单元，这里只兜"单元含词表子串"的残余情况：
    // 取最长词表前缀切分（退化 [1, size-1]）。
    if (chars.size() >= 2) {
        std::size_t split_at = 1;
        for (std::size_t length = chars.size() - 1; length >= 2; --length) {
            if (in_lexicon(join_range(chars, 0, length))) {
                split_at = length;
                break;
            }
        }
        const std::string head = join_range(chars, 0, split_at);
        const std::string tail = join_range(chars, split_at, chars.size());
        if (in_must_neural(head) || in_must_neural(last_two_range(chars, 0, split_at))) {
            set_tone(finals[split_at - 1], '5');
        }
        if (in_must_neural(tail) || in_must_neural(last_two_range(chars, split_at, chars.size()))) {
            set_tone(finals.back(), '5');
        }
    }
}

/// 上游 TS._three_sandhi：三声连读。
void three_sandhi(Unit& unit) {
    const auto& chars = unit.chars;
    auto& finals = unit.finals;

    if (chars.size() == 2 && all_tone_three(finals)) {
        // 双字全三声：前字变二声，如"你好"。
        set_tone(finals[0], '2');
    } else if (chars.size() == 3) {
        const std::size_t split_at = split_position_for_three(unit);
        if (all_tone_three(finals)) {
            if (split_at == 2) {
                // 双字 + 单字，如"蒙古包"。
                set_tone(finals[0], '2');
                set_tone(finals[1], '2');
            } else {
                // 单字 + 双字，如"纸老虎"。
                set_tone(finals[1], '2');
            }
        } else {
            // 上游对两半各自判断（finals_list 循环）：
            //   前半（split_at==2 时两字）全三声 → 前半首字变二声；
            //   i==1 分支（后半首字三声、前半末字三声、后半非全三声）
            //   在单字后半上自相矛盾、恒不触发 —— 只有两字后半可能命中。
            if (split_at == 2) {
                if (ends_with_tone(finals[0], '3') && ends_with_tone(finals[1], '3')) {
                    set_tone(finals[0], '2');
                }
            } else {
                // 后半（两字）非全三声，且交界两字都三声 → 前半末字变二声，
                // 如"好 / 喜欢"。
                if (ends_with_tone(finals[0], '3') && ends_with_tone(finals[1], '3') &&
                    !ends_with_tone(finals[2], '3')) {
                    set_tone(finals[0], '2');
                }
            }
        }
    } else if (chars.size() == 4) {
        // 四字成语拆两半、各自双字连读。
        for (const std::size_t base : {std::size_t{0}, std::size_t{2}}) {
            if (ends_with_tone(finals[base], '3') && ends_with_tone(finals[base + 1], '3')) {
                set_tone(finals[base], '2');
            }
        }
    }
}

}  // namespace

void modified_tone(Unit& unit) {
    if (unit.chars.empty()) {
        return;
    }
    bu_sandhi(unit);
    yi_sandhi(unit);
    neural_sandhi(unit);
    three_sandhi(unit);
}

void merge_erhua(Unit& unit) {
    if (unit.chars.empty()) {
        return;
    }
    auto& finals = unit.finals;
    const std::string& last_char = unit.chars.back();

    // 1) 词尾"儿"读 er1 的修正为 er2（上游 EH 第 1 步）。
    if (last_char == "儿" && finals.back() == "er1") {
        finals.back() = "er2";
    }
    // 2) 跳过判断。上游还查 POS ∈ {a, j, nr}（形容词/简称/人名不做儿化）；
    //    无 POS 只查词表（audit §9.6）。
    const std::string word = unit.word();
    const bool in_must = must_erhua().find(word) != must_erhua().end();
    const bool in_not = not_erhua().find(word) != not_erhua().end();
    if (!in_must && in_not) {
        return;
    }
    // 3) 长度校验（上游防"……"之类长度不一致的符号；我们恒相等，防御保留）。
    if (finals.size() != unit.chars.size()) {
        return;
    }
    // 4) 词尾"儿"且读 er2/er5：改读前字的声调，如"花儿" → hua1 er1。
    //    注意这里查的是 word[-2:]（上游语义），不是整词。
    if (unit.chars.size() >= 2 && last_char == "儿" &&
        (finals.back() == "er2" || finals.back() == "er5") &&
        not_erhua().find(unit.last_two()) == not_erhua().end() &&
        !finals[finals.size() - 2].empty()) {
        finals.back() = "er";
        finals.back() += finals[finals.size() - 2].back();
    }
}

}  // namespace genie::text::sandhi
