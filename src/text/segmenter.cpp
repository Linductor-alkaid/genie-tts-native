// src/text/segmenter.cpp —— 见 segmenter.hpp 的规则说明

#include "text/segmenter.hpp"

#include "genie/utf8.hpp"

namespace genie::text {
namespace {

/// 句末终止符 —— 段在这里立即落刀。
bool is_hard_terminator(std::uint32_t cp) noexcept {
    switch (cp) {
        case 0x3002:  // 。
        case 0xFF01:  // ！
        case 0xFF1F:  // ？
        case 0x0021:  // !
        case 0x003F:  // ?
        case 0x2026:  // …
        case 0x000A:  // \n
        case 0x000D:  // \r
            return true;
        default:
            return false;
    }
}

/// 逗号级标点 —— 只有段超长时才在这里落刀。
bool is_soft_terminator(std::uint32_t cp) noexcept {
    switch (cp) {
        case 0xFF0C:  // ，
        case 0x002C:  // ,
        case 0xFF1B:  // ；
        case 0x003B:  // ;
        case 0xFF1A:  // ：
        case 0x003A:  // :
        case 0x3001:  // 、
            return true;
        default:
            return false;
    }
}

/// 收尾引号 —— 粘在前一句的终止符后面，不落在段首。
bool is_closer(std::uint32_t cp) noexcept {
    switch (cp) {
        case 0x201D:  // ”
        case 0x2019:  // ’
        case 0x300F:  // 』
        case 0x300D:  // 」
            return true;
        default:
            return false;
    }
}

/// 段首尾可丢弃的空白。
///
/// 含换行：换行本身是终止符，只会出现在段边界（被落刀吞掉），
/// 留在段边缘的换行没有发音价值，按空白修剪。
bool is_ignorable_space(std::uint32_t cp) noexcept {
    return cp == 0x0020 || cp == 0x0009 || cp == 0x3000 ||  // 半角空格 / 制表符 / 全角空格
           cp == 0x000A || cp == 0x000D;                    // \n / \r
}

}  // namespace

std::vector<std::string> split_for_synthesis(std::string_view text, std::size_t max_chars) {
    std::vector<std::string> segments;
    if (text.empty()) {
        return segments;
    }

    std::size_t seg_start = 0;   // 当前段起点（字节偏移）
    std::size_t pos = 0;         // 扫描位置（字节偏移）
    std::size_t char_count = 0;  // 当前段的字符数
    std::size_t last_soft_end = 0;  // 当前段内最近一个软终止符之后的字节偏移；0 = 还没出现
    bool has_content = false;  // 当前段有没有非空白、非标点的内容

    // 交付 [seg_start, end)：去掉首尾空白后收进结果，空白段丢弃。
    // 段长上限是 max_chars（≈100 字符），重扫一遍的开销可以忽略。
    const auto emit = [&](std::size_t end) {
        std::size_t begin = seg_start;
        utf8::Char ch;
        while (begin < end) {
            std::size_t probe = begin;
            if (!utf8::decode_next(text, probe, ch)) {
                break;
            }
            if (!is_ignorable_space(ch.codepoint)) {
                break;
            }
            begin = probe;
        }
        std::size_t tail = end;
        while (tail > begin) {
            // 找 tail 前一个字符的起点（从 begin 重扫）。
            std::size_t last_begin = begin;
            std::size_t probe = begin;
            while (probe < tail) {
                last_begin = probe;
                if (!utf8::decode_next(text, probe, ch)) {
                    break;
                }
            }
            // decode_next 的 offset 是**入参引用** —— 直接传 last_begin 会被
            // 推进到下一个字符，tail 就再也不缩了（死循环）。用副本解码。
            std::size_t current = last_begin;
            if (!utf8::decode_next(text, current, ch) || !is_ignorable_space(ch.codepoint)) {
                break;
            }
            tail = last_begin;
        }
        if (tail > begin) {
            segments.emplace_back(text.substr(begin, tail - begin));
        }
    };

    // 下一个字符是不是天然边界（终止符 / 文本末尾）？只窥视、不消费 pos。
    const auto next_is_boundary = [&]() {
        std::size_t peek = pos;
        utf8::Char next;
        if (!utf8::decode_next(text, peek, next)) {
            return true;  // 文本结束 —— 边界交给收尾的 emit
        }
        return is_hard_terminator(next.codepoint) || is_soft_terminator(next.codepoint);
    };

    utf8::Char ch;
    while (utf8::decode_next(text, pos, ch)) {
        const std::uint32_t cp = ch.codepoint;
        if (is_hard_terminator(cp)) {
            // 连续终止符与紧随的收尾引号并入同一次落刀：
            // "你好！！！"是一段；"他说：“好。”" 的引号粘进前句。
            // 注意 decode_next 是"读了就走"——必须先窥视、确认是终止符
            // 类再提交 pos，否则第一个非终止符会被白吃掉一个。
            for (;;) {
                std::size_t peek = pos;
                if (!utf8::decode_next(text, peek, ch)) {
                    pos = peek;  // 到末尾
                    break;
                }
                if (!is_hard_terminator(ch.codepoint) && !is_closer(ch.codepoint)) {
                    break;  // pos 停在第一个非终止符上
                }
                pos = peek;
            }
            if (has_content) {
                emit(pos);
            }
            seg_start = pos;
            char_count = 0;
            last_soft_end = 0;
            has_content = false;
            continue;
        }
        if (!is_soft_terminator(cp) && !is_closer(cp) && !is_ignorable_space(cp)) {
            has_content = true;
        }
        if (is_soft_terminator(cp)) {
            last_soft_end = pos;
        }
        ++char_count;
        if (max_chars != 0 && char_count >= max_chars) {
            if (last_soft_end > seg_start) {
                // 回退到最近的软终止符落刀（终止符本身留在前一段）。
                emit(last_soft_end);
                seg_start = last_soft_end;
                char_count = 0;
                last_soft_end = 0;
                // 切点之后的余段回到"无内容"状态：紧随的落单终止符（比如
                // 句号的句读残余）走纯标点前缀的丢弃规则，不单独成段。
                has_content = false;
            } else if (next_is_boundary()) {
                // 超限但下一个字符就是天然边界（句号、逗号、文本末尾……）
                // —— 这一刀让给它：句尾标点留在它所属的句子里，不被硬切
                // 撕成落单的一段。**不能**在这里重置计数：当前字符还属于
                // 未落刀的段，下一个终止符迭代要靠 cc >= max 与 has_content
                // 把整段（含句尾标点）收走。
            } else {
                // 段内没有任何软点、下一个字符也不是边界 → 按字符硬切。
                emit(pos);
                seg_start = pos;
                char_count = 0;
                last_soft_end = 0;
                has_content = false;
            }
        }
    }
    if (seg_start < text.size()) {
        emit(text.size());
    }
    return segments;
}

}  // namespace genie::text
