// src/runtime/symbols.cpp

#include "genie/symbols.hpp"

#include <algorithm>
#include <map>
#include <mutex>
#include <set>

namespace genie::symbols {
namespace {

// ---------------------------------------------------------------------------
// 上游的各个符号集合（逐条对应 SymbolsV2.py）
// ---------------------------------------------------------------------------

// 中文普通话声母。注意 "zh"/"ch"/"sh" 必须排在 "z"/"c"/"s" 之前 ——
// 上游的 get_initials() 是前缀匹配，顺序错了 "zhong" 会被切成 "z" + "hong"。
constexpr std::string_view kPinyinInitials[] = {
    "AA", "EE", "OO", "b", "c", "ch", "d",  "f", "g", "h", "j", "k", "l",
    "m",  "n",  "p",  "q", "r", "s",  "sh", "t", "w", "x", "y", "z", "zh",
};

// 基础韵母（不带声调）。
constexpr std::string_view kPinyinFinalsBase[] = {
    "E",  "En", "a",   "ai",   "an",  "ang",  "ao", "e",   "ei",   "en", "eng", "er", "i",
    "i0", "ia", "ian", "iang", "iao", "ie",   "in", "ing", "iong", "ir", "iu",  "o",  "ong",
    "ou", "u",  "ua",  "uai",  "uan", "uang", "ui", "un",  "uo",   "v",  "van", "ve", "vn",
};

// 日语（罗马字）音素。
constexpr std::string_view kJapaneseSymbols[] = {
    "I",  "N", "U",  "a", "b",  "by", "ch", "cl", "d",  "dy", "e",  "f", "g",
    "gy", "h", "hy", "i", "j",  "k",  "ky", "m",  "my", "n",  "ny", "o", "p",
    "py", "r", "ry", "s", "sh", "t",  "ts", "u",  "v",  "w",  "y",  "z",
};

// 标点与特殊分隔符。
constexpr std::string_view kPunctuationSymbols[] = {
    "!", "?", "…", ",", ".", "-", "SP", "SP2", "SP3", "UNK",
};

// 英语 ARPAbet 音素。
constexpr std::string_view kArpabetSymbols[] = {
    "AH0", "S",   "AH1", "EY2", "AE2", "EH0", "OW2", "UH0", "NG",  "B",   "G",   "AY0",
    "M",   "AA0", "F",   "AO0", "ER2", "UH1", "IY1", "AH2", "DH",  "IY0", "EY1", "IH0",
    "K",   "N",   "W",   "IY2", "T",   "AA1", "ER1", "EH2", "OY0", "UH2", "UW1", "Z",
    "AW2", "AW1", "V",   "UW2", "AA2", "ER",  "AW0", "UW0", "R",   "OW1", "EH1", "ZH",
    "AE0", "IH2", "IH",  "Y",   "JH",  "P",   "AY1", "EY0", "OY2", "TH",  "HH",  "D",
    "ER0", "CH",  "AO1", "AE1", "AO2", "OY1", "AY2", "IH1", "OW0", "L",   "SH",
};

// 韩语音素（上游是一个字符串，排序前先去重）。
constexpr std::string_view kKoreanSymbols =
    "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㅏㅓㅗㅜㅡㅣㅐㅔ空停";

// 粤语音素（上游是一个 set，排序前先去重）。
constexpr std::string_view kCantoneseSymbols[] = {
    "Yeot3",  "Yip1",   "Yyu3",   "Yeng4",  "Yut5",   "Yaan5",  "Ym5",    "Yaan6",  "Yang1",
    "Yun4",   "Yon2",   "Yui5",   "Yun2",   "Yat3",   "Ye",     "Yeot1",  "Yoeng5", "Yoek2",
    "Yam2",   "Yeon6",  "Yu6",    "Yiu3",   "Yaang6", "Yp5",    "Yai4",   "Yoek4",  "Yit6",
    "Yam5",   "Yoeng6", "Yg1",    "Yk3",    "Yoe4",   "Yam3",   "Yc",     "Yyu4",   "Yyut1",
    "Yiu4",   "Ying3",  "Yip3",   "Yaap3",  "Yau3",   "Yan4",   "Yau1",   "Yap4",   "Yk6",
    "Yok3",   "Yai1",   "Yeot6",  "Yan2",   "Yoek6",  "Yt1",    "Yoi1",   "Yit5",   "Yn4",
    "Yaau3",  "Yau4",   "Yuk6",   "Ys",     "Yuk",    "Yin6",   "Yung6",  "Ya",     "You",
    "Yaai5",  "Yau5",   "Yoi3",   "Yaak3",  "Yaat3",  "Ying2",  "Yok5",   "Yeng2",  "Yyut3",
    "Yam1",   "Yip5",   "You1",   "Yam6",   "Yaa5",   "Yi6",    "Yek4",   "Yyu2",   "Yuk5",
    "Yaam1",  "Yang2",  "Yai",    "Yiu6",   "Yin4",   "Yok4",   "Yot3",   "Yui2",   "Yeoi5",
    "Yyun6",  "Yyu5",   "Yoi5",   "Yeot2",  "Yim4",   "Yeoi2",  "Yaan1",  "Yang6",  "Yong1",
    "Yaang4", "Yung5",  "Yeon1",  "Yin2",   "Ya3",    "Yaang3", "Yg",     "Yk2",    "Yaau5",
    "Yut1",   "Yt5",    "Yip4",   "Yung4",  "Yj",     "Yong3",  "Ya1",    "Yg6",    "Yaau6",
    "Yit3",   "Yun3",   "Ying1",  "Yn2",    "Yg4",    "Yl",     "Yp3",    "Yn3",    "Yak1",
    "Yang5",  "Yoe6",   "You2",   "Yap2",   "Yak2",   "Yt3",    "Yot5",   "Yim2",   "Yi1",
    "Yn6",    "Yaat5",  "Yaam3",  "Yoek5",  "Ye3",    "Yeon4",  "Yaa2",   "Yu3",    "Yim6",
    "Ym",     "Yoe3",   "Yaai2",  "Ym2",    "Ya6",    "Yeng6",  "Yik4",   "Yot4",   "Yaai4",
    "Yyun3",  "Yu1",    "Yoeng1", "Yaap2",  "Yuk3",   "Yoek3",  "Yeng5",  "Yeoi1",  "Yiu2",
    "Yok1",   "Yo1",    "Yoek1",  "Yoeng2", "Yeon5",  "Yiu1",   "Yoeng4", "Yuk2",   "Yat4",
    "Yg5",    "Yut4",   "Yan6",   "Yin3",   "Yaa6",   "Yap1",   "Yg2",    "Yoe5",   "Yt4",
    "Ya5",    "Yo4",    "Yyu1",   "Yak3",   "Yeon2",  "Yong4",  "Ym1",    "Ye2",    "Yaang5",
    "Yoi2",   "Yeng3",  "Yn",     "Yyut4",  "Yau",    "Yaak2",  "Yaan4",  "Yek2",   "Yin1",
    "Yi5",    "Yoe2",   "Yei5",   "Yaat6",  "Yak5",   "Yp6",    "Yok6",   "Yei2",   "Yaap1",
    "Yyut5",  "Yi4",    "Yim1",   "Yk5",    "Ye4",    "Yok2",   "Yaam6",  "Yat2",   "Yon6",
    "Yei3",   "Yyu6",   "Yeot5",  "Yk4",    "Yai6",   "Yd",     "Yg3",    "Yei6",   "Yau2",
    "Yok",    "Yau6",   "Yung3",  "Yim5",   "Yut6",   "Yit1",   "Yon3",   "Yat1",   "Yaam2",
    "Yyu2",   "Yui6",   "Yt2",    "Yek6",   "Yuk",    "Ye6",    "Yang3",  "Ying6",  "Yaau1",
    "Yeon3",  "Yng",    "Yh",     "Yang4",  "Ying5",  "Yaap6",  "Yoeng3", "Yyun4",  "You3",
    "Yan5",   "Yat5",   "Yot1",   "Yun1",   "Yi3",    "Yaa1",   "Yaap4",  "You6",   "Yaang2",
    "Yaap5",  "Yaa3",   "Yaak6",  "Yeng1",  "Yaak1",  "Yo5",    "Yoi4",   "Yam4",   "Yik1",
    "Ye1",    "Yai5",   "Yung1",  "Yp2",    "Yui4",   "Yaak4",  "Yung2",  "Yak4",   "Yaat4",
    "Yeoi4",  "Yut2",   "Yin5",   "Yaau4",  "Yap6",   "Yb",     "Yaam4",  "Yw",     "Yut3",
    "Yong2",  "Yt6",    "Yaai6",  "Yap5",   "Yik5",   "Yun6",   "Yaam5",  "Yun5",   "Yik3",
    "Ya2",    "Yyut6",  "Yon4",   "Yk1",    "Yit4",   "Yak6",   "Yaan2",  "Yuk1",   "Yai2",
    "Yik2",   "Yaat2",  "Yo3",    "Ykw",    "Yn5",    "Yaa",    "Ye5",    "Yu4",    "Yei1",
    "Yai3",   "Yyut2",  "Yt",     "Yyun5",  "Yip2",   "Yaau2",  "Yiu5",   "Ym4",    "Yeoi6",
    "Yk",     "Ym6",    "Yoe1",   "Yeoi3",  "Yon",    "Yuk4",   "Yaai3",  "Yaa4",   "Yot6",
    "Yaang1", "Yei4",   "Yek1",   "Yo",     "Yp",     "Yo6",    "Yp4",    "Yan3",   "Yoi",
    "Yap3",   "Yek3",   "Yim3",   "Yz",     "Yot2",   "Yoi6",   "Yit2",   "Yu5",    "Yaan3",
    "Yan1",   "Yon5",   "Yp1",    "Yong5",  "Ygw",    "Yak",    "Yat6",   "Ying4",  "Yu2",
    "Yf",     "Ya4",    "Yon1",   "You4",   "Yik6",   "Yui1",   "Yaat1",  "Yeot4",  "Yi2",
    "Yaai1",  "Yek5",   "Ym3",    "Yong6",  "You5",   "Yyun1",  "Yn1",    "Yo2",    "Yip6",
    "Yui3",   "Yaak5",  "Yyun2",
};

/// 按 UTF-8 字节序排序并去重 —— 等价于 Python 的 sorted(set(...))。
/// 按 UTF-8 码位切分字符串。用来把上游的韩语音素串拆成单个字符。
[[nodiscard]] std::vector<std::string> split_utf8(std::string_view text) {
    std::vector<std::string> result;
    for (std::size_t i = 0; i < text.size();) {
        const auto lead = static_cast<unsigned char>(text[i]);
        std::size_t width = 1;
        if ((lead & 0xE0U) == 0xC0U) {
            width = 2;
        } else if ((lead & 0xF0U) == 0xE0U) {
            width = 3;
        } else if ((lead & 0xF8U) == 0xF0U) {
            width = 4;
        }
        width = std::min(width, text.size() - i);
        result.emplace_back(text.substr(i, width));
        i += width;
    }
    return result;
}

template <typename Range>
[[nodiscard]] std::vector<std::string> sorted_unique(const Range& items) {
    std::vector<std::string> result;
    result.reserve(items.size());
    for (const auto& item : items) {
        result.emplace_back(item);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

/// 生成符号表。规则见 symbols.hpp 的注释。
[[nodiscard]] std::vector<std::string> build_master_v2() {
    // 第 1 步：main_symbols 是一个 set（去重 + 排序）。
    std::set<std::string, std::less<>> main_symbols;
    main_symbols.insert("_");
    for (const auto item : kPinyinInitials) {
        main_symbols.emplace(item);
    }
    // 基础韵母 × 声调 1..5。注意 tone 从 1 开始、final 在内层 —— 顺序不影响
    // 结果（set 会排序），但保持与上游一致以便对照。
    for (int tone = 1; tone <= 5; ++tone) {
        for (const auto final : kPinyinFinalsBase) {
            main_symbols.emplace(std::string(final) + std::to_string(tone));
        }
    }
    for (const auto item : kJapaneseSymbols) {
        main_symbols.emplace(item);
    }
    for (const auto item : kPunctuationSymbols) {
        main_symbols.emplace(item);
    }
    for (const auto item : kArpabetSymbols) {
        main_symbols.emplace(item);
    }

    std::vector<std::string> master;
    master.reserve(main_symbols.size() + 64);
    for (const auto& symbol : main_symbols) {
        master.push_back(symbol);
    }

    // 第 3 步：方括号。
    master.emplace_back("[");
    master.emplace_back("]");

    // 第 4 步：韩语音素（上游是一个字符串，按**码位**拆成字符后去重排序）。
    // 注意不能按字节拆 —— 韩文字符是 3 字节 UTF-8。
    for (auto& item : sorted_unique(split_utf8(kKoreanSymbols))) {
        master.push_back(std::move(item));
    }

    // 第 5 步：粤语音素。
    for (auto& item : sorted_unique(std::vector<std::string_view>(
             kCantoneseSymbols, kCantoneseSymbols + std::size(kCantoneseSymbols)))) {
        master.push_back(std::move(item));
    }

    return master;
}

/// 符号 → ID 的索引。用二分查找：表是排序的，但只有主符号段有序，
/// 后面追加的方括号/韩文/粤文段是有序追加的，整体并非严格有序，
/// 所以用 std::lower_bound 之前先建 map。
struct Index {
    std::vector<std::string> symbols;
    std::map<std::string, std::int64_t, std::less<>> ids;
};

[[nodiscard]] const Index& index() {
    static const Index kIndex = [] {
        Index built;
        built.symbols = build_master_v2();
        for (std::size_t i = 0; i < built.symbols.size(); ++i) {
            built.ids.emplace(built.symbols[i], static_cast<std::int64_t>(i));
        }
        return built;
    }();
    return kIndex;
}

}  // namespace

const std::vector<std::string>& master_v2() {
    return index().symbols;
}

std::int64_t id_of(std::string_view symbol) noexcept {
    const auto& ids = index().ids;
    const auto it = ids.find(symbol);
    return it == ids.end() ? -1 : it->second;
}

std::string_view name_of(std::int64_t id) noexcept {
    const auto& symbols = index().symbols;
    if (id < 0 || static_cast<std::size_t>(id) >= symbols.size()) {
        return {};
    }
    return symbols[static_cast<std::size_t>(id)];
}

std::optional<std::int64_t> find(std::string_view symbol) noexcept {
    const auto id = id_of(symbol);
    if (id < 0) {
        return std::nullopt;
    }
    return id;
}

bool contains(std::string_view symbol) noexcept {
    return id_of(symbol) >= 0;
}

std::size_t size() noexcept {
    return index().symbols.size();
}

}  // namespace genie::symbols
