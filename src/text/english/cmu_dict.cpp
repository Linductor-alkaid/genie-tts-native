// src/text/english/cmu_dict.cpp —— CMU 发音词典容器的加载与查表。
//
// 上游（Genie-TTS）的英文 G2P 用 CMU 发音词典查词：
// `G2P/English/EnglishG2P.py` 的 `_read_cmu_dict` 把词典读成
// `{word: [读音]}`，**每个词只保留第一个读音**，词尾的 `(2)` 序号被剥掉，
// 词统一小写。
//
// 容器由 tools/deploy/convert_cmu_dict.py 生成，落在
// third_party/english/cmudict.bin。词典本体是 CMU Pronouncing Dictionary
// （BSD 许可，见 third_party/english/LICENSE）。
//
// 词按字节序排序 → 运行时二分查找。

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

constexpr char kMagic[] = "GENIECMU";
constexpr std::uint32_t kVersion = 1;

struct Reader {
    const std::vector<unsigned char>& data;
    std::size_t pos = 0;

    void take(void* out, std::size_t n) {
        if (pos + n > data.size()) {
            throw_error(ErrorCode::InvalidArgument, "cmu dict container truncated");
        }
        std::memcpy(out, data.data() + pos, n);
        pos += n;
    }
    std::uint32_t u32() {
        std::uint32_t v = 0;
        take(&v, sizeof(v));
        return v;
    }
    std::uint16_t u16() {
        std::uint16_t v = 0;
        take(&v, sizeof(v));
        return v;
    }
};

}  // namespace

CmuDict CmuDict::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw_error(ErrorCode::InvalidArgument, "cmu dict container not found: " + path);
    }
    std::vector<unsigned char> data((std::istreambuf_iterator<char>(file)),
                                    std::istreambuf_iterator<char>());
    if (data.size() < sizeof(kMagic) - 1) {
        throw_error(ErrorCode::InvalidArgument, "cmu dict container too small");
    }
    if (std::memcmp(data.data(), kMagic, sizeof(kMagic) - 1) != 0) {
        throw_error(ErrorCode::InvalidArgument, "cmu dict container magic mismatch");
    }

    CmuDict dict;
    Reader reader{data, sizeof(kMagic) - 1};
    const std::uint32_t version = reader.u32();
    if (version != kVersion) {
        throw_error(ErrorCode::InvalidArgument, "cmu dict container version mismatch");
    }
    const std::uint32_t word_count = reader.u32();
    const std::uint32_t phoneme_name_count = reader.u32();
    const std::uint32_t phoneme_count = reader.u32();

    dict.phoneme_names_.reserve(phoneme_name_count);
    for (std::uint32_t i = 0; i < phoneme_name_count; ++i) {
        const std::uint32_t length = reader.u32();
        std::string name(length, '\0');
        reader.take(name.data(), length);
        dict.phoneme_names_.push_back(std::move(name));
    }

    struct RawEntry {
        std::uint32_t word_offset;
        std::uint32_t word_length;
        std::uint32_t pron_offset;
        std::uint32_t pron_length;
    };
    std::vector<RawEntry> raw(word_count);
    for (std::uint32_t i = 0; i < word_count; ++i) {
        raw[i].word_offset = reader.u32();
        raw[i].word_length = reader.u32();
        raw[i].pron_offset = reader.u32();
        raw[i].pron_length = reader.u32();
    }

    const std::size_t words_start = reader.pos;
    // 词字节区长度 = max(word_offset + word_length + 1)（偏移相对词区起点）
    // 那个 +1 是词后的 NUL 分隔符（见 tools/deploy/convert_cmu_dict.py）。
    const std::size_t words_end = words_start + [&] {
        std::size_t end = 0;
        for (const auto& entry : raw) {
            end = std::max(end, static_cast<std::size_t>(entry.word_offset) + entry.word_length + 1);
        }
        return end;
    }();
    if (words_end > data.size()) {
        throw_error(ErrorCode::InvalidArgument, "cmu dict container truncated (words)");
    }
    dict.words_.assign(reinterpret_cast<const char*>(data.data()) + words_start,
                       reinterpret_cast<const char*>(data.data()) + words_end);
    // 跳过词字节区，接着读音素 ID
    reader.pos = words_end;

    dict.phoneme_ids_.resize(phoneme_count);
    for (std::uint32_t i = 0; i < phoneme_count; ++i) {
        dict.phoneme_ids_[i] = reader.u16();
    }
    if (reader.pos != data.size()) {
        throw_error(ErrorCode::InvalidArgument, "cmu dict container has trailing bytes");
    }

    dict.entries_.reserve(word_count);
    for (const auto& entry : raw) {
        dict.entries_.push_back(
            Entry{entry.word_offset, entry.word_length, entry.pron_offset, entry.pron_length});
    }
    dict.count_ = word_count;
    return dict;
}

std::vector<std::string> CmuDict::lookup(std::string_view word) const {
    // 二分查找：词按字节序排序
    std::size_t low = 0;
    std::size_t high = entries_.size();
    while (low < high) {
        const std::size_t mid = low + (high - low) / 2;
        const Entry& entry = entries_[mid];
        const std::string_view candidate =
            std::string_view(words_).substr(entry.word_offset, entry.word_length);
        if (candidate < word) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    if (low >= entries_.size()) {
        return {};
    }
    const Entry& entry = entries_[low];
    const std::string_view candidate =
        std::string_view(words_).substr(entry.word_offset, entry.word_length);
    if (candidate != word) {
        return {};
    }
    std::vector<std::string> phonemes;
    phonemes.reserve(entry.pron_length);
    for (std::uint32_t i = 0; i < entry.pron_length; ++i) {
        const std::uint16_t id = phoneme_ids_[entry.pron_offset + i];
        if (id < phoneme_names_.size()) {
            phonemes.push_back(phoneme_names_[id]);
        }
    }
    return phonemes;
}

}  // namespace genie::text::english
