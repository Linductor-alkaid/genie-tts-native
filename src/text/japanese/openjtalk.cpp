// src/text/japanese/openjtalk.cpp —— OpenJTalk 前端的 C++ 封装。
//
// 链路与 pyopenjtalk 的 `run_frontend` + `make_label` 一致：
//
//   text2mecab(buff, text)
//   Mecab_analysis(mecab, buff)
//   mecab2njd(njd, Mecab_get_feature(mecab), Mecab_get_size(mecab))
//   njd_set_pronunciation / njd_set_digit / njd_set_accent_phrase
//   njd_set_accent_type / njd_set_unvoiced_vowel / njd_set_long_vowel
//   njd2jpcommon(jpcommon, njd)
//   JPCommon_make_label(jpcommon)
//
// 与 pyopenjtalk 的差别：Python 那边为了跨语言边界，把中间结果转成
// dict 又转回 NJD 节点；C++ 这边不需要绕一圈 ✓（同样的数据 ✓）。

#include "text/japanese/openjtalk.hpp"

// 头文件的依赖顺序有要求：mecab2njd.h 需要 NJD ✓，
// njd2jpcommon.h 需要 NJD 与 JPCommon ✓ —— 这些第三方头**不是自包含**的 ✓
// （它们不 include 自己的依赖 ✓）。
//
// 所以这里用**空行分组** ✓：基础头一组 ✓、依赖头一组 ✓ ——
// clang-format 只在组内排序 ✓（IncludeBlocks: Preserve ✓），
// 组与组之间的顺序不动 ✓。合成一组的话排序会把依赖头排到前面 ✓
// → 编不过 ✓。
#include <jpcommon.h>
#include <mecab.h>
#include <njd.h>

#include <mecab2njd.h>
#include <njd2jpcommon.h>
#include <njd_set_accent_phrase.h>
#include <njd_set_accent_type.h>
#include <njd_set_digit.h>
#include <njd_set_long_vowel.h>
#include <njd_set_pronunciation.h>
#include <njd_set_unvoiced_vowel.h>
#include <text2mecab.h>

#include <cstring>
#include <string>
#include <vector>

namespace genie::text::japanese {

struct OpenJTalk::Impl {
    Mecab mecab{};
    NJD njd{};
    JPCommon jpcommon{};
    bool loaded = false;
};

OpenJTalk::OpenJTalk() : impl_(std::make_unique<Impl>()) {
    Mecab_initialize(&impl_->mecab);
    NJD_initialize(&impl_->njd);
    JPCommon_initialize(&impl_->jpcommon);
}

OpenJTalk::~OpenJTalk() {
    if (impl_ == nullptr) {
        return;
    }
    Mecab_clear(&impl_->mecab);
    NJD_clear(&impl_->njd);
    JPCommon_clear(&impl_->jpcommon);
}

bool OpenJTalk::load(const std::string& dictionary_dir) {
    if (dictionary_dir.empty()) {
        return false;
    }
    // 词典目录由调用方给出（音色包 ✓）—— mecab 的 dicdir 是运行时参数 ✓。
    if (Mecab_load(&impl_->mecab, dictionary_dir.c_str()) == 0) {
        return false;
    }
    impl_->loaded = true;
    return true;
}

bool OpenJTalk::loaded() const noexcept {
    return impl_ != nullptr && impl_->loaded;
}

bool OpenJTalk::labels(const std::string& text, std::vector<std::string>& out) {
    out.clear();
    if (!loaded()) {
        return false;
    }
    // text2mecab 的缓冲区固定 8192（与上游一致 ✓）。
    std::vector<char> buffer(8192, '\0');
    text2mecab(buffer.data(), text.c_str());

    if (Mecab_analysis(&impl_->mecab, buffer.data()) == 0) {
        return false;
    }
    mecab2njd(&impl_->njd, Mecab_get_feature(&impl_->mecab), Mecab_get_size(&impl_->mecab));
    njd_set_pronunciation(&impl_->njd);
    njd_set_digit(&impl_->njd);
    njd_set_accent_phrase(&impl_->njd);
    njd_set_accent_type(&impl_->njd);
    njd_set_unvoiced_vowel(&impl_->njd);
    njd_set_long_vowel(&impl_->njd);

    njd2jpcommon(&impl_->jpcommon, &impl_->njd);
    JPCommon_make_label(&impl_->jpcommon);

    const int size = JPCommon_get_label_size(&impl_->jpcommon);
    char** features = JPCommon_get_label_feature(&impl_->jpcommon);
    if (features != nullptr) {
        out.reserve(static_cast<std::size_t>(size));
        for (int i = 0; i < size; ++i) {
            if (features[i] != nullptr) {
                out.emplace_back(features[i]);
            }
        }
    }

    // 释放中间状态（下一轮要用就重新填 ✓）。
    JPCommon_refresh(&impl_->jpcommon);
    NJD_refresh(&impl_->njd);
    Mecab_refresh(&impl_->mecab);
    return true;
}

}  // namespace genie::text::japanese
