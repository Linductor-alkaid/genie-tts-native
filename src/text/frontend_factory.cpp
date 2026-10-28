// src/text/frontend_factory.cpp
//
// 文本前端的工厂。
//
// 目前只内置了中文前端。英文 / 日文 / 韩文的上游实现分别依赖 nltk、
// pyopenjtalk-plus、jamo —— 全是 Python 生态，且英语的 CMU 词典有 3.5MB，
// 不适合内嵌进端侧库。这些语言留成可插拔接口：
// 使用者自己实现 `TextFrontend` 接进来即可。
//
// 差异记录在 docs/upstream-audit.md。

#include <memory>

#include "genie/text.hpp"
#include "text/frontend_factory_internal.hpp"

namespace genie::text {

std::unique_ptr<TextFrontend> make_frontend(Language language) {
    switch (language) {
        case Language::Chinese:
        case Language::HybridChineseEnglish:
            return make_chinese_frontend();
        case Language::Auto: {
            // Auto 交给调用方先跑 detect_language()。这里按中文处理，
            // 至少不会把文本整个丢掉。
            return make_chinese_frontend();
        }
        case Language::English:
            // 英文前端需要词典 —— 没有资源就返回 nullptr（调用方据此报错，
            // 而不是静默产出空音素）。
            return nullptr;
        case Language::Korean:
            return make_korean_frontend();
        case Language::Japanese:
            return nullptr;
    }
    return nullptr;
}

std::unique_ptr<TextFrontend> make_frontend(Language language,
                                            const EnglishG2pResources& resources) {
    if (language == Language::English) {
        return make_english_frontend(resources);
    }
    return make_frontend(language);
}

}  // namespace genie::text
