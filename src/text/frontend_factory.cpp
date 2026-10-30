// src/text/frontend_factory.cpp
//
// 文本前端的工厂。
//
// 内置中文前端；英文 / 日文 / 韩文前端也内置了（各带一套词典与
// 模型资源，见 src/text/english、src/text/japanese、src/text/korean），
// 但资源不随仓库发布 —— 由音色包角色给出（缺资源时明确报错，
// 不静默回退到中文前端）。
//
// 需要自定义前端的使用者可以自己实现 `TextFrontend` 接进来。
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
            // 日文前端需要 OpenJTalk + 词典 —— 由引擎那一侧注入 ✓
            // （make_frontend(Language, resources) 的重载负责 ✓）。
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

std::unique_ptr<TextFrontend> make_frontend(Language language,
                                            const JapaneseG2pResources& resources) {
#ifdef GENIE_HAVE_OPENJTALK
    if (language == Language::Japanese) {
        return make_japanese_frontend(resources);
    }
#endif
    // 没编 OpenJTalk 时没有日文前端 ✓（调用方据此报错 ✓）。
    return make_frontend(language);
}

}  // namespace genie::text
