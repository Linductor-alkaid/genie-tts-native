// src/text/japanese/openjtalk.hpp —— OpenJTalk 前端
//
// 日语音素转换必须靠 OpenJTalk 的整条链路：
//   mecab 形态分析 → 全上下文 HMM（njd）→ 韵律标签（jpcommon）
//
// OpenJTalk 是 BSD 许可的 C++ 代码，内置在 third_party/openjtalk/
// （见该目录的 COPYING）。**词典不在仓库里** ——
// open_jtalk_dic_utf_8-1.11 约 103 MB，由音色包在运行时给出路径。

#ifndef GENIE_SRC_TEXT_JAPANESE_OPENJTALK_HPP
#define GENIE_SRC_TEXT_JAPANESE_OPENJTALK_HPP

#include <memory>
#include <string>
#include <vector>

namespace genie::text::japanese {

/// OpenJTalk 的文本前端（一个实例内部有状态 —— 不要跨线程共享 ✓）。
class OpenJTalk {
public:
    OpenJTalk();
    ~OpenJTalk();

    OpenJTalk(const OpenJTalk&) = delete;
    OpenJTalk& operator=(const OpenJTalk&) = delete;

    /// 载入词典目录（open_jtalk_dic_utf_8-1.11 ✓）。
    ///
    /// @return 失败返回 false（词典不存在 / 损坏 ✓）。
    bool load(const std::string& dictionary_dir);

    /// 是否已经载入词典 ✓。
    [[nodiscard]] bool loaded() const noexcept;

    /// 文本 → 全上下文标签 ✓。
    ///
    /// 标签形如 "o^N-n+i=ch/A:-2+3+3/B:.../F:5_5#0_xx@1_1|1_5/..." ✓
    /// （OpenJTalk 的标准全上下文格式 ✓）。
    [[nodiscard]] bool labels(const std::string& text, std::vector<std::string>& out);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace genie::text::japanese

#endif  // GENIE_SRC_TEXT_JAPANESE_OPENJTALK_HPP
