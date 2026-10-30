# third_party/openjtalk/CMakeLists.txt —— 内置的 OpenJTalk
#
# Open JTalk（HTS Working Group，BSD 3-Clause）的**运行时**部分：
# jpcommon / mecab / mecab2njd / njd / njd2jpcommon / njd_set_* / text2mecab。
#
# 来源：https://github.com/r9y9/open_jtalk （1.11 分支，与上游
# Genie-TTS 用的 pyopenjtalk 同源 —— 上游的 pyopenjtalk 就是它 +
# Cython 绑定）。授权原文见同目录 COPYING。
#
# 为什么内置：日语文音转换必须靠 OpenJTalk 的整条链路
# （mecab 形态分析 → 全上下文 HMM → 韵律标签），
# 自己重写不现实（词典 100 MB，规则是几十年积累的）。
#
# 与上游 CMakeLists 的差别：
#   1. configure_file 的产物写到构建目录（不污染源码树 ✓）；
#   2. 目标名加 _vendor 后缀，避免与系统库撞名 ✓；
#   3. 不编译命令行工具（bin/）与词典编译器 ✓；
#   4. 源文件显式列出（不用 aux_source_directory）——
#      mecab/src/mecab-dict-index.cpp 带全局 main()，
#      链进可执行文件会抢走程序入口 ✓。
#
# **词典不在这里** —— open_jtalk_dic_utf_8-1.11 约 103 MB，
# 由音色包在运行时给出路径（见 src/text/japanese/）✓。

cmake_minimum_required(VERSION 3.16)

set(OPEN_JTALK_VERSION 1.11)
project(openjtalk_vendor C CXX)

# 源文件列表**显式写出**，不用 aux_source_directory ✓。
#
# 为什么：mecab/src/mecab-dict-index.cpp 里有一个全局 `int main()`
# （词典索引工具的入口）。它一旦被链进可执行文件，链接器就会把它
# 当成程序的入口 —— 我们的测试程序于是"变成"了 mecab 的词典编译器：
# 启动时去当前目录找 unk.def、编译词典、然后因为没有词典而崩掉。
#
# 这个坑非常隐蔽（症状是"测试莫名其妙地跑起了 mecab"），
# 所以这里逐个列出源文件，把带 main 的那个排除掉 ✓。
set(OPENJTALK_SOURCES
    jpcommon/jpcommon.c
    jpcommon/jpcommon_label.c
    jpcommon/jpcommon_node.c
    mecab/src/char_property.cpp
    mecab/src/connector.cpp
    mecab/src/context_id.cpp
    mecab/src/dictionary.cpp
    mecab/src/dictionary_compiler.cpp
    mecab/src/dictionary_generator.cpp
    mecab/src/dictionary_rewriter.cpp
    mecab/src/eval.cpp
    mecab/src/feature_index.cpp
    mecab/src/iconv_utils.cpp
    mecab/src/lbfgs.cpp
    mecab/src/learner.cpp
    mecab/src/learner_tagger.cpp
    mecab/src/libmecab.cpp
    mecab/src/mecab.cpp
    mecab/src/nbest_generator.cpp
    mecab/src/param.cpp
    mecab/src/string_buffer.cpp
    mecab/src/tagger.cpp
    mecab/src/tokenizer.cpp
    mecab/src/utils.cpp
    mecab/src/viterbi.cpp
    mecab/src/writer.cpp
    mecab2njd/mecab2njd.c
    njd/njd.c
    njd/njd_node.c
    njd2jpcommon/njd2jpcommon.c
    njd_set_accent_phrase/njd_set_accent_phrase.c
    njd_set_accent_type/njd_set_accent_type.c
    njd_set_digit/njd_set_digit.c
    njd_set_long_vowel/njd_set_long_vowel.c
    njd_set_pronunciation/njd_set_pronunciation.c
    njd_set_unvoiced_vowel/njd_set_unvoiced_vowel.c
    text2mecab/text2mecab.c
)


add_library(openjtalk_vendor STATIC ${OPENJTALK_SOURCES})

# 生成 mecab/config.h —— 写到构建目录，源码树保持干净 ✓
include(CheckIncludeFiles)
include(CheckFunctionExists)
include(CheckLibraryExists)
include(CheckTypeSize)
check_include_files(ctype.h HAVE_CTYPE_H)
check_include_files(dirent.h HAVE_DIRENT_H)
check_include_files(fcntl.h HAVE_FCNTL_H)
check_include_files(inttypes.h HAVE_INTTYPES_H)
check_include_files(io.h HAVE_IO_H)
check_include_files(memory.h HAVE_MEMORY_H)
check_include_files(setjmp.h HAVE_SETJMP_H)
check_include_files(stdint.h HAVE_STDINT_H)
check_include_files(stdlib.h HAVE_STDLIB_H)
check_include_files(strings.h HAVE_STRINGS_H)
check_include_files(string.h HAVE_STRING_H)
check_include_files(sys/mman.h HAVE_SYS_MMAN_H)
check_include_files(sys/stat.h HAVE_SYS_STAT_H)
check_include_files(sys/times.h HAVE_SYS_TIMES_H)
check_include_files(sys/types.h HAVE_SYS_TYPES_H)
check_include_files(unistd.h HAVE_UNISTD_H)
check_include_files(windows.h HAVE_WINDOWS_H)
check_function_exists(getenv HAVE_GETENV)
check_function_exists(getpagesize HAVE_GETPAGESIZE)
check_function_exists(mmap HAVE_MMAP)
check_function_exists(opendir HAVE_OPENDIR)
check_function_exists(setjmp HAVE_SETJMP)
check_function_exists(sqrt HAVE_SQRT)
check_function_exists(strstr HAVE_STRSTR)
check_library_exists(iconv iconv_open "" ICONV_CONST)
check_library_exists(m sqrt "" HAVE_LIBM)
check_type_size(char SIZEOF_CHAR)
check_type_size(int SIZEOF_INT)
check_type_size(long SIZEOF_LONG)
check_type_size("long long" SIZEOF_LONG_LONG)
check_type_size(short SIZEOF_SHORT)
check_type_size(size_t SIZEOF_SIZE_T)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/mecab/config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/mecab/config.h
    @ONLY)
target_include_directories(openjtalk_vendor PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}/mecab
    ${CMAKE_CURRENT_SOURCE_DIR}/jpcommon
    ${CMAKE_CURRENT_SOURCE_DIR}/mecab/src
    ${CMAKE_CURRENT_SOURCE_DIR}/mecab2njd
    ${CMAKE_CURRENT_SOURCE_DIR}/njd
    ${CMAKE_CURRENT_SOURCE_DIR}/njd2jpcommon
    ${CMAKE_CURRENT_SOURCE_DIR}/njd_set_accent_phrase
    ${CMAKE_CURRENT_SOURCE_DIR}/njd_set_accent_type
    ${CMAKE_CURRENT_SOURCE_DIR}/njd_set_digit
    ${CMAKE_CURRENT_SOURCE_DIR}/njd_set_long_vowel
    ${CMAKE_CURRENT_SOURCE_DIR}/njd_set_pronunciation
    ${CMAKE_CURRENT_SOURCE_DIR}/njd_set_unvoiced_vowel
    ${CMAKE_CURRENT_SOURCE_DIR}/text2mecab)

target_compile_definitions(openjtalk_vendor PUBLIC
    HAVE_CONFIG_H
    DIC_VERSION=102
    MECAB_DEFAULT_RC="dummy"
    MECAB_WITHOUT_SHARE_DIC
    PACKAGE="open_jtalk"
    VERSION="${OPEN_JTALK_VERSION}")

# 字符集固定 UTF-8（上游也是 utf8 ✓）。
target_compile_definitions(openjtalk_vendor PUBLIC
    CHARSET_UTF_8 MECAB_CHARSET=utf-8 MECAB_UTF8_USE_ONLY)

# 第三方代码 —— 不套用本项目的严格告警 ✓（那是给本项目代码用的 ✓）。
target_compile_options(openjtalk_vendor PRIVATE -w)
set_target_properties(openjtalk_vendor PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_STANDARD 99
    CXX_STANDARD 17)
