# aarch64-linux-gnu.cmake — Linux ARM64 交叉编译工具链
#
# 用法（Ubuntu/Debian，需先安装交叉工具链）：
#   sudo apt-get install g++-aarch64-linux-gnu file
#   cmake --preset arm64-release -DGENIE_ONNXRUNTIME_ROOT=<aarch64 ORT 包根目录>
#
# ORT 用官方的 onnxruntime-linux-aarch64-<版本>.tgz（与 host x64 包同名布局）。
# 交叉环境只做**编译验证**：产物是 aarch64 ELF，不能在 x86 宿主机上运行。
#
# 说明：
# - 编译器名是 Ubuntu/Debian 包 gcc-aarch64-linux-gnu / g++-aarch64-linux-gnu
#   装出来的标准名字；其他发行版如前缀不同，在此处改两个字段即可。
# - CMAKE_FIND_ROOT_PATH 收敛到交叉 sysroot，防止 host 上装过的 x86
#   onnxruntime config 包（find_package(CONFIG)）混进来。GENIE_ONNXRUNTIME_ROOT
#   的解析（cmake/Dependencies.cmake）用的是绝对路径 HINTS + NO_DEFAULT_PATH，
#   不受 root 重定向影响，所以显式传的 aarch64 ORT 包一定胜出。

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# 交叉 gcc 的内置 sysroot 就是 /usr/aarch64-linux-gnu（gcc-aarch64-linux-gnu 包约定），
# 这里只影响 find_* 的搜索范围，不影响编译器自身。
set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)

# 程序（ninja、git、python3 —— FetchContent 需要）一律用宿主机的。
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
