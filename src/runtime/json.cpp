// src/runtime/json.cpp — 最小 JSON 解析器实现

#include "genie/json.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <utility>

#include "genie/errors.hpp"

namespace genie::json {
namespace {

/// 嵌套深度上限。manifest 不需要深嵌套，这个上限纯粹是防栈溢出。
constexpr int kMaxDepth = 64;

/// 解析器状态。
class Parser {
public:
    explicit Parser(std::string_view text) : text_(text) {}

    Value parse_document() {
        skip_whitespace();
        Value value = parse_value(0);
        skip_whitespace();
        if (offset_ != text_.size()) {
            fail("trailing characters after JSON value");
        }
        return value;
    }

private:
    [[noreturn]] void fail(const std::string& message) const {
        // 行列信息：行号按换行计数，列号是当前行内的偏移 + 1。
        std::size_t line = 1;
        std::size_t column = 1;
        for (std::size_t i = 0; i < offset_ && i < text_.size(); ++i) {
            if (text_[i] == '\n') {
                ++line;
                column = 1;
            } else {
                ++column;
            }
        }
        throw_error(ErrorCode::InvalidArgument, "JSON parse error at line " + std::to_string(line) +
                                                    ", column " + std::to_string(column) + ": " +
                                                    message);
    }

    [[nodiscard]] bool done() const noexcept { return offset_ >= text_.size(); }
    [[nodiscard]] char peek() const noexcept { return text_[offset_]; }

    void skip_whitespace() {
        while (!done()) {
            const char ch = text_[offset_];
            if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
                ++offset_;
            } else {
                break;
            }
        }
    }

    void expect(char ch) {
        if (done() || text_[offset_] != ch) {
            fail(std::string("expected '") + ch + "'");
        }
        ++offset_;
    }

    Value parse_value(int depth) {
        if (depth > kMaxDepth) {
            fail("nesting too deep");
        }
        if (done()) {
            fail("unexpected end of input");
        }
        switch (peek()) {
            case '{':
                return parse_object(depth);
            case '[':
                return parse_array(depth);
            case '"':
                return Value(parse_string());
            case 't':
                expect_literal("true");
                return Value(true);
            case 'f':
                expect_literal("false");
                return Value(false);
            case 'n':
                expect_literal("null");
                return Value();
            default:
                return parse_number();
        }
    }

    void expect_literal(std::string_view literal) {
        if (text_.substr(offset_, literal.size()) != literal) {
            fail("invalid literal");
        }
        offset_ += literal.size();
    }

    Value parse_object(int depth) {
        expect('{');
        ObjectType object;
        skip_whitespace();
        if (!done() && peek() == '}') {
            ++offset_;
            return Value(std::move(object));
        }
        for (;;) {
            skip_whitespace();
            if (done() || peek() != '"') {
                fail("expected object key");
            }
            std::string key = parse_string();
            skip_whitespace();
            expect(':');
            skip_whitespace();
            object.emplace(std::move(key), parse_value(depth + 1));
            skip_whitespace();
            if (done()) {
                fail("unterminated object");
            }
            if (peek() == ',') {
                ++offset_;
                continue;
            }
            if (peek() == '}') {
                ++offset_;
                break;
            }
            fail("expected ',' or '}'");
        }
        return Value(std::move(object));
    }

    Value parse_array(int depth) {
        expect('[');
        ArrayType array;
        skip_whitespace();
        if (!done() && peek() == ']') {
            ++offset_;
            return Value(std::move(array));
        }
        for (;;) {
            skip_whitespace();
            array.push_back(parse_value(depth + 1));
            skip_whitespace();
            if (done()) {
                fail("unterminated array");
            }
            if (peek() == ',') {
                ++offset_;
                continue;
            }
            if (peek() == ']') {
                ++offset_;
                break;
            }
            fail("expected ',' or ']'");
        }
        return Value(std::move(array));
    }

    /// 把码点编码为 UTF-8，追加到 out。
    static void append_utf8(std::string& out, std::uint32_t codepoint) {
        if (codepoint <= 0x7FU) {
            out.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7FFU) {
            out.push_back(static_cast<char>(0xC0U | (codepoint >> 6)));
            out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
        } else if (codepoint <= 0xFFFFU) {
            out.push_back(static_cast<char>(0xE0U | (codepoint >> 12)));
            out.push_back(static_cast<char>(0x80U | ((codepoint >> 6) & 0x3FU)));
            out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
        } else {
            out.push_back(static_cast<char>(0xF0U | (codepoint >> 18)));
            out.push_back(static_cast<char>(0x80U | ((codepoint >> 12) & 0x3FU)));
            out.push_back(static_cast<char>(0x80U | ((codepoint >> 6) & 0x3FU)));
            out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
        }
    }

    std::uint32_t parse_hex4() {
        if (offset_ + 4 > text_.size()) {
            fail("truncated \\u escape");
        }
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const char ch = text_[offset_++];
            value <<= 4;
            if (ch >= '0' && ch <= '9') {
                value |= static_cast<std::uint32_t>(ch - '0');
            } else if (ch >= 'a' && ch <= 'f') {
                value |= static_cast<std::uint32_t>(ch - 'a' + 10);
            } else if (ch >= 'A' && ch <= 'F') {
                value |= static_cast<std::uint32_t>(ch - 'A' + 10);
            } else {
                fail("invalid hex digit in \\u escape");
            }
        }
        return value;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        for (;;) {
            if (done()) {
                fail("unterminated string");
            }
            const char ch = text_[offset_++];
            if (ch == '"') {
                break;
            }
            if (ch != '\\') {
                if (static_cast<unsigned char>(ch) < 0x20U) {
                    fail("unescaped control character in string");
                }
                out.push_back(ch);
                continue;
            }
            if (done()) {
                fail("unterminated escape sequence");
            }
            const char escape = text_[offset_++];
            switch (escape) {
                case '"':
                    out.push_back('"');
                    break;
                case '\\':
                    out.push_back('\\');
                    break;
                case '/':
                    out.push_back('/');
                    break;
                case 'b':
                    out.push_back('\b');
                    break;
                case 'f':
                    out.push_back('\f');
                    break;
                case 'n':
                    out.push_back('\n');
                    break;
                case 'r':
                    out.push_back('\r');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                case 'u': {
                    std::uint32_t codepoint = parse_hex4();
                    // 代理对：高低代理必须成对
                    if (codepoint >= 0xD800U && codepoint <= 0xDBFFU) {
                        if (offset_ + 1 < text_.size() && text_[offset_] == '\\' &&
                            text_[offset_ + 1] == 'u') {
                            offset_ += 2;
                            const std::uint32_t low = parse_hex4();
                            if (low < 0xDC00U || low > 0xDFFFU) {
                                fail("invalid low surrogate in \\u escape");
                            }
                            codepoint = 0x10000U + ((codepoint - 0xD800U) << 10) + (low - 0xDC00U);
                        } else {
                            fail("unpaired high surrogate in \\u escape");
                        }
                    } else if (codepoint >= 0xDC00U && codepoint <= 0xDFFFU) {
                        fail("unpaired low surrogate in \\u escape");
                    }
                    append_utf8(out, codepoint);
                    break;
                }
                default:
                    fail("invalid escape sequence");
            }
        }
        return out;
    }

    Value parse_number() {
        const std::size_t start = offset_;
        if (!done() && peek() == '-') {
            ++offset_;
        }
        // 整数部分：'0' 之后不能再跟数字（JSON 不允许前导零）
        if (!done() && peek() == '0') {
            ++offset_;
        } else {
            while (!done() && peek() >= '0' && peek() <= '9') {
                ++offset_;
            }
        }
        bool is_double = false;
        if (!done() && peek() == '.') {
            is_double = true;
            ++offset_;
            while (!done() && peek() >= '0' && peek() <= '9') {
                ++offset_;
            }
        }
        if (!done() && (peek() == 'e' || peek() == 'E')) {
            is_double = true;
            ++offset_;
            if (!done() && (peek() == '+' || peek() == '-')) {
                ++offset_;
            }
            while (!done() && peek() >= '0' && peek() <= '9') {
                ++offset_;
            }
        }
        if (offset_ == start) {
            fail("invalid number");
        }
        const std::string text(text_.substr(start, offset_ - start));
        if (!is_double) {
            try {
                std::size_t consumed = 0;
                const auto value = std::stoll(text, &consumed);
                if (consumed == text.size()) {
                    return Value(static_cast<std::int64_t>(value));
                }
            } catch (const std::exception&) {
                // 超出 int64 范围 → 退化成 double
            }
        }
        try {
            return Value(std::stod(text));
        } catch (const std::exception&) {
            fail("invalid number: " + text);
        }
    }

    std::string_view text_;
    std::size_t offset_ = 0;
};

/// 把字符串按 JSON 规则转义。
void escape_into(const std::string& input, std::string& out) {
    out.push_back('"');
    for (const char ch : input) {
        switch (ch) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20U) {
                    static constexpr char kHex[] = "0123456789abcdef";
                    out += "\\u00";
                    out.push_back(kHex[(static_cast<unsigned char>(ch) >> 4) & 0x0FU]);
                    out.push_back(kHex[static_cast<unsigned char>(ch) & 0x0FU]);
                } else {
                    out.push_back(ch);
                }
                break;
        }
    }
    out.push_back('"');
}

}  // namespace

std::int64_t Value::as_int(std::int64_t fallback) const noexcept {
    if (type_ == Type::Int) {
        return int_;
    }
    if (type_ == Type::Double) {
        return static_cast<std::int64_t>(double_);
    }
    return fallback;
}

double Value::as_double(double fallback) const noexcept {
    if (type_ == Type::Double) {
        return double_;
    }
    if (type_ == Type::Int) {
        return static_cast<double>(int_);
    }
    return fallback;
}

const Value* Value::find(std::string_view key) const {
    if (type_ != Type::Object) {
        return nullptr;
    }
    const auto it = object_.find(std::string(key));
    return it == object_.end() ? nullptr : &it->second;
}

std::int64_t Value::get_int(std::string_view key, std::int64_t fallback) const {
    const Value* value = find(key);
    return value != nullptr ? value->as_int(fallback) : fallback;
}

bool Value::get_bool(std::string_view key, bool fallback) const {
    const Value* value = find(key);
    return value != nullptr ? value->as_bool(fallback) : fallback;
}

std::string Value::get_string(std::string_view key, std::string fallback) const {
    const Value* value = find(key);
    if (value == nullptr || !value->is_string()) {
        return fallback;
    }
    return value->as_string();
}

std::string Value::dump() const {
    std::string out;
    switch (type_) {
        case Type::Null:
            out += "null";
            break;
        case Type::Bool:
            out += bool_ ? "true" : "false";
            break;
        case Type::Int:
            out += std::to_string(int_);
            break;
        case Type::Double: {
            std::ostringstream stream;
            stream << double_;
            out += stream.str();
            break;
        }
        case Type::String:
            escape_into(string_, out);
            break;
        case Type::Array: {
            out.push_back('[');
            bool first = true;
            for (const auto& item : array_) {
                if (!first) {
                    out.push_back(',');
                }
                first = false;
                out += item.dump();
            }
            out.push_back(']');
            break;
        }
        case Type::Object: {
            out.push_back('{');
            bool first = true;
            for (const auto& [key, item] : object_) {
                if (!first) {
                    out.push_back(',');
                }
                first = false;
                escape_into(key, out);
                out.push_back(':');
                out += item.dump();
            }
            out.push_back('}');
            break;
        }
    }
    return out;
}

Value parse(std::string_view text) {
    Parser parser(text);
    return parser.parse_document();
}

Value parse_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw_error(ErrorCode::IoError, "cannot open JSON file: " + path);
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    const std::string content = buffer.str();
    try {
        return parse(content);
    } catch (const Error& error) {
        throw_error(ErrorCode::InvalidArgument, path + ": " + error.what());
    }
}

}  // namespace genie::json
