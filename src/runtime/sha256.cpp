// src/runtime/sha256.cpp — SHA-256 (FIPS 180-4)

#include "genie/sha256.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <utility>

#include "genie/errors.hpp"

namespace genie {
namespace {

constexpr std::array<std::uint32_t, 64> kRoundConstants{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U,
    0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU,
    0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
    0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU,
    0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U,
    0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U,
    0xc67178f2U};

constexpr std::uint32_t rotr(std::uint32_t value, std::uint32_t bits) noexcept {
    return (value >> bits) | (value << (32U - bits));
}

/// 增量式 SHA-256 上下文。
class Sha256Context {
public:
    void update(const std::uint8_t* data, std::size_t size) {
        for (std::size_t i = 0; i < size; ++i) {
            buffer_[buffer_size_++] = data[i];
            if (buffer_size_ == 64) {
                transform(buffer_.data());
                bit_length_ += 512;
                buffer_size_ = 0;
            }
        }
    }

    [[nodiscard]] Sha256Digest finish() {
        std::size_t i = buffer_size_;
        if (buffer_size_ < 56) {
            buffer_[i++] = 0x80;
            while (i < 56) {
                buffer_[i++] = 0x00;
            }
        } else {
            buffer_[i++] = 0x80;
            while (i < 64) {
                buffer_[i++] = 0x00;
            }
            transform(buffer_.data());
            std::memset(buffer_.data(), 0, 56);
        }

        bit_length_ += static_cast<std::uint64_t>(buffer_size_) * 8U;
        buffer_[63] = static_cast<std::uint8_t>(bit_length_);
        buffer_[62] = static_cast<std::uint8_t>(bit_length_ >> 8U);
        buffer_[61] = static_cast<std::uint8_t>(bit_length_ >> 16U);
        buffer_[60] = static_cast<std::uint8_t>(bit_length_ >> 24U);
        buffer_[59] = static_cast<std::uint8_t>(bit_length_ >> 32U);
        buffer_[58] = static_cast<std::uint8_t>(bit_length_ >> 40U);
        buffer_[57] = static_cast<std::uint8_t>(bit_length_ >> 48U);
        buffer_[56] = static_cast<std::uint8_t>(bit_length_ >> 56U);
        transform(buffer_.data());

        Sha256Digest digest{};
        for (std::size_t j = 0; j < 8; ++j) {  // 8 个状态字 → 32 字节
            digest[j * 4 + 0] = static_cast<std::uint8_t>(state_[j] >> 24U);
            digest[j * 4 + 1] = static_cast<std::uint8_t>(state_[j] >> 16U);
            digest[j * 4 + 2] = static_cast<std::uint8_t>(state_[j] >> 8U);
            digest[j * 4 + 3] = static_cast<std::uint8_t>(state_[j]);
        }
        return digest;
    }

private:
    void transform(const std::uint8_t* chunk) {
        std::array<std::uint32_t, 64> w{};
        for (std::size_t i = 0; i < 16; ++i) {
            w[i] = (static_cast<std::uint32_t>(chunk[i * 4 + 0]) << 24U) |
                   (static_cast<std::uint32_t>(chunk[i * 4 + 1]) << 16U) |
                   (static_cast<std::uint32_t>(chunk[i * 4 + 2]) << 8U) |
                   static_cast<std::uint32_t>(chunk[i * 4 + 3]);
        }
        for (std::size_t i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3U);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10U);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];

        for (std::size_t i = 0; i < 64; ++i) {
            const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 = h + s1 + ch + kRoundConstants[i] + w[i];
            const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = s0 + maj;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                                        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t buffer_size_ = 0;
    std::uint64_t bit_length_ = 0;
};

}  // namespace

Sha256Digest sha256(const std::vector<std::uint8_t>& data) {
    Sha256Context context;
    if (!data.empty()) {
        context.update(data.data(), data.size());
    }
    return context.finish();
}

Sha256Digest sha256(std::string_view data) {
    Sha256Context context;
    if (!data.empty()) {
        context.update(reinterpret_cast<const std::uint8_t*>(data.data()), data.size());
    }
    return context.finish();
}

std::string sha256_hex(std::string_view data) {
    return to_hex(sha256(data));
}

Sha256Digest sha256_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw_error(ErrorCode::IoError, "cannot open file for hashing: " + path);
    }
    Sha256Context context;
    std::vector<char> chunk(64 * 1024);
    while (file) {
        file.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        const auto read = static_cast<std::size_t>(file.gcount());
        if (read > 0) {
            context.update(reinterpret_cast<const std::uint8_t*>(chunk.data()), read);
        }
    }
    return context.finish();
}

std::string to_hex(const Sha256Digest& digest) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(digest.size() * 2);
    for (const std::uint8_t byte : digest) {
        out.push_back(kHex[(byte >> 4U) & 0x0FU]);
        out.push_back(kHex[byte & 0x0FU]);
    }
    return out;
}

}  // namespace genie
