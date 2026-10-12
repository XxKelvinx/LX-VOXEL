#pragma once

#include <cstdlib>

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#include <intrin.h>
#endif

namespace lx_cpu {

enum class Path : unsigned char { SSE41, AVX2_FMA, NEON };

constexpr bool is_x86() noexcept {
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    return true;
#else
    return false;
#endif
}

constexpr bool is_arm() noexcept {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    return true;
#else
    return false;
#endif
}

inline bool has_avx2_fma() noexcept {
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__i386__) || defined(__x86_64__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
    int leaf1[4]{};
    __cpuid(leaf1, 1);
    constexpr int kFma = 1 << 12;
    constexpr int kOsxsave = 1 << 27;
    constexpr int kAvx = 1 << 28;
    if ((leaf1[2] & (kFma | kOsxsave | kAvx)) != (kFma | kOsxsave | kAvx)) return false;
    if ((_xgetbv(0) & 0x6) != 0x6) return false;
    int leaf0[4]{};
    __cpuid(leaf0, 0);
    if (leaf0[0] < 7) return false;
    int leaf7[4]{};
    __cpuidex(leaf7, 7, 0);
    return (leaf7[1] & (1 << 5)) != 0;
#else
    return false;
#endif
}

inline bool force_baseline() noexcept {
    const char* value = std::getenv("LXVOXEL_FORCE_BASELINE");
    if (!value) value = std::getenv("LXVOXEL_FORCE_SCALAR"); // old research switch
    return value && std::atoi(value) != 0;
}

inline Path best_path() noexcept {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(__ARM_NEON__)
    return Path::NEON;
#elif defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    return (!force_baseline() && has_avx2_fma()) ? Path::AVX2_FMA : Path::SSE41;
#else
#error "LX-VOXEL production targets require x86 SSE4.1 or ARM NEON"
#endif
}

inline const char* name(Path p) noexcept {
    switch (p) {
        case Path::SSE41: return "sse4.1";
        case Path::AVX2_FMA: return "avx2+fma";
        case Path::NEON: return "neon";
    }
    return "unknown";
}

} // namespace lx_cpu
