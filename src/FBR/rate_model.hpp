#pragma once
#include <array>
#include <algorithm>
#include <cstddef>

namespace lx::rate_model {

struct Knot { float kbps; float rho; };

// R339 physical-rate recalibration after the R338 hybrid sideinfo coder.
// 96..512 kbps is calibrated on 14 music tracks (first 20 s each). The corpus
// mean is intentionally kept only slightly above nominal so lossless entropy
// savings are reinvested into quantizer precision instead of silently lowering
// quality. 64/80 kbps retain their previous calibration. 96..512 use monotone
// 16-kbps interpolation through measured anchors; no separate <140-kbps oracle.
inline constexpr std::array<Knot, 29> kPhysicalRate = {{
    { 64.0f, 1.494139727341997f},
    { 80.0f, 1.349609148447557f},
    { 96.0f, 1.296277800816986f},
    {112.0f, 1.239352162858943f},
    {128.0f, 1.195906138057405f},
    {144.0f, 1.167823674467191f},
    {160.0f, 1.139741210876978f},
    {176.0f, 1.125299161375329f},
    {192.0f, 1.110857111873679f},
    {208.0f, 1.103354484289471f},
    {224.0f, 1.095851856705262f},
    {240.0f, 1.088349229121053f},
    {256.0f, 1.080846601536845f},
    {272.0f, 1.078755081389961f},
    {288.0f, 1.076663561243077f},
    {304.0f, 1.074572041096193f},
    {320.0f, 1.072480520949309f},
    {336.0f, 1.075192889567989f},
    {352.0f, 1.077905258186669f},
    {368.0f, 1.080617626805350f},
    {384.0f, 1.083329995424030f},
    {400.0f, 1.089455741206534f},
    {416.0f, 1.095581486989038f},
    {432.0f, 1.101707232771542f},
    {448.0f, 1.107832978554046f},
    {464.0f, 1.115552798239051f},
    {480.0f, 1.123272617924056f},
    {496.0f, 1.130992437609062f},
    {512.0f, 1.138712257294067f},
}};

inline float rho(float kbps) noexcept {
    if (kbps <= kPhysicalRate.front().kbps) return kPhysicalRate.front().rho;
    if (kbps >= kPhysicalRate.back().kbps) return kPhysicalRate.back().rho;
    const float u = (kbps - 64.0f) * (1.0f / 16.0f);
    const std::size_t i = static_cast<std::size_t>(u);
    const float t = u - static_cast<float>(i);
    return kPhysicalRate[i].rho + t * (kPhysicalRate[i + 1].rho - kPhysicalRate[i].rho);
}

} // namespace lx::rate_model
