#pragma once
#include <cmath>
#include <cstdint>
#include <algorithm>
#include "quant/warp.hpp"
#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(LXVOXEL_NEON) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace lx_conditional_gain {

// v0.26 compatibility: occupancy-only Laplace scalar-quantizer correction.

struct DnMoments {
    double q1 = 0.0; // sum |q|
    double q2 = 0.0; // sum q^2
    double nz = 0.0; // non-zero coordinates
};

#if defined(__AVX2__)
static inline double hsum256_ps(__m256 v) noexcept {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v,1);
    __m128 s = _mm_add_ps(lo,hi);
    s = _mm_hadd_ps(s,s);
    s = _mm_hadd_ps(s,s);
    return double(_mm_cvtss_f32(s));
}
#endif

// Signal-branch-free inner loop.  The D_n parity repair changes at most one
// coordinate; the observed final lattice vector already contains that repair,
// so its shell moments are consumed directly rather than guessed from occupancy.
static inline DnMoments dn_moments(const int32_t* q, int n) noexcept {
    DnMoments out{};
    int i=0;
#if defined(__AVX2__)
    __m256 a1=_mm256_setzero_ps(), a2=_mm256_setzero_ps(), anz=_mm256_setzero_ps();
    const __m256 z=_mm256_setzero_ps(), one=_mm256_set1_ps(1.0f);
    for(;i+8<=n;i+=8){
        const __m256i qi=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q+i));
        const __m256 qf=_mm256_cvtepi32_ps(qi);
        const __m256 af=_mm256_andnot_ps(_mm256_set1_ps(-0.0f),qf);
        a1=_mm256_add_ps(a1,af);
        a2=_mm256_add_ps(a2,_mm256_mul_ps(qf,qf));
        const __m256 mask=_mm256_cmp_ps(af,z,_CMP_NEQ_OQ);
        anz=_mm256_add_ps(anz,_mm256_and_ps(mask,one));
    }
    out.q1=hsum256_ps(a1); out.q2=hsum256_ps(a2); out.nz=hsum256_ps(anz);
#elif defined(LXVOXEL_NEON) || defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t a1=vdupq_n_f32(0.0f), a2=vdupq_n_f32(0.0f);
    uint32x4_t anz=vdupq_n_u32(0);
    const float32x4_t z=vdupq_n_f32(0.0f);
    for(;i+4<=n;i+=4){
        const int32x4_t qi=vld1q_s32(q+i);
        const float32x4_t qf=vcvtq_f32_s32(qi);
        const float32x4_t af=vabsq_f32(qf);
        a1=vaddq_f32(a1,af);
        a2=vmlaq_f32(a2,qf,qf);
        anz=vsubq_u32(anz,vcgtq_f32(af,z)); // true lanes are 0xffffffff
    }
#if defined(__aarch64__) || defined(_M_ARM64)
    out.q1=double(vaddvq_f32(a1));
    out.q2=double(vaddvq_f32(a2));
    out.nz=double(vaddvq_u32(anz));
#else
    float32x2_t s1=vadd_f32(vget_low_f32(a1),vget_high_f32(a1)); s1=vpadd_f32(s1,s1);
    float32x2_t s2=vadd_f32(vget_low_f32(a2),vget_high_f32(a2)); s2=vpadd_f32(s2,s2);
    uint32x2_t sn=vadd_u32(vget_low_u32(anz),vget_high_u32(anz)); sn=vpadd_u32(sn,sn);
    out.q1=double(vget_lane_f32(s1,0)); out.q2=double(vget_lane_f32(s2,0)); out.nz=double(vget_lane_u32(sn,0));
#endif
#endif
    for(;i<n;++i){
        const double a=std::fabs(double(q[i]));
        out.q1+=a; out.q2+=a*a; out.nz+=double(a!=0.0);
    }
    return out;
}

static inline double shell_a_from_q1_nz(double q1, double nz, uint32_t coordinates) noexcept {
    if(!(q1>0.0) || coordinates==0u) return 0.0;
    const double N=double(coordinates);
    // Exact MLE for s=exp(-a/2) under a Laplace source observed through
    // unit-step nearest-integer quantization.  The quantized pmf is
    // P(q=0)=1-s and P(|q|=k)=s^(2k-1)(1-s^2), k>=1.
    // Its sufficient statistics are q1=sum|q| and nz=count(q!=0).
    // Setting d log L / ds = 0 yields:
    //   (N+2q1)s^2 + (N-nz)s - (2q1-nz) = 0.
    const double A=N+2.0*q1;
    const double B=N-nz;
    const double C=2.0*q1-nz;
    if(!(C>0.0)) return 0.0;
    const double disc=B*B+4.0*A*C;
    const double s=(2.0*C)/(B+std::sqrt(disc));
    return (s>0.0 && s<1.0) ? -2.0*std::log(s) : 0.0;
}

static inline float conditional_from_weighted_moments(double q1w,double q2w,
                                                       double nzw,double zw,
                                                       double a) noexcept {
    if(!(a>0.0) || !(q2w>0.0)) return 1.0f;
    constexpr double h=0.5;
    const double ah=a*h;
    // s = exp(-a/2).  Use identities in s so the hot decoder path needs no
    // sinh/cosh calls and only the single log used to obtain a.
    const double s=std::exp(-ah);
    const double s2=s*s;
    const double d=1.0-s2;
    const double coth=(1.0+s2)/d;
    const double invsinh2=(4.0*s2)/(d*d);
    const double ia=1.0/a, ia2=ia*ia;
    const double mu=ia-h*coth;
    const double var=ia2-(h*h)*invsinh2;
    const double z2=var+mu*mu;
    const double z02=(2.0*ia2-s*(h*h+2.0*h*ia+2.0*ia2))/(1.0-s);
    const double ep=q2w+2.0*mu*q1w+z2*nzw+z02*zw;
    return float(std::sqrt(std::max(ep,0.0)/q2w));
}

static inline double conditional_dn_q2_offset_from_moments(double q1,double nz,
                                                            uint32_t coords) noexcept {
    if(coords==0u || !(q1>0.0)) return 0.0;
    const double N=double(coords);
    const double A=N+2.0*q1;
    const double B=N-nz;
    const double C=2.0*q1-nz;
    if(!(C>0.0)) return 0.0;
    const double disc=B*B+4.0*A*C;
    const double s=(2.0*C)/(B+std::sqrt(disc));
    if(!(s>0.0 && s<1.0)) return 0.0;
    const double a=-2.0*std::log(s);
    const double s2=s*s, d=1.0-s2;
    const double coth=(1.0+s2)/d;
    const double invsinh2=(4.0*s2)/(d*d);
    constexpr double h=0.5;
    const double ia=1.0/a, ia2=ia*ia;
    const double mu=ia-h*coth;
    const double var=ia2-(h*h)*invsinh2;
    const double z2=var+mu*mu;
    const double z02=(2.0*ia2-s*(h*h+2.0*h*ia+2.0*ia2))/(1.0-s);
    return 2.0*mu*q1+z2*nz+z02*(double(coords)-nz);
}

static inline double conditional_dn_expected_q2_from_moments(double q1,double q2,double nz,
                                                               uint32_t coords) noexcept {
    return std::max(q2+conditional_dn_q2_offset_from_moments(q1,nz,coords),0.0);
}

static inline float conditional_dn_gain_from_moments(double q1,double q2,double nz,
                                                      uint32_t coords) noexcept {
    if(!(q2>0.0)) return 1.0f;
    const double ep=conditional_dn_expected_q2_from_moments(q1,q2,nz,coords);
    return float(std::sqrt(ep/q2));
}

// Production v0.27+ estimator; v0.28 applies it to every Direct-LVQ ERB band.  No gain/shape split and no side information:
// infer the Laplace width from the final D_n shell, then condition source energy
// on the actual received lattice magnitudes.  With warp disabled (production
// default), delta cancels exactly and the AVX2 loop only accumulates q moments.
static inline float conditional_dn_gain(const int32_t* qa,const int32_t* qb,
                                        int n,int nch,int band,float delta) noexcept {
    if(n<=0 || nch<=0) return 1.0f;
    const DnMoments ma=dn_moments(qa,n);
    DnMoments m=ma;
    if(nch==2){ const DnMoments mb=dn_moments(qb,n); m.q1+=mb.q1; m.q2+=mb.q2; m.nz+=mb.nz; }
    const uint32_t coords=uint32_t(n*nch);
    if(!(m.q2>0.0)) return 1.0f;
    const double a=shell_a_from_q1_nz(m.q1,m.nz,coords);
    if(!(a>0.0)) return 1.0f;

    const int wm=lx_warp_mode_cfg();
    const float ws=lx_warp_strength_cfg();
    if(wm<=0 || band<lx_warp_start_cfg() || n<=1 || ws==0.0f){
        const double z=double(coords)-m.nz;
        return conditional_from_weighted_moments(m.q1,m.q2,m.nz,z,a);
    }

    // Optional warp path: preserve exact per-coordinate delta/warp geometry.
    // This path is outside the production default; no signal-value branches.
    double q1w=0.0,q2w=0.0,nzw=0.0,tw=0.0;
    for(int i=0;i<n;++i){
        const double w=double(lx_warp_weight(band,i,n));
        const double de=double(delta)/w, de2=de*de;
        const double aa=std::fabs(double(qa[i]));
        const double za=double(aa!=0.0);
        q1w+=de2*aa; q2w+=de2*aa*aa; nzw+=de2*za; tw+=de2;
        if(nch==2){
            const double ab=std::fabs(double(qb[i]));
            const double zb=double(ab!=0.0);
            q1w+=de2*ab; q2w+=de2*ab*ab; nzw+=de2*zb; tw+=de2;
        }
    }
    return conditional_from_weighted_moments(q1w,q2w,nzw,tw-nzw,a);
}

} // namespace lx_conditional_gain
