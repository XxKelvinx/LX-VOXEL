#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace lx_sfc {

#ifndef LX_SFC_EXC_SCALE
#define LX_SFC_EXC_SCALE 0.25f
#endif
#ifndef LX_SFC_CONTEXT_MUL
#define LX_SFC_CONTEXT_MUL 2
#endif
#ifndef LX_SFC_MS_DOMAIN
#define LX_SFC_MS_DOMAIN 0
#endif

// SFC AUTO generator
// ------------------
// Decoder-side local spectral continuation in the transform domain.
//
// The core model is AR(2), the lowest-order real linear recurrence capable of
// representing a damped sinusoid.  It is fitted to the immediately available
// decoded spectral context with Burg's method.  Burg reflection coefficients
// are stable by construction; the tiny numerical projection below only guards
// round-off at |k|=1.
//
// Continuation is not a hard "tone/noise" mode switch.  The same AR model is
// always used.  Its prediction-error energy drives a deterministic residual
// excitation.  Tonal context => small residual => coherent continuation.
// Diffuse context => large residual => naturally diffuse continuation.
// A transmitted target L2 gain is applied only after shape generation.
//
// No signal-dependent branch exists in the generated-sample inner loop.

struct Ar2 {
    float a1 = 0.0f;          // x[n] + a1*x[n-1] + a2*x[n-2] = e[n]
    float a2 = 0.0f;
    float residual_rms = 0.0f;
    double context_energy = 0.0;
};

static inline float unit_hash_noise(uint32_t frame, uint32_t channel, uint32_t bin) {
    // Integer hash -> deterministic approximately uniform [-sqrt(3), +sqrt(3)]
    // with unit variance.  This is decoder-synchronous and has no state.
    uint32_t x = bin ^ (frame * 0x9E3779B9u) ^ (channel * 0x85EBCA6Bu);
    x ^= x >> 16; x *= 0x7FEB352Du;
    x ^= x >> 15; x *= 0x846CA68Bu;
    x ^= x >> 16;
    constexpr float inv24 = 1.0f / 16777216.0f;
    constexpr float sqrt3 = 1.7320508075688772935f;
    return (float(x >> 8) * inv24 * 2.0f - 1.0f) * sqrt3;
}

// Least-squares AR(2).  AR(2) is the minimum-order real recurrence that can
// represent one damped sinusoid exactly.  The unconstrained fit is projected
// onto the exact Schur-stable AR(2) triangle before synthesis.
static inline Ar2 fit_ar2(const float* x, int n) {
    Ar2 m{};
    if (n < 3) return m;

    // Solve x[n] = c1*x[n-1] + c2*x[n-2] in normal-equation form.
    double s11=0.0,s22=0.0,s12=0.0,t1=0.0,t2=0.0,ey=0.0;
    for (int i=2;i<n;++i) {
        const double u=double(x[i-1]), v=double(x[i-2]), y=double(x[i]);
        s11+=u*u; s22+=v*v; s12+=u*v; t1+=u*y; t2+=v*y; ey+=y*y;
    }
    const double det=s11*s22-s12*s12;
    const double scale=s11*s22+s12*s12+std::numeric_limits<double>::min();
    const double ok=double(std::fabs(det)>std::numeric_limits<double>::epsilon()*scale);
    const double idet=ok/std::max(std::fabs(det),std::numeric_limits<double>::min())*std::copysign(1.0,det);
    double c1=(t1*s22-t2*s12)*idet;
    double c2=(t2*s11-t1*s12)*idet;

    // Convert to x[n] + a1*x[n-1] + a2*x[n-2] = e[n].
    double a1=-c1, a2=-c2;

    // Exact Schur stability region for a real AR(2):
    //   -1 < a2 < 1,  -1-a2 < a1 < 1+a2.
    const double om=std::nextafter(1.0,0.0);
    a2=std::clamp(a2,-om,om);
    const double lim=1.0+a2;
    a1=std::clamp(a1,-lim*om,lim*om);

    // Residual SSE from the normal-equation sufficient statistics.
    // This is algebraically identical to a second residual scan after the
    // Schur projection, but removes that entire memory pass.
    const double pc1=-a1, pc2=-a2;
    double re=ey - 2.0*pc1*t1 - 2.0*pc2*t2
                  + pc1*pc1*s11 + 2.0*pc1*pc2*s12 + pc2*pc2*s22;
    re=std::max(re,0.0);
    const int rn=n-2;
    m.a1=float(a1);m.a2=float(a2);
    m.residual_rms=float(std::sqrt(re/double(std::max(rn,1))));
    m.context_energy=ey+double(x[0])*double(x[0])+double(x[1])*double(x[1]);
    return m;
}

// Continue one missing spectral interval from its lower-frequency boundary.
// target_l2 is the transmitted gain for the complete interval.  Passing zero
// produces exact zero output.  Context is never copied into the target region;
// only the fitted AR(2) state and residual statistics cross the boundary.
static inline void continue_forward(const float* context, int context_n,
                                    float* dst, int dst_n, float target_l2,
                                    uint32_t frame, uint32_t channel,
                                    uint32_t first_bin, double* context_energy_out=nullptr) {
    if (dst_n <= 0) return;
    if (context_n < 2 || !(target_l2 > 0.0f)) {
        std::fill(dst, dst + dst_n, 0.0f);
        return;
    }

    const Ar2 m = fit_ar2(context, context_n);
    float x2 = context[context_n - 2];
    float x1 = context[context_n - 1];
    double e = 0.0;

    // Decoder-known AR unpredictability.  The innovation term is continuously
    // attenuated when the local context is well predicted, while diffuse
    // context retains proportionally more innovation.  No signal-value branch
    // occurs in the generated-coefficient loop.
    const double ce=m.context_energy;
    if(context_energy_out) *context_energy_out=ce;
    const double context_rms = std::sqrt(ce / double(context_n));
    const double rr = double(m.residual_rms);
    const float unpredictability = float(rr / std::sqrt(rr*rr + context_rms*context_rms + std::numeric_limits<double>::min()));
    const float exc_scale = LX_SFC_EXC_SCALE * std::sqrt(unpredictability);

    // Same continuous formula for every generated coefficient.
    for (int i = 0; i < dst_n; ++i) {
        const float exc = unit_hash_noise(frame, channel, first_bin + uint32_t(i));
        const float y = -m.a1 * x1 - m.a2 * x2 + (exc_scale * m.residual_rms) * exc;
        dst[i] = y;
        x2 = x1;
        x1 = y;
        e += double(y) * double(y);
    }

    // Exact zero-context fallback.  A preceding ZERO SFC band can leave no
    // predictive state at all.  In that case AR(2) produces the all-zero
    // vector, which cannot be gain-normalized.  Use bounded deterministic
    // unit-variance excitation as the unique decoder-synchronous base shape.
    // This path is only entered when the generated vector has exactly no energy.
    if (!(e > 0.0) || !std::isfinite(e)) {
        e = 0.0;
        for (int i = 0; i < dst_n; ++i) {
            const float y = unit_hash_noise(frame, channel, first_bin + uint32_t(i));
            dst[i] = y;
            e += double(y) * double(y);
        }
    }

    const double den = std::sqrt(std::max(e, std::numeric_limits<double>::min()));
    const float s = float(double(target_l2) / den);
    for (int i = 0; i < dst_n; ++i) dst[i] *= s;
}

// Bidirectional gap continuation.  A gap with decoded context on both sides is
// predicted from both boundaries.  The two predictions are combined by the
// minimum-degree partition of unity wL=1-u, wR=u; no signal-value switching.
// Scratch must hold 2*gap_n floats.

} // namespace lx_sfc

namespace lx_sfc {

// Beta SFC profile: the top eight FEAR/ERB bands are eligible for implicit
// gain-only continuation.  An active band carries one predicted combined
// stereo gain; spatial structure is inherited from the decoder-side local
// continuation, so no channel-ratio side field is required.
constexpr int SFC_FIRST_BAND = 40;
constexpr int SFC_BANDS = 8;
constexpr int SFC_MAX_ACTIVE = 4;
#ifndef LX_SFC_GAIN_BITS
#define LX_SFC_GAIN_BITS 8
#endif
constexpr int SFC_GAIN_BITS = LX_SFC_GAIN_BITS;
static_assert(SFC_GAIN_BITS >= 2 && SFC_GAIN_BITS <= 8);
constexpr int SFC_GAIN_LEVELS = 1 << SFC_GAIN_BITS;
constexpr uint32_t SFC_GAIN_MASK = (uint32_t(1) << SFC_GAIN_BITS) - 1u;
constexpr int SFC_AR_BITS = 4;
constexpr int SFC_AR_LEVELS = 1 << SFC_AR_BITS;
constexpr uint32_t SFC_AR_MASK = SFC_AR_LEVELS - 1u;

// 3-bit intraband spectral-tilt side information for every active SFC band.
// The encoder measures only a normalized spectral-energy shape; total SFC gain
// remains independently coded by SFC_GAIN_BITS.  Three mathematically defined
// quantizers are retained for research; production chooses one at compile time.
constexpr int SFC_TILT_BITS = 3;
constexpr int SFC_TILT_LEVELS = 1 << SFC_TILT_BITS;
constexpr double SFC_PI = 3.141592653589793238462643383279502884;


// decoder-synchronous spectral folding.
// The source is an already decoded lower ERB band.  We resample its stereo
// shape onto the target band and close the combined L2 gain exactly.  No
// source samples or source indices are signaled: the source is always b-1.
// All index decisions depend only on band geometry, never on signal values.
static inline void fold_stereo_band_to(const float* srcL,const float* srcR,
                                       int dst_lo,int dst_hi,
                                       float target_l2,
                                       float* dstL,float* dstR) {
    const int dn=dst_hi-dst_lo;
    const int src_lo=dst_lo-dn;
    if(src_lo<0 || dn<=0 || !(target_l2>0.0f)) {
        for(int i=0;i<dn;++i){dstL[i]=0.0f;dstR[i]=0.0f;}
        return;
    }
    double energy=0.0;
    for(int i=0;i<dn;++i){
        const float l=srcL[src_lo+i];
        const float r=srcR[src_lo+i];
        dstL[i]=l; dstR[i]=r;
        energy+=double(l)*l+double(r)*r;
    }
    if(!(energy>0.0)) {
        for(int i=0;i<dn;++i){dstL[i]=0.0f;dstR[i]=0.0f;}
        return;
    }
    const float gain=float(double(target_l2)/std::sqrt(energy));
    for(int i=0;i<dn;++i){dstL[i]*=gain;dstR[i]*=gain;}
}

static inline void fold_stereo_band(float* L,float* R,
                                    int src_lo,int src_hi,
                                    int dst_lo,int dst_hi,
                                    float target_l2) {
    const int dn=dst_hi-dst_lo;
    // Sliding fSFC: use the immediately preceding reconstructed spectral slice
    // with exactly the target dimension. No resampling and no side information.
    src_hi = dst_lo;
    src_lo = dst_lo - dn;
    const int sn=src_hi-src_lo;
    if(src_lo<0 || sn<=0 || dn<=0 || !(target_l2>0.0f)) {
        for(int i=dst_lo;i<dst_hi;++i){L[i]=0.0f;R[i]=0.0f;}
        return;
    }
    double e=0.0;
    for(int t=0;t<dn;++t){
        const float l=L[src_lo+t];
        const float r=R[src_lo+t];
        L[dst_lo+t]=l; R[dst_lo+t]=r;
        e+=double(l)*l+double(r)*r;
    }
    if(!(e>0.0)) {
        for(int i=dst_lo;i<dst_hi;++i){L[i]=0.0f;R[i]=0.0f;}
        return;
    }
    const float g=float(double(target_l2)/std::sqrt(e));
    for(int i=dst_lo;i<dst_hi;++i){L[i]*=g;R[i]*=g;}
}

static inline uint8_t quantize_spectral_tilt3(const float* xL,const float* xR,int lo,int hi) {
    const int n=hi-lo, mid=lo+(n>>1);
    double e0=0.0,e1=0.0;
    for(int i=lo;i<mid;++i) e0 += double(xL[i])*xL[i] + double(xR[i])*xR[i];
    for(int i=mid;i<hi;++i) e1 += double(xL[i])*xL[i] + double(xR[i])*xR[i];
    const double et=e0+e1;
    if(!(et>0.0)) return 0;
    // Current v0.32 SFC tilt is the 8-level uniform-angle codebook.  Because
    // p=sin^2(theta) is monotone on [0,pi/2], quantization can compare the
    // energy fraction directly against the seven exact midpoint thresholds.
    // This is bitstream-equivalent away from measure-zero decision boundaries
    // and removes atan2/sqrt from the encoder decision path.
    static constexpr double kMidP[7]={
        0.012536043909088196,0.10908425876598511,0.28305813044122097,
        0.5,0.71694186955877903,0.89091574123401474,0.98746395609091187
    };
    const double p=e1/et;
    return uint8_t((p>=kMidP[0])+(p>=kMidP[1])+(p>=kMidP[2])+(p>=kMidP[3])+
                   (p>=kMidP[4])+(p>=kMidP[5])+(p>=kMidP[6]));
}

static inline void apply_spectral_tilt3(float* yL,float* yR,int lo,int hi,float target_l2,uint8_t q) {
    const int n=hi-lo;
    if(n<2 || !(target_l2>0.0f)) return;
    static constexpr double kHighFrac[8]={
        0.0,0.049515566048790434,0.18825509907063323,0.38873953302184272,
        0.61126046697815717,0.8117449009293668,0.95048443395120963,1.0
    };
    const double ph=kHighFrac[q&7u];
    const int mid=lo+(n>>1); double e0=0.0,e1=0.0;
    for(int i=lo;i<mid;++i)e0+=double(yL[i])*yL[i]+double(yR[i])*yR[i];
    for(int i=mid;i<hi;++i)e1+=double(yL[i])*yL[i]+double(yR[i])*yR[i];
    const double Et=double(target_l2)*target_l2, tiny=std::numeric_limits<double>::min();
    const float s0=float(std::sqrt(Et*(1.0-ph)/std::max(e0,tiny)));
    const float s1=float(std::sqrt(Et*ph/std::max(e1,tiny)));
    for(int i=lo;i<mid;++i){yL[i]*=s0;yR[i]*=s0;}
    for(int i=mid;i<hi;++i){yL[i]*=s1;yR[i]*=s1;}
}

// SFC Diffuse Guard.
//
// Persistent parametric HF reconstruction can become perceptually conspicuous even
// when each individual frame is locally plausible.  The guard therefore adds a
// decoder-synchronous *temporal exposure prior* to the already transmitted 3-bit
// tilt; it sends no extra bits and never changes the SFC band L2 gain.
//
// exposure is measured in units of one longest (2048-sample) transform support and
// is accumulated only across consecutive frames in which this exact SFC band is
// active.  The bounded confidence
//
//      g = exposure / (1 + exposure)
//
// grows continuously from 0 to 1.  At full exposure the decoded tilt angle is
// shifted downward by at most half one existing 3-bit angular quantizer step.  That
// bound is not an empirical safety constant: it is exactly the native worst-case
// angular quantization radius of the 3-bit tilt representation.  Consequently the
// guard cannot add more systematic envelope bias than the tilt quantizer already
// admits locally, while long SFC runs receive a slightly darker, less conspicuous
// HF balance.
static inline double sfc_diffuse_guard_weight(double exposure_long_blocks) {
    const double e=std::max(exposure_long_blocks,0.0);
    return e/(1.0+e);
}

static inline void apply_sfc_diffuse_guard(float* yL,float* yR,int lo,int hi,
                                            float target_l2,uint8_t q,
                                            double exposure_long_blocks) {
    const int n=hi-lo;
    if(n<2 || !(target_l2>0.0f) || !(exposure_long_blocks>0.0)) return;
    constexpr double step=(0.5*SFC_PI)/double(SFC_TILT_LEVELS-1);
    const double theta_q=step*double(q);
    const double g=sfc_diffuse_guard_weight(exposure_long_blocks);
    const double theta=std::max(0.0,theta_q-0.5*step*g);
    const double sh=std::sin(theta), ph=sh*sh;
    const int mid=lo+(n>>1);
    double e0=0.0,e1=0.0;
    for(int i=lo;i<mid;++i)e0+=double(yL[i])*yL[i]+double(yR[i])*yR[i];
    for(int i=mid;i<hi;++i)e1+=double(yL[i])*yL[i]+double(yR[i])*yR[i];
    const double Et=double(target_l2)*target_l2;
    // Support-gated tilt projection: preserve requested energy while assigning
    // no multiplicative target to a half-band that has zero support.
    const double a0=1.0-ph, a1=ph;
    const double tau=std::numeric_limits<double>::epsilon()*(Et+e0+e1);
    const double h0=e0/(e0+tau);
    const double h1=e1/(e1+tau);
    const double w0=a0*h0, w1=a1*h1;
    const double invw=1.0/(w0+w1+std::numeric_limits<double>::epsilon());
    const float s0=float(std::sqrt(Et*(w0*invw)/(e0+tau)));
    const float s1=float(std::sqrt(Et*(w1*invw)/(e1+tau)));
    for(int i=lo;i<mid;++i){yL[i]*=s0;yR[i]*=s0;}
    for(int i=mid;i<hi;++i){yL[i]*=s1;yR[i]*=s1;}
}

// Encoder-only selector for mode 4.  It searches the eight transmitted codewords
// using the exact squared-error sufficient statistics of the two half-bands.
// q=0 is the current decoder prediction and therefore provides a hard no-regression
// candidate in coefficient-domain SSE.  For modes 1..3 this reduces to the direct
// target-derived quantizer above.
static inline uint8_t select_spectral_tilt3(const float* xL,const float* xR,
                                             const float*,const float*,
                                             int n,float) {
    return quantize_spectral_tilt3(xL,xR,0,n);
}

struct SharedAr2Q { uint8_t k1=uint8_t(SFC_AR_LEVELS/2-1), k2=uint8_t(SFC_AR_LEVELS/2-1); };

// optional Innovation Filter Correction (IFC).  IFC is deliberately a
// tiny decoder-synchronous local correction layered on top of SFC.  Four
// Gabor modes and sixteen normalized band positions are addressed directly;
// a three-bit signed mid-rise coefficient controls the correction strength.
constexpr int SFC_IFC_FILTER_BITS = 2;
constexpr int SFC_IFC_FILTERS = 1 << SFC_IFC_FILTER_BITS;
constexpr int SFC_IFC_POS_BITS = 4;
constexpr int SFC_IFC_POSITIONS = 1 << SFC_IFC_POS_BITS;
constexpr int SFC_IFC_AMP_BITS = 3;
constexpr int SFC_IFC_AMP_LEVELS = 1 << SFC_IFC_AMP_BITS;
constexpr int SFC_IFC_PAYLOAD_BITS = SFC_IFC_FILTER_BITS + SFC_IFC_POS_BITS + SFC_IFC_AMP_BITS;

static inline float decode_ifc_amp3(uint8_t q) {
    return (2.0f*float(q)+1.0f-float(SFC_IFC_AMP_LEVELS))/float(SFC_IFC_AMP_LEVELS);
}
static inline uint8_t quantize_ifc_amp3(double a) {
    const double u=(a+1.0)*0.5*double(SFC_IFC_AMP_LEVELS);
    return uint8_t(std::clamp(int(std::floor(u)),0,SFC_IFC_AMP_LEVELS-1));
}

// Unitless Gabor atom.  Position spacing itself fixes the Gaussian scale:
// z=1 is one position cell, so there is no separately tuned width constant.
static inline float ifc_gabor_sample(uint8_t mode,uint8_t pos,int i,int n) {
    const double u=(double(i)+0.5)/double(std::max(n,1));
    const double c=(double(pos)+0.5)/double(SFC_IFC_POSITIONS);
    const double z=double(SFC_IFC_POSITIONS)*(u-c);
    const double env=std::exp(-0.5*z*z);
    return float(env*std::cos(3.1415926535897932384626433832795*double(mode)*z));
}

// Apply one IFC state and then close the band exactly back onto the transmitted
// SFC gain sphere.  Stereo orientation is inherited continuously from the
// decoder's current SFC reconstruction; no spatial side information is needed.
static inline void apply_ifc_band(float* L,float* R,int lo,int hi,float target_l2,
                                  uint8_t mode,uint8_t pos,uint8_t qamp) {
    const int n=hi-lo; if(n<=0 || !(target_l2>0.0f)) return;
    double eL=0.0,eR=0.0,ea=std::numeric_limits<double>::min();
    for(int i=lo;i<hi;++i){eL+=double(L[i])*L[i];eR+=double(R[i])*R[i];const double a=ifc_gabor_sample(mode,pos,i-lo,n);ea+=a*a;}
    const double eps=std::numeric_limits<double>::min();
    const double es=eL+eR+2.0*eps;
    const float wL=float(std::sqrt((eL+eps)/es)), wR=float(std::sqrt((eR+eps)/es));
    const float inva=float(1.0/std::sqrt(ea));
    const float A=decode_ifc_amp3(qamp)*target_l2;
    double e=0.0;
    for(int i=lo;i<hi;++i){const float a=ifc_gabor_sample(mode,pos,i-lo,n)*inva;L[i]+=A*wL*a;R[i]+=A*wR*a;e+=double(L[i])*L[i]+double(R[i])*R[i];}
    const float sc=float(double(target_l2)/std::sqrt(std::max(e,eps)));
    for(int i=lo;i<hi;++i){L[i]*=sc;R[i]*=sc;}
}

// Mid-rise PARCOR quantizer. Every reconstruction level is strictly inside
// (-1,1), hence every decoded AR(2) polynomial is Schur-stable by construction.
static inline float decode_parcor4(uint8_t q) {
    return (2.0f*float(q)+1.0f-float(SFC_AR_LEVELS))/float(SFC_AR_LEVELS);
}
static inline uint8_t quantize_parcor4(double k) {
    const double u=(k+1.0)*0.5*double(SFC_AR_LEVELS);
    return uint8_t(std::clamp(int(std::floor(u)),0,SFC_AR_LEVELS-1));
}

// Pooled-stereo Burg AR(2). Burg reflection coefficients satisfy |k_m|<=1
// directly from Cauchy-Schwarz; no unstable LPC projection is required.
static inline SharedAr2Q fit_shared_ar2_q(const float* L,const float* R,int n) {
    double num1=0.0,den1=std::numeric_limits<double>::min();
    for(int ch=0;ch<2;++ch){const float* x=ch?R:L;for(int i=1;i<n;++i){const double f=x[i],b=x[i-1];num1+=f*b;den1+=f*f+b*b;}}
    const double k1=-2.0*num1/den1;
    double num2=0.0,den2=std::numeric_limits<double>::min();
    for(int ch=0;ch<2;++ch){const float* x=ch?R:L;for(int i=2;i<n;++i){const double f=double(x[i])+k1*double(x[i-1]);const double b=double(x[i-2])+k1*double(x[i-1]);num2+=f*b;den2+=f*f+b*b;}}
    const double k2=-2.0*num2/den2;
    return {quantize_parcor4(k1),quantize_parcor4(k2)};
}

static inline void continue_stereo_band_guided(float* L,float* R,int lo,int hi,float target_l2,uint32_t frame,uint8_t qk1,uint8_t qk2) {
    const int n=hi-lo; if(n<=0) return;
    if(!(target_l2>0.0f)){std::fill(L+lo,L+hi,0.0f);std::fill(R+lo,R+hi,0.0f);return;}
    const float k1=decode_parcor4(qk1),k2=decode_parcor4(qk2);
    const float a1=k1*(1.0f+k2),a2=k2;
    double et=0.0;
    for(uint32_t ch=0;ch<2;++ch){float* y=ch?R:L;const int cn=std::min(lo,n*LX_SFC_CONTEXT_MUL);const Ar2 m=fit_ar2(y+lo-cn,cn);double ce=0.0;for(int i=lo-cn;i<lo;++i)ce+=double(y[i])*y[i];const double cr=std::sqrt(ce/double(std::max(cn,1)));const double rr=double(m.residual_rms);const float un=float(rr/std::sqrt(rr*rr+cr*cr+std::numeric_limits<double>::min()));const float ex=LX_SFC_EXC_SCALE*std::sqrt(un)*m.residual_rms;float x2=lo>=2?y[lo-2]:0.0f,x1=lo>=1?y[lo-1]:0.0f;for(int i=lo;i<hi;++i){const float v=-a1*x1-a2*x2+ex*unit_hash_noise(frame,ch,uint32_t(i));y[i]=v;x2=x1;x1=v;et+=double(v)*v;}}
    if(!(et>0.0) || !std::isfinite(et)){et=0.0;for(uint32_t ch=0;ch<2;++ch){float* y=ch?R:L;for(int i=lo;i<hi;++i){const float v=unit_hash_noise(frame,ch,uint32_t(i));y[i]=v;et+=double(v)*v;}}}
    const float sc=float(double(target_l2)/std::sqrt(std::max(et,std::numeric_limits<double>::min())));
    for(int i=lo;i<hi;++i){L[i]*=sc;R[i]*=sc;}
}
constexpr int sfc_subset_bits(int k) { return k==1?3:k==2?5:k==3?6:k==4?7:0; }
#ifndef LX_SFC_GAIN_MIN_DB
#define LX_SFC_GAIN_MIN_DB -120.0f
#endif
constexpr float SFC_GAIN_MIN_DB = LX_SFC_GAIN_MIN_DB;
constexpr float SFC_GAIN_MAX_DB = 0.0f;
// q=0 is an exact ZERO state; q=1..255 cover -120..0 dBFS.
// This prevents sub-floor HF energy from being promoted to the -70 dB floor.
constexpr int SFC_GAIN_NONZERO_LEVELS = SFC_GAIN_LEVELS - 1;
constexpr float SFC_GAIN_STEP_DB = (SFC_GAIN_MAX_DB-SFC_GAIN_MIN_DB)/float(SFC_GAIN_NONZERO_LEVELS-1);


// final: gain-only anti-collapse for completely collapsed direct HF bands.
// Bands 40..47 carry one predicted gain when their transmitted
// direct-lattice shape is all-zero.  The all-zero shape is decoder-inferable,
// therefore anti-collapse needs no mode flag.  Reconstruction uses one fixed,
// deterministic four-uniform Irwin-Hall excitation and exact stereo L2 closure.
static inline float ac_hash01(uint32_t frame,uint32_t channel,uint32_t bin,uint32_t salt){
    uint32_t x=bin ^ (frame*0x9E3779B9u) ^ (channel*0x85EBCA6Bu) ^ salt;
    x^=x>>16; x*=0x7FEB352Du; x^=x>>15; x*=0x846CA68Bu; x^=x>>16;
    return float(x>>8)*(1.0f/16777216.0f);
}
static inline float ac_irwin_hall4(uint32_t frame,uint32_t channel,uint32_t bin){
    const float u0=ac_hash01(frame,channel,bin,0x243F6A88u);
    const float u1=ac_hash01(frame,channel,bin,0x13198A2Eu);
    const float u2=ac_hash01(frame,channel,bin,0xA4093822u);
    const float u3=ac_hash01(frame,channel,bin,0x299F31D0u);
    return (u0+u1+u2+u3)-2.0f;
}
static inline void anti_collapse_stereo_band(float* L,float* R,int lo,int hi,float target_l2,
                                              uint32_t frame,bool intensity,float ct,float st,bool ms){
    const int n=hi-lo;
    if(n<=0 || !(target_l2>0.0f)){for(int i=lo;i<hi;++i){L[i]=0.0f;R[i]=0.0f;}return;}
    double e=0.0;
    if(intensity){
        for(int i=0;i<n;++i){const float a=ac_irwin_hall4(frame,0,uint32_t(lo+i));L[lo+i]=ct*a;R[lo+i]=st*a;e+=double(a)*a;}
    }else if(ms){
        for(int i=0;i<n;++i){const float a=ac_irwin_hall4(frame,0,uint32_t(lo+i));const float b=ac_irwin_hall4(frame,1,uint32_t(lo+i));const float l=(a+b)*0.7071067811865475f,r=(a-b)*0.7071067811865475f;L[lo+i]=l;R[lo+i]=r;e+=double(l)*l+double(r)*r;}
    }else{
        for(int i=0;i<n;++i){const float l=ac_irwin_hall4(frame,0,uint32_t(lo+i));const float r=ac_irwin_hall4(frame,1,uint32_t(lo+i));L[lo+i]=l;R[lo+i]=r;e+=double(l)*l+double(r)*r;}
    }
    const float sc=float(double(target_l2)/std::sqrt(std::max(e,std::numeric_limits<double>::min())));
    for(int i=lo;i<hi;++i){L[i]*=sc;R[i]*=sc;}
}

// SFC keeps the historical strictly-positive gain semantics. Anti-collapse
// deliberately does not use this: its perceptual gate may quantize to q=0.
static inline uint8_t quantize_gain32_ac(float gain_rms) {
    // q=0 and q=1 form an ordinary nearest-neighbour boundary in linear
    // L2 amplitude.  q=1 reconstructs the minimum positive AC RMS gain;
    // source gains closer to zero than to that representable level must stay
    // zero instead of being promoted to the -70 dBFS floor.
    static const float ac_min_gain = std::pow(10.0f,SFC_GAIN_MIN_DB/20.0f);
    const float zero_boundary = 0.5f * ac_min_gain;
    if (!(gain_rms >= zero_boundary)) return 0;
    const float db = 20.0f*std::log10(gain_rms);
    const int qi = 1 + int(std::nearbyint((db-SFC_GAIN_MIN_DB)/SFC_GAIN_STEP_DB));
    return uint8_t(std::clamp(qi,1,SFC_GAIN_LEVELS-1));
}
static inline float decode_gain32(uint8_t q) {
    if (q == 0) return 0.0f;
    return std::pow(10.0f,(SFC_GAIN_MIN_DB+SFC_GAIN_STEP_DB*float(q-1))/20.0f);
}

// Stereo continuation for one target band.  Both channels are generated from
// their own immediately preceding decoded context.  The context length equals
// the target-band width: a local continuation rule with no source-tile search.
// Their relative energy is inherited from that context and only the transmitted
// combined-stereo gain fixes absolute target energy.

// Encoder-only candidate evaluator.  It produces exactly the same target-band
// samples as continue_stereo_band() when the surrounding spectrum is the
// unmodified baseline, but writes only n target coefficients.  This avoids
// copying two complete K_LONG spectra for each of the eight SFC RD candidates.
static inline void continue_stereo_band_eval(const float* L, const float* R,
                                             int lo, int hi, float target_l2,
                                             uint32_t frame,
                                             float* __restrict outL,
                                             float* __restrict outR) {
    const int n = hi-lo;
    if(n<=0) return;
    const int cn = std::min(lo,n*LX_SFC_CONTEXT_MUL);
    if(cn<2 || !(target_l2>0.0f)) {
        std::fill(outL,outL+n,0.0f);
        std::fill(outR,outR+n,0.0f);
        return;
    }
#if LX_SFC_MS_DOMAIN
    std::array<float,512> mctx{},sctx{},mout{},sout{};
    const int cuse=std::min(cn,512),nuse=std::min(n,512);
    for(int i=0;i<cuse;++i){
        const float a=L[lo-cuse+i],b=R[lo-cuse+i];
        mctx[i]=(a+b)*0.7071067811865475244f;
        sctx[i]=(a-b)*0.7071067811865475244f;
    }
    continue_forward(mctx.data(),cuse,mout.data(),nuse,1.0f,frame,0u,uint32_t(lo));
    continue_forward(sctx.data(),cuse,sout.data(),nuse,1.0f,frame,1u,uint32_t(lo));
    double e=0.0;
    for(int i=0;i<nuse;++i){
        const float l=(mout[i]+sout[i])*0.7071067811865475244f;
        const float r=(mout[i]-sout[i])*0.7071067811865475244f;
        outL[i]=l;outR[i]=r;e+=double(l)*l+double(r)*r;
    }
    const float sc=float(double(target_l2)/std::sqrt(std::max(e,std::numeric_limits<double>::min())));
    for(int i=0;i<nuse;++i){outL[i]*=sc;outR[i]*=sc;}
#else
    double cEL=0.0,cER=0.0;
    continue_forward(L+lo-cn,cn,outL,n,1.0f,frame,0u,uint32_t(lo),&cEL);
    continue_forward(R+lo-cn,cn,outR,n,1.0f,frame,1u,uint32_t(lo),&cER);
    const double eps=std::numeric_limits<double>::min();
    const double denC=cEL+cER+2.0*eps;
    const float wL=float(std::sqrt((cEL+eps)/denC));
    const float wR=float(std::sqrt((cER+eps)/denC));
    double e=0.0;
    for(int i=0;i<n;++i){outL[i]*=wL;outR[i]*=wR;e+=double(outL[i])*outL[i]+double(outR[i])*outR[i];}
    const float sc=float(double(target_l2)/std::sqrt(std::max(e,std::numeric_limits<double>::min())));
    for(int i=0;i<n;++i){outL[i]*=sc;outR[i]*=sc;}
#endif
}

static inline void continue_stereo_band(float* L, float* R, int lo, int hi,
                                        float target_l2, uint32_t frame) {
    const int n = hi-lo;
    if(n<=0) return;
    const int cn = std::min(lo,n*LX_SFC_CONTEXT_MUL);
    if(cn<2 || !(target_l2>0.0f)) {
        std::fill(L+lo,L+hi,0.0f);
        std::fill(R+lo,R+hi,0.0f);
        return;
    }

#if LX_SFC_MS_DOMAIN
    std::array<float, 512> mctx{}, sctx{}, mout{}, sout{};
    const int cuse=std::min(cn,512), nuse=std::min(n,512);
    for(int i=0;i<cuse;++i){
        const float a=L[lo-cuse+i], b=R[lo-cuse+i];
        mctx[i]=(a+b)*0.7071067811865475244f;
        sctx[i]=(a-b)*0.7071067811865475244f;
    }
    continue_forward(mctx.data(),cuse,mout.data(),nuse,1.0f,frame,0u,uint32_t(lo));
    continue_forward(sctx.data(),cuse,sout.data(),nuse,1.0f,frame,1u,uint32_t(lo));
    double e=0.0;
    for(int i=0;i<nuse;++i){
        L[lo+i]=(mout[i]+sout[i])*0.7071067811865475244f;
        R[lo+i]=(mout[i]-sout[i])*0.7071067811865475244f;
        e+=double(L[lo+i])*L[lo+i]+double(R[lo+i])*R[lo+i];
    }
    const float sc=float(double(target_l2)/std::sqrt(std::max(e,std::numeric_limits<double>::min())));
    for(int i=0;i<nuse;++i){L[lo+i]*=sc;R[lo+i]*=sc;}
#else
    double cEL=0.0,cER=0.0;
    continue_forward(L+lo-cn,cn,L+lo,n,1.0f,frame,0u,uint32_t(lo),&cEL);
    continue_forward(R+lo-cn,cn,R+lo,n,1.0f,frame,1u,uint32_t(lo),&cER);
    // If the lower context is exactly ZERO, inherit a neutral stereo split
    // rather than multiplying the fallback shape by zero and re-normalizing 0/0.
    const double eps=std::numeric_limits<double>::min();
    const double denC=cEL+cER+2.0*eps;
    const float wL=float(std::sqrt((cEL+eps)/denC));
    const float wR=float(std::sqrt((cER+eps)/denC));

    double e=0.0;
    for(int i=lo;i<hi;++i){ L[i]*=wL; R[i]*=wR; e+=double(L[i])*L[i]+double(R[i])*R[i]; }
    const float sc=float(double(target_l2)/std::sqrt(std::max(e,std::numeric_limits<double>::min())));
    for(int i=lo;i<hi;++i){ L[i]*=sc; R[i]*=sc; }
#endif
}


// Precision-guided SFC refinement.
// Sparse Precision coordinates are decoder-known control points.  They are
// applied before the final gain closure; SFC then supplies only the remaining
// band energy.  Thus an important Precision peak automatically reduces the
// energy left to the synthetic field instead of being added on top of a
// separately normalized SFC band.
static inline void guide_stereo_band_with_precision(
        float* L, float* R, int lo, int hi, float target_l2,
        const int32_t* qa, const int32_t* qb, int nch, bool ms,
        float ct, float st, float pdelta) {
    const int n=hi-lo;
    if(n<=0) return;
    std::array<uint8_t,512> anchor{};
    const int nu=std::min(n,int(anchor.size()));
    int na=0;
    for(int j=0;j<nu;++j){
        const int32_t a=qa[j], b=(nch==2?qb[j]:0);
        const uint8_t m=uint8_t((a!=0)|(b!=0));
        anchor[j]=m; na+=int(m);
        const float fa=float(a)*pdelta, fb=float(b)*pdelta;
        float dl,dr;
        if(nch==1){dl=ct*fa;dr=st*fa;}
        else if(!ms){dl=fa;dr=fb;}
        else{dl=(fa+fb)*0.7071067811865475244f;dr=(fa-fb)*0.7071067811865475244f;}
        L[lo+j]+=float(m)*dl;
        R[lo+j]+=float(m)*dr;
    }
    if(na==0) return;

    // Precision owns its exact anchor energy.  SFC is renormalized only in the
    // unprotected coordinates to consume the residual transmitted band energy.
    double ea=0.0,en=0.0;
    for(int j=0;j<nu;++j){
        const double e=double(L[lo+j])*L[lo+j]+double(R[lo+j])*R[lo+j];
        ea+=double(anchor[j])*e;
        en+=double(1u-anchor[j])*e;
    }
    const double target2=double(target_l2)*double(target_l2);
    const double rem=std::max(target2-ea,0.0);
    const float sn=float(std::sqrt(rem/std::max(en,std::numeric_limits<double>::min())));
    for(int j=0;j<nu;++j){
        const float m=float(anchor[j]);
        const float s=m+(1.0f-m)*sn;
        L[lo+j]*=s;R[lo+j]*=s;
    }
}

} // namespace lx_sfc
