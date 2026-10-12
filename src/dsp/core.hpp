#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(LXVOXEL_NEON) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace pulsar {

// LX-VOXEL product core
// ---------------------
// 4096-sample low-overlap MDCT -> 2048 coefficients -> 48 ERB bands (0..20 kHz)
// -> FEAR temporal importance -> global waterfill -> one exact D_n vector per band.
// Entropy coding is kept independent from the transform and psychoacoustic core.
// The hot coefficient loops contain no
// signal-value branches except the mathematically required D_n parity correction.
constexpr int K = 2048;              // MDCT coefficients
constexpr int N = 2 * K;             // MDCT input samples
constexpr int H = K;                 // hop
constexpr int ERB_BANDS = 48;
constexpr int MIN_BAND_COEFFS = 3;
constexpr double CODED_HZ = 20000.0;
constexpr double PI = 3.1415926535897932384626433832795;
constexpr double LN2 = 0.693147180559945309417232121458176568;

static inline double hz_to_erb(double f) { return 21.4 * std::log10(1.0 + 0.00437 * f); }
static inline double erb_to_hz(double e) { return (std::pow(10.0, e / 21.4) - 1.0) / 0.00437; }

struct ErbLayout {
    std::array<int, ERB_BANDS + 1> edge{};
    void init(double sr) {
        // PULSAR coding domain: exactly 48 ERB-rate bands over 0..20 kHz.
        // Every band contains at least MIN_BAND_COEFFS coefficients.
        // At 48 kHz the remaining MDCT bins (>20 kHz) are outside the coded domain.
        const double fmax = std::min(CODED_HZ, 0.5 * sr);
        const int coded_bins = std::clamp(
            (int)std::llround(fmax * (2.0 * K) / sr), ERB_BANDS * MIN_BAND_COEFFS, K);
        const double emax = hz_to_erb(fmax);
        edge[0] = 0;
        for (int b = 1; b < ERB_BANDS; ++b) {
            const double f = erb_to_hz(emax * (double)b / ERB_BANDS);
            int k = (int)std::llround(f * (2.0 * K) / sr); // boundary in MDCT-bin units
            edge[b] = std::clamp(k, edge[b - 1] + MIN_BAND_COEFFS,
                                 coded_bins - MIN_BAND_COEFFS * (ERB_BANDS - b));
        }
        edge[ERB_BANDS] = coded_bins;
    }

    int coded_bins() const { return edge[ERB_BANDS]; }
};

// Low-overlap TDAC window. Only OVERLAP samples participate in the transition;
// the transform itself uses the N/4 complex-FFT factorization below.
constexpr int OVERLAP = 128; // fixed 128-sample low-overlap transition = 2.667 ms at 48 kHz.
struct PBWindow {
    std::array<float, OVERLAP> w{};
    void init() {
        for (int n=0;n<OVERLAP;n++) {
            const double u=(n+0.5)/(double)OVERLAP;
            w[n]=(float)std::sin(0.5*PI*u);
        }
    }
};

// Product transform below follows the standard low-overlap MDCT factorization:
// window+fold -> pre-rotation -> N/4 complex FFT -> post-rotation.
// This is the same algebraic factorization used by CELT-style MDCTs, but the FFT
// and SIMD implementation here is self-contained.

// Radix-4 FFT specialized to the MDCT's 1024 complex samples (4096/4).
// 1024 = 4^5, so the transform is five radix-4 stages with no radix-2 tail.
struct FFT1024Radix4 {
    static constexpr int F=1024;
    static constexpr int STAGES=5;
    alignas(64) std::array<float,F> re{},im{},tr{},ti{};
    std::array<uint16_t,F> perm{};
    alignas(64) std::array<float,F> wr{},wi{};
    FFT1024Radix4(){
        for(int k=0;k<F;k++){
            const double a=-2.0*PI*k/F;
            wr[k]=(float)std::cos(a); wi[k]=(float)std::sin(a);
        }
        for(int i=0;i<F;i++){
            int x=i,y=0;
            for(int s=0;s<STAGES;s++){ y=(y<<2)|(x&3); x>>=2; }
            perm[i]=(uint16_t)y;
        }
    }
    static inline void mul(float ar,float ai,float br,float bi,float& rr,float& ri){
        rr=ar*br-ai*bi; ri=ar*bi+ai*br;
    }
    void forward(){
        for(int m=F;m>=4;m>>=2){
            const int q=m>>2,step=F/m;
            for(int base=0;base<F;base+=m){
                for(int j=0;j<q;j++){
                    const int p0=base+j,p1=p0+q,p2=p1+q,p3=p2+q;
                    const float a0r=re[p0],a0i=im[p0],a1r=re[p1],a1i=im[p1];
                    const float a2r=re[p2],a2i=im[p2],a3r=re[p3],a3i=im[p3];
                    const float s02r=a0r+a2r,s02i=a0i+a2i,d02r=a0r-a2r,d02i=a0i-a2i;
                    const float s13r=a1r+a3r,s13i=a1i+a3i,d13r=a1r-a3r,d13i=a1i-a3i;
                    re[p0]=s02r+s13r; im[p0]=s02i+s13i;
                    const float y1r=d02r+d13i,y1i=d02i-d13r;
                    const float y2r=s02r-s13r,y2i=s02i-s13i;
                    const float y3r=d02r-d13i,y3i=d02i+d13r;
                    const int k=j*step;
                    mul(y1r,y1i,wr[k],wi[k],re[p1],im[p1]);
                    mul(y2r,y2i,wr[(2*k)&(F-1)],wi[(2*k)&(F-1)],re[p2],im[p2]);
                    mul(y3r,y3i,wr[(3*k)&(F-1)],wi[(3*k)&(F-1)],re[p3],im[p3]);
                }
            }
        }
        for(int i=0;i<F;i++){tr[i]=re[perm[i]];ti[i]=im[perm[i]];}
        re.swap(tr);im.swap(ti);
    }
};

struct MDCT2048 {
    FFT1024Radix4 fft;
    // trig[i] = cos(2*pi*(i+1/8)/N), i=0..N/2-1.
    alignas(64) std::array<float,N/2> trig{};
    alignas(64) std::array<float,N/2> fold{};
    MDCT2048(){
        for(int i=0;i<N/2;i++) trig[i]=(float)std::cos(2.0*PI*(i+0.125)/N);
    }

    void forward(const float* x,const float* window,float* out){
        constexpr int N2=N/2,N4=N/4,O=OVERLAP;
        const float* xp1=x+(O>>1);
        const float* xp2=x+N2-1+(O>>1);
        float* yp=fold.data();
        const float* wp1=window+(O>>1);
        const float* wp2=window+(O>>1)-1;
        int i=0;

        // Window + shuffle + fold. Branching depends only on fixed loop regions, never signal values.
        for(;i<((O+3)>>2);i++){
            *yp++ = xp1[N2]*(*wp2) + (*xp2)*(*wp1);
            *yp++ = (*xp1)*(*wp1) - xp2[-N2]*(*wp2);
            xp1+=2;xp2-=2;wp1+=2;wp2-=2;
        }
        wp1=window;wp2=window+O-1;
        for(;i<N4-((O+3)>>2);i++){
            *yp++=*xp2; *yp++=*xp1; xp1+=2;xp2-=2;
        }
        for(;i<N4;i++){
            *yp++ = -xp1[-N2]*(*wp1) + (*xp2)*(*wp2);
            *yp++ = (*xp1)*(*wp2) + xp2[N2]*(*wp1);
            xp1+=2;xp2-=2;wp1+=2;wp2-=2;
        }

        // Pre-rotation. scale=1/N4 makes the FFT convention match the normalized MDCT kernel.
        constexpr float scale=1.0f/N4;
        yp=fold.data();
#if defined(__AVX2__)
        for(i=0;i<N4;i+=8){
            // Eight complex values: deinterleave 16 folded floats with two AVX loads.
            const __m256 a=_mm256_load_ps(yp+2*i);
            const __m256 b=_mm256_load_ps(yp+2*i+8);
            alignas(32) float v[16];
            _mm256_store_ps(v,a);_mm256_store_ps(v+8,b);
            for(int lane=0;lane<8;lane++){
                const float re=v[2*lane],im=v[2*lane+1];
                const float t0=trig[i+lane],t1=trig[N4+i+lane];
                fft.re[i+lane]=(re*t0-im*t1)*scale;
                fft.im[i+lane]=(im*t0+re*t1)*scale;
            }
        }
#elif defined(LXVOXEL_NEON) || defined(__ARM_NEON) || defined(__ARM_NEON__)
        const float32x4_t vscale=vdupq_n_f32(scale);
        for(i=0;i+4<=N4;i+=4){
            const float32x4x2_t z=vld2q_f32(yp+2*i);
            const float32x4_t t0=vld1q_f32(trig+i);
            const float32x4_t t1=vld1q_f32(trig+N4+i);
            const float32x4_t rr=vmulq_f32(vsubq_f32(vmulq_f32(z.val[0],t0),vmulq_f32(z.val[1],t1)),vscale);
            const float32x4_t ii=vmulq_f32(vaddq_f32(vmulq_f32(z.val[1],t0),vmulq_f32(z.val[0],t1)),vscale);
            vst1q_f32(fft.re.data()+i,rr);
            vst1q_f32(fft.im.data()+i,ii);
        }
        for(;i<N4;i++){
            const float re=yp[2*i],im=yp[2*i+1];
            const float t0=trig[i],t1=trig[N4+i];
            fft.re[i]=(re*t0-im*t1)*scale;
            fft.im[i]=(im*t0+re*t1)*scale;
        }
#else
        for(i=0;i<N4;i++){
            const float re=*yp++,im=*yp++;
            const float t0=trig[i],t1=trig[N4+i];
            fft.re[i]=(re*t0-im*t1)*scale;
            fft.im[i]=(im*t0+re*t1)*scale;
        }
#endif
        fft.forward();

        // Post-rotation writes the even coefficients from the front and odd coefficients from the back.
        float* y1=out; float* y2=out+N2-1;
        for(i=0;i<N4;i++){
            const float fr=fft.re[i],fi=fft.im[i];
            const float t0=trig[i],t1=trig[N4+i];
            const float yr=fi*t1-fr*t0;
            const float yi=fr*t1+fi*t0;
            *y1=yr; *y2=yi; y1+=2;y2-=2;
        }
    }
};

// Exact nearest-point quantizer for D_n = { z in Z^n : sum_i z_i is even }.
//
// Let r_i = round(u_i), e_i=u_i-r_i, |e_i|<=1/2.  If sum r_i is even,
// r is already in D_n and is the Euclidean nearest lattice point. If the parity
// is odd, every valid D_n point differs from r in an odd number of integer steps.
// The minimum correction is therefore exactly one +/-1 step. Its extra squared
// error is (1-|e_i|)^2-|e_i|^2 = 1-2|e_i|, so the exact optimum is the coordinate
// with maximum |e_i|, moved toward u_i.  Hence nearest-D_n is O(n), no codebook,
// no tree, no search iterations.

// -----------------------------------------------------------------------------
// FEAR Stable-V1 auditory control, adapted to the codec's *same* hard 48-ERB
// partition over 0..20 kHz.  Unlike the standalone FEAR bank, no second
// interpolation bank is built: each MDCT coefficient belongs to exactly one
// coding/FEAR band.  The stable law's Gammatone-4 spreading kernel is preserved.
//
// ISO 226:2023 Table 1 threshold of hearing T_f. The standard defines T_f
// only at these 29 preferred third-octave frequencies and specifies no
// interpolation. LX-VOXEL uses a shape-preserving PCHIP in log-frequency
// between those support points. Above 12.5 kHz, where ISO 226:2023 has no
// T_f data, the previous FEAR-V6 high-frequency law is retained only as a
// continuous out-of-standard extension anchored at the ISO 12.5-kHz value.
inline double lx_ath_legacy_v6_db_spl(double f_hz) {
    const double fk=std::max(f_hz*1e-3,0.02);
    const double base=3.64*std::pow(fk,-0.8)
                    - 6.5*std::exp(-0.6*(fk-3.3)*(fk-3.3));
    constexpr double q=461.0/480.0;
    const double h=0.001*std::pow(fk,4.0);
    return base + std::expm1(q*std::log1p(h))/q;
}

inline double lx_ath_iso226_2023_db_spl(double f_hz) {
    static constexpr std::array<double,29> F = {
        20.0,25.0,31.5,40.0,50.0,63.0,80.0,100.0,125.0,160.0,
        200.0,250.0,315.0,400.0,500.0,630.0,800.0,1000.0,1250.0,
        1600.0,2000.0,2500.0,3150.0,4000.0,5000.0,6300.0,8000.0,
        10000.0,12500.0
    };
    static constexpr std::array<double,29> T = {
        78.1,68.7,59.5,51.1,44.0,37.5,31.5,26.5,22.1,17.9,
        14.4,11.4,8.6,6.2,4.4,3.0,2.2,2.4,3.5,1.7,-1.3,-4.2,
        -6.0,-5.4,-1.5,6.0,12.6,13.9,12.3
    };
    if (f_hz <= F.front()) return T.front();
    if (f_hz >= F.back()) {
        return T.back() + lx_ath_legacy_v6_db_spl(f_hz)
                        - lx_ath_legacy_v6_db_spl(F.back());
    }

    std::array<double,29> x{},m{};
    std::array<double,28> h{},d{};
    for (size_t i=0;i<F.size();++i) x[i]=std::log(F[i]);
    for (size_t i=0;i+1<F.size();++i) {
        h[i]=x[i+1]-x[i];
        d[i]=(T[i+1]-T[i])/h[i];
    }
    auto same_sign=[](double a,double b){ return (a>0.0&&b>0.0)||(a<0.0&&b<0.0); };
    for (size_t i=1;i+1<F.size();++i) {
        if (!same_sign(d[i-1],d[i])) { m[i]=0.0; continue; }
        const double w1=2.0*h[i]+h[i-1];
        const double w2=h[i]+2.0*h[i-1];
        m[i]=(w1+w2)/(w1/d[i-1]+w2/d[i]);
    }
    m[0]=((2.0*h[0]+h[1])*d[0]-h[0]*d[1])/(h[0]+h[1]);
    if (!same_sign(m[0],d[0])) m[0]=0.0;
    else if (!same_sign(d[0],d[1]) && std::abs(m[0])>3.0*std::abs(d[0])) m[0]=3.0*d[0];
    constexpr size_t N=29;
    m[N-1]=((2.0*h[N-2]+h[N-3])*d[N-2]-h[N-2]*d[N-3])/(h[N-2]+h[N-3]);
    if (!same_sign(m[N-1],d[N-2])) m[N-1]=0.0;
    else if (!same_sign(d[N-2],d[N-3]) && std::abs(m[N-1])>3.0*std::abs(d[N-2])) m[N-1]=3.0*d[N-2];

    const double xf=std::log(f_hz);
    const auto it=std::upper_bound(x.begin(),x.end(),xf);
    const size_t k=size_t(it-x.begin()-1);
    const double u=(xf-x[k])/h[k],u2=u*u,u3=u2*u;
    const double h00=2.0*u3-3.0*u2+1.0;
    const double h10=u3-2.0*u2+u;
    const double h01=-2.0*u3+3.0*u2;
    const double h11=u3-u2;
    return h00*T[k] + h10*h[k]*m[k] + h01*T[k+1] + h11*h[k]*m[k+1];
}

// Important: the ATH term needs a physical full-scale SPL calibration.  It is
// therefore optional and is never replaced by an arbitrary digital "magic dB".
// Without calibration, ATH=0 and steering is masking/reference-excitation only.
struct Fear48 {
    alignas(64) std::array<float,ERB_BANDS> center_hz{};
    alignas(64) std::array<float,ERB_BANDS> erb_hz{};
    alignas(64) std::array<float,ERB_BANDS> ath{};
    alignas(64) std::array<std::array<float,ERB_BANDS>,ERB_BANDS> spread{}; // [dst][src]

    static inline double erb_width_hz(double f) { return 24.7 * (1.0 + 0.00437*f); }
    static inline double ath_db_spl(double f_hz) { return lx_ath_iso226_2023_db_spl(f_hz); }

    void init(double sr,const ErbLayout& layout,double full_scale_spl=100.0){
        const double df=sr/(2.0*K);
        for(int b=0;b<ERB_BANDS;b++){
            const double lo=layout.edge[b]*df;
            const double hi=layout.edge[b+1]*df;
            // Center in ERB-rate, not arithmetic Hz, so the auditory geometry
            // stays consistent even where min-3 projection moved low-band edges.
            const double ec=0.5*(hz_to_erb(lo)+hz_to_erb(hi));
            const double fc=erb_to_hz(ec);
            center_hz[b]=(float)fc;
            erb_hz[b]=(float)erb_width_hz(fc);
            ath[b]=(float)(0.5*std::pow(10.0,(ath_db_spl(fc)-full_scale_spl)/10.0));
        }
        for(int d=0;d<ERB_BANDS;d++){
            for(int src=0;src<ERB_BANDS;src++){
                const double u=2.0*((double)center_hz[d]-(double)center_hz[src])/(double)erb_hz[src];
                const double u2=u*u;
                spread[d][src]=(float)(1.0/((1.0+u2)*(1.0+u2)*(1.0+u2)*(1.0+u2)));
            }
        }
    }

    // Direct hard-band reference power. No spectral interpolation and no
    // signal-dependent branches in the coefficient loop.
    void reference_energy(const float* X,const ErbLayout& layout,std::array<float,ERB_BANDS>& e) const {
        e.fill(0.0f);
        for(int b=0;b<ERB_BANDS;b++){
            const int lo=layout.edge[b], hi=layout.edge[b+1];
#if defined(__AVX2__)
            __m256 acc=_mm256_setzero_ps(); int i=lo;
            for(;i+8<=hi;i+=8){const __m256 v=_mm256_loadu_ps(X+i); acc=_mm256_fmadd_ps(v,v,acc);} 
            alignas(32) float t[8]; _mm256_store_ps(t,acc); float s=t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
            for(;i<hi;i++) s+=X[i]*X[i]; e[b]=s;
#else
            float s=0.0f; for(int i=lo;i<hi;i++) s+=X[i]*X[i]; e[b]=s;
#endif
        }
    }

    // Xref = ATH + reference auditory excitation, exactly the denominator used
    // by FEAR Stable-V1. Dense 48x48, but only 2304 FMAs/frame.
    void xref_from_energy(const std::array<float,ERB_BANDS>& ref,std::array<float,ERB_BANDS>& xref) const {
        for(int d=0;d<ERB_BANDS;d++){
#if defined(__AVX2__)
            __m256 acc=_mm256_setzero_ps();
            for(int s=0;s<ERB_BANDS;s+=8){
                const __m256 k=_mm256_loadu_ps(spread[d].data()+s);
                const __m256 r=_mm256_loadu_ps(ref.data()+s);
                acc=_mm256_fmadd_ps(k,r,acc);
            }
            alignas(32) float t[8]; _mm256_store_ps(t,acc);
            xref[d]=ath[d]+t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
#else
            float v=ath[d]; for(int s=0;s<ERB_BANDS;s++) v+=spread[d][s]*ref[s]; xref[d]=v;
#endif
        }
    }
};

// A D_n symbol lives on an integer dyadic sublattice.  The *only* physical
// spectral unit is base_delta, known from the codec FBR/profile.  FEAR chooses
// a dyadic dilation 2^s, but that scale is not written as side information:
// the transmitted lattice vector itself is q = 2^s v.  Decoder reconstruction
// is simply xhat_i = base_delta*q_i.  Thus common low zero bitplanes are the
// scale, inherently contained in the lattice symbol.
//
// FEAR control is derived from its own audibility denominator Xref. Quantization
// error power scales as Delta^2, so equal relative audibility gives
// Delta_b proportional to sqrt(Xref_b).  We reference the smallest Xref in the
// frame, hence the most audible band uses the base lattice and every other band
// is an analytically coarser dyadic sublattice. No fitted psychoacoustic number.

struct FearFrameDiag {
    std::array<float,ERB_BANDS> ref{};
    std::array<float,ERB_BANDS> xref{};
    std::array<int,ERB_BANDS> shift{};
};

struct DQuantResult { double mse=0.0; double energy=0.0; int dim=0; };

struct alignas(64) DnScratch {
    std::array<int32_t, K> q{};
};

// Search/quantization only. Returns parity-corrected integer D_n vector.
// Signal-dependent decisions are vector comparisons / masks. The only scalar
// parity branch is once per whole ERB vector, outside the coefficient loop.
static inline void dn_nearest_i32_avx2(const float* __restrict x, int n, float inv_delta,
                                       int32_t* __restrict q) {
#if defined(__AVX2__)
    const __m256 vinv = _mm256_set1_ps(inv_delta);
    __m256 vmax = _mm256_set1_ps(-1.0f);
    __m256i vidxmax = _mm256_setzero_si256();
    __m256i vxorp = _mm256_setzero_si256();
    const __m256i lane = _mm256_setr_epi32(0,1,2,3,4,5,6,7);
    const __m256i eight = _mm256_set1_epi32(8);
    const __m256 signmask = _mm256_set1_ps(-0.0f);
    __m256i idx = lane;

    int i=0;
    for (; i+8<=n; i+=8) {
        const __m256 u = _mm256_mul_ps(_mm256_loadu_ps(x+i), vinv);
        const __m256 rn = _mm256_round_ps(u, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        const __m256i qi = _mm256_cvtps_epi32(rn);
        _mm256_storeu_si256((__m256i*)(q+i), qi);

        // XOR-reduction later needs only bit 0 of each lane; xor full ints is equivalent.
        vxorp = _mm256_xor_si256(vxorp, qi);

        const __m256 ae = _mm256_andnot_ps(signmask, _mm256_sub_ps(u,rn));
        const __m256 take = _mm256_cmp_ps(ae, vmax, _CMP_GT_OQ);
        vmax = _mm256_blendv_ps(vmax, ae, take);
        vidxmax = _mm256_blendv_epi8(vidxmax, idx, _mm256_castps_si256(take));
        idx = _mm256_add_epi32(idx,eight);
    }

    alignas(32) float maxv[8];
    alignas(32) int32_t maxi[8], xp[8];
    _mm256_store_ps(maxv,vmax);
    _mm256_store_si256((__m256i*)maxi,vidxmax);
    _mm256_store_si256((__m256i*)xp,vxorp);
    float best_abs=-1.0f; int best_i=0; int parity=0;
    for(int l=0;l<8;l++) {
        parity ^= xp[l];
        if(maxv[l]>best_abs){best_abs=maxv[l];best_i=maxi[l];}
    }
    for(;i<n;i++) {
        const float u=x[i]*inv_delta;
        const float rn=std::nearbyint(u);
        const int32_t qi=(int32_t)rn;
        q[i]=qi; parity^=qi;
        const float ae=std::fabs(u-rn);
        if(ae>best_abs){best_abs=ae;best_i=i;}
    }
    parity &= 1;
    if(parity) {
        const float u=x[best_i]*inv_delta;
        const float rn=(float)q[best_i];
        q[best_i] += 1 - 2*(int)(u<rn);
    }
#else
    int parity=0,best_i=0; float best_abs=-1.0f;
    for(int i=0;i<n;i++){
        const float u=x[i]*inv_delta, rn=std::nearbyint(u), ae=std::fabs(u-rn);
        const int32_t qi=(int32_t)rn; q[i]=qi; parity^=qi;
        if(ae>best_abs){best_abs=ae;best_i=i;}
    }
    if(parity&1){ const float u=x[best_i]*inv_delta; q[best_i]+=1-2*(int)(u<(float)q[best_i]); }
#endif
}

#if defined(__AVX512F__)
// AVX-512 version: 16 coefficients/iteration.  Kept separate so binaries built
// for AVX2-only CPUs remain valid; -march=native selects this path on capable CPUs.
static inline void dn_nearest_i32_avx512(const float* __restrict x, int n, float inv_delta,
                                         int32_t* __restrict q) {
    const __m512 vinv=_mm512_set1_ps(inv_delta);
    __m512 vmax=_mm512_set1_ps(-1.0f);
    __m512i vidxmax=_mm512_setzero_si512();
    __m512i vxorp=_mm512_setzero_si512();
    __m512i idx=_mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    const __m512i sixteen=_mm512_set1_epi32(16);
    const __m512 signmask=_mm512_set1_ps(-0.0f);
    int i=0;
    for(;i+16<=n;i+=16){
        const __m512 u=_mm512_mul_ps(_mm512_loadu_ps(x+i),vinv);
        const __m512 rn=_mm512_roundscale_ps(u,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
        const __m512i qi=_mm512_cvtps_epi32(rn);
        _mm512_storeu_si512((void*)(q+i),qi);
        vxorp=_mm512_xor_si512(vxorp,qi);
        const __m512 ae=_mm512_andnot_ps(signmask,_mm512_sub_ps(u,rn));
        const __mmask16 take=_mm512_cmp_ps_mask(ae,vmax,_CMP_GT_OQ);
        vmax=_mm512_mask_mov_ps(vmax,take,ae);
        vidxmax=_mm512_mask_mov_epi32(vidxmax,take,idx);
        idx=_mm512_add_epi32(idx,sixteen);
    }
    alignas(64) float maxv[16]; alignas(64) int32_t maxi[16],xp[16];
    _mm512_store_ps(maxv,vmax); _mm512_store_si512((void*)maxi,vidxmax); _mm512_store_si512((void*)xp,vxorp);
    float best_abs=-1.0f; int best_i=0,parity=0;
    for(int l=0;l<16;l++){parity^=xp[l]; if(maxv[l]>best_abs){best_abs=maxv[l];best_i=maxi[l];}}
    for(;i<n;i++){
        const float u=x[i]*inv_delta,rn=std::nearbyint(u),ae=std::fabs(u-rn); const int32_t qi=(int32_t)rn;
        q[i]=qi; parity^=qi; if(ae>best_abs){best_abs=ae;best_i=i;}
    }
    if(parity&1){const float u=x[best_i]*inv_delta; q[best_i]+=1-2*(int)(u<(float)q[best_i]);}
}
#endif

static inline void dn_nearest_i32(const float* __restrict x,int n,float inv_delta,int32_t* __restrict q){
#if defined(__AVX512F__)
    dn_nearest_i32_avx512(x,n,inv_delta,q);
#elif defined(__AVX2__)
    dn_nearest_i32_avx2(x,n,inv_delta,q);
#else
    dn_nearest_i32_avx2(x,n,inv_delta,q);
#endif
}


// Quantize one band on FEAR-selected sublattice 2^shift D_n, but materialize the
// final symbol in the finest integer lattice coordinates.  No scale field exists.


// -----------------------------------------------------------------------------
// Static-profile byte rANS for D_n lattice symbols.
//
// The entropy model is decoder-known from bits_per_coeff.  It uses a symmetric
// discrete-Laplacian (two-sided geometric) prior -- the natural model for
// transform coefficients and the same source family motivating PVQ.  Its rho is
// solved analytically/numerically so the prior entropy equals the profile rate R;
// there are no trained tables and no signal-derived model side bits.
//
// Integer q in [-127,127] maps directly to one of 255 signed symbols; 255 is ESC.
// Escaped int32 values are kept as ZigZag LEB128 in a deterministic escape tail.
// This preserves the exact implicit-scale lattice symbol q: no gain/scale field.

// -----------------------------------------------------------------------------
// Temporal FEAR + global bit reservoir (current LX-VOXEL psychoacoustic path).

// FEAR-V2 error-audibility importance.
//
// `threshold` is the external auditory threshold for this ERB band
// (ATH + masking from the other bands, with self masking removed).
// The previous allocator gated importance by the *quantizer* noise q0,
// which made quiet but clearly audible structure lose priority merely
// because its absolute level was low.  That couples psychoacoustics to
// the current operating point and is the wrong dependency.
//
// Instead we use the source audibility ratio itself.  The factor
//
//      energy / (energy + threshold)
//
// is the continuous occupancy of the audible region: it tends to zero
// only when the source vanishes below its auditory threshold, and tends
// to one as soon as real, unmasked structure dominates that threshold.
// Once occupied, marginal quantization-error importance is therefore
// controlled by 1/threshold, not by absolute source loudness.
//
// No threshold branch, no quiet boost, no codec-dependent q0 term.
// FEAR-V3: causal error audibility.
//
// V2 fixed quiet passages by removing the codec-step-dependent Q0 gate.
// V3 adds the smallest useful causal memory without thresholds:
//   - the auditory ERB time constant is lower-bounded by one codec hop,
//     because sub-hop states are unobservable at this frame rate;
//   - a second state integrates over twice that time;
//   - persistent structure uses the geometric mean sqrt(E * slow);
//   - transient novelty is the normalized squared fast/slow disagreement.
// The equations are continuous and contain no signal-dependent branches.
inline double fear_v3_error_importance(double energy,
                                       double slow_energy,
                                       double threshold,
                                       double novelty) {
    const double tiny = std::numeric_limits<double>::min();
    const double persistent = std::sqrt(std::max(0.0, energy * slow_energy));
    const double occupancy = persistent / (persistent + threshold + tiny);
    return (1.0 + novelty) * occupancy / (threshold + tiny);
}

struct FearTemporalState48 {
    std::array<float,ERB_BANDS> fast{},slow{},aFast{},aSlow{};
    bool initialized=false;

    void init(const Fear48& fear,double hop_seconds){
        for(int b=0;b<ERB_BANDS;b++){
            const double auditory_tau=1.0/(2.0*PI*(double)fear.erb_hz[b]);
            const double tau_fast=std::max(auditory_tau, hop_seconds);
            const double tau_slow=2.0*tau_fast;
            aFast[b]=(float)std::exp(-hop_seconds/tau_fast);
            aSlow[b]=(float)std::exp(-hop_seconds/tau_slow);
        }
        fast.fill(0.0f); slow.fill(0.0f); initialized=false;
    }

    void analyze(const std::array<float,ERB_BANDS>& e,
                 std::array<float,ERB_BANDS>& slow_out,
                 std::array<float,ERB_BANDS>& novelty){
        if(!initialized){ fast=e; slow=e; initialized=true; }
        const float tiny=std::numeric_limits<float>::min();
#if defined(__AVX2__)
        for(int b=0;b<ERB_BANDS;b+=8){
            const __m256 ve=_mm256_loadu_ps(e.data()+b);
            __m256 vf=_mm256_loadu_ps(fast.data()+b);
            __m256 vs=_mm256_loadu_ps(slow.data()+b);
            const __m256 af=_mm256_loadu_ps(aFast.data()+b);
            const __m256 as=_mm256_loadu_ps(aSlow.data()+b);
            const __m256 one=_mm256_set1_ps(1.0f);
            vf=_mm256_add_ps(_mm256_mul_ps(af,vf),_mm256_mul_ps(_mm256_sub_ps(one,af),ve));
            vs=_mm256_add_ps(_mm256_mul_ps(as,vs),_mm256_mul_ps(_mm256_sub_ps(one,as),ve));
            _mm256_storeu_ps(fast.data()+b,vf);
            _mm256_storeu_ps(slow.data()+b,vs);
            _mm256_storeu_ps(slow_out.data()+b,vs);

            const __m256 d=_mm256_sub_ps(vf,vs);
            const __m256 num=_mm256_mul_ps(d,d);
            const __m256 ff=_mm256_mul_ps(vf,vf);
            const __m256 ss=_mm256_mul_ps(vs,vs);
            const __m256 den=_mm256_max_ps(_mm256_add_ps(ff,ss),_mm256_set1_ps(tiny));
            _mm256_storeu_ps(novelty.data()+b,_mm256_div_ps(num,den));
        }
#else
        for(int b=0;b<ERB_BANDS;b++){
            fast[b]=aFast[b]*fast[b]+(1.0f-aFast[b])*e[b];
            slow[b]=aSlow[b]*slow[b]+(1.0f-aSlow[b])*e[b];
            slow_out[b]=slow[b];
            const float d=fast[b]-slow[b];
            novelty[b]=(d*d)/std::max(fast[b]*fast[b]+slow[b]*slow[b],tiny);
        }
#endif
    }
};

struct TemporalGbrCell {
    float loga=0.0f;
    float bpc=0.0f;
    uint16_t dim=0;
};


} // namespace pulsar
