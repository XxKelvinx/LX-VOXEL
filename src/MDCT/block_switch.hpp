
#pragma once
#include "dsp/core.hpp"
#include "MDCT/mdct.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <atomic>
#include <thread>
#include <stdexcept>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace pulsar_bs {

using namespace pulsar;

constexpr int K_ERB_MASTER = 2048;
// Physical MDCT lengths. K_1024/K_256 and the BlockSize enum labels are
// retained only as internal legacy identifiers; they do not denote those lengths.
constexpr int K_LONG   = 1024;
constexpr int K_1024   = 768;
constexpr int K_SHORT  = 512;
constexpr int K_256    = 384;
constexpr int K_MIN    = K_256;
constexpr int OVL      = OVERLAP;

// Two-bit block syntax.  The low bit deliberately preserves the old v0.10
// meaning for the legacy pair: 00=2048 and 01=512.  v0.11 adds one packed high
// bit per frame, giving 10=1024 and 11=256 without perturbing any other control
// flags in the byte-aligned side syntax.
enum class BlockSize : uint8_t {
    Long2048   = 0, // 00
    Short512   = 1, // 01 (legacy short code)
    Medium1024 = 2, // 10
    Micro256   = 3  // 11
};

constexpr int block_samples(BlockSize b) {
    switch(b){
        case BlockSize::Long2048:   return K_LONG;
        case BlockSize::Medium1024: return K_1024;
        case BlockSize::Short512:   return K_SHORT;
        case BlockSize::Micro256:   return K_256;
    }
    return K_LONG;
}
constexpr unsigned block_syntax(BlockSize b) { return unsigned(uint8_t(b)) & 3u; }
constexpr unsigned block_low_bit(BlockSize b) { return block_syntax(b) & 1u; }
constexpr unsigned block_high_bit(BlockSize b){ return block_syntax(b) >> 1; }

// Temporal-resolution order, independent of the backward-compatible syntax code.
constexpr int block_level(BlockSize b) {
    switch(b){
        case BlockSize::Long2048:   return 0;
        case BlockSize::Medium1024: return 1;
        case BlockSize::Short512:   return 2;
        case BlockSize::Micro256:   return 3;
    }
    return 0;
}
constexpr BlockSize block_from_level(int level) {
    return level<=0 ? BlockSize::Long2048
         : level==1 ? BlockSize::Medium1024
         : level==2 ? BlockSize::Short512
                    : BlockSize::Micro256;
}

struct FrameHeader {
    uint8_t stereo_ms = 0;
    uint8_t block_code = 0; // 2-bit BlockSize syntax code

    uint8_t packed3() const {
        return uint8_t((stereo_ms & 1u) | ((block_code & 3u) << 1));
    }
};

// Canonical 48-band analysis/VQ partition.
//
// The 2048-domain widths are the master geometry.  1024 and 512 are exact
// integer quotients.  At 256, the 20-kHz coded domain is only 213.5 ideal bins,
// so exact /8 widths would create half-bin ERBs.  Instead of clamping bands or
// merging them heuristically, project the ideal scaled widths onto the positive
// integer simplex: minimize sum_b (w_b - w*_b)^2 subject to w_b>=1 and a fixed
// rounded total coded-bin count.  The greedy unit-cost updates below are exact
// for this separable convex integer problem.
inline constexpr std::array<uint16_t, ERB_BANDS> LX_ERB_WIDTH_2048 = {
      4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
      8,  8,  8,  8,  8,  8,
     12, 12, 12, 12,
     16, 16,
     20, 20, 20,
     24,
     28, 28,
     32,
     36, 36,
     40,
     44,
     48,
     56,
     60,
     64,
     72,
     80,
     88,
     96,
    104,
    116,
    128,
    140,
    152
};

inline constexpr int LX_CODED_BINS_2048=[](){int s=0;for(auto w:LX_ERB_WIDTH_2048)s+=int(w);return s;}();

template<int KC>
struct ErbLayoutN {
    static_assert(KC == K_LONG || KC == K_1024 || KC == K_SHORT || KC == K_256,
                  "LX product ERB partition supports 2048/1024/512/256");

    std::array<int, ERB_BANDS + 1> edge{};

    void init(double sr) {
        (void)sr; // Product profile is fixed at 48 kHz / 20 kHz coded bandwidth.
        constexpr int target=(LX_CODED_BINS_2048*KC + K_ERB_MASTER/2)/K_ERB_MASTER;
        std::array<int,ERB_BANDS> width{};
        int total=0;
        for(int b=0;b<ERB_BANDS;++b){
            const int num=int(LX_ERB_WIDTH_2048[b])*KC;
            width[b]=std::max(1,num/K_LONG);
            total+=width[b];
        }
        while(total<target){
            int best=0; double best_cost=std::numeric_limits<double>::infinity();
            for(int b=0;b<ERB_BANDS;++b){
                const double ideal=double(LX_ERB_WIDTH_2048[b])*double(KC)/double(K_ERB_MASTER);
                const double d=double(width[b])-ideal;
                const double cost=2.0*d+1.0; // SSE increase for w -> w+1
                if(cost<best_cost){best_cost=cost;best=b;}
            }
            ++width[best]; ++total;
        }
        while(total>target){
            int best=-1; double best_cost=std::numeric_limits<double>::infinity();
            for(int b=0;b<ERB_BANDS;++b){
                if(width[b]<=1) continue;
                const double ideal=double(LX_ERB_WIDTH_2048[b])*double(KC)/double(K_ERB_MASTER);
                const double d=double(width[b])-ideal;
                const double cost=-2.0*d+1.0; // SSE increase for w -> w-1
                if(cost<best_cost){best_cost=cost;best=b;}
            }
            if(best<0) throw std::runtime_error("invalid ERB simplex projection");
            --width[best]; --total;
        }
        edge[0]=0;
        for(int b=0;b<ERB_BANDS;++b) edge[b+1]=edge[b]+width[b];
    }

    int coded_bins() const { return edge[ERB_BANDS]; }
};

template<int KC>
struct Fear48N {
    alignas(64) std::array<float,ERB_BANDS> center_hz{}, erb_hz{}, ath{};
    alignas(64) std::array<std::array<float,ERB_BANDS>,ERB_BANDS> spread{};

    static inline double erb_width_hz(double f) { return 24.7 * (1.0 + 0.00437*f); }
    static inline double ath_db_spl(double f_hz) { return pulsar::lx_ath_iso226_2023_db_spl(f_hz); }

    void init(double sr,const ErbLayoutN<KC>& layout,double full_scale_spl=100.0) {
        const double df=sr/(2.0*KC);
        for(int b=0;b<ERB_BANDS;++b) {
            const double lo=layout.edge[b]*df, hi=layout.edge[b+1]*df;
            const double ec=0.5*(hz_to_erb(lo)+hz_to_erb(hi));
            const double fc=erb_to_hz(ec);
            center_hz[b]=(float)fc;
            erb_hz[b]=(float)erb_width_hz(fc);
            ath[b]=(float)(0.5*std::pow(10.0,(ath_db_spl(fc)-full_scale_spl)/10.0));
        }
        for(int d=0;d<ERB_BANDS;++d) for(int s=0;s<ERB_BANDS;++s) {
            const double u=2.0*(double(center_hz[d])-double(center_hz[s]))/double(erb_hz[s]);
            const double z=1.0+u*u;
            spread[d][s]=(float)(1.0/(z*z*z*z));
        }
    }

    void reference_energy(const float* X,const ErbLayoutN<KC>& layout,
                          std::array<float,ERB_BANDS>& e) const {
        e.fill(0.0f);
        for(int b=0;b<ERB_BANDS;++b) {
            const int lo=layout.edge[b], hi=layout.edge[b+1];
#if defined(__AVX2__)
            __m256 acc=_mm256_setzero_ps(); int i=lo;
            for(;i+8<=hi;i+=8){const __m256 v=_mm256_loadu_ps(X+i);acc=_mm256_fmadd_ps(v,v,acc);}
            alignas(32) float t[8];_mm256_store_ps(t,acc);
            float s=t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
            for(;i<hi;++i)s+=X[i]*X[i];
            e[b]=s;
#else
            float s=0;for(int i=lo;i<hi;++i)s+=X[i]*X[i];e[b]=s;
#endif
        }
    }

    void xref_from_energy(const std::array<float,ERB_BANDS>& ref,
                          std::array<float,ERB_BANDS>& xref) const {
        for(int d=0;d<ERB_BANDS;++d) {
#if defined(__AVX2__)
            __m256 acc=_mm256_setzero_ps();
            for(int s=0;s<ERB_BANDS;s+=8)
                acc=_mm256_fmadd_ps(_mm256_loadu_ps(spread[d].data()+s),
                                    _mm256_loadu_ps(ref.data()+s),acc);
            alignas(32) float t[8];_mm256_store_ps(t,acc);
            xref[d]=ath[d]+t[0]+t[1]+t[2]+t[3]+t[4]+t[5]+t[6]+t[7];
#else
            float v=ath[d];for(int s=0;s<ERB_BANDS;++s)v+=spread[d][s]*ref[s];xref[d]=v;
#endif
        }
    }
};

// Single causal FEAR memory shared across both block sizes.
// The only thing that changes with the syntax bit is the precomputed decay for
// that frame duration. State itself is never reset at a block transition.
struct FearTemporalVariable48 {
    std::array<std::array<float,ERB_BANDS>,4> af{},as{};
    std::array<float,ERB_BANDS> fast{},slow{};
    bool initialized=false;

    template<class Fear>
    static void make_decay(const Fear& fear,double hop,
                           std::array<float,ERB_BANDS>& af_out,
                           std::array<float,ERB_BANDS>& as_out) {
        for(int b=0;b<ERB_BANDS;++b) {
            const double auditory_tau=1.0/(2.0*PI*double(fear.erb_hz[b]));
            const double tf=std::max(auditory_tau,hop);
            af_out[b]=(float)std::exp(-hop/tf);
            as_out[b]=(float)std::exp(-hop/(2.0*tf));
        }
    }

    template<class Fear>
    void init(const Fear& fear,double sr) {
        make_decay(fear,K_LONG /sr,af[0],as[0]);
        make_decay(fear,K_1024 /sr,af[1],as[1]);
        make_decay(fear,K_SHORT/sr,af[2],as[2]);
        make_decay(fear,K_256  /sr,af[3],as[3]);
        fast.fill(0);slow.fill(0);initialized=false;
    }

    void analyze(const std::array<float,ERB_BANDS>& e,BlockSize bs,
                 std::array<float,ERB_BANDS>& slow_out,
                 std::array<float,ERB_BANDS>& novelty) {
        if(!initialized){fast=e;slow=e;initialized=true;}
        const int bi=block_level(bs);
        const auto& afb=af[bi];
        const auto& asb=as[bi];
        const float tiny=std::numeric_limits<float>::min();
#if defined(__AVX2__)
        for(int b=0;b<ERB_BANDS;b+=8) {
            const __m256 ve=_mm256_loadu_ps(e.data()+b);
            __m256 vf=_mm256_loadu_ps(fast.data()+b);
            __m256 vs=_mm256_loadu_ps(slow.data()+b);
            const __m256 va=_mm256_loadu_ps(afb.data()+b);
            const __m256 vb=_mm256_loadu_ps(asb.data()+b);
            const __m256 one=_mm256_set1_ps(1.0f);
            vf=_mm256_add_ps(_mm256_mul_ps(va,vf),_mm256_mul_ps(_mm256_sub_ps(one,va),ve));
            vs=_mm256_add_ps(_mm256_mul_ps(vb,vs),_mm256_mul_ps(_mm256_sub_ps(one,vb),ve));
            _mm256_storeu_ps(fast.data()+b,vf);_mm256_storeu_ps(slow.data()+b,vs);
            _mm256_storeu_ps(slow_out.data()+b,vs);
            const __m256 d=_mm256_sub_ps(vf,vs);
            const __m256 num=_mm256_mul_ps(d,d);
            const __m256 den=_mm256_max_ps(_mm256_add_ps(_mm256_mul_ps(vf,vf),_mm256_mul_ps(vs,vs)),
                                           _mm256_set1_ps(tiny));
            _mm256_storeu_ps(novelty.data()+b,_mm256_div_ps(num,den));
        }
#else
        for(int b=0;b<ERB_BANDS;++b){
            fast[b]=afb[b]*fast[b]+(1-afb[b])*e[b];
            slow[b]=asb[b]*slow[b]+(1-asb[b])*e[b];
            slow_out[b]=slow[b];
            const float d=fast[b]-slow[b];
            novelty[b]=d*d/std::max(fast[b]*fast[b]+slow[b]*slow[b],tiny);
        }
#endif
    }
};

// Generic radix-4 FFT for the two exact MDCT sizes used by LX-VOXEL.
// F=1024 for K=2048 and F=256 for K=512.
template<int F>
struct FFTPow4 {
    static constexpr int stages=[](){int x=F,s=0;while(x>1){x>>=2;++s;}return s;}();
    alignas(64) std::array<float,F> re{},im{},tr{},ti{},wr{},wi{};
    std::array<uint16_t,F> perm{};
    FFTPow4(){
        static_assert((1<<(2*stages))==F);
        for(int k=0;k<F;++k){const double a=-2.0*PI*k/F;wr[k]=(float)std::cos(a);wi[k]=(float)std::sin(a);}
        for(int i=0;i<F;++i){int x=i,y=0;for(int s=0;s<stages;++s){y=(y<<2)|(x&3);x>>=2;}perm[i]=(uint16_t)y;}
    }
    static inline void mul(float ar,float ai,float br,float bi,float&rr,float&ri){
        rr=ar*br-ai*bi;ri=ar*bi+ai*br;
    }
    void forward(){
        for(int m=F;m>=4;m>>=2){
            const int q=m>>2,step=F/m;
            for(int base=0;base<F;base+=m)for(int j=0;j<q;++j){
                const int p0=base+j,p1=p0+q,p2=p1+q,p3=p2+q;
                const float a0r=re[p0],a0i=im[p0],a1r=re[p1],a1i=im[p1],
                            a2r=re[p2],a2i=im[p2],a3r=re[p3],a3i=im[p3];
                const float s02r=a0r+a2r,s02i=a0i+a2i,d02r=a0r-a2r,d02i=a0i-a2i;
                const float s13r=a1r+a3r,s13i=a1i+a3i,d13r=a1r-a3r,d13i=a1i-a3i;
                re[p0]=s02r+s13r;im[p0]=s02i+s13i;
                const float y1r=d02r+d13i,y1i=d02i-d13r;
                const float y2r=s02r-s13r,y2i=s02i-s13i;
                const float y3r=d02r-d13i,y3i=d02i+d13r;
                const int k=j*step;
                mul(y1r,y1i,wr[k],wi[k],re[p1],im[p1]);
                mul(y2r,y2i,wr[(2*k)&(F-1)],wi[(2*k)&(F-1)],re[p2],im[p2]);
                mul(y3r,y3i,wr[(3*k)&(F-1)],wi[(3*k)&(F-1)],re[p3],im[p3]);
            }
        }
        for(int i=0;i<F;++i){tr[i]=re[perm[i]];ti[i]=im[perm[i]];}
        re.swap(tr);im.swap(ti);
    }
};

template<int KC>
struct LowOverlapMDCT {
    static constexpr int NC=2*KC,N2=KC,N4=KC/2,F=KC/2;
    FFTPow4<F> fft;
    alignas(64) std::array<float,KC> trig{},fold{};
    LowOverlapMDCT(){for(int i=0;i<KC;++i)trig[i]=(float)std::cos(2.0*PI*(i+0.125)/NC);}

    void forward(const float*x,const float*window,float*out){
        const float*xp1=x+(OVL>>1),*xp2=x+N2-1+(OVL>>1);
        float*yp=fold.data();const float*wp1=window+(OVL>>1),*wp2=window+(OVL>>1)-1;
        int i=0;
        for(;i<((OVL+3)>>2);++i){
            *yp++=xp1[N2]*(*wp2)+(*xp2)*(*wp1);
            *yp++=(*xp1)*(*wp1)-xp2[-N2]*(*wp2);
            xp1+=2;xp2-=2;wp1+=2;wp2-=2;
        }
        wp1=window;wp2=window+OVL-1;
        for(;i<N4-((OVL+3)>>2);++i){*yp++=*xp2;*yp++=*xp1;xp1+=2;xp2-=2;}
        for(;i<N4;++i){
            *yp++=-xp1[-N2]*(*wp1)+(*xp2)*(*wp2);
            *yp++=(*xp1)*(*wp2)+xp2[N2]*(*wp1);
            xp1+=2;xp2-=2;wp1+=2;wp2-=2;
        }
        constexpr float scale=1.0f/N4;yp=fold.data();
        for(i=0;i<N4;++i){
            const float re=yp[2*i],im=yp[2*i+1],t0=trig[i],t1=trig[N4+i];
            fft.re[i]=(im*t0+re*t1)*scale;
            fft.im[i]=(re*t0-im*t1)*scale;
        }
        fft.forward();
        for(i=0;i<N4;++i){
            const float re=fft.im[i],im=fft.re[i],t0=trig[i],t1=trig[N4+i];
            out[2*i]=re*t0+im*t1;
            out[2*i+1]=re*t1-im*t0;
        }
    }
};

struct FramePlan {
    size_t sample_pos=0;
    BlockSize block=BlockSize::Long2048;
    size_t coeff_off=0;
    size_t q_off=0;
    uint16_t k=K_LONG;
    uint16_t coded_bins=0;
};


// LX-VOXEL v0.2.14: legacy Opus/CELT transient and LPC/tone selector removed.
// Block decisions are made exclusively by LX-TD M9 in lx_voxel_rans4.cpp.

} // namespace pulsar_bs
