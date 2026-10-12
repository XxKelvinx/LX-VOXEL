
#include <array>
#include <bit>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <thread>
#include <vector>
#include <fstream>
#include <filesystem>
#include <optional>
#include <sstream>
#if defined(LXVOXEL_NEON) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shlobj.h>
#endif

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif


#include "bitstream/format.hpp"
#include "rANS/rans.hpp"
#include "rANS/sideinfo_rans.hpp"
#include "io/fast_io.hpp"
#include "io/audio_input.hpp"
#include "metadata/metadata.hpp"
#include "FEAR/fear.hpp"
#include "PitchPredictor/pitch_predictor.hpp"
#include "quant/conditional_gain.hpp"
#include "quant/warp.hpp"
#include "FBR/rate_model.hpp"
#include "synthesis/cpm.hpp"
#include "synthesis/sfc.hpp"
#include "synthesis/gain_predictor.hpp"
#include "MDCT/block_switch.hpp"

#ifndef LX_SFC_SEL_MODE
#define LX_SFC_SEL_MODE 12
#endif

#ifndef LX_FEAR_TC_MODE
#define LX_FEAR_TC_MODE 17
#endif

#ifndef LX_FEAR_V4_GAIN
#define LX_FEAR_V4_GAIN 1.0
#endif

#ifndef LX_SFC_TONE_MODE
// production default: normalized target-energy-weighted FEAR distortion.
// 20 s / 128 kbit/s sweep: robust winner on all eight validation tracks.
#define LX_SFC_TONE_MODE 1
#endif
#ifndef LX_PREC_RES_MODE
#define LX_PREC_RES_MODE 0
#endif
#ifndef LX_CPM_PREC_SHIFT
#define LX_CPM_PREC_SHIFT 4
#endif
#ifndef LX_SFC_AUX_MODE
#define LX_SFC_AUX_MODE 0
#endif
#ifndef LX_PREC_RATE_MUL
#define LX_PREC_RATE_MUL 4.0
#endif

using namespace pulsar;
using namespace pulsar_bs;
using namespace lx_nsmdct;
using Clock=std::chrono::steady_clock;

namespace {
static std::filesystem::path path_from_utf8(const char* value){
    const auto* first=reinterpret_cast<const char8_t*>(value);
    return std::filesystem::path(std::u8string(first,first+std::strlen(value)));
}

static void notify_output_updated(const std::filesystem::path& path) noexcept{
#if defined(_WIN32)
    // Explorer and Media Player cache properties and thumbnails by pathname.
    // Tell the shell that a newly written (or overwritten) VOX file must be
    // queried again through the LX-VOXEL property/thumbnail handler.
    SHChangeNotify(SHCNE_UPDATEITEM,
        SHCNF_PATHW | SHCNF_FLUSHNOWAIT,
        path.c_str(),nullptr);
#else
    (void)path;
#endif
}

static void set_environment(const char* name,const char* value){
#if defined(_WIN32)
    if(_putenv_s(name,value)!=0) throw std::runtime_error("cannot set process environment");
#else
    if(setenv(name,value,1)!=0) throw std::runtime_error("cannot set process environment");
#endif
}

static inline double lx_fear_loga(double lf,double iso,double meanlf){
    const double c=lf-meanlf;
    const double h=2.0*std::sqrt(iso)/(1.0+std::sqrt(iso));
    // Isotropy may shape FEAR contrast, but it is not an audibility term.
    // The former +log(iso)/h offset could suppress sparse/tonal, unmasked
    // bands by tens of dB after FEAR had already identified them as audible.
    return meanlf+h*c;
}
constexpr double SR=48000.0;
constexpr int MODEL_SUPERFRAME=32;

constexpr int CLASS_FREQ_CTX = ERB_BANDS/2; // 2 ERBs/class frequency context
constexpr int CLASS_TEMP_STATES = 4;
constexpr int CLASS_TUPLE_CTX = CLASS_FREQ_CTX*3*CLASS_TEMP_STATES; // local count x temporal support-density state
constexpr int MAG_BASE_CTX = 11;
constexpr int MAG_PRED_STATES = 4;
constexpr int MAG_LOCAL_STATES = 3;
constexpr int MAG_CTX = MAG_BASE_CTX * MAG_PRED_STATES * MAG_LOCAL_STATES;
constexpr std::array<uint8_t,ERB_BANDS> MAG_CTX_MAP = {0,0,0,1,1,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,10,10,10,10,10,10};
constexpr int MAG_ESC=31;
constexpr int MAG_ALPHA=MAG_ESC+1;

template<int A,int C>
struct CtxRansModels {
    struct DivMeta { uint16_t start,freq; };
    struct alignas(16) Q44Meta { uint64_t recip; uint32_t xmax, sc; }; // sc=start | complement<<12
    static_assert(sizeof(DivMeta)==4); static_assert(sizeof(Q44Meta)==16);
    alignas(64) std::array<lx_rans4_fast::EncMeta,A*C> flat{};
    alignas(64) std::array<DivMeta,A*C> divflat{};
    alignas(64) std::array<Q44Meta,A*C> q44{};

    static void build_one(const std::array<uint32_t,A>& hist,
                          lx_rans4_fast::EncMeta* enc,DivMeta* div){
        using namespace lx_rans4_fast;
        uint64_t total_mass=0;
        for(int s=0;s<A;++s) total_mass+=2ull*hist[s]+1ull;
        constexpr uint64_t rem=uint64_t(M)-A;
        uint64_t cum=0,prev_q=0;
        uint32_t c=0;
        for(int s=0;s<A;++s){
            cum+=2ull*hist[s]+1ull;
            const uint64_t q=(cum*rem)/total_mass;
            const uint32_t f=1u+uint32_t(q-prev_q); prev_q=q;
            uint32_t rcp,sh,bias;
            if(f<2){rcp=0xffffffffu;sh=0;bias=c+M-1;}
            else{uint32_t ceil_log2=0;while(f>(1u<<ceil_log2))++ceil_log2;rcp=uint32_t(((uint64_t(1)<<(ceil_log2+31))+f-1)/f);sh=ceil_log2-1;bias=c;}
            enc[s]={rcp,bias|((M-f)<<13)|(sh<<25)};
            div[s]={uint16_t(c),uint16_t(f)};
            c+=f;
        }
    }
    void sync_q44(){for(size_t i=0;i<size_t(A*C);++i){const uint32_t f=divflat[i].freq,st=divflat[i].start;const uint64_t r=((uint64_t(1)<<44)+f-1u)/f;q44[i]={r,f<<20,st|((4096u-f)<<12)};}}
    CtxRansModels(){std::array<uint32_t,A> h{};h.fill(1);for(int c=0;c<C;++c) build_one(h,flat.data()+size_t(c)*A,divflat.data()+size_t(c)*A);sync_q44();}
    void rebuild_encoder_only(const std::array<std::array<uint32_t,A>,C>& h){for(int c=0;c<C;++c) build_one(h[c],flat.data()+size_t(c)*A,divflat.data()+size_t(c)*A);sync_q44();}
};


template<int A,int C>
struct CtxQ44SoAModels {
    static_assert(lx_rans4_fast::M == 4096u);
    static_assert(lx_rans4_fast::L == (1u<<16));
    static_assert(A > 1 && A < 4096);
    alignas(64) std::array<uint64_t,A*C> qrecip{};
    alignas(64) std::array<uint32_t,A*C> qxmax{}, qsc{};
    static void build_one(const std::array<uint32_t,A>& hist,uint64_t* rr,uint32_t* xx,uint32_t* ss){
        uint64_t total_mass=0; for(int s=0;s<A;++s) total_mass+=2ull*hist[s]+1ull;
        constexpr uint64_t rem=uint64_t(lx_rans4_fast::M)-A;
        uint64_t cum=0,prev_q=0; uint32_t c=0;
        for(int s=0;s<A;++s){
            cum+=2ull*hist[s]+1ull;
            const uint64_t qq=(cum*rem)/total_mass;
            const uint32_t f=1u+uint32_t(qq-prev_q); prev_q=qq;
            // Exact reciprocal on the post-renorm rANS domain x < f*2^20.
            // Q44 is the minimum universal fixed shift for every 1<=f<=4095:
            // ceil(2^44/f) cannot advance floor(x/f) because f^2 < 2^24.
            const uint64_t r=((uint64_t(1)<<44)+f-1u)/f;
            rr[s]=r; xx[s]=f<<20; ss[s]=c|((4096u-f)<<12); c+=f;
        }
    }
    CtxQ44SoAModels(){std::array<uint32_t,A> h{};h.fill(1);for(int c=0;c<C;++c){size_t o=size_t(c)*A;build_one(h,qrecip.data()+o,qxmax.data()+o,qsc.data()+o);}}
    void rebuild_encoder_only(const std::array<std::array<uint32_t,A>,C>&h){for(int c=0;c<C;++c){size_t o=size_t(c)*A;build_one(h[c],qrecip.data()+o,qxmax.data()+o,qsc.data()+o);}}
};

template<int A1,int C1,int A2,int C2>
static std::pair<lx_rans4_fast::Encoded,lx_rans4_fast::Encoded> encode4_q44_key_pair(
 const uint16_t*k1,size_t n1,const CtxQ44SoAModels<A1,C1>&m1,lx_rans4_fast::Scratch&o1,
 const uint16_t*k2,size_t n2,const CtxQ44SoAModels<A2,C2>&m2,lx_rans4_fast::Scratch&o2){
 using namespace lx_rans4_fast;uint8_t*b1=o1.buf.data(),*b2=o2.buf.data();uint16_t*w1=(uint16_t*)(b1+o1.buf.size()),*w2=(uint16_t*)(b2+o2.buf.size());uint32_t a0=L,a1=L,a2=L,a3=L,z0=L,z1=L,z2=L,z3=L;size_t i1=n1,i2=n2;
#define Q1(X,J) do{const uint32_t _k=k1[J];const uint64_t _r=m1.qrecip[_k];const uint32_t _xm=m1.qxmax[_k],_sc=m1.qsc[_k];uint32_t v=X;if(v>=_xm){*--w1=uint16_t(v);v>>=16;}const uint32_t q=uint32_t((uint64_t(v)*_r)>>44);X=v+(_sc&4095u)+q*(_sc>>12);}while(0)
#define Q2(X,J) do{const uint32_t _k=k2[J];const uint64_t _r=m2.qrecip[_k];const uint32_t _xm=m2.qxmax[_k],_sc=m2.qsc[_k];uint32_t v=X;if(v>=_xm){*--w2=uint16_t(v);v>>=16;}const uint32_t q=uint32_t((uint64_t(v)*_r)>>44);X=v+(_sc&4095u)+q*(_sc>>12);}while(0)
 while(i1&3u){size_t j=--i1;switch(j&3u){case 0:Q1(a0,j);break;case 1:Q1(a1,j);break;case 2:Q1(a2,j);break;default:Q1(a3,j);}}while(i2&3u){size_t j=--i2;switch(j&3u){case 0:Q2(z0,j);break;case 1:Q2(z1,j);break;case 2:Q2(z2,j);break;default:Q2(z3,j);}}
 while(i1&&i2){i1-=4;i2-=4;Q1(a3,i1+3);Q2(z3,i2+3);Q1(a2,i1+2);Q2(z2,i2+2);Q1(a1,i1+1);Q2(z1,i2+1);Q1(a0,i1);Q2(z0,i2);}while(i1){i1-=4;Q1(a3,i1+3);Q1(a2,i1+2);Q1(a1,i1+1);Q1(a0,i1);}while(i2){i2-=4;Q2(z3,i2+3);Q2(z2,i2+2);Q2(z1,i2+1);Q2(z0,i2);}
#undef Q1
#undef Q2
 uint8_t*p1=(uint8_t*)w1-16;memcpy(p1,&a0,4);memcpy(p1+4,&a1,4);memcpy(p1+8,&a2,4);memcpy(p1+12,&a3,4);uint8_t*p2=(uint8_t*)w2-16;memcpy(p2,&z0,4);memcpy(p2+4,&z1,4);memcpy(p2+8,&z2,4);memcpy(p2+12,&z3,4);return {{p1,size_t(b1+o1.buf.size()-p1),n1},{p2,size_t(b2+o2.buf.size()-p2),n2}};
}

#if defined(__AVX2__)
template<int A,int C>
static __attribute__((noinline)) lx_rans4_fast::Encoded encode4_ctx_simd(
        const uint8_t* sym,const uint8_t* ctx,size_t n,
        const CtxRansModels<A,C>& models,lx_rans4_fast::Scratch& scratch){
    using namespace lx_rans4_fast;
    uint8_t* const base=scratch.buf.data();
    uint8_t* p=base+scratch.buf.size();
    alignas(32) uint64_t xs[4]={L,L,L,L};
    size_t i=n;

    // Exact scalar tail preserves lane assignment.
    auto scalar_step=[&](size_t j){
        const unsigned lane=unsigned(j)&3u;
        uint32_t x=uint32_t(xs[lane]);
        const EncMeta& em=models.flat[size_t(ctx[j])*A+sym[j]];
        const uint32_t xmax=em.freq()<<20;
        if(x>=xmax){p-=2;const uint16_t w=uint16_t(x);std::memcpy(p,&w,2);x>>=16;}
        const uint32_t qq=uint32_t((uint64_t(x)*em.rcp)>>32)>>em.shift();
        xs[lane]=x+em.bias()+qq*em.cmpl();
    };
    while(i&3u)scalar_step(--i);

    __m256i X=_mm256_load_si256((const __m256i*)xs);
    const __m256i mask12=_mm256_set1_epi64x(4095);
    const __m256i mask13=_mm256_set1_epi64x(8191);
    const __m256i thirtytwo=_mm256_set1_epi64x(32);
    const __m256i MM=_mm256_set1_epi64x(M);

    while(i){
        i-=4;
        uint32_t sv,cv;
        std::memcpy(&sv,sym+i,4);std::memcpy(&cv,ctx+i,4);
        const __m128i si=_mm_cvtepu8_epi32(_mm_cvtsi32_si128(int(sv)));
        const __m128i ci=_mm_cvtepu8_epi32(_mm_cvtsi32_si128(int(cv)));
        const __m128i ids=_mm_add_epi32(si,_mm_mullo_epi32(ci,_mm_set1_epi32(A)));
        const __m256i z=_mm256_i32gather_epi64((const long long*)models.flat.data(),ids,8);

        const __m256i rcp=_mm256_and_si256(z,_mm256_set1_epi64x(0xffffffffull));
        const __m256i pp=_mm256_srli_epi64(z,32);
        const __m256i bias=_mm256_and_si256(pp,mask13);
        const __m256i cmpl=_mm256_and_si256(_mm256_srli_epi64(pp,13),mask12);
        const __m256i freq=_mm256_sub_epi64(MM,cmpl);
        const __m256i xmax=_mm256_slli_epi64(freq,20);
        const __m256i needv=_mm256_cmpgt_epi64(X,_mm256_sub_epi64(xmax,_mm256_set1_epi64x(1)));
        const unsigned need=unsigned(_mm256_movemask_pd(_mm256_castsi256_pd(needv)))&15u;

        // Four independent rANS states. Compress exactly the renormalizing low
        // words in lane order; the fixed 8-byte store only touches still-free
        // backward-buffer space and is overwritten by later groups as needed.
        const unsigned cnt=__builtin_popcount(need);
        if(cnt){
            // Four lanes only: scalar compaction of the renormalization words
            // avoids the AVX-512 VBMI2 dependency while the expensive rANS
            // state update remains fully AVX2.
            alignas(32) uint64_t lane_x[4];
            _mm256_store_si256((__m256i*)lane_x,X);
            uint16_t words[4]; unsigned w=0;
            for(unsigned l=0;l<4;++l) if(need&(1u<<l)) words[w++]=uint16_t(lane_x[l]);
            p-=2*cnt;
            std::memcpy(p,words,2*cnt);
        }

        const __m256i shifts=_mm256_and_si256(needv,_mm256_set1_epi64x(16));
        const __m256i XR=_mm256_srlv_epi64(X,shifts);
        const __m256i sh=_mm256_add_epi64(_mm256_srli_epi64(pp,25),thirtytwo);
        const __m256i prod=_mm256_mul_epu32(XR,rcp);
        __m256i q=_mm256_srlv_epi64(prod,sh);
        // Reciprocal quotient is at most one away from floor(X/f). Correct it
        // branchlessly so the SIMD path is bit-identical to canonical scalar rANS.
        __m256i qf=_mm256_mul_epu32(q,freq);
        const __m256i too_hi=_mm256_cmpgt_epi64(qf,XR);
        q=_mm256_sub_epi64(q,_mm256_and_si256(too_hi,_mm256_set1_epi64x(1)));
        qf=_mm256_mul_epu32(q,freq);
        const __m256i rem0=_mm256_sub_epi64(XR,qf);
        const __m256i need_up=_mm256_cmpgt_epi64(rem0,_mm256_sub_epi64(freq,_mm256_set1_epi64x(1)));
        q=_mm256_add_epi64(q,_mm256_and_si256(need_up,_mm256_set1_epi64x(1)));
        qf=_mm256_mul_epu32(q,freq);
        const __m256i rem=_mm256_sub_epi64(XR,qf);
        const __m256i is_f1=_mm256_cmpeq_epi64(freq,_mm256_set1_epi64x(1));
        const __m256i start=_mm256_sub_epi64(bias,_mm256_and_si256(is_f1,_mm256_set1_epi64x(M-1)));
        X=_mm256_add_epi64(_mm256_add_epi64(_mm256_slli_epi64(q,SCALE_BITS),rem),start);
    }

    _mm256_store_si256((__m256i*)xs,X);
    p-=16;
    for(int l=0;l<4;++l){const uint32_t x=uint32_t(xs[l]);std::memcpy(p+4*l,&x,4);}
    return {p,size_t(base+scratch.buf.size()-p),n};
}
#endif


#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512VBMI2__)
template<int A,int C>
static __attribute__((noinline)) lx_rans4_fast::Encoded encode8_ctx_avx512(
        const uint8_t* sym,const uint8_t* ctx,size_t n,
        const CtxRansModels<A,C>& models,lx_rans4_fast::Scratch& scratch){
    using namespace lx_rans4_fast;
    uint8_t* const base=scratch.buf.data();
    uint8_t* p=base+scratch.buf.size();
    alignas(64) uint64_t xs[8]={L,L,L,L,L,L,L,L};
    size_t i=n;

    auto scalar_step=[&](size_t j){
        const unsigned lane=unsigned(j)&7u;
        uint32_t x=uint32_t(xs[lane]);
        const EncMeta& em=models.flat[size_t(ctx[j])*A+sym[j]];
        const uint32_t xmax=em.freq()<<20;
        if(x>=xmax){p-=2;const uint16_t w=uint16_t(x);std::memcpy(p,&w,2);x>>=16;}
        const uint32_t qq=uint32_t((uint64_t(x)*em.rcp)>>32)>>em.shift();
        xs[lane]=x+em.bias()+qq*em.cmpl();
    };
    while(i&7u)scalar_step(--i);

    __m512i X=_mm512_load_si512((const __m512i*)xs);
    const __m512i mask12=_mm512_set1_epi64(4095);
    const __m512i mask13=_mm512_set1_epi64(8191);
    const __m512i thirtytwo=_mm512_set1_epi64(32);
    const __m512i MM=_mm512_set1_epi64(M);

    while(i){
        i-=8;
        uint64_t sv,cv;
        std::memcpy(&sv,sym+i,8);std::memcpy(&cv,ctx+i,8);
        const __m256i si=_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)sv));
        const __m256i ci=_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)cv));
        const __m256i ids32=_mm256_add_epi32(si,_mm256_mullo_epi32(ci,_mm256_set1_epi32(A)));
        const __m512i ids64=_mm512_cvtepi32_epi64(ids32);
        const __m512i z=_mm512_i64gather_epi64(ids64,(const void*)models.flat.data(),8);

        const __m512i rcp=_mm512_and_si512(z,_mm512_set1_epi64(0xffffffffull));
        const __m512i pp=_mm512_srli_epi64(z,32);
        const __m512i bias=_mm512_and_si512(pp,mask13);
        const __m512i cmpl=_mm512_and_si512(_mm512_srli_epi64(pp,13),mask12);
        const __m512i freq=_mm512_sub_epi64(MM,cmpl);
        const __m512i xmax=_mm512_slli_epi64(freq,20);
        const __mmask8 need=_mm512_cmp_epu64_mask(X,xmax,_MM_CMPINT_GE);
        const unsigned cnt=__builtin_popcount(unsigned(need));
        if(cnt){
            const __m128i words=_mm512_cvtepi64_epi16(X);
            const __m128i packed=_mm_maskz_compress_epi16(need,words);
            p-=2*cnt;
            std::memcpy(p,&packed,2*cnt);
        }

        const __m512i XR=_mm512_mask_srli_epi64(X,need,X,16);
        const __m512i sh=_mm512_add_epi64(_mm512_srli_epi64(pp,25),thirtytwo);
        const __m512i prod=_mm512_mul_epu32(XR,rcp);
        const __m512i q=_mm512_srlv_epi64(prod,sh);
        const __m512i qc=_mm512_mul_epu32(q,cmpl);
        X=_mm512_add_epi64(_mm512_add_epi64(XR,bias),qc);
    }

    _mm512_store_si512((__m512i*)xs,X);
    p-=32;
    for(int l=0;l<8;++l){const uint32_t x=uint32_t(xs[l]);std::memcpy(p+4*l,&x,4);}
    return {p,size_t(base+scratch.buf.size()-p),n};
}
#endif


#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512VBMI2__)
template<int A,int C>
static __attribute__((noinline)) lx_rans4_fast::Encoded encode16_ctx_avx512(
        const uint8_t* sym,const uint8_t* ctx,size_t n,
        const CtxRansModels<A,C>& models,lx_rans4_fast::Scratch& scratch){
    using namespace lx_rans4_fast;
    uint8_t* const base=scratch.buf.data();
    uint8_t* p=base+scratch.buf.size();
    alignas(64) uint64_t xs[16];
    for(auto& x:xs)x=L;
    size_t i=n;

    auto scalar_step=[&](size_t j){
        const unsigned lane=unsigned(j)&15u;
        uint32_t x=uint32_t(xs[lane]);
        const EncMeta& em=models.flat[size_t(ctx[j])*A+sym[j]];
        const uint32_t xmax=em.freq()<<20;
        if(x>=xmax){p-=2;const uint16_t w=uint16_t(x);std::memcpy(p,&w,2);x>>=16;}
        const uint32_t qq=uint32_t((uint64_t(x)*em.rcp)>>32)>>em.shift();
        xs[lane]=x+em.bias()+qq*em.cmpl();
    };
    while(i&15u)scalar_step(--i);

    __m512i X0=_mm512_load_si512((const __m512i*)(xs+0));
    __m512i X1=_mm512_load_si512((const __m512i*)(xs+8));
    const __m512i mask12=_mm512_set1_epi64(4095),mask13=_mm512_set1_epi64(8191);
    const __m512i thirtytwo=_mm512_set1_epi64(32),MM=_mm512_set1_epi64(M);

    while(i){
        i-=16;
        __m128i sb=_mm_loadu_si128((const __m128i*)(sym+i));
        __m128i cb=_mm_loadu_si128((const __m128i*)(ctx+i));
        const __m256i s0=_mm256_cvtepu8_epi32(sb);
        const __m256i c0=_mm256_cvtepu8_epi32(cb);
        const __m128i sbh=_mm_srli_si128(sb,8),cbh=_mm_srli_si128(cb,8);
        const __m256i s1=_mm256_cvtepu8_epi32(sbh);
        const __m256i c1=_mm256_cvtepu8_epi32(cbh);
        const __m256i id0=_mm256_add_epi32(s0,_mm256_mullo_epi32(c0,_mm256_set1_epi32(A)));
        const __m256i id1=_mm256_add_epi32(s1,_mm256_mullo_epi32(c1,_mm256_set1_epi32(A)));
        const __m512i z0=_mm512_i64gather_epi64(_mm512_cvtepi32_epi64(id0),(const void*)models.flat.data(),8);
        const __m512i z1=_mm512_i64gather_epi64(_mm512_cvtepi32_epi64(id1),(const void*)models.flat.data(),8);

        auto prep=[&](const __m512i Z,const __m512i X,
                      __m512i& rcp,__m512i& bias,__m512i& cmpl,__m512i& pp,__mmask8& need){
            rcp=_mm512_and_si512(Z,_mm512_set1_epi64(0xffffffffull));
            pp=_mm512_srli_epi64(Z,32);
            bias=_mm512_and_si512(pp,mask13);
            cmpl=_mm512_and_si512(_mm512_srli_epi64(pp,13),mask12);
            const __m512i freq=_mm512_sub_epi64(MM,cmpl);
            need=_mm512_cmp_epu64_mask(X,_mm512_slli_epi64(freq,20),_MM_CMPINT_GE);
        };
        __m512i r0,b0,m0,p0,r1,b1,m1,p1;__mmask8 n0,n1;
        prep(z0,X0,r0,b0,m0,p0,n0);prep(z1,X1,r1,b1,m1,p1,n1);

        const unsigned c0n=__builtin_popcount(unsigned(n0));
        const unsigned c1n=__builtin_popcount(unsigned(n1));
        const unsigned total=c0n+c1n;
        if(total){
            const __m128i w0=_mm_maskz_compress_epi16(n0,_mm512_cvtepi64_epi16(X0));
            const __m128i w1=_mm_maskz_compress_epi16(n1,_mm512_cvtepi64_epi16(X1));
            p-=2*total;
            std::memcpy(p,&w0,2*c0n);
            std::memcpy(p+2*c0n,&w1,2*c1n);
        }

        auto step=[&](__m512i X,const __m512i rcp,const __m512i bias,const __m512i cmpl,const __m512i pp,const __mmask8 need){
            const __m512i XR=_mm512_mask_srli_epi64(X,need,X,16);
            const __m512i sh=_mm512_add_epi64(_mm512_srli_epi64(pp,25),thirtytwo);
            const __m512i prod=_mm512_mul_epu32(XR,rcp);
            const __m512i q=_mm512_srlv_epi64(prod,sh);
            return _mm512_add_epi64(_mm512_add_epi64(XR,bias),_mm512_mul_epu32(q,cmpl));
        };
        X0=step(X0,r0,b0,m0,p0,n0);
        X1=step(X1,r1,b1,m1,p1,n1);
    }

    _mm512_store_si512((__m512i*)(xs+0),X0);
    _mm512_store_si512((__m512i*)(xs+8),X1);
    p-=64;
    for(int l=0;l<16;++l){const uint32_t x=uint32_t(xs[l]);std::memcpy(p+4*l,&x,4);}
    return {p,size_t(base+scratch.buf.size()-p),n};
}
#endif

static inline int scale_predict(const std::vector<uint16_t>& s,size_t f,int b){
    if(f==0 && b==0)return 0;
    if(f==0)return int(s[size_t(b-1)]);
    if(b==0)return int(s[(f-1)*ERB_BANDS]);
    const int left=int(s[f*ERB_BANDS+size_t(b-1)]);
    const int top =int(s[(f-1)*ERB_BANDS+size_t(b)]);
    const int tl  =int(s[(f-1)*ERB_BANDS+size_t(b-1)]);
    // JPEG-LS median edge predictor: median(left,top,left+top-tl).
    if(tl>=std::max(left,top))return std::min(left,top);
    if(tl<=std::min(left,top))return std::max(left,top);
    return left+top-tl;
}

static inline int lr_scale_predict(const std::vector<int16_t>& s,size_t f,int b){
    if(f==0 && b==0)return 0;
    if(f==0)return int(s[size_t(b-1)]);
    if(b==0)return int(s[(f-1)*ERB_BANDS]);
    const int left=int(s[f*ERB_BANDS+size_t(b-1)]);
    const int top =int(s[(f-1)*ERB_BANDS+size_t(b)]);
    const int tl  =int(s[(f-1)*ERB_BANDS+size_t(b-1)]);
    if(tl>=std::max(left,top))return std::min(left,top);
    if(tl<=std::min(left,top))return std::max(left,top);
    return left+top-tl;
}

static inline uint32_t zigzag_i32(int32_t x){
    return x>=0 ? uint32_t(x)<<1 : (uint32_t(-int64_t(x))<<1)-1u;
}
constexpr float INV_SQRT2=0.7071067811865475244f;
enum class StereoMode:uint8_t{LR=0,MS=1};

// Energy-preserving high-band Intensity Stereo.  Bands 40..47 are eligible.
// The shared shape is radial, |S_i|=sqrt(L_i^2+R_i^2), so stereo reduction
// cannot erase spectral energy through L/R projection cancellation.  One
// 6-bit L/R energy angle is shared by each adjacent 2-ERB pair.  Activation
// is mask-weighted RD (metric 2), not a hard signal threshold.  No
// signal-dependent branch is introduced in coefficient hot loops.
constexpr int IS_MAX_BANDS = 8;
constexpr int IS_FIRST_BAND = ERB_BANDS - IS_MAX_BANDS;
constexpr int IS_ANGLE_BITS = 2; // legacy syntax default; optional path is runtime-configurable
constexpr float LX_PI = 3.14159265358979323846f;

// research Intensity Stereo controls.  These are process-constant research
// parameters: no signal-dependent branches are introduced in coefficient hot loops.
static inline int is_style_cfg(){return 10;}
static inline int is_metric_cfg(){
    static const int v=[](){const char*p=::lxvoxel_release_env("LX_IS_METRIC");return std::clamp(p?std::atoi(p):24,1,26);}();return v;
}
static inline int is_bits_cfg(){return 6;}
static inline int is_code_count_cfg(){
    const int st=is_style_cfg(); return (st==2||st==4||st==6)?1:((st==9||st==10)?(IS_MAX_BANDS/2):IS_MAX_BANDS);
}
static inline uint8_t quantize_is_angle_bits(float theta,int bits){
    if(bits==0) return 0; // zero-sideinfo profile: fixed Mid axis
    if(bits==1) return uint8_t(theta>=0.0f); // {-45,+45} degrees
    const int L=1<<bits;
    // Nested projective grid, anchored on the canonical L/M/S axes.
    // bits=2: {-45,0,+45,+90}; each added bit bisects the angular cells.
    const float u=theta*(float(L)/LX_PI)+float(L/2-1);
    int q=int(std::floor(u+0.5f)); q%=L; if(q<0)q+=L; return uint8_t(q);
}
static inline float decode_is_angle_bits(uint8_t q,int bits){
    if(bits==0) return 0.25f*LX_PI;
    if(bits==1) return q ? 0.25f*LX_PI : -0.25f*LX_PI;
    const int L=1<<bits;
    return (float(int(q)-(L/2-1)))*(LX_PI/float(L));
}
static inline uint32_t pack_is_angle_byte(const std::array<uint8_t,ERB_BANDS>&q){
    const int st=is_style_cfg(), bits=is_bits_cfg();
    if(bits==0)return 0u;
    const uint32_t mask=(uint32_t(1)<<bits)-1u; uint32_t v=0;
    if(st==2||st==4||st==6) return uint32_t(q[IS_FIRST_BAND])&mask;
    if(st==9||st==10){
        for(int g=0;g<IS_MAX_BANDS/2;++g) v|=(uint32_t(q[IS_FIRST_BAND+2*g])&mask)<<(bits*g);
        return v;
    }
    for(int j=0;j<IS_MAX_BANDS;++j) v|=(uint32_t(q[IS_FIRST_BAND+j])&mask)<<(bits*j);
    return v;
}
static const std::array<float,1024> LX_SHADOW_LOG2_MANT=[](){
    std::array<float,1024> t{};
    for(size_t i=0;i<t.size();++i){
        const double x=1.0+(double(i)+0.5)/1024.0;
        t[i]=float(std::log2(x));
    }
    return t;
}();

static inline double shadow_signed_integer_bits(double q) noexcept{
    const uint32_t m=uint32_t(std::llabs((long long)q));
    const uint32_t x=1u+(m<<1);
    const float xf=float(x);
    const uint32_t u=std::bit_cast<uint32_t>(xf);
    const uint32_t e=(u>>23)-127u;
    const uint32_t idx=(u>>13)&1023u;
    return double(e)+double(LX_SHADOW_LOG2_MANT[idx]);
}

static inline double qlap_entropy_pc(double l1,int dim,double delta){
    if(dim<=0 || !(l1>0.0) || !(delta>0.0)) return 0.0;
    const double bs=l1/double(dim), d=delta/bs;
    if(d>80.0) return 0.0;
    const double r=std::exp(-d), q=std::exp(-0.5*d);
    const double p0=1.0-q, B=0.5*(1.0-r)*std::exp(0.5*d), S=0.5*q;
    const double T=S/std::max(1.0-r,std::numeric_limits<double>::min());
    double H=0.0;
    if(p0>0.0) H-=p0*std::log2(p0);
    if(S>0.0 && B>0.0 && r>0.0) H-=2.0*(S*std::log2(B)+T*std::log2(r));
    return std::max(H,0.0);
}


// SFC transport: bits [5:3] of the control byte carry active-band cardinality k.
// The mask is coded by its combinatorial rank among C(8,k) subsets, so payload
// length is derived exactly from k and gain precision; bits [7:6] remain reserved.
struct SfcFrameDecision {
    uint16_t nbits=0;
    uint8_t mask=0;
    std::array<uint8_t,lx_sfc::SFC_BANDS> gain{};
    std::array<uint8_t,lx_sfc::SFC_BANDS> ar_k1{};
    std::array<uint8_t,lx_sfc::SFC_BANDS> ar_k2{};
    std::array<uint8_t,lx_sfc::SFC_BANDS> tilt{};
    std::array<uint8_t,lx_sfc::SFC_BANDS> mode{}; // v0.15: 0=pSFC prediction, 1=fSFC folding from b-1
    std::array<uint8_t,lx_sfc::SFC_BANDS> ifc_on{};
    std::array<uint8_t,lx_sfc::SFC_BANDS> ifc_mode{};
    std::array<uint8_t,lx_sfc::SFC_BANDS> ifc_pos{};
    std::array<uint8_t,lx_sfc::SFC_BANDS> ifc_amp{};
};

struct CpmFrameDecision {
    uint8_t on=0, anchor=0, band=0, pos=0;
    uint16_t signs=0;
    uint16_t nbits=0;
};
struct CpmTrackState {
    bool valid=false; uint8_t band=0,pos=0; size_t frame=0; float freq_hz=0.f;
    float al=0.f,bl=0.f,ar=0.f,br=0.f; // global cos/sin amplitudes in physical L/R space
};
static inline double cpm_int_rate(int32_t q){
    return std::log2(1.0 + 2.0*std::fabs(double(q)));
}

static inline uint32_t sfc_rank_mask(uint8_t mask){
    const int k=std::popcount(unsigned(mask)); uint32_t r=0;
    for(unsigned m=1;m<unsigned(mask);++m) r += unsigned(std::popcount(m)==k);
    return r;
}
static inline uint8_t quantize_is_angle(float theta){return quantize_is_angle_bits(theta,is_bits_cfg());}

struct IntensityAxisF { float cosine, sine; };
struct IntensityAxisD { double cosine, sine; };

// The transmitted intensity angle is a finite codebook value.  Decode its
// trigonometric axis once per process instead of evaluating sin/cos repeatedly
// in quantization and reconstruction hot paths.
static inline IntensityAxisF intensity_axis_f(uint8_t code){
    static const auto table=[](){
        std::array<IntensityAxisF,256> t{};
        for(unsigned q=0;q<t.size();++q){
            const float angle=decode_is_angle_bits(uint8_t(q),is_bits_cfg());
            t[q]={std::cos(angle),std::sin(angle)};
        }
        return t;
    }();
    return table[code];
}
static inline int is_shape_mode_cfg(){
    static const int v=[](){const char* p=::lxvoxel_release_env("LX_IS_SHAPE_MODE");return p?std::atoi(p):1;}();
    return v;
}
static inline float intensity_shape_sample(float L,float R,uint8_t code,float sa,float sb,float delta){
    const int m=is_shape_mode_cfg();
    if(m==0) return sa*L+sb*R;
    const auto ax=intensity_axis_f(code);
    const float p=ax.cosine*L+ax.sine*R;
    const float e=L*L+R*R;
    if(m==1) return std::copysign(std::sqrt(std::max(e,0.0f)),p==0.0f?1.0f:p);
    if(m==2) return sa*L+sb*R; // normalized merge uses style 11 weights
    if(m==3){
        const float r2=std::max(0.0f,e-p*p);
        const float q2=delta*delta*(1.0f/12.0f);
        const float alpha=r2/(r2+q2+std::numeric_limits<float>::min());
        return std::copysign(std::sqrt(std::max(0.0f,p*p+alpha*r2)),p==0.0f?1.0f:p);
    }
    return sa*L+sb*R;
}

static inline IntensityAxisD intensity_axis_d(uint8_t code){
    static const auto table=[](){
        std::array<IntensityAxisD,256> t{};
        for(unsigned q=0;q<t.size();++q){
            const double angle=double(decode_is_angle_bits(uint8_t(q),is_bits_cfg()));
            t[q]={std::cos(angle),std::sin(angle)};
        }
        return t;
    }();
    return table[code];
}

#if defined(__linux__)
#else
#endif


static inline void deinterleave_local(const float* src,size_t n,float* L,float* R){
    size_t i=0;
#if defined(__AVX2__)
    for(;i+8<=n;i+=8){
        const __m256 a=_mm256_loadu_ps(src+2*i);
        const __m256 b=_mm256_loadu_ps(src+2*i+8);
        const __m256 l0=_mm256_shuffle_ps(a,b,_MM_SHUFFLE(2,0,2,0));
        const __m256 r0=_mm256_shuffle_ps(a,b,_MM_SHUFFLE(3,1,3,1));
        _mm256_storeu_ps(L+i,_mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(l0),_MM_SHUFFLE(3,1,2,0))));
        _mm256_storeu_ps(R+i,_mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(r0),_MM_SHUFFLE(3,1,2,0))));
    }
#elif defined(__SSE4_1__) || defined(LXVOXEL_SSE41)
    for(;i+4<=n;i+=4){
        const __m128 a=_mm_loadu_ps(src+2*i);
        const __m128 b=_mm_loadu_ps(src+2*i+4);
        _mm_storeu_ps(L+i,_mm_shuffle_ps(a,b,_MM_SHUFFLE(2,0,2,0)));
        _mm_storeu_ps(R+i,_mm_shuffle_ps(a,b,_MM_SHUFFLE(3,1,3,1)));
    }
#elif defined(LXVOXEL_NEON) || defined(__ARM_NEON) || defined(__ARM_NEON__)
    for(;i+4<=n;i+=4){
        const float32x4x2_t lr=vld2q_f32(src+2*i);
        vst1q_f32(L+i,lr.val[0]);
        vst1q_f32(R+i,lr.val[1]);
    }
#endif
    for(;i<n;++i){L[i]=src[2*i];R[i]=src[2*i+1];}
}


// ERB Group Vectors (EGV)
// -----------------------
// 2048/1024/512/256 blocks use 16/8/4/2 contiguous ERB groups.
// The intermediate tables form a dyadic hierarchy in transform duration.
//
// Groups terminate only on the canonical 48-band boundaries.  The partition is
// fixed bitstream geometry, not a signal-dependent decision.
//
// The EGV itself is one stereo D_n vector.  Until FEAR-delta shaping is enabled,
// its dyadic scale is the dimension-weighted geometric-volume equivalent of the
// old per-ERB scales:
//
//   bpc_EGV = sum_b n_b*bpc_b / sum_b n_b
//
// This preserves the first-order log cell volume represented by the old cells
// while replacing their Cartesian product by one higher-dimensional D_n cell.
struct EgvRange { uint8_t b0, b1; }; // [b0,b1)

inline constexpr std::array<EgvRange,16> EGV_LONG = {{
    { 0,19}, {19,26}, {26,30}, {30,33},
    {33,35}, {35,37}, {37,39}, {39,40},
    {40,41}, {41,42}, {42,43}, {43,44},
    {44,45}, {45,46}, {46,47}, {47,48}
}};

inline constexpr std::array<EgvRange,8> EGV_1024 = {{
    { 0,26}, {26,33}, {33,37}, {37,40},
    {40,42}, {42,44}, {44,46}, {46,48}
}};

inline constexpr std::array<EgvRange,4> EGV_SHORT = {{
    {0,34}, {34,41}, {41,45}, {45,48}
}};

inline constexpr std::array<EgvRange,2> EGV_256 = {{
    {0,30}, {30,48}
}};


// 0.376287 dB FEAR-delta grid: 16 fractional steps per octave.
// Integer octave scaling remains implicit in q via powers of two; only the
// 4-bit fractional phase r must be carried per ERB for decoder reconstruction.
inline constexpr std::array<float,16> LX_DELTA_FRAC16 = {{
    1.000000000f, 1.044273782f, 1.090507733f, 1.138788635f,
    1.189207115f, 1.241857812f, 1.296839555f, 1.354255547f,
    1.414213562f, 1.476826146f, 1.542210825f, 1.610490332f,
    1.681792831f, 1.756252160f, 1.834008086f, 1.915206561f
}};
static_assert(sizeof(LX_DELTA_FRAC16)==64);

// Energy-aware D_n parity repair.  The energy term is expressed as radial
// Euclidean distance, (sqrt(E_src)-sqrt(E_q))^2, so it has the same physical
// units as coefficient MSE and needs no empirical weighting constant.  A
// second radial term preserves the source frame-to-frame amplitude change.
static inline double lx_energy_repair_increment(double mse_inc,
        double src_e,double q_e,double q_e_after,
        float prev_src_amp,float prev_q_amp,bool have_prev) noexcept {
    const double sa=std::sqrt(std::max(src_e,0.0));
    const double qa=std::sqrt(std::max(q_e,0.0));
    const double qan=std::sqrt(std::max(q_e_after,0.0));
    const double r0=(sa-qa)*(sa-qa);
    const double r1=(sa-qan)*(sa-qan);
    double dt=0.0;
    if(have_prev){
        const double target_delta=sa-double(prev_src_amp);
        const double d0=target_delta-(qa-double(prev_q_amp));
        const double d1=target_delta-(qan-double(prev_q_amp));
        dt=d1*d1-d0*d0;
    }
    return mse_inc+(r1-r0)+dt;
}


// Exact parity-preserving nearest-shell refinement for one ERB.
//
// Product implementation notes:
//   * no heap allocation in the per-band path;
//   * exact grouping by the D_n shell increment h=(q+s)^2-q^2;
//   * fixed-size open-address table, O(N), replacing heap vectors/std::sort;
//   * only the two least-MSE distinct coordinates are retained per h because
//     the decoder-synchronous radial objective depends on a move solely through
//     h (q2), while q1/nz changes are uniquely determined by h for +/-1 moves.
//
// A D_n lattice remains in D_n when two distinct coordinates are changed by
// +/-1.  The search therefore evaluates the complete nearest-neighbour shell
// graph while preserving the bitstream and decoder exactly.
struct LxShellMove {
    float dmse=0.0f;
    uint16_t bin=0;
    uint16_t idx=0;
    uint8_t ch=0;
    int8_t dir=0;
};
struct LxShellCandidate {
    LxShellMove move;
    int32_t h;
};
struct LxShellGroup {
    int32_t h=0;
    LxShellMove a{},b{};
    uint8_t n=0;
};

// Exact per-band capacity from the product ERB geometry.  The widest refined
// ERB in the 1024-bin transform is 76 bins; full stereo therefore contributes
// at most 152 coordinates and two +/-1 move keys per coordinate.
static constexpr int LX_SHELL_MAX_BAND_BINS = 76;
static constexpr int LX_SHELL_MAX_COORD = 2*LX_SHELL_MAX_BAND_BINS;
static constexpr int LX_SHELL_MAX_GROUP = 2*LX_SHELL_MAX_COORD; // <=304 distinct h keys
static constexpr int LX_SHELL_HASH_SIZE = 512;                  // exact capacity, load <=0.594
static_assert((LX_SHELL_HASH_SIZE&(LX_SHELL_HASH_SIZE-1))==0);
static_assert(LX_SHELL_MAX_GROUP < LX_SHELL_HASH_SIZE);

static inline void lx_shell_insert_two(LxShellGroup& g,const LxShellMove& m) noexcept {
    if(g.n==0){g.a=m;g.n=1;return;}
    if(m.idx==g.a.idx){if(m.dmse<g.a.dmse)g.a=m;return;}
    if(g.n==1){
        if(m.dmse<g.a.dmse){g.b=g.a;g.a=m;}else g.b=m;
        g.n=2;return;
    }
    if(m.idx==g.b.idx){
        if(m.dmse<g.b.dmse){g.b=m;if(g.b.dmse<g.a.dmse)std::swap(g.a,g.b);}return;
    }
    if(m.dmse<g.a.dmse){g.b=g.a;g.a=m;}
    else if(m.dmse<g.b.dmse)g.b=m;
}

struct LxShellScratch {
    alignas(64) LxShellGroup groups[LX_SHELL_MAX_GROUP];
    alignas(64) LxShellCandidate candidates[2*LX_SHELL_MAX_COORD+1];
    alignas(64) int32_t keys[LX_SHELL_HASH_SIZE];
    alignas(64) uint16_t group_index[LX_SHELL_HASH_SIZE];
    alignas(64) uint16_t epoch[LX_SHELL_HASH_SIZE]{};
    uint16_t current_epoch=0;
};

static inline uint32_t lx_shell_hash(int32_t h) noexcept {
    // Shell increments h=1+2*q are always odd. Removing their fixed low
    // bit is an exact injection modulo the power-of-two table size.
    // Probing still resolves every collision, including large coefficients.
    return (uint32_t(h)>>1)&(LX_SHELL_HASH_SIZE-1);
}

static inline LxShellGroup& lx_shell_group_for(LxShellScratch& s,int32_t h,int& ng) noexcept {
    uint32_t p=lx_shell_hash(h);
    for(;;){
        if(s.epoch[p]!=s.current_epoch){
            s.epoch[p]=s.current_epoch; s.keys[p]=h; s.group_index[p]=uint16_t(ng);
            LxShellGroup& g=s.groups[ng++]; g=LxShellGroup{}; g.h=h; return g;
        }
        if(s.keys[p]==h) return s.groups[s.group_index[p]];
        p=(p+1)&(LX_SHELL_HASH_SIZE-1);
    }
}

template<class Layout>
static inline double lx_refine_dn_shell_pairs(
        const float* __restrict l,const float* __restrict r,bool ms,
        const Layout& erb,int is_start,const uint8_t* __restrict angle_code,
        const float* __restrict src_l,const float* __restrict src_r,
        const float* __restrict band_delta,int32_t* __restrict qA,int32_t* __restrict qB,
        float* __restrict prev_src_amp,float* __restrict prev_q_amp,uint8_t* __restrict prev_valid,
        int refine_b0=0,int refine_b1=46){
    double total_dmse=0.0;
    // Thread-local scratch removes allocator traffic while remaining safe if
    // encoders are run concurrently on different threads.
    thread_local LxShellScratch scratch;

    // R195: keep D_n shell refinement through ERB edge 46 (~16.55 kHz).
    // R194 stopped at SFC_FIRST_BAND=40 (~9.36 kHz), leaving the upper
    // quantized spectrum without the same decoder-synchronous radial/temporal
    // point refinement. Bands 46..47 (16.55..20 kHz) retain the original path;
    // extending the same temporal objective there was not uniformly beneficial.
    constexpr int LX_SHELL_REFINE_END_BAND = 46;
    static_assert(LX_SHELL_REFINE_END_BAND <= ERB_BANDS);
    // R334 encoder default: calibrated energy + temporal fidelity in ERB 38..43.
    // Measured per-band RD refinement weights. No new transmitted gain or bits.
    // All other bands retain the R333 objective unchanged. The exact weights
    // were checked at 128/160/192/256 kbit/s and on full-song ZIM windows.
    constexpr int HF_REFINE_FIRST_BAND = 38;
    constexpr int HF_REFINE_END_BAND = 44;
    constexpr double HF_REFINE_ENERGY_WEIGHT = 2.0;
    constexpr double HF_REFINE_TEMPORAL_WEIGHT = 2.0;
    const int refine_end=std::min(refine_b1,LX_SHELL_REFINE_END_BAND);
    for(int band=std::max(0,refine_b0);band<refine_end;++band){
        const int lo=erb.edge[band], hi=erb.edge[band+1];
        if(hi-lo<=0) continue;
        const double hf_mask=double(band>=HF_REFINE_FIRST_BAND && band<HF_REFINE_END_BAND);
        const double energy_w=1.0+(HF_REFINE_ENERGY_WEIGHT-1.0)*hf_mask;
        const double temporal_w=1.0+(HF_REFINE_TEMPORAL_WEIGHT-1.0)*hf_mask;
        const double d=double(band_delta[band]), d2=d*d;
        double src_e=0.0,q2=0.0,q1=0.0,nz=0.0;
        int coord=0,ng=0;
        float min_move=std::numeric_limits<float>::infinity();
        float second_move=std::numeric_limits<float>::infinity();
        // First pass: decoder-synchronous moments in the original accumulation
        // order, plus a cheap lower bound for every +/-1 move. No hash traffic.
        auto accumulate_coord=[&](float x,int32_t* qp) {
            const int32_t q=*qp;
            const double xd=double(x), qd=double(q), e0=xd-d*qd;
            src_e+=xd*xd; q2+=qd*qd; q1+=std::abs(qd); nz+=double(q!=0);
            const float m0=float(d2+2.0*d*e0);
            const float m1=float(d2-2.0*d*e0);
            // Cheapest +/-1 move per coordinate, then two cheapest distinct
            // coordinates. No conditionals on the signal in the coefficient loop.
            const float c=std::min(m0,m1);
            second_move=std::min(second_move,std::max(min_move,c));
            min_move=std::min(min_move,c);
            ++coord;
        };
        if(band>=is_start){
            for(int i=lo;i<hi;++i){
                const float z=intensity_shape_sample(l[i],r[i],angle_code[band],src_l[band],src_r[band],float(d));
                src_e+=double(l[i])*l[i]+double(r[i])*r[i]-double(z)*z;
                accumulate_coord(z,qA+i);
            }
        }else{
            for(int i=lo;i<hi;++i){
                const float L=l[i],R=r[i];
                const float a=ms?(L+R)*INV_SQRT2:L;
                const float b=ms?(L-R)*INV_SQRT2:R;
                accumulate_coord(a,qA+i); accumulate_coord(b,qB+i);
            }
        }

        if(coord<2){
            prev_src_amp[band]=float(std::sqrt(std::max(src_e,0.0)));
            const float gg=lx_conditional_gain::conditional_dn_gain_from_moments(q1,q2,nz,uint32_t(coord));
            prev_q_amp[band]=float(d*std::sqrt(std::max(q2,0.0))*double(gg));
            prev_valid[band]=1; continue;
        }

        const double sa=std::sqrt(std::max(src_e,0.0));
        const double ep0=lx_conditional_gain::conditional_dn_expected_q2_from_moments(q1,q2,nz,uint32_t(coord));
        const double qa=d*std::sqrt(ep0);
        // For a +/-1 pair, total dq1 is exactly {-2,0,+2} and total dnz is
        // {-2..+2}.  Conditional reconstruction energy is affine in q2 once
        // q1/nz are fixed, so factor the expensive log-based geometry out of
        // the O(G^2) pair search.  The hot candidate path becomes add+sqrt.
        double ep_offset[3][5]{};
        ep_offset[1][2]=ep0-q2;
        uint16_t ep_valid=uint16_t(1u<<(1*5+2));
        auto get_ep_offset=[&](int iq,int iz) noexcept -> double {
            const uint16_t bit=uint16_t(1u<<(iq*5+iz));
            if((ep_valid&bit)==0){
                const double dq1=2.0*double(iq-1);
                const double dnz=double(iz-2);
                ep_offset[iq][iz]=lx_conditional_gain::conditional_dn_q2_offset_from_moments(
                    q1+dq1,nz+dnz,uint32_t(coord));
                ep_valid|=bit;
            }
            return ep_offset[iq][iz];
        };
        const double eamp0=sa-qa;
        const bool hp=prev_valid[band]!=0;
        const double td0=hp ? ((sa-double(prev_src_amp[band]))-(qa-double(prev_q_amp[band]))) : 0.0;
        const double base_obj=energy_w*eamp0*eamp0 + (hp?temporal_w*td0*td0:0.0);
        const double W=energy_w+(hp?temporal_w:0.0);
        const double temporal_target=hp ? (sa-double(prev_src_amp[band])+double(prev_q_amp[band])) : 0.0;
        const double K=energy_w*sa+(hp?temporal_w*temporal_target:0.0);
        const double qa2=qa*qa;

        // Rigorous early rejection: every valid two-coordinate change has
        // move cost >= 2*min_move, and its radial/temporal objective >= 0.
        // The original ordered pair search also exits immediately in this case.
        // This check is band-level: no signal-dependent branch in the hot
        // coefficient loop.
        if(double(min_move)+double(second_move)>base_obj){
            prev_src_amp[band]=float(sa);
            prev_q_amp[band]=float(qa);
            prev_valid[band]=1;
            continue;
        }

        if(++scratch.current_epoch==0){
            std::memset(scratch.epoch,0,sizeof(scratch.epoch));scratch.current_epoch=1;
        }
        // R288: exact admissible move sieve. For any improving pair (u,v),
        // move(u)+move(v) < baseline radial+temporal objective. Since the
        // cheapest possible move has cost min_move, a move satisfying
        // move(u)+min_move > base_obj can never participate in any
        // improving pair, regardless of its partner or conditional gain.
        // Compaction uses unconditional stores and arithmetic count updates:
        // no data-dependent branch in the per-coordinate scan.
        int candidate_coord=0;
        uint32_t kept=0;
        auto add_candidate=[&](float x,int32_t* qp,uint8_t channel,uint16_t bin) __attribute__((always_inline)) {
            const int32_t q=*qp;
            const double xd=double(x), qd=double(q), e0=xd-d*qd;
            LxShellMove m0,m1;
            m0.dmse=float(d2+2.0*d*e0);m0.bin=bin;m0.idx=uint16_t(candidate_coord);m0.ch=channel;m0.dir=-1;
            m1.dmse=float(d2-2.0*d*e0);m1.bin=bin;m1.idx=uint16_t(candidate_coord);m1.ch=channel;m1.dir=+1;
            scratch.candidates[kept]=LxShellCandidate{m0,1-2*q};
            kept+=uint32_t(double(m0.dmse)+double(min_move)<=base_obj);
            scratch.candidates[kept]=LxShellCandidate{m1,1+2*q};
            kept+=uint32_t(double(m1.dmse)+double(min_move)<=base_obj);
            ++candidate_coord;
        };
        if(band>=is_start){
            for(int i=lo;i<hi;++i){
                const float z=intensity_shape_sample(l[i],r[i],angle_code[band],src_l[band],src_r[band],float(d));
                add_candidate(z,qA+i,0,uint16_t(i));
            }
        }else{
            for(int i=lo;i<hi;++i){
                const float L=l[i],R=r[i];
                const float a=ms?(L+R)*INV_SQRT2:L;
                const float b=ms?(L-R)*INV_SQRT2:R;
                add_candidate(a,qA+i,0,uint16_t(i));
                add_candidate(b,qB+i,1,uint16_t(i));
            }
        }
        for(uint32_t k=0;k<kept;++k){
            const LxShellCandidate& c=scratch.candidates[k];
            lx_shell_insert_two(lx_shell_group_for(scratch,c.h,ng),c.move);
        }


        // Exact branch-and-bound ordering.  group.a is the minimum-MSE move
        // for that h, hence a_i+a_j is a rigorous lower bound for every valid
        // distinct-coordinate pair from the two groups.  Stable insertion sort
        // keeps the tiny (~10 group) working set in L1 and lets the inner loop
        // terminate as soon as the lower bound reaches base_obj.
        uint16_t order[LX_SHELL_MAX_GROUP];
        for(int i=0;i<ng;++i) order[i]=uint16_t(i);
        for(int i=1;i<ng;++i){
            const uint16_t v=order[i]; const float c=scratch.groups[v].a.dmse;
            int k=i;
            while(k>0){
                const uint16_t u=order[k-1]; const float p=scratch.groups[u].a.dmse;
                if(p<c || (p==c && u<v)) break;
                order[k]=u; --k;
            }
            order[k]=v;
        }

        double best_delta=0.0; LxShellMove bm1{},bm2{}; int32_t bh1=0,bh2=0; bool found=false;
        double best_qamp=qa;
        int best_oi=0,best_oj=0;
        for(int si=0;si<ng;++si){
            const int i=int(order[si]); const auto& A=scratch.groups[i];
            // Exact outer bound: all later groups have minimum move cost >= A.a.dmse.
            // Therefore every remaining pair costs at least 2*A.a.dmse before the
            // non-negative reconstruction objective is added.
            const double prune_limit=base_obj+best_delta;
            if(2.0*double(A.a.dmse)>prune_limit) break;
            for(int sj=si;sj<ng;++sj){
                const int j=int(order[sj]); const auto& B=scratch.groups[j];
                if(double(A.a.dmse)+double(B.a.dmse)>base_obj+best_delta) break;
                LxShellMove m1{},m2{};
                bool valid=false;
                if(i==j){
                    if(A.n>1){m1=A.a;m2=A.b;valid=true;}
                }else if(A.a.idx!=B.a.idx){
                    m1=A.a;m2=B.a;valid=true;
                }else{
                    const bool v0=B.n>1, v1=A.n>1;
                    if(v0&&v1){
                        const float d0=A.a.dmse+B.b.dmse;
                        const float d1=A.b.dmse+B.a.dmse;
                        if(d0<=d1){m1=A.a;m2=B.b;}else{m1=A.b;m2=B.a;}
                        valid=true;
                    }else if(v0){m1=A.a;m2=B.b;valid=true;}
                    else if(v1){m1=A.b;m2=B.a;valid=true;}
                }
                if(!valid) continue;
                const double move_dmse=double(m1.dmse)+double(m2.dmse);
                if(move_dmse>base_obj+best_delta) continue;
                const double q2n=q2+double(A.h+B.h);
                if(q2n<0.0) continue;
                const int dq1i=(A.h>0?1:-1)+(B.h>0?1:-1);
                const int dnzi=(A.h==1?1:(A.h==-1?-1:0))+(B.h==1?1:(B.h==-1?-1:0));
                const double epn=std::max(q2n+get_ep_offset((dq1i+2)>>1,dnzi+2),0.0);
                const double y=d2*epn;
                if(K>0.0){
                    const double A=move_dmse+W*(y-qa2)+2.0*K*qa-best_delta;
                    if(A>=0.0 && A*A>=4.0*K*K*y) continue;
                }
                const double qan=std::sqrt(y);
                const double delta_obj=move_dmse+W*(y-qa2)-2.0*K*(qan-qa);
                const int oi=std::min(i,j), oj=std::max(i,j);
                if(delta_obj<best_delta || (found && delta_obj==best_delta && (oi<best_oi || (oi==best_oi && oj<best_oj)))){
                    best_delta=delta_obj; bm1=m1; bm2=m2; bh1=A.h; bh2=B.h; found=true; best_qamp=qan; best_oi=oi; best_oj=oj;
                }
            }
        }

        if(found){
            int32_t* p1=(bm1.ch?qB:qA)+bm1.bin;
            int32_t* p2=(bm2.ch?qB:qA)+bm2.bin;
            *p1+=bm1.dir; *p2+=bm2.dir;
            total_dmse+=double(bm1.dmse)+double(bm2.dmse);
            q2+=double(bh1+bh2);
            const auto dq1h=[](int32_t h)->double{return h>0?1.0:-1.0;};
            const auto dnzh=[](int32_t h)->double{return h==1?1.0:(h==-1?-1.0:0.0);};
            q1+=dq1h(bh1)+dq1h(bh2);
            nz+=dnzh(bh1)+dnzh(bh2);
        }
        prev_src_amp[band]=float(sa);
        prev_q_amp[band]=float(found?best_qamp:qa);
        prev_valid[band]=1;
    }
    return total_dmse;
}

// FEAR-delta weighted EGV: direct stereo-basis quantization.
// No intermediate joint vector is materialized.  L/R is loaded once; for M/S
// the orthonormal rotation and both anisotropic D_n coordinate streams are
// quantized in the same SIMD traversal.  All ERBs in the EGV share one parity.
template<class Layout>
static DQuantResult quantize_egv_fear_direct(
        const float* __restrict l,const float* __restrict r,int p,int n,bool ms,
        const Layout& erb,int b0,int b1,const TemporalGbrCell* __restrict fc,
        float base_delta,float maxb,const uint16_t* __restrict scale,
        const float* __restrict band_delta,
        int32_t* __restrict qA,int32_t* __restrict qB,
        float* __restrict prev_src_amp,float* __restrict prev_q_amp,
        uint8_t* __restrict prev_valid){
    DQuantResult out;out.dim=2*n;if(n<=0)return out;
    const int warp_mode = lx_warp_mode_cfg();
    const float warp_strength = lx_warp_strength_cfg();
    const int warp_start = lx_warp_start_cfg();
    int parity=0;float best_cost=std::numeric_limits<float>::infinity();
    int best_idx=0,best_dir=1,best_oct=0;
#if defined(__AVX512F__)
    __m512 msev=_mm512_setzero_ps(),env=_mm512_setzero_ps();
    __m512 vbest=_mm512_set1_ps(std::numeric_limits<float>::infinity());
    __m512i ibest=_mm512_setzero_si512(),dbest=_mm512_set1_epi32(1),obest=_mm512_setzero_si512(),xorp=_mm512_setzero_si512();
    const __m512 one=_mm512_set1_ps(1.f),two=_mm512_set1_ps(2.f),signmask=_mm512_set1_ps(-0.0f),g=_mm512_set1_ps(INV_SQRT2);
#endif
    const int base=erb.edge[b0];
    int b=b0;
#if defined(__AVX2__)
    __m256i parity_acc256=_mm256_setzero_si256();
#endif
    while(b<b1){
        const int sk=int(scale[b]);
        const int oct=sk>>4;
        int e=b+1;
        while(e<b1){
            const int nk=int(scale[e]);
            if(nk!=sk)break;
            ++e;
        }
        const int lo=erb.edge[b]-base,hi=erb.edge[e]-base;
        const float delta=band_delta[b],inv=1.f/delta,d2=delta*delta;
#if defined(__AVX512F__)
        const __m512 vd=_mm512_set1_ps(delta),vi=_mm512_set1_ps(inv),vd2=_mm512_set1_ps(d2);const __m512i vo=_mm512_set1_epi32(oct);
        for(int i=lo;i<hi;i+=16){
            const int rem=hi-i;const __mmask16 active=(rem>=16)?__mmask16(0xffff):__mmask16((1u<<rem)-1u);
            const __m512 vl=_mm512_maskz_loadu_ps(active,l+p+i),vr=_mm512_maskz_loadu_ps(active,r+p+i);
            const __m512 xa=ms?_mm512_mul_ps(_mm512_add_ps(vl,vr),g):vl;
            const __m512 xb=ms?_mm512_mul_ps(_mm512_sub_ps(vl,vr),g):vr;
#define LX_Q_SIDE(X,QOUT,IDBASE) do { \
            const __m512 u=_mm512_mul_ps((X),vi); \
            const __m512 rn=_mm512_roundscale_ps(u,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC); \
            const __m512i qi=_mm512_cvtps_epi32(rn); xorp=_mm512_xor_si512(xorp,_mm512_maskz_mov_epi32(active,qi)); \
            const __m512 ae=_mm512_andnot_ps(signmask,_mm512_sub_ps(u,rn)); \
            const __m512 cost=_mm512_mul_ps(vd2,_mm512_sub_ps(one,_mm512_mul_ps(two,ae))); \
            const __mmask16 take=active&_mm512_cmp_ps_mask(cost,vbest,_CMP_LT_OQ);vbest=_mm512_mask_mov_ps(vbest,take,cost); \
            const int _ib=(IDBASE); const __m512i ids=_mm512_setr_epi32(_ib+0,_ib+1,_ib+2,_ib+3,_ib+4,_ib+5,_ib+6,_ib+7,_ib+8,_ib+9,_ib+10,_ib+11,_ib+12,_ib+13,_ib+14,_ib+15); \
            ibest=_mm512_mask_mov_epi32(ibest,take,ids); \
            const __mmask16 neg=_mm512_cmp_ps_mask(u,rn,_CMP_LT_OQ);const __m512i dirs=_mm512_mask_blend_epi32(neg,_mm512_set1_epi32(1),_mm512_set1_epi32(-1)); \
            dbest=_mm512_mask_mov_epi32(dbest,take,dirs);obest=_mm512_mask_mov_epi32(obest,take,vo); \
            const __m512 y=_mm512_mul_ps(_mm512_cvtepi32_ps(qi),vd),er=_mm512_sub_ps((X),y);msev=_mm512_mask3_fmadd_ps(er,er,msev,active);env=_mm512_mask3_fmadd_ps((X),(X),env,active); \
            _mm512_mask_storeu_epi32((void*)((QOUT)+p+i),active,qi); \
        } while(0)
            LX_Q_SIDE(xa,qA,i);
            LX_Q_SIDE(xb,qB,n+i);
#undef LX_Q_SIDE
        }
#elif defined(__AVX2__)
        // Product fast path: the normative/default warp is identity.  Keep the
        // research warp path scalar, but quantize the overwhelmingly common
        // identity geometry eight coefficients at a time.  The branch is a
        // codec-configuration branch outside the signal hot loop.
        const bool identity_warp=(warp_mode<=0 || warp_strength==0.0f || e<=warp_start);
        if(identity_warp){
            const __m256 vi=_mm256_set1_ps(inv);
            const __m256 vg=_mm256_set1_ps(INV_SQRT2);
            int i=lo;
            for(;i+8<=hi;i+=8){
                const __m256 vl=_mm256_loadu_ps(l+p+i), vr=_mm256_loadu_ps(r+p+i);
                const __m256 xa=ms?_mm256_mul_ps(_mm256_add_ps(vl,vr),vg):vl;
                const __m256 xb=ms?_mm256_mul_ps(_mm256_sub_ps(vl,vr),vg):vr;
#define LX_Q8_SIDE(X,QOUT,IDBASE) do { \
                const __m256 u=_mm256_mul_ps((X),vi); \
                const __m256 rn=_mm256_round_ps(u,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC); \
                const __m256i qi=_mm256_cvtps_epi32(rn); \
                _mm256_storeu_si256((__m256i*)((QOUT)+p+i),qi); \
                /* D_n parity is associative. Accumulate complete SIMD vectors
                   and horizontally reduce once per EGV, not once per 8 coefficients. */ \
                parity_acc256=_mm256_xor_si256(parity_acc256,qi); \
            } while(0)
                LX_Q8_SIDE(xa,qA,i);
                LX_Q8_SIDE(xb,qB,n+i);
#undef LX_Q8_SIDE
            }
            for(;i<hi;++i){
                const float L=l[p+i],R=r[p+i],xa=ms?(L+R)*INV_SQRT2:L,xb=ms?(L-R)*INV_SQRT2:R;
                const float vv[2]={xa,xb};int32_t*qq[2]={qA,qB};
                for(int ch=0;ch<2;++ch){const float ox=vv[ch],u=ox*inv,rn=std::nearbyint(u);const int32_t qi=(int32_t)rn;parity^=qi;qq[ch][p+i]=qi;}
            }
        }else{
            for(int i=lo;i<hi;++i){const float L=l[p+i],R=r[p+i],xa=ms?(L+R)*INV_SQRT2:L,xb=ms?(L-R)*INV_SQRT2:R;const float vv[2]={xa,xb};int32_t*qq[2]={qA,qB};const float ww=lx_warp_weight(b,i-lo,hi-lo);const float de=delta/ww, inve=1.0f/de, de2=de*de;for(int ch=0;ch<2;++ch){const float ox=vv[ch],u=ox*inve,rn=std::nearbyint(u);const int32_t qi=(int32_t)rn;parity^=qi;qq[ch][p+i]=qi;}}
        }
#else
        for(int i=lo;i<hi;++i){const float L=l[p+i],R=r[p+i],xa=ms?(L+R)*INV_SQRT2:L,xb=ms?(L-R)*INV_SQRT2:R;const float vv[2]={xa,xb};int32_t*qq[2]={qA,qB};const float ww=lx_warp_weight(b,i-lo,hi-lo);const float de=delta/ww, inve=1.0f/de, de2=de*de;for(int ch=0;ch<2;++ch){const float ox=vv[ch],u=ox*inve,rn=std::nearbyint(u);const int32_t qi=(int32_t)rn;parity^=qi;qq[ch][p+i]=qi;}}
#endif
        b=e;
    }
#if defined(__AVX2__)
    {
        __m128i qx=_mm_xor_si128(_mm256_castsi256_si128(parity_acc256),_mm256_extracti128_si256(parity_acc256,1));
        qx=_mm_xor_si128(qx,_mm_srli_si128(qx,8)); qx=_mm_xor_si128(qx,_mm_srli_si128(qx,4));
        parity^=_mm_cvtsi128_si32(qx);
    }
#endif
#if defined(__AVX512F__)
    alignas(64)float bc[16],mm[16],ee[16];alignas(64)int32_t bi[16],bd[16],bo[16],xp[16];_mm512_store_ps(bc,vbest);_mm512_store_ps(mm,msev);_mm512_store_ps(ee,env);_mm512_store_si512((void*)bi,ibest);_mm512_store_si512((void*)bd,dbest);_mm512_store_si512((void*)bo,obest);_mm512_store_si512((void*)xp,xorp);
    for(int k=0;k<16;++k){parity^=xp[k];out.mse+=mm[k];out.energy+=ee[k];if(bc[k]<best_cost){best_cost=bc[k];best_idx=bi[k];best_dir=bd[k];best_oct=bo[k];}}
#endif
    // Re-score the mandatory odd-parity repair against local ERB energy and
    // its temporal change.  This is outside the coefficient hot loop.
    std::array<double,ERB_BANDS> src_e{},q_e{};
    for(int bb=b0;bb<b1;++bb){
        const int lo=erb.edge[bb],hi=erb.edge[bb+1]; const float d=band_delta[bb];
        for(int i=lo;i<hi;++i){
            const float L=l[i],R=r[i];
            src_e[bb]+=double(L)*L+double(R)*R;
            q_e[bb]+=double(d)*d*(double(qA[i])*qA[i]+double(qB[i])*qB[i]);
        }
    }
    if(parity&1){
        double best_joint=std::numeric_limits<double>::infinity(); int32_t* jq=nullptr; int jdir=1; double jmse=0.0; int jb=b0;
        for(int bb=b0;bb<b1;++bb){
            const int lo=erb.edge[bb],hi=erb.edge[bb+1]; const float d=band_delta[bb],inv=1.f/d,d2=d*d;
            for(int i=lo;i<hi;++i){
                const float L=l[i],R=r[i]; const float x[2]={ms?(L+R)*INV_SQRT2:L,ms?(L-R)*INV_SQRT2:R}; int32_t* q[2]={qA+i,qB+i};
                for(int ch=0;ch<2;++ch){
                    const int32_t qi=*q[ch]; const float u=x[ch]*inv; const int dir=(u<float(qi))?-1:1;
                    const double mse_inc=double(d2)*(1.0-2.0*std::fabs(double(u)-double(qi)));
                    const double de=double(d2)*(double((qi+dir)*(qi+dir))-double(qi)*qi);
                    const double score=lx_energy_repair_increment(mse_inc,src_e[bb],q_e[bb],q_e[bb]+de,prev_src_amp[bb],prev_q_amp[bb],prev_valid[bb]!=0);
                    if(score<best_joint){best_joint=score;jq=q[ch];jdir=dir;jmse=mse_inc;jb=bb;}
                }
            }
        }
        if(jq){ const int32_t qi=*jq; *jq+=jdir; const double d2=double(band_delta[jb])*band_delta[jb]; q_e[jb]+=d2*(double((*jq)*(*jq))-double(qi)*qi); }
    }
    for(int bb=b0;bb<b1;++bb){prev_src_amp[bb]=float(std::sqrt(std::max(src_e[bb],0.0)));prev_q_amp[bb]=float(std::sqrt(std::max(q_e[bb],0.0)));prev_valid[bb]=1;}
    return out;
}


static inline uint64_t sgc_choose(int n,int k){
    if(k<0||k>n)return 0; if(k>n-k)k=n-k; uint64_t r=1; for(int i=1;i<=k;++i)r=(r*uint64_t(n-k+i))/uint64_t(i); return r;
}
static inline unsigned sgc_rank_bits(int n,int k){uint64_t c=sgc_choose(n,k); unsigned b=0; while((uint64_t(1)<<b)<c && b<63)++b; return b;}
static inline uint64_t sgc_rank_mask_n(uint64_t mask,int n,int k){uint64_t rank=0;int left=k;for(int i=n-1;i>=0&&left>0;--i){if(mask&(uint64_t(1)<<i)){rank+=sgc_choose(i,left);--left;}}return rank;}

static inline int stereo_detector_cfg(){static const int v=[](){const char*p=::lxvoxel_release_env("LX_STEREO_DETECTOR");return p?std::atoi(p):4;}();return v;}

static inline int psq_enable_cfg();
static inline int indep_lr_cfg();
static inline int psq_detector_cfg();
static inline int psq_curve_cfg();
static inline float psq_side_gain(int b);

// PSQ experimental quantizer: per-ERB L/R vs M/S plus a deterministic,
// cell-volume preserving M/S anisotropy.  It intentionally uses the scalar
// path for implementation clarity; the selected geometry can be SIMD-specialized
// after the sweep.  D_n parity still spans the complete EGV.
template<class Layout>
static DQuantResult quantize_egv_psq_direct(
        const float* __restrict l,const float* __restrict r,int p,int n,bool frame_ms,
        const uint8_t* __restrict band_ms,const Layout& erb,int b0,int b1,
        float base_delta,const uint16_t* __restrict scale,
        const float* __restrict band_delta,const float* __restrict lr_gain,
        int32_t* __restrict qA,int32_t* __restrict qB){
    DQuantResult out; out.dim=2*n; if(n<=0)return out;
    int parity=0; float best_cost=std::numeric_limits<float>::infinity();
    int32_t* best_q=nullptr; int best_dir=1;
    for(int b=b0;b<b1;++b){
        const float d=band_delta[b];
        const bool indep=indep_lr_cfg()!=0;
        const bool refine=!indep && (psq_detector_cfg()>=10) && (band_ms[b]!=0);
        const bool ms=indep ? frame_ms : ((psq_detector_cfg()>=10) ? (refine?true:frame_ms) : (band_ms[b]!=0));
        const float g=indep ? (ms?1.0f:lr_gain[b]) : ((psq_detector_cfg()>=10) ? (refine?psq_side_gain(b):1.0f) : (ms?psq_side_gain(b):1.0f));
        const float dA=d*g;
        const float dB=d/g;
        const float invA=1.0f/dA,invB=1.0f/dB;
        const int lo=erb.edge[b],hi=erb.edge[b+1];
        for(int i=lo;i<hi;++i){
            const float L=l[i],R=r[i];
            const float xa=ms?(L+R)*INV_SQRT2:L;
            const float xb=ms?(L-R)*INV_SQRT2:R;
            const float x[2]={xa,xb}, inv[2]={invA,invB}, dd[2]={dA,dB};
            int32_t* q[2]={qA+i,qB+i};
            for(int ch=0;ch<2;++ch){
                const float u=x[ch]*inv[ch],rn=std::nearbyint(u),ae=std::fabs(u-rn);
                const int32_t qi=int32_t(rn); *q[ch]=qi; parity^=qi;
                const float d2=dd[ch]*dd[ch];
                const float cost=d2*(1.f-2.f*ae);
                if(cost<best_cost){best_cost=cost;best_q=q[ch];best_dir=(u<rn)?-1:1;}

            }
        }
    }
    if((parity&1) && best_q){*best_q+=best_dir;}
    return out;
}


// R332 production true-independent L/R quantizer.  This is the anisotropic
// counterpart of quantize_egv_fear_direct for physical L/R frames.  The two
// axes use reciprocal step scales, so D_n cell volume is unchanged:
//   dL = d*g, dR = d/g.
// One gain is shared by all ERBs in the current EGV, as signalled by v0.34.
// Keep the coefficient loop branch-free and SIMD-friendly.
template<class Layout>
static DQuantResult quantize_egv_indep_lr_direct(
        const float* __restrict l,const float* __restrict r,int p,int n,
        const Layout& erb,int b0,int b1,const float* __restrict band_delta,
        const float* __restrict lr_gain,int32_t* __restrict qA,int32_t* __restrict qB){
    DQuantResult out; out.dim=2*n; if(n<=0)return out;
    int parity=0; float best_cost=std::numeric_limits<float>::infinity();
    int32_t* best_q=nullptr; int best_dir=1;
#if defined(__AVX2__)
    const __m256 vone=_mm256_set1_ps(1.0f), vtwo=_mm256_set1_ps(2.0f);
    const __m256 vsgn=_mm256_set1_ps(-0.0f);
#endif
#if defined(__AVX2__)
    __m256i parity_acc256=_mm256_setzero_si256();
#endif
    for(int b=b0;b<b1;++b){
        const float d=band_delta[b];
        const float g=lr_gain[b];
        const float dL=d*g, dR=d/g;
        const float iL=1.0f/dL, iR=1.0f/dR;
        const float dL2=dL*dL, dR2=dR*dR;
        const int lo=erb.edge[b], hi=erb.edge[b+1];
#if defined(__AVX2__)
        const __m256 viL=_mm256_set1_ps(iL),viR=_mm256_set1_ps(iR);
        const __m256 vdL2=_mm256_set1_ps(dL2),vdR2=_mm256_set1_ps(dR2);
        int i=lo;
        for(;i+8<=hi;i+=8){
            const __m256 xL=_mm256_loadu_ps(l+i), xR=_mm256_loadu_ps(r+i);
#define LX_Q8_INDEP(X,INV,D2,QOUT) do { \
            const __m256 u=_mm256_mul_ps((X),(INV)); \
            const __m256 rn=_mm256_round_ps(u,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC); \
            const __m256i qi=_mm256_cvtps_epi32(rn); \
            _mm256_storeu_si256((__m256i*)((QOUT)+i),qi); \
            parity_acc256=_mm256_xor_si256(parity_acc256,qi); \
            const __m256 ae=_mm256_andnot_ps(vsgn,_mm256_sub_ps(u,rn)); \
            const __m256 cost=_mm256_mul_ps((D2),_mm256_sub_ps(vone,_mm256_mul_ps(vtwo,ae))); \
            const __m128 clo=_mm256_castps256_ps128(cost),chi=_mm256_extractf128_ps(cost,1); \
            __m128 cm=_mm_min_ps(clo,chi); cm=_mm_min_ps(cm,_mm_movehl_ps(cm,cm)); cm=_mm_min_ss(cm,_mm_shuffle_ps(cm,cm,1)); \
            const float cmin=_mm_cvtss_f32(cm); \
            if(cmin<best_cost){ \
                const int mask=_mm256_movemask_ps(_mm256_cmp_ps(cost,_mm256_set1_ps(cmin),_CMP_EQ_OQ)); \
                const int jj=__builtin_ctz((unsigned)mask); \
                alignas(32) float uv[8]; alignas(32) int32_t qv[8]; \
                _mm256_store_ps(uv,u); _mm256_store_si256((__m256i*)qv,qi); \
                best_cost=cmin; best_q=(QOUT)+i+jj; best_dir=(uv[jj]<float(qv[jj]))?-1:1; \
            } \
        } while(0)
            LX_Q8_INDEP(xL,viL,vdL2,qA);
            LX_Q8_INDEP(xR,viR,vdR2,qB);
#undef LX_Q8_INDEP
        }
        for(;i<hi;++i){
            const float x[2]={l[i],r[i]}, inv[2]={iL,iR}, d2[2]={dL2,dR2};
            int32_t* q[2]={qA+i,qB+i};
            for(int ch=0;ch<2;++ch){
                const float u=x[ch]*inv[ch],rn=std::nearbyint(u),ae=std::fabs(u-rn);
                const int32_t qi=int32_t(rn); *q[ch]=qi; parity^=qi;
                const float cost=d2[ch]*(1.0f-2.0f*ae);
                if(cost<best_cost){best_cost=cost;best_q=q[ch];best_dir=(u<rn)?-1:1;}
            }
        }
#else
        for(int i=lo;i<hi;++i){
            const float x[2]={l[i],r[i]},inv[2]={iL,iR},d2[2]={dL2,dR2};
            int32_t* q[2]={qA+i,qB+i};
            for(int ch=0;ch<2;++ch){
                const float u=x[ch]*inv[ch],rn=std::nearbyint(u),ae=std::fabs(u-rn);
                const int32_t qi=int32_t(rn); *q[ch]=qi; parity^=qi;
                const float cost=d2[ch]*(1.0f-2.0f*ae);
                if(cost<best_cost){best_cost=cost;best_q=q[ch];best_dir=(u<rn)?-1:1;}
            }
        }
#endif
    }
#if defined(__AVX2__)
    {
        __m128i qx=_mm_xor_si128(_mm256_castsi256_si128(parity_acc256),_mm256_extracti128_si256(parity_acc256,1));
        qx=_mm_xor_si128(qx,_mm_srli_si128(qx,8)); qx=_mm_xor_si128(qx,_mm_srli_si128(qx,4));
        parity^=_mm_cvtsi128_si32(qx);
    }
#endif
    if((parity&1) && best_q)*best_q+=best_dir;
    return out;
}

// Mixed full-stereo / intensity EGV path.  It is only called for an EGV that
// overlaps the active high-band intensity suffix; all other EGVs stay on the
// original AVX-512 direct quantizer above.
template<class Layout>
static DQuantResult quantize_egv_fear_is_direct(
        const float* __restrict l,const float* __restrict r,int p,int n,bool ms,
        const Layout& erb,int b0,int b1,const TemporalGbrCell* __restrict fc,
        float base_delta,float maxb,const uint16_t* __restrict scale,
        const float* __restrict band_delta,
        int is_start,const uint8_t* __restrict angle_code,
        const float* __restrict src_l,const float* __restrict src_r,
        int32_t* __restrict qA,int32_t* __restrict qB,
        float* __restrict prev_src_amp,float* __restrict prev_q_amp,
        uint8_t* __restrict prev_valid){
    DQuantResult out;out.dim=2*n;if(n<=0)return out;
    int parity=0;float best_cost=std::numeric_limits<float>::infinity();
    int32_t* best_q=nullptr;int best_dir=1;
    for(int b=b0;b<b1;++b){
        const float delta=band_delta[b];
        const float inv=1.f/delta,d2=delta*delta;
        const int lo=erb.edge[b],hi=erb.edge[b+1];
        if(b>=is_start){
            const auto axis=intensity_axis_f(angle_code[b]);
            const float ct=axis.cosine,st=axis.sine;
            for(int i=lo;i<hi;++i){
                const float L=l[i],R=r[i];
                const float z=intensity_shape_sample(L,R,angle_code[b],src_l[b],src_r[b],delta);
                const float u=z*inv,rn=std::nearbyint(u),ae=std::fabs(u-rn);
                const int32_t qi=int32_t(rn);qA[i]=qi;qB[i]=0;parity^=qi;
                const float cost=d2*(1.f-2.f*ae);
                if(cost<best_cost){best_cost=cost;best_q=qA+i;best_dir=(u<rn)?-1:1;}
            }
        }else{
            for(int i=lo;i<hi;++i){
                const float L=l[i],R=r[i];
                const float a=ms?(L+R)*INV_SQRT2:L;
                const float bb=ms?(L-R)*INV_SQRT2:R;
                const float x[2]={a,bb};int32_t* q[2]={qA+i,qB+i};
                for(int ch=0;ch<2;++ch){
                    const float u=x[ch]*inv,rn=std::nearbyint(u),ae=std::fabs(u-rn);
                    const int32_t qi=int32_t(rn);*q[ch]=qi;parity^=qi;
                    const float cost=d2*(1.f-2.f*ae);
                    if(cost<best_cost){best_cost=cost;best_q=q[ch];best_dir=(u<rn)?-1:1;}

                }
            }
        }
    }
    std::array<double,ERB_BANDS> src_e{},q_e{};
    for(int bb=b0;bb<b1;++bb){
        const float d=band_delta[bb]; const int lo=erb.edge[bb],hi=erb.edge[bb+1];
        if(bb>=is_start){
            for(int i=lo;i<hi;++i){src_e[bb]+=double(l[i])*l[i]+double(r[i])*r[i];q_e[bb]+=double(d)*d*double(qA[i])*qA[i];}
        }else{
            for(int i=lo;i<hi;++i){const float L=l[i],R=r[i],a=ms?(L+R)*INV_SQRT2:L,c=ms?(L-R)*INV_SQRT2:R;src_e[bb]+=double(a)*a+double(c)*c;q_e[bb]+=double(d)*d*(double(qA[i])*qA[i]+double(qB[i])*qB[i]);}
        }
    }
    if(parity&1){
        double best_joint=std::numeric_limits<double>::infinity(); int32_t* jq=nullptr; int jdir=1; double jmse=0.0; int jb=b0;
        for(int bb=b0;bb<b1;++bb){
            const float d=band_delta[bb],inv=1.f/d,d2=d*d; const int lo=erb.edge[bb],hi=erb.edge[bb+1];
            if(bb>=is_start){
                for(int i=lo;i<hi;++i){const float z=intensity_shape_sample(l[i],r[i],angle_code[bb],src_l[bb],src_r[bb],d),u=z*inv;const int32_t qi=qA[i];const int dir=(u<float(qi))?-1:1;const double mi=double(d2)*(1.0-2.0*std::fabs(double(u)-double(qi)));const double de=double(d2)*(double((qi+dir)*(qi+dir))-double(qi)*qi);const double sc=lx_energy_repair_increment(mi,src_e[bb],q_e[bb],q_e[bb]+de,prev_src_amp[bb],prev_q_amp[bb],prev_valid[bb]!=0);if(sc<best_joint){best_joint=sc;jq=qA+i;jdir=dir;jmse=mi;jb=bb;}}
            }else{
                for(int i=lo;i<hi;++i){const float L=l[i],R=r[i],x[2]={ms?(L+R)*INV_SQRT2:L,ms?(L-R)*INV_SQRT2:R};int32_t* q[2]={qA+i,qB+i};for(int ch=0;ch<2;++ch){const int32_t qi=*q[ch];const float u=x[ch]*inv;const int dir=(u<float(qi))?-1:1;const double mi=double(d2)*(1.0-2.0*std::fabs(double(u)-double(qi)));const double de=double(d2)*(double((qi+dir)*(qi+dir))-double(qi)*qi);const double sc=lx_energy_repair_increment(mi,src_e[bb],q_e[bb],q_e[bb]+de,prev_src_amp[bb],prev_q_amp[bb],prev_valid[bb]!=0);if(sc<best_joint){best_joint=sc;jq=q[ch];jdir=dir;jmse=mi;jb=bb;}}}
            }
        }
        if(jq){const int32_t qi=*jq;*jq+=jdir;const double d2=double(band_delta[jb])*band_delta[jb];q_e[jb]+=d2*(double((*jq)*(*jq))-double(qi)*qi);}
    }
    for(int bb=b0;bb<b1;++bb){prev_src_amp[bb]=float(std::sqrt(std::max(src_e[bb],0.0)));prev_q_amp[bb]=float(std::sqrt(std::max(q_e[bb],0.0)));prev_valid[bb]=1;}
    return out;
}


// PSQ experimental controls. These are runtime-only experimental switches so a
// single binary can sweep stereo geometry without recompilation.
static inline int psq_enable_cfg(){static const int v=[](){const char*p=::lxvoxel_release_env("LX_PSQ_ENABLE");return p?std::atoi(p):0;}();return v;}
static inline int indep_lr_cfg(){return 1;}
static inline int psq_detector_cfg(){static const int v=[](){const char*p=::lxvoxel_release_env("LX_PSQ_DETECTOR");return p?std::atoi(p):0;}();return v;}
static inline int psq_curve_cfg(){static const int v=[](){const char*p=::lxvoxel_release_env("LX_PSQ_CURVE");return p?std::atoi(p):0;}();return v;}

// SGC experimental controls: per-band stereo geometry correction after
// quantization/TNS inversion and before SFC.  Runtime-only for sweeps.
static inline int sgc_enable_cfg(){static const int v=[](){const char*p=::lxvoxel_release_env("LX_SGC_ENABLE");return p?std::atoi(p):1;}();return v;}

static inline float psq_side_gain(int b){
    // Cell-volume preserving stereo precision redistribution:
    // Delta_M = Delta*g, Delta_S = Delta/g. Product remains Delta^2.
    const double x=double(b)/double(ERB_BANDS-1); // 0=LF, 1=HF
    double a=0.0,p=1.0;
    switch(psq_curve_cfg()){
        case 1:a=0.125;p=1.0;break;
        case 2:a=0.250;p=1.0;break;
        case 3:a=0.500;p=1.0;break;
        case 4:a=0.750;p=1.0;break;
        case 5:a=0.250;p=2.0;break;
        case 6:a=0.500;p=2.0;break;
        case 7:a=0.750;p=2.0;break;
        case 8:a=1.000;p=2.0;break;
        default:return 1.0f;
    }
    return float(std::exp2(a*std::pow(1.0-x,p)));
}


struct Result{uint64_t bytes=0;double mse=0,energy=0;size_t frames=0,b1024=0,b512=0,b256=0,ms=0,is_frames=0,is_bands=0;uint64_t is_side_bits=0,sfc_bits=0;size_t sfc_frames=0;uint64_t prec_bins=0,prec_coords=0;double entropy_ns=0;uint64_t entropy_symbols=0;float effective_bpc=0.0f;uint64_t tns_side_bits=0,tns_frames=0;bool rate_saturated=false;double tfbr_target_kbps=0.0,kkt_mu=0.0,kkt_C=0.0;};
struct PilotStats{std::vector<double> packet_bits,entropy_bits,overhead_bits;};

// Frame-level state consumed by FBR. LX-FEAR owns local psychoacoustics;
// FBR uses only the resulting frame pressure and centered temporal structure.
struct FbrFrameFeature {
    float fear_loggeo=0.0f;
    float fear_novelty=0.0f;
    float fear_signed_novelty=0.0f;
    float log_qabs=0.0f;
    float dim=0.0f;
};


// research: local RFFT Tensor x Ridge evidence. One 512-point RFFT per
// coded frame, projected directly onto the codec's 48 ERB grid.  The feature
// is encoder-only and zero-common-mode when used for allocation.
struct Rfft512LocalTR {
    lx_nsmdct::FFTPow2D<256> fft;
    std::array<float,214> preva{}, cura{};
    std::array<float,256> wr{}, wi{};
    std::array<float,512> win{};
    bool primed=false;
    Rfft512LocalTR(){
        for(int k=0;k<256;++k){ const double a=-2.0*pulsar::PI*double(k)/512.0; wr[k]=float(std::cos(a)); wi[k]=float(std::sin(a)); }
        for(int n=0;n<512;++n){ const double a=pulsar::PI*(double(n)+0.5)/512.0; win[n]=float(std::sin(a)); }
    }
    void analyse(const float* interleaved,size_t ns,size_t center,
                 std::array<float,ERB_BANDS>& bt,
                 std::array<float,ERB_BANDS>& br,
                 std::array<float,ERB_BANDS>& bj,
                 std::array<float,ERB_BANDS>& bp){
        constexpr int H=256;
        const size_t base=(center>=H?center-H:0);
        for(int n=0;n<H;++n){
            const size_t i0=base+size_t(2*n),i1=i0+1;
            const float x0=(i0<ns)?0.5f*(interleaved[2*i0]+interleaved[2*i0+1]):0.0f;
            const float x1=(i1<ns)?0.5f*(interleaved[2*i1]+interleaved[2*i1+1]):0.0f;
            fft.re[n]=win[2*n]*x0; fft.im[n]=win[2*n+1]*x1;
        }
        fft.forward();
        for(int k=1;k<=213;++k){
            const int j=(256-k)&255;
            const double ar=fft.re[k],ai=fft.im[k],br0=fft.re[j],bi=-fft.im[j];
            const double sr=ar+br0,si=ai+bi,dr=ar-br0,di=ai-bi;
            const double tr=double(wr[k])*dr-double(wi[k])*di;
            const double ti=double(wr[k])*di+double(wi[k])*dr;
            const double xr=0.5*(sr+ti),xi=0.5*(si-tr);
            cura[k]=float(std::sqrt(xr*xr+xi*xi));
        }
        std::array<double,ERB_BANDS> sw{},st{},sridge{},sjoint{},spersist{};
        const double emax=hz_to_erb(CODED_HZ);
        for(int k=3;k<=211;++k){
            const double am2=cura[k-2],am1=cura[k-1],ac=cura[k],ap1=cura[k+1],ap2=cura[k+2];
            const double pm1=preva[k-1],pc=preva[k],pp1=preva[k+1];
            const double gf=ap1-am1,gt=ac-pc,gfm=ac-am2,gtm=am1-pm1,gfp=ap2-ac,gtp=ap1-pp1;
            const double jff=0.25*(gfm*gfm+2.0*gf*gf+gfp*gfp);
            const double jtt=0.25*(gtm*gtm+2.0*gt*gt+gtp*gtp);
            const double jft=0.25*(gfm*gtm+2.0*gf*gt+gfp*gtp),jd=jff-jtt;
            const double tensor=std::sqrt(jd*jd+4.0*jft*jft)/(jff+jtt+1e-30);
            const double sm5=(am2+4.0*am1+6.0*ac+4.0*ap1+ap2)*(1.0/16.0);
            const double ridge=2.0*ac/(ac+sm5+1e-30);
            const double persist=primed?(2.0*ac*pc/(ac*ac+pc*pc+1e-30)):1.0;
            const double hz=double(k)*double(SR)/512.0;
            const int b=std::clamp(int(hz_to_erb(std::min(hz,CODED_HZ))*double(ERB_BANDS)/emax),0,ERB_BANDS-1);
            const double w=ac+sm5;
            sw[b]+=w; st[b]+=w*tensor; sridge[b]+=w*ridge; sjoint[b]+=w*std::sqrt(std::max(0.0,tensor*ridge)); spersist[b]+=w*persist;
        }
        for(int b=0;b<ERB_BANDS;++b){
            const double inv=1.0/(sw[b]+1e-30);
            bt[b]=float(st[b]*inv); br[b]=float(sridge[b]*inv); bj[b]=float(sjoint[b]*inv); bp[b]=float(spersist[b]*inv);
        }
        for(int k=1;k<=213;++k)preva[k]=cura[k];
        primed=true;
    }
};

struct LXTransientFeatures {
    float peak = 1.0f;
    float hp_peak = 1.0f;
    float tv = 0.0f;
    float hp_tv = 0.0f;
    float rise = 0.0f;
    float coarse_mask = 0.0f;
    float duty = 1.0f;
    float hp_duty = 1.0f;
};

// LX-VOXEL temporal geometry detector: 16 x 128-sample cells over the
// prospective 2048-sample long block. It measures temporal concentration and
// non-stationarity directly; no LPC, recursive masking scan or tone search.
static inline LXTransientFeatures lx_transient_features_2048(const float* L,const float* R) {
    constexpr int TCELLS = K_LONG/128;
    static_assert(TCELLS >= 2);
    alignas(64) float e[TCELLS], h[TCELLS];
    float prevL=L[-1], prevR=R[-1];
    float sumE=0.0f,sumH=0.0f,maxE=0.0f,maxH=0.0f;
    for(int c=0;c<TCELLS;++c){
        float se=0.0f,sh=0.0f;
#if defined(__AVX2__)
        __m256 ae=_mm256_setzero_ps(), ah=_mm256_setzero_ps();
        const int base=c*128;
        int i=0;
        for(;i<128;i+=8){
            const __m256 l=_mm256_loadu_ps(L+base+i);
            const __m256 r=_mm256_loadu_ps(R+base+i);
            const __m256 lp=(base+i==0)
                ? _mm256_set_ps(L[base+i+6],L[base+i+5],L[base+i+4],L[base+i+3],L[base+i+2],L[base+i+1],L[base+i],prevL)
                : _mm256_loadu_ps(L+base+i-1);
            const __m256 rp=(base+i==0)
                ? _mm256_set_ps(R[base+i+6],R[base+i+5],R[base+i+4],R[base+i+3],R[base+i+2],R[base+i+1],R[base+i],prevR)
                : _mm256_loadu_ps(R+base+i-1);
            ae=_mm256_fmadd_ps(l,l,ae); ae=_mm256_fmadd_ps(r,r,ae);
            const __m256 dl=_mm256_sub_ps(l,lp), dr=_mm256_sub_ps(r,rp);
            ah=_mm256_fmadd_ps(dl,dl,ah); ah=_mm256_fmadd_ps(dr,dr,ah);
        }
        alignas(32) float te[8],th[8]; _mm256_store_ps(te,ae); _mm256_store_ps(th,ah);
        for(int k=0;k<8;++k){se+=te[k];sh+=th[k];}
#else
        const int base=c*128;
        for(int i=0;i<128;++i){
            const float l=L[base+i],r=R[base+i];
            const float dl=l-prevL,dr=r-prevR;
            se+=l*l+r*r; sh+=dl*dl+dr*dr;
            prevL=l;prevR=r;
        }
#endif
#if defined(__AVX2__)
        prevL=L[c*128+127];prevR=R[c*128+127];
#endif
        e[c]=se;h[c]=sh;sumE+=se;sumH+=sh;maxE=std::max(maxE,se);maxH=std::max(maxH,sh);
    }
    const float tiny=std::numeric_limits<float>::min();
    LXTransientFeatures f;
    f.peak=float(TCELLS)*maxE/(sumE+tiny);
    f.hp_peak=float(TCELLS)*maxH/(sumH+tiny);
    float tv=0.0f,htv=0.0f,rise=0.0f;
    for(int c=0;c<TCELLS-1;++c){
        const float de=2.0f*(e[c+1]-e[c])/(e[c+1]+e[c]+tiny);
        const float dh=2.0f*(h[c+1]-h[c])/(h[c+1]+h[c]+tiny);
        tv += std::fabs(de);
        htv+= std::fabs(dh);
        const float re=std::max(de,0.0f);
        rise+=re;
    }
    f.tv=tv*(1.0f/float(TCELLS-1));f.hp_tv=htv*(1.0f/float(TCELLS-1));
    f.rise=rise*(2.0f/float(TCELLS-1));
    float srE=0.0f,srH=0.0f;
    for(int c=0;c<TCELLS;++c){srE+=std::sqrt(std::max(e[c],0.0f));srH+=std::sqrt(std::max(h[c],0.0f));}
    f.duty=(srE*srE)/(float(TCELLS)*sumE+tiny);
    f.hp_duty=(srH*srH)/(float(TCELLS)*sumH+tiny);
    // Coarse bidirectional masking on the 16 high-pass energy cells.
    // a=b=1/4: short memory matched to the 128-sample cell grid.
    float fw[TCELLS],bw[TCELLS],m=0.0f,mx=0.0f;
    for(int c=0;c<TCELLS;++c){m=h[c]+0.25f*m;fw[c]=0.75f*m;}
    m=0.0f;
    for(int c=TCELLS-1;c>=0;--c){m=fw[c]+0.25f*m;bw[c]=0.75f*m;mx=std::max(mx,bw[c]);}
    const float g=std::sqrt((sumH*(1.0f/float(TCELLS)))*mx+tiny);
    float cm=0.0f;
    for(int c=0;c<TCELLS;++c)cm+=std::sqrt(g/(bw[c]+tiny));
    f.coarse_mask=cm*(1.0f/float(TCELLS));
    return f;
}

struct EncoderStaticTables {
    ErbLayoutN<K_LONG>  eL;
    ErbLayoutN<K_1024>  e1024;
    ErbLayoutN<K_SHORT> eS;
    ErbLayoutN<K_256>   e256;
    Fear48N<K_LONG>  fL;
    Fear48N<K_1024>  f1024;
    Fear48N<K_SHORT> fS;
    Fear48N<K_256>   f256;
    lx_fear_v4::Static fear4;
    VariableLowOverlapMDCT lap;
    DCT4Ortho<1024> dLong, dLongR;
    DCT4Ortho<768> d1024, d1024R;
    DCT4Ortho<512>  dShort, dShortR;
    DCT4Ortho<384>  d256, d256R;
    struct IfcAtomBand { int n=0; std::vector<float> a,a2; };
    std::array<IfcAtomBand,lx_sfc::SFC_BANDS> ifcL{},ifc1024{},ifcS{},ifc256{};

    EncoderStaticTables(){
        eL.init(SR);e1024.init(SR);eS.init(SR);e256.init(SR);
        fL.init(SR,eL,100.0);f1024.init(SR,e1024,100.0);fS.init(SR,eS,100.0);f256.init(SR,e256,100.0);
        // FEAR-v4's fixed physical-Hz lattice is the 512-domain lattice.
        fear4.init(eS);
        // IFC atoms depend only on codec geometry.  Build all 4x16 normalized
        // Gabor vectors once, outside the song hot path.
        auto build_ifc=[&](const auto& e,auto& bank){
            for(int j=0;j<lx_sfc::SFC_BANDS;++j){
                const int b=lx_sfc::SFC_FIRST_BAND+j,n=e.edge[b+1]-e.edge[b];
                auto& q=bank[j]; q.n=n; const size_t total=size_t(lx_sfc::SFC_IFC_FILTERS*lx_sfc::SFC_IFC_POSITIONS*n); q.a.resize(total); q.a2.resize(total);
                for(int m=0;m<lx_sfc::SFC_IFC_FILTERS;++m)for(int pp=0;pp<lx_sfc::SFC_IFC_POSITIONS;++pp){
                    float* dst=q.a.data()+size_t(m*lx_sfc::SFC_IFC_POSITIONS+pp)*size_t(n);
                    double ee=std::numeric_limits<double>::min();
                    for(int i=0;i<n;++i){const float v=lx_sfc::ifc_gabor_sample(uint8_t(m),uint8_t(pp),i,n);dst[i]=v;ee+=double(v)*v;}
                    const float inv=float(1.0/std::sqrt(ee)); const size_t off=size_t(m*lx_sfc::SFC_IFC_POSITIONS+pp)*size_t(n); for(int i=0;i<n;++i){dst[i]*=inv;q.a2[off+size_t(i)]=dst[i]*dst[i];}
                }
            }
        };
        build_ifc(eL,ifcL); build_ifc(e1024,ifc1024); build_ifc(eS,ifcS); build_ifc(e256,ifc256);
    }
};

// Product tables/twiddles are codec constants. Construct them lazily so a
// runtime-dispatch build initializes only the selected scalar/AVX2 variant.
static EncoderStaticTables& encoder_tables(){
    static EncoderStaticTables tables;
    return tables;
}


static inline double tfbr_mu_ref(double kbps){
    // T-FBR normative complexity reference, calibrated on the 11-track
    // development corpus. Values are the zero-mean constrained KKT reference
    // for a +/-2% track-rate envelope. Linear interpolation is sufficient
    // because mu(R) is smooth and monotone over the product range.
    static constexpr std::array<double,10> R={64,96,128,160,192,256,320,384,448,512};
    static constexpr std::array<double,10> M={
        9.121585814697212,7.622216791460730,6.610964899590472,5.811401742596978,
        5.126909584709582,3.898751362438007,2.725524764123036,1.585461695814081,
        0.457380379289331,-0.670608995121929};
    if(kbps<=R.front()) return M.front()+(kbps-R.front())*(M[1]-M[0])/(R[1]-R[0]);
    if(kbps>=R.back()) return M.back()+(kbps-R.back())*(M[9]-M[8])/(R[9]-R[8]);
    size_t i=0; while(i+1<R.size() && kbps>R[i+1]) ++i;
    const double t=(kbps-R[i])/(R[i+1]-R[i]);
    return M[i]+t*(M[i+1]-M[i]);
}

// Encoder hot path is split textually to keep translation-unit source modules readable.
// These .inl files are included at namespace scope; preprocessing reconstructs the
// exact original encode() definition, so there is no ABI/runtime/inlining change.
#include "encode_core_part1.inl"
#include "encode_core_part2.inl"

} // namespace

#if defined(LXVOXEL_ENCODER_VARIANT)
extern "C" int LXVOXEL_ENCODER_ENTRY(int argc,char**argv){
#else
static int main_impl(int argc,char**argv){
#endif
    // Deliberately do not pin to CPU0: on VMs/shared hosts CPU0 carries disproportionate scheduler/IRQ load.
    if(argc>1 && (std::strcmp(argv[1],"--help")==0 || std::strcmp(argv[1],"-h")==0)){
        std::cout
            << "LX-VOXEL encoder\n"
            << "usage:\n"
            << "  lxvoxel-enc INPUT.flac --kbps RATE [OUTPUT.vox]\n"
            << "  lxvoxel-enc INPUT.flac --kbps RATE -o OUTPUT.vox [--no-metadata]\n"
            << "  lxvoxel-enc INPUT.f32 --kbps RATE [--fast-fbr|--fbr] [RECON.f32|-] [OUTPUT.vox]\n"
            << "  lxvoxel-enc INPUT.f32 BPC [RECON.f32|-] [OUTPUT.vox]\n\n"
            << "FLAC input is decoded to 48-kHz stereo float PCM. By default, title, artist,\n"
            << "album, album artist, genre, date, track/disc fields, composer, comments,\n"
            << "copyright, encoder and embedded cover art are copied into the .vox trailer.\n\n"
            << "output options:\n"
            << "  -o, --output FILE   output .vox path (FLAC default: INPUT.vox)\n"
            << "  --recon FILE        optional reconstructed stereo float32 output\n"
            << "  --no-metadata       do not import FLAC tags or cover art\n\n"
            << "runtime dispatch:\n"
            << "  --cpu-info           print the selected SSE4.1, AVX2+FMA, or NEON encoder path\n"
            << "  LXVOXEL_FORCE_BASELINE=1 forces the x86 SSE4.1 baseline for diagnostics\n\n"
            << "rate-control modes:\n"
            << "  --fast-fbr   default: one-pass global noncausal perceptual FBR; no startup pilot or whole-track probe\n"
            << "  --tfbr       constrained track-level Fast-FBR; default envelope +/-2% of nominal rate\n"
            << "  --tfbr-span F  T-FBR fractional envelope (e.g. 0.02 = +/-2%)\n"
            << "  --fbr        luxury mode: outer whole-track rate probes around the same noncausal core\n  --exact-fbr  compatibility alias for --fbr\n\n"
            << "Fast FBR is the normative product/default path. FBR is opt-in; causal FBR is reserved for streaming.\n";
        return 0;
    }

    if(argc<2) throw std::runtime_error("missing input file; use --help for syntax");
    const std::filesystem::path input_path=path_from_utf8(argv[1]);
    const bool flac_input=lxio::is_flac_path(input_path);
    float bpc=14.0f,target_kbps=0.0f;
    int argi=2;
    if(argc>3 && std::strcmp(argv[2],"--kbps")==0){ target_kbps=std::stof(argv[3]); argi=4; }
    else if(argc>2 && argv[2][0]!='-'){ bpc=std::stof(argv[2]); argi=3; }

    // Fast FBR is intentionally the product default.  Exact FBR is an explicit
    // luxury option for offline encodes where tighter final-rate convergence is
    // worth multiple whole-track probes.  CLI flags override compatibility env.
    bool fast_fbr=true;
    if(const char* q=::lxvoxel_release_env("LX_R81_ONEPASS_RATE")) fast_fbr=std::atoi(q)!=0;
    if(const char* m=::lxvoxel_release_env("LX_FBR_MODE")){
        if(std::strcmp(m,"fbr")==0 || std::strcmp(m,"exact")==0) fast_fbr=false;
        else if(std::strcmp(m,"fast")==0) fast_fbr=true;
    }
    bool tfbr=false;
    bool import_metadata=true;
    std::optional<std::filesystem::path> explicit_output;
    std::optional<std::filesystem::path> explicit_recon;
    std::vector<const char*> positional;
    for(int i=argi;i<argc;++i){
        if(std::strcmp(argv[i],"--fast-fbr")==0) fast_fbr=true;
        else if(std::strcmp(argv[i],"--tfbr")==0){ fast_fbr=true; tfbr=true; }
        else if(std::strcmp(argv[i],"--tfbr-span")==0 && i+1<argc){ fast_fbr=true; tfbr=true; set_environment("LX_TFBR_SPAN",argv[++i]); }
        else if(std::strcmp(argv[i],"--fbr")==0 || std::strcmp(argv[i],"--exact-fbr")==0) fast_fbr=false;
        else if((std::strcmp(argv[i],"-o")==0 || std::strcmp(argv[i],"--output")==0) && i+1<argc)
            explicit_output=path_from_utf8(argv[++i]);
        else if(std::strcmp(argv[i],"--recon")==0 && i+1<argc)
            explicit_recon=path_from_utf8(argv[++i]);
        else if(std::strcmp(argv[i],"--no-metadata")==0) import_metadata=false;
        else if(argv[i][0]=='-') throw std::runtime_error(std::string("unknown option: ")+argv[i]);
        else positional.push_back(argv[i]);
    }
    set_environment("LX_TFBR",tfbr?"1":"0");
    // The encoder core reads this historical knob on each encode call.  Publish
    // the selected product mode once so Fast/Exact semantics are identical for
    // CLI, future GUI front-ends and existing automation.
    set_environment("LX_R81_ONEPASS_RATE", fast_fbr?"1":"0");
    // File encoding default: global noncausal FBR.  The legacy causal controller
    // remains available for streaming/research with LX_FBR_NC_MODE=0.
    if(!::lxvoxel_release_env("LX_FBR_NC_MODE")) set_environment("LX_FBR_NC_MODE","2");

    std::optional<std::filesystem::path> recon_path=explicit_recon;
    std::optional<std::filesystem::path> vox_path=explicit_output;
    if(flac_input){
        if(positional.size()==1 && !vox_path) vox_path=path_from_utf8(positional[0]);
        else if(positional.size()>=2){
            if(!recon_path && std::strcmp(positional[0],"-")!=0)
                recon_path=path_from_utf8(positional[0]);
            if(!vox_path) vox_path=path_from_utf8(positional[1]);
        }
        if(!vox_path){ vox_path=input_path; vox_path->replace_extension(".vox"); }
    }else{
        if(!recon_path && positional.size()>0 && std::strcmp(positional[0],"-")!=0)
            recon_path=path_from_utf8(positional[0]);
        if(!vox_path && positional.size()>1)
            vox_path=path_from_utf8(positional[1]);
    }
    if(positional.size()>2) throw std::runtime_error("too many positional output paths");

    auto io0=Clock::now();
    auto input=lxio::read_audio_input(input_path);
    const size_t nf=input.interleaved.size();
    lxvoxel::metadata::Metadata imported_metadata;
    if(flac_input && import_metadata) imported_metadata=lxvoxel::metadata::read_flac(input_path);
    const size_t ns=nf/2;double dur=double(ns)/48000.0;
    const bool pitch_enabled=[](){const char* q=::lxvoxel_release_env("LX_PITCH");return !q || std::atoi(q)!=0;}();
    lx_pitch::Schedule pitch_schedule;
    std::vector<uint8_t> pitch_payload;
    std::vector<float> pitch_pcm;
    const float* p=input.interleaved.data();
    if(pitch_enabled){
        // R336: pitch analysis and prefilter are one permanent encoder path.
        pitch_schedule=lx_pitch::analyse(p,ns,&pitch_pcm);
        pitch_payload=lx_pitch::encode_payload(pitch_schedule);
        p=pitch_pcm.data();
    }

    if(const char* sw=::lxvoxel_release_env("LX_TR_SWEEP")){
        if(std::atoi(sw)!=0){
            for(int cfg=0;cfg<=100;++cfg){
                char zbuf[16]; std::snprintf(zbuf,sizeof(zbuf),"%d",cfg); set_environment("LX_TR_LOCAL",zbuf);
                auto a0=Clock::now(); vox_beta::Writer vw0(ns,0.0f,0); if(pitch_enabled) vw0.set_pitch_payload(pitch_payload);
                auto rr=encode(p,ns,bpc,nullptr,&vw0,0.0f); const uint64_t vb=vw0.write("/dev/null"); auto a1=Clock::now();
                const size_t nlong=rr.frames-rr.b1024-rr.b512-rr.b256; ErbLayoutN<K_LONG> el;el.init(SR);ErbLayoutN<K_1024> e1;e1.init(SR);ErbLayoutN<K_SHORT> e2;e2.init(SR);ErbLayoutN<K_256> e3;e3.init(SR);
                const double dims=2.0*(double(nlong)*el.coded_bins()+double(rr.b1024)*e1.coded_bins()+double(rr.b512)*e2.coded_bins()+double(rr.b256)*e3.coded_bins());
                const double sec=std::chrono::duration<double>(a1-a0).count();
                std::cout<<"cfg="<<cfg<<" vox_kbps="<<(8.0*double(vb)/dur/1000.0)<<" mse="<<(rr.mse/dims)<<" snr="<<(10*std::log10(rr.energy/rr.mse))<<" RT="<<(dur/sec)<<"\n";
            }
            return 0;
        }
    }
    auto io1=Clock::now();
    std::vector<float> recon;
    const bool want_recon=recon_path.has_value();
    const double bd=std::sqrt(12.0/double(K_LONG))*std::exp2(-double(bpc));
    float encode_target_kbps=target_kbps;
    if(const char* q=::lxvoxel_release_env("LX_FBR_ENCODE_TARGET")) encode_target_kbps=float(std::atof(q));
    double probe0_kbps=0.0,probe1_kbps=0.0;
    auto t0=Clock::now();

    // FBR luxury mode: solve the outer total-.vox rate fixed point with
    // whole-track probes.  Fast FBR skips this block entirely and performs only
    // the final encode plus the tiny local SF0 pilot below.
    if(target_kbps>0.0f && !fast_fbr){
        struct Probe{double rate;bool saturated;};
        auto probe_rate=[&](double t)->Probe{
            // Outer total-rate probes stay free of the local SF0 pilot.
            // Otherwise the fixed-point solver and the startup Newton step form
            // a coupled feedback loop. Calibrate song-scale qbase first; apply
            // the local pilot only once in the final encode.
            vox_beta::Writer probe(ns,float(bd),0); if(pitch_enabled) probe.set_pitch_payload(pitch_payload);
            const auto pr=encode(p,ns,bpc,nullptr,&probe,float(t),target_kbps);
            const uint64_t pb=probe.write("/dev/null");
            return {8.0*double(pb)/dur/1000.0,pr.rate_saturated};
        };
        const double t0=double(target_kbps);
        const Probe p0=probe_rate(t0); probe0_kbps=p0.rate;
        if(p0.saturated && p0.rate<=double(target_kbps)){
            encode_target_kbps=float(t0);
        }else{
            const double tiny=std::numeric_limits<double>::min();
            const double t1=std::max(tiny,t0*double(target_kbps)/std::max(p0.rate,tiny));
            const Probe p1=probe_rate(t1); probe1_kbps=p1.rate;
            double tf=(std::abs(p1.rate-double(target_kbps))<std::abs(p0.rate-double(target_kbps)))?t1:t0;
            if(!(p1.saturated && p1.rate<=double(target_kbps)) && p1.rate!=p0.rate && t1!=t0){
                const double slope=(p1.rate-p0.rate)/(t1-t0);
                if(slope>0.0){const double ts=t1+(double(target_kbps)-p1.rate)/slope;if(std::isfinite(ts)&&ts>0.0)tf=ts;}
            }
            encode_target_kbps=float(tf);
        }
    }

    const int main_nc_mode=[](){const char* p=::lxvoxel_release_env("LX_FBR_NC_MODE");return p?std::atoi(p):2;}();
    const int startup_pilot_sfs=(main_nc_mode==2)?0:[&](){const char* q=::lxvoxel_release_env("LX_FBR_PILOT_SFS");return q?std::clamp(std::atoi(q),1,16):(fast_fbr?3:1);}();
    PilotStats startup_pilot{};
    if(encode_target_kbps>0.0f && startup_pilot_sfs>0){
        vox_beta::Writer pilot(ns,float(bd),0); if(pitch_enabled) pilot.set_pitch_payload(pitch_payload);
        (void)encode(p,ns,bpc,nullptr,&pilot,encode_target_kbps,target_kbps,size_t(startup_pilot_sfs),0.0,0.0,&startup_pilot,nullptr);
    }
    if(const char* dbg=::lxvoxel_release_env("LX_DEBUG_PILOT")){ if(std::atoi(dbg)!=0){ double pp=0,pe=0,po=0; for(double x:startup_pilot.packet_bits)pp+=x; for(double x:startup_pilot.entropy_bits)pe+=x; for(double x:startup_pilot.overhead_bits)po+=x; std::cout<<"pilot_packet_bits="<<pp<<" pilot_entropy_bits="<<pe<<" pilot_overhead_bits="<<po<<" "; }}
    std::unique_ptr<vox_beta::Writer> vw; if(vox_path){vw=std::make_unique<vox_beta::Writer>(ns,float(bd),0);vw->set_metadata(imported_metadata);if(pitch_enabled)vw->set_pitch_payload(pitch_payload);}
    auto r=encode(p,ns,bpc,want_recon?&recon:nullptr,vw.get(),encode_target_kbps,target_kbps,std::numeric_limits<size_t>::max(),0.0,0.0,nullptr,&startup_pilot);auto t1=Clock::now();
    uint64_t vox_bytes=0;
    if(vw){
        vox_bytes=vw->write(*vox_path);
        notify_output_updated(*vox_path);
    }
    if(want_recon){if(pitch_enabled)lx_pitch::postfilter(recon,ns,pitch_schedule);std::ofstream fo(*recon_path,std::ios::binary);fo.write((const char*)recon.data(),std::streamsize(recon.size()*sizeof(float)));}double sec=std::chrono::duration<double>(t1-t0).count();
    const double io_ms=std::chrono::duration<double,std::milli>(io1-io0).count();
    ErbLayoutN<K_LONG> _el;_el.init(SR);ErbLayoutN<K_1024> _e1024;_e1024.init(SR);ErbLayoutN<K_SHORT> _es;_es.init(SR);ErbLayoutN<K_256> _e256;_e256.init(SR);
    const size_t nlong=r.frames-r.b1024-r.b512-r.b256;
    const double _dims=2.0*(double(nlong)*_el.coded_bins()+double(r.b1024)*_e1024.coded_bins()+double(r.b512)*_es.coded_bins()+double(r.b256)*_e256.coded_bins());
    const char* cpu_path=std::getenv("LXVOXEL_CPU_PATH");
    std::cout<<std::setprecision(10)<<"cpu_path="<<(cpu_path?cpu_path:"single")<<" fbr_mode="<<(fast_fbr?"fast":"fbr")<<" nc_mode="<<main_nc_mode<<" pilot_sfs="<<startup_pilot_sfs<<" bpc="<<r.effective_bpc<<" target_kbps="<<target_kbps<<" tfbr_target_kbps="<<r.tfbr_target_kbps<<" encode_target_kbps="<<encode_target_kbps<<" probe0_kbps="<<probe0_kbps<<" probe1_kbps="<<probe1_kbps<<" kbps="<<8.0*r.bytes/dur/1000.0<<" io_ms="<<io_ms<<" encode_s="<<sec<<" RT="<<dur/sec
             <<" frames="<<r.frames<<" block1024="<<nlong<<" block768="<<r.b1024<<" block512="<<r.b512<<" block384="<<r.b256
             <<" header_bits="<<(4*r.frames+r.is_side_bits)<<" MS="<<r.ms<<" LR="<<(r.frames-r.ms)
             <<" IS_frames="<<r.is_frames<<" IS_bands="<<r.is_bands<<" IS_side_bits="<<r.is_side_bits
             <<" SFC_frames="<<r.sfc_frames<<" SFC_payload_bits="<<r.sfc_bits<<" TNS_frames=0 TNS_side_bits=0 PREC_bins="<<r.prec_bins<<" PREC_coords="<<r.prec_coords
             <<" spectral_MSE="<<(r.mse/_dims)<<" spectral_SNR_dB="<<10*std::log10(r.energy/r.mse)
             <<" rans_ns_per_symbol="<<(r.entropy_symbols?r.entropy_ns/double(r.entropy_symbols):0.0)
             <<" rans_symbols="<<r.entropy_symbols<<" pitch_side_bits="<<(pitch_enabled?8ull*pitch_payload.size():0ull)
             <<(vox_bytes?" vox_kbps=":"")<<(vox_bytes?8.0*double(vox_bytes)/dur/1000.0:0.0)<<"\n";
    return 0;
}

#if !defined(LXVOXEL_ENCODER_VARIANT) && defined(_WIN32)
int wmain(int argc,wchar_t**wide_argv){
    std::vector<std::string> utf8_args;
    std::vector<char*> argv;
    utf8_args.reserve(size_t(argc)); argv.reserve(size_t(argc));
    for(int i=0;i<argc;++i){
        const int bytes=WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,wide_argv[i],-1,nullptr,0,nullptr,nullptr);
        if(bytes<=0){std::cerr<<"error: invalid Unicode command line\n";return 2;}
        std::string value(size_t(bytes),'\0');
        if(WideCharToMultiByte(CP_UTF8,WC_ERR_INVALID_CHARS,wide_argv[i],-1,value.data(),bytes,nullptr,nullptr)<=0){
            std::cerr<<"error: cannot convert Unicode command line\n";return 2;
        }
        value.resize(size_t(bytes-1));
        utf8_args.push_back(std::move(value));
    }
    for(auto& value:utf8_args)argv.push_back(value.data());
#elif !defined(LXVOXEL_ENCODER_VARIANT)
int main(int argc,char**argv){
#endif
#if !defined(LXVOXEL_ENCODER_VARIANT)
    try {
#if defined(_WIN32)
        return main_impl(argc,argv.data());
#else
        return main_impl(argc,argv);
#endif
    }
    catch(const std::exception& e){ std::cerr<<"error: "<<e.what()<<"\n"; return 2; }
    catch(...){ std::cerr<<"error: unknown fatal exception\n"; return 3; }
}
#endif
