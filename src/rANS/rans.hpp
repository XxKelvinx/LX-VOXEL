#pragma once

#if defined(_MSC_VER)
#define LX_NOINLINE __declspec(noinline)
#else
#define LX_NOINLINE __attribute__((noinline))
#endif
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace lx_rans4_fast {
constexpr unsigned SCALE_BITS=12;
constexpr uint32_t M=1u<<SCALE_BITS;
constexpr uint32_t MASK=M-1;
constexpr uint32_t L=1u<<16;

struct EncMeta {
    uint32_t rcp;
    // bias 13 bits | complement freq 12 bits | reciprocal shift 5 bits.
    uint32_t p;
    inline uint32_t bias() const {return p&8191u;}
    inline uint32_t cmpl() const {return (p>>13)&4095u;}
    inline uint32_t freq() const {return M-cmpl();}
    inline uint32_t shift() const {return p>>25;}
};
static_assert(sizeof(EncMeta)==8);


template<int A>
struct Model {
    alignas(64) std::array<EncMeta,A> enc{};
    alignas(64) std::array<uint8_t,M> dec{};
    std::array<uint16_t,A> freq{},start{};

    void build(const std::array<uint32_t,A>& hist){
        // O(A) cumulative apportionment.  Using integer mass 2*h+1 is the
        // exact +0.5 pseudocount model without sorting fractional remainders.
        // The cumulative quantizer guarantees freq>=1 and sum(freq)==M.
        uint64_t total_mass=0;
        for(int s=0;s<A;++s)total_mass+=2ull*hist[s]+1ull;
        constexpr uint64_t rem=uint64_t(M)-A;
        uint64_t cum=0,prev_q=0;
        for(int s=0;s<A;++s){
            cum+=2ull*hist[s]+1ull;
            const uint64_t q=(cum*rem)/total_mass;
            freq[s]=uint16_t(1u+uint32_t(q-prev_q));
            prev_q=q;
        }
        uint32_t c=0;
        for(int s=0;s<A;++s){
            start[s]=uint16_t(c);const uint32_t f=freq[s];uint32_t rcp,sh,bias;
            if(f<2){rcp=0xffffffffu;sh=0;bias=c+M-1;}
            else{uint32_t ceil_log2=0;while(f>(1u<<ceil_log2))++ceil_log2;rcp=uint32_t(((uint64_t(1)<<(ceil_log2+31))+f-1)/f);sh=ceil_log2-1;bias=c;}
            const uint32_t cmpl=M-f;
            enc[s]={rcp,bias|(cmpl<<13)|(sh<<25)};
            for(uint32_t j=0;j<f;++j)dec[c+j]=uint8_t(s);
            c+=f;
        }
    }

    void build_decoder_only(const std::array<uint32_t,A>& hist){
        // Decoder needs only the exact normalized frequencies, starts and the
        // slot->symbol table.  Reciprocal encoder metadata is deliberately not
        // constructed here; it is never read by rANS decode.
        uint64_t total_mass=0;
        for(int s=0;s<A;++s)total_mass+=2ull*hist[s]+1ull;
        constexpr uint64_t rem=uint64_t(M)-A;
        uint64_t cum=0,prev_q=0;
        for(int s=0;s<A;++s){
            cum+=2ull*hist[s]+1ull;
            const uint64_t q=(cum*rem)/total_mass;
            freq[s]=uint16_t(1u+uint32_t(q-prev_q));
            prev_q=q;
        }
        uint32_t c=0;
        for(int s=0;s<A;++s){
            start[s]=uint16_t(c);
            const uint32_t f=freq[s];
            std::memset(dec.data()+c,s,f);
            c+=f;
        }
    }

    void build_encoder_only(const std::array<uint32_t,A>& hist){
        // O(A) cumulative apportionment.  Using integer mass 2*h+1 is the
        // exact +0.5 pseudocount model without sorting fractional remainders.
        // The cumulative quantizer guarantees freq>=1 and sum(freq)==M.
        uint64_t total_mass=0;
        for(int s=0;s<A;++s)total_mass+=2ull*hist[s]+1ull;
        constexpr uint64_t rem=uint64_t(M)-A;
        uint64_t cum=0,prev_q=0;
        for(int s=0;s<A;++s){
            cum+=2ull*hist[s]+1ull;
            const uint64_t q=(cum*rem)/total_mass;
            freq[s]=uint16_t(1u+uint32_t(q-prev_q));
            prev_q=q;
        }
        uint32_t c=0;
        for(int s=0;s<A;++s){
            start[s]=uint16_t(c);const uint32_t f=freq[s];uint32_t rcp,sh,bias;
            if(f<2){rcp=0xffffffffu;sh=0;bias=c+M-1;}
            else{uint32_t ceil_log2=0;while(f>(1u<<ceil_log2))++ceil_log2;rcp=uint32_t(((uint64_t(1)<<(ceil_log2+31))+f-1)/f);sh=ceil_log2-1;bias=c;}
            enc[s]={rcp,bias|((M-f)<<13)|(sh<<25)};c+=f;
        }
    }
};

struct Hist {
    std::array<uint32_t,129> qA{},qB{};
    std::array<uint32_t,256> residual{};
};
struct Tables {
    Model<129> qA,qB;Model<256> residual;
    Tables(){Hist h;h.qA.fill(1);h.qB.fill(1);h.residual.fill(1);rebuild(h);}
    void rebuild(const Hist&h){qA.build(h.qA);qB.build(h.qB);residual.build(h.residual);}
    void rebuild_encoder_only(const Hist&h){qA.build_encoder_only(h.qA);qB.build_encoder_only(h.qB);residual.build_encoder_only(h.residual);}
};

static inline uint32_t update(uint32_t x,const EncMeta&m){
    const uint32_t q=uint32_t((uint64_t(x)*m.rcp)>>32)>>m.shift();
    return x+m.bias()+q*m.cmpl();
}

struct Scratch {std::vector<uint8_t> buf;Scratch(){buf.resize(1u<<18);}};
struct Encoded {const uint8_t* data=nullptr;size_t bytes=0,symbols=0;};

// 4 independent states, 16-bit renormalization. For L=2^16,M=2^12,
// x_max=f*2^20 and one renorm word is sufficient for every state update.
template<int A>
static LX_NOINLINE Encoded encode4(const uint8_t*sym,size_t n,const Model<A>&model,Scratch&s){
    uint8_t*const base=s.buf.data();
    uint16_t* wptr=reinterpret_cast<uint16_t*>(base+s.buf.size());
    uint32_t x0=L,x1=L,x2=L,x3=L;size_t i=n;
#define LX_RANS_STEP(X,SYM) do { \
    const EncMeta&m=model.enc[(SYM)]; const uint32_t xmax=m.freq()<<20; \
    if((X)>=xmax){*--wptr=uint16_t(X);(X)>>=16;} \
    (X)=update((X),m); \
} while(0)
    while(i&3u){const size_t j=--i;switch(j&3u){case 0:LX_RANS_STEP(x0,sym[j]);break;case 1:LX_RANS_STEP(x1,sym[j]);break;case 2:LX_RANS_STEP(x2,sym[j]);break;default:LX_RANS_STEP(x3,sym[j]);break;}}
    while(i){i-=4;LX_RANS_STEP(x3,sym[i+3]);LX_RANS_STEP(x2,sym[i+2]);LX_RANS_STEP(x1,sym[i+1]);LX_RANS_STEP(x0,sym[i]);}
#undef LX_RANS_STEP
    uint8_t*ptr=reinterpret_cast<uint8_t*>(wptr)-16;
    std::memcpy(ptr,&x0,4);std::memcpy(ptr+4,&x1,4);std::memcpy(ptr+8,&x2,4);std::memcpy(ptr+12,&x3,4);
    return {ptr,size_t(base+s.buf.size()-ptr),n};
}


template<int A>
static inline bool decode4(const uint8_t*data,size_t bytes,uint8_t*out,size_t n,const Model<A>&model){
    if(bytes<16)return false;uint32_t x[4];std::memcpy(x,data,16);const uint8_t*ptr=data+16,*end=data+bytes;
    for(size_t i=0;i<n;++i){const unsigned lane=unsigned(i)&3u;uint32_t v=x[lane];const uint32_t slot=v&MASK;const uint8_t sy=model.dec[slot];out[i]=sy;v=uint32_t(model.freq[sy])*(v>>SCALE_BITS)+slot-model.start[sy];if(v<L){if(ptr+2>end)return false;uint16_t w;std::memcpy(&w,ptr,2);ptr+=2;v=(v<<16)|w;}x[lane]=v;}
    return ptr==end;
}

// No output stores, used to measure the arithmetic state kernel itself.


#if defined(__AVX2__)
#endif


#if defined(__AVX512F__)
template<int A>
struct PairMeta {
    alignas(64) std::array<EncMeta,2*A> enc{};
    void build(const Model<A>&a,const Model<A>&b){for(int i=0;i<A;++i){enc[i]=a.enc[i];enc[A+i]=b.enc[i];}}
};

#endif


#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512VBMI2__)
template<int A>
struct PairEncoded {Encoded a,b;};


#endif

} // namespace
