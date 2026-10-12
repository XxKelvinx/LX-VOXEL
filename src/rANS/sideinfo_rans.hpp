#pragma once
#include <array>
#include <cstdint>
#include <vector>
#include "rANS/rans.hpp"
#include "synthesis/gain_predictor.hpp"

// v0.36 lossless SFC/AC side-information residual coder.
// All predicted scalar residuals share one four-lane rANS stream per superframe.
// The model family is the exact rANS analogue of a Rice scale: probability mass
// falls by 1/2 every 2^k ZigZag symbols. The encoder exhaustively chooses k=0..7
// by actual encoded byte count; only k is transmitted, so no histogram overhead.
namespace lx_sideinfo_rans {
constexpr uint8_t ESC = 255;
constexpr unsigned ALPHA = 256;

inline std::array<uint32_t,ALPHA> model_hist(unsigned k){
    std::array<uint32_t,ALPHA> h{};
    for(unsigned s=0;s<ALPHA;++s){
        const unsigned bucket=std::min(15u, s >> k);
        h[s]=1u << (15u-bucket);
    }
    return h;
}
inline lx_rans4_fast::Model<ALPHA> model(unsigned k){
    lx_rans4_fast::Model<ALPHA> m; m.build(model_hist(k)); return m;
}
inline uint8_t symbol(int q,int p){
    if(p<0) return ESC; // bootstrap is raw and emits no rANS symbol; caller ignores this.
    const unsigned z=lx_gainpred::zigzag(q-p);
    return uint8_t(z<ESC?z:ESC);
}

template<class W> inline unsigned put_raw_if_needed(W& w,int q,int p,unsigned bits){
    if(p<0){w.put(unsigned(q),bits);return bits;}
    if(lx_gainpred::zigzag(q-p)>=ESC){w.put(unsigned(q),bits);return bits;}
    return 0;
}

template<class R> inline bool get(R& r,const std::vector<uint8_t>& syms,size_t& si,
                                  int p,unsigned bits,uint8_t& q){
    uint64_t v=0;
    if(p<0){if(!r.get(bits,v))return false;q=uint8_t(v);return true;}
    if(si>=syms.size())return false;
    const uint8_t s=syms[si++];
    if(s==ESC){if(!r.get(bits,v))return false;q=uint8_t(v);return true;}
    const int d=p+lx_gainpred::unzigzag(unsigned(s));
    if(d<0 || d>=int(1u<<bits))return false;
    q=uint8_t(d);return true;
}

template<int A> inline lx_rans4_fast::Encoded encode_prefix_then_main(const uint8_t* pre,size_t np,const lx_rans4_fast::Model<A>& pm,
                                                                     const uint8_t* main,size_t nm,const lx_rans4_fast::Model<A>& mm,
                                                                     lx_rans4_fast::Scratch& sc){
    uint8_t*const base=sc.buf.data(); uint16_t* wptr=reinterpret_cast<uint16_t*>(base+sc.buf.size());
    uint32_t x[4]={lx_rans4_fast::L,lx_rans4_fast::L,lx_rans4_fast::L,lx_rans4_fast::L};
    const size_t n=np+nm;
    auto step=[&](size_t j,uint8_t sy,const lx_rans4_fast::Model<A>& m){
        uint32_t& X=x[j&3u]; const auto& em=m.enc[sy]; const uint32_t xmax=em.freq()<<20;
        if(X>=xmax){*--wptr=uint16_t(X);X>>=16;} X=lx_rans4_fast::update(X,em);
    };
    for(size_t j=n;j-->0;){ if(j<np) step(j,pre[j],pm); else step(j,main[j-np],mm); }
    uint8_t*ptr=reinterpret_cast<uint8_t*>(wptr)-16;
    std::memcpy(ptr,x,16); return {ptr,size_t(base+sc.buf.size()-ptr),n};
}

}
