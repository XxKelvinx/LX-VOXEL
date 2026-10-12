#pragma once
#include <algorithm>
#include <array>
#include <cstdint>

// v0.35: shared deterministic gain-residual syntax for SFC and anti-collapse.
// The state is reset at each superframe and is independently coded for SFC/AC.
// A Rice escape after GAIN_CODE_BITS quotient bits carries an unmodified gain
// code, bounding every residual even on completely discontinuous audio.
namespace lx_gainpred {
constexpr unsigned GAIN_CODE_BITS=8;
constexpr unsigned MAX_QUOTIENT=GAIN_CODE_BITS;
constexpr unsigned MAX_RICE_K=7;

struct State {
    std::array<int,8> last{}, prior{}, current{};
    int frame=-1;
    State(){last.fill(-1);prior.fill(-1);current.fill(-1);}
    void begin_frame(int f){
        if(frame==f)return;
        prior=last; current.fill(-1);frame=f;
    }
    int predict(int j)const{
        const int t=prior[j];
        const int a=j?current[j-1]:-1;
        const int c=j?prior[j-1]:-1;
        if(t>=0 && a>=0 && c>=0){
            // JPEG-LS median edge predictor, independent of channel amplitude.
            const int planar=t+a-c;
            return std::clamp(planar,std::min(t,a),std::max(t,a));
        }
        if(t>=0)return t;
        if(a>=0)return a;
        return -1; // bootstrap: raw 8-bit symbol, no guessed initial gain.
    }
    void observe(int j,int q){current[j]=q;last[j]=q;}
};
inline unsigned zigzag(int v){return v>=0?unsigned(2*v):unsigned(-2*v-1);}
inline int unzigzag(unsigned v){return (v&1u)?-int((v+1u)>>1):int(v>>1);}
inline unsigned bit_cost(int q,int p,unsigned k){
    if(p<0)return GAIN_CODE_BITS;
    const unsigned quot=zigzag(q-p)>>k;
    return quot<MAX_QUOTIENT?(quot+1u+k):(MAX_QUOTIENT+GAIN_CODE_BITS);
}

template<typename W> inline unsigned put(W& w,int q,int p,unsigned k){
    if(p<0){w.put(unsigned(q),GAIN_CODE_BITS);return GAIN_CODE_BITS;}
    const unsigned z=zigzag(q-p), quot=z>>k;
    if(quot>=MAX_QUOTIENT){
        w.put((1u<<MAX_QUOTIENT)-1u,MAX_QUOTIENT);
        w.put(unsigned(q),GAIN_CODE_BITS);
        return MAX_QUOTIENT+GAIN_CODE_BITS;
    }
    if(quot)w.put((1u<<quot)-1u,quot);
    w.put(0u,1);
    if(k)w.put(z&((1u<<k)-1u),k);
    return 1u+quot+k;
}
template<typename R> inline bool get(R& r,int p,unsigned k,uint8_t& q){
    uint64_t v=0;
    if(p<0){if(!r.get(GAIN_CODE_BITS,v))return false;q=uint8_t(v);return true;}
    unsigned n=0;
    for(;n<MAX_QUOTIENT;++n){
        if(!r.get(1,v))return false;
        if(!v)break;
    }
    if(n==MAX_QUOTIENT){
        if(!r.get(GAIN_CODE_BITS,v))return false;
        q=uint8_t(v);return true;
    }
    uint64_t low=0;
    if(k && !r.get(k,low))return false;
    const unsigned z=(n<<k)|unsigned(low);
    const int decoded=p+unzigzag(z);
    if(decoded<0||decoded>=int(1u<<GAIN_CODE_BITS))return false;
    q=uint8_t(decoded);
    return true;
}
}
