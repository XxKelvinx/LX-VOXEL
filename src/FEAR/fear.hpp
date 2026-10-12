#pragma once
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

// LX-FEAR Core v4: fixed 46.875-Hz analysis lattice derived directly from the
// codec MDCT.  Long (2048) blocks collapse four adjacent coefficients into one
// cell; 512 blocks already live on exactly this lattice; 1024 groups pairs and
// 256 splits each physical bin over two cells.  Thus the
// psychoacoustic analyser adds no second FFT/RFFT.
//
// All signal-value operations in the hot feature loops are continuous.  Signal
// dependent branches are deliberately absent.  The only branches are geometry /
// ISA selection known before analysing a block.
namespace lx_fear_v4 {

constexpr int GRID = 427;             // ceil(20 kHz / 46.875 Hz)
constexpr int FAMILIES = 6;           // geometry, harmonic, periodic, masking, temporal, stereo
constexpr float EPS = 1.0e-20f;

struct RatioMap {
    alignas(64) std::array<int32_t, GRID> idx{};
    alignas(64) std::array<float, GRID> valid{};
};

struct Static {
    std::array<uint8_t, GRID> erb{};
    std::array<float, 48> inv_count{};
    std::array<float, 48> norm_freq{};
    std::array<RatioMap, 5> harmonic{};   // 1/3, 1/2, 2, 3, 4
    std::array<RatioMap, 5> periodic{};   // 1,2,3,4 semitones + perfect fifth
    std::array<float, GRID> harmonic_count{}, periodic_count{};

    template<class Layout>
    void init(const Layout& short_layout) {
        std::array<int,48> count{};
        int b=0;
        for(int i=0;i<GRID;++i){
            while(b<47 && i>=short_layout.edge[b+1]) ++b;
            erb[i]=uint8_t(b); count[b]++;
        }
        for(int k=0;k<48;++k) inv_count[k]=count[k]?1.0f/float(count[k]):0.0f;
        std::array<float,48> fsum{};
        for(int i=0;i<GRID;++i) fsum[erb[i]] += (float(i)+0.5f)/float(GRID);
        for(int k=0;k<48;++k) norm_freq[k]=count[k]?fsum[k]/float(count[k]):0.0f;

        const float hr[5]={1.0f/3.0f,0.5f,2.0f,3.0f,4.0f};
        const float pr[5]={1.0594630943592953f,1.122462048309373f,1.189207115002721f,
                           1.2599210498948732f,1.5f};
        auto build=[&](RatioMap& m,float r){
            for(int i=0;i<GRID;++i){
                const float x=(float(i)+0.5f)*r-0.5f;
                const int j=int(std::lrint(x));
                const int jc=std::clamp(j,0,GRID-1);
                m.idx[i]=jc;
                m.valid[i]=(j>=0 && j<GRID)?1.0f:0.0f;
            }
        };
        for(int q=0;q<5;++q){build(harmonic[q],hr[q]);build(periodic[q],pr[q]);}
        for(int i=0;i<GRID;++i){
            float hc=0.0f,pc=0.0f;
            for(int q=0;q<5;++q){hc+=harmonic[q].valid[i];pc+=periodic[q].valid[i];}
            harmonic_count[i]=hc; periodic_count[i]=pc;
        }
    }
};

struct State {
    alignas(64) std::array<float, GRID> prev{};
    alignas(64) std::array<float, GRID> prev2{};
    bool primed=false;

    void reset(){prev.fill(0.0f);prev2.fill(0.0f);primed=false;}
};

struct Scratch {
    alignas(64) std::array<float, GRID> a{}, l2{}, r2{}, lr{};
    // gt is reused by the wide temporal gradient. The frequency gradient itself
    // is consumed immediately when jff/jft are formed, so do not spill it.
    alignas(64) std::array<float, GRID> gt{}, jff{}, jtt{}, jft{};
    alignas(64) std::array<float, GRID> geom{}, harm{}, periodic{}, mask{}, temp{}, stereo{}; std::array<float, GRID> ridge_tonal{}, tmr{};
    alignas(64) std::array<float, GRID+1> prefix{};
};


// The product codec has exactly four MDCT lengths: 1024, 768, 512 and 384.
// Their physical-frequency projection geometry is signal independent, so build it
// once and keep the hot psychoacoustic path free of floor/ceil/min/max/divisions.
struct GridProjection {
    alignas(64) std::array<int16_t,1024> g0{},g1{};
    alignas(64) std::array<float,1024> q0{},q1{};
};

static inline GridProjection make_projection(int k){
    GridProjection p{};
    const double w=512.0/double(k);
    for(int i=0;i<k;++i){
        const double a=double(i)*w, z=double(i+1)*w;
        const int lo=std::max(0,int(std::floor(a)));
        const int hi=std::min(GRID-1,int(std::ceil(z)-1.0));
        p.g0[i]=int16_t(lo); p.g1[i]=int16_t(hi);
        const double ov0=std::max(0.0,std::min(z,double(lo+1))-std::max(a,double(lo)));
        p.q0[i]=float(ov0/w);
        if(hi!=lo){
            const double ov1=std::max(0.0,std::min(z,double(hi+1))-std::max(a,double(hi)));
            p.q1[i]=float(ov1/w);
        } else p.q1[i]=0.0f;
    }
    return p;
}

static inline const GridProjection& projection_for(int k){
    static const GridProjection p1024=make_projection(1024);
    static const GridProjection p768 =make_projection(768);
    static const GridProjection p512 =make_projection(512);
    static const GridProjection p384 =make_projection(384);
    switch(k){
        case 1024:return p1024;
        case 768: return p768;
        case 512: return p512;
        default:  return p384;
    }
}

// Build the fixed-frequency amplitude lattice directly from existing MDCT
// coefficients using the precomputed exact box-projection map.
static inline int build_grid(const float* l,const float* r,int k,int coded_bins,Scratch& s){
    // Two product geometries are exact one-cell projections:
    //   k=512  : one MDCT coefficient per 46.875-Hz cell
    //   k=1024 : two adjacent coefficients per cell
    // Their precomputed q0 is exactly 1 and q1 exactly 0 throughout the coded
    // range, so avoid the generic second zero-weight read/modify/write entirely.
    // This is a geometry specialization only; signal equations are unchanged.
    if(k==512){
        const int n=coded_bins;
        for(int i=0;i<coded_bins;++i){
            const float x=l[i], y=r[i];
            const float ex=x*x, ey=y*y, ec=x*y;
            s.l2[i]=ex; s.r2[i]=ey; s.lr[i]=ec;
            s.a[i]=std::sqrt(0.5f*(ex+ey)+EPS);
        }
        return n;
    }
    if(k==1024){
        const int n=(coded_bins+1)>>1;
        int g=0, i=0;
        for(;i+1<coded_bins;i+=2,++g){
            const float x0=l[i], y0=r[i], x1=l[i+1], y1=r[i+1];
            float el=x0*x0; el+=x1*x1;
            float er=y0*y0; er+=y1*y1;
            float ec=x0*y0; ec+=x1*y1;
            s.l2[g]=el; s.r2[g]=er; s.lr[g]=ec;
            s.a[g]=std::sqrt(0.5f*(el+er)+EPS);
        }
        if(i<coded_bins){
            const float x=l[i],y=r[i];
            const float el=x*x,er=y*y;
            s.l2[g]=el;s.r2[g]=er;s.lr[g]=x*y;
            s.a[g]=std::sqrt(0.5f*(el+er)+EPS);
        }
        return n;
    }

    const auto& p=projection_for(k);
    const int n=coded_bins>0 ? int(p.g1[coded_bins-1])+1 : 0;
    std::fill_n(s.l2.data(),n,0.0f); std::fill_n(s.r2.data(),n,0.0f);
    std::fill_n(s.lr.data(),n,0.0f);
    for(int i=0;i<coded_bins;++i){
        const float x=l[i], y=r[i];
        const float ex=x*x, ey=y*y, ec=x*y;
        const int a=int(p.g0[i]), b=int(p.g1[i]);
        const float qa=p.q0[i], qb=p.q1[i];
        s.l2[a]+=qa*ex; s.r2[a]+=qa*ey; s.lr[a]+=qa*ec;
        s.l2[b]+=qb*ex; s.r2[b]+=qb*ey; s.lr[b]+=qb*ec;
    }
    for(int g=0;g<n;++g) s.a[g]=std::sqrt(0.5f*(s.l2[g]+s.r2[g])+EPS);
    return n;
}

static inline void gradients_avx2(const float* a,const float* prev,int n,Scratch& s){
    s.gt[0]=a[0]-prev[0];
    int i=1;
#if defined(__AVX512F__)
    for(;i+16<=n-1;i+=16){
        const __m512 am=_mm512_loadu_ps(a+i-1), ap=_mm512_loadu_ps(a+i+1);
        const __m512 ac=_mm512_loadu_ps(a+i), pv=_mm512_loadu_ps(prev+i);
        const __m512 gf=_mm512_sub_ps(ap,am), gt=_mm512_sub_ps(ac,pv);
        _mm512_storeu_ps(s.gt.data()+i,gt);
        _mm512_storeu_ps(s.jff.data()+i,_mm512_mul_ps(gf,gf));
        _mm512_storeu_ps(s.jtt.data()+i,_mm512_mul_ps(gt,gt));
        _mm512_storeu_ps(s.jft.data()+i,_mm512_mul_ps(gf,gt));
    }
#elif defined(__AVX2__)
    for(;i+8<=n-1;i+=8){
        const __m256 am=_mm256_loadu_ps(a+i-1), ap=_mm256_loadu_ps(a+i+1);
        const __m256 ac=_mm256_loadu_ps(a+i), pv=_mm256_loadu_ps(prev+i);
        const __m256 gf=_mm256_sub_ps(ap,am), gt=_mm256_sub_ps(ac,pv);
        _mm256_storeu_ps(s.gt.data()+i,gt);
        _mm256_storeu_ps(s.jff.data()+i,_mm256_mul_ps(gf,gf));
        _mm256_storeu_ps(s.jtt.data()+i,_mm256_mul_ps(gt,gt));
        _mm256_storeu_ps(s.jft.data()+i,_mm256_mul_ps(gf,gt));
    }
#endif
    for(;i<n-1;++i){const float gf=a[i+1]-a[i-1],gt=a[i]-prev[i];s.gt[i]=gt;s.jff[i]=gf*gf;s.jtt[i]=gt*gt;s.jft[i]=gf*gt;}
    if(n>1){s.gt[n-1]=a[n-1]-prev[n-1];s.jff[n-1]=0.0f;s.jtt[n-1]=s.gt[n-1]*s.gt[n-1];s.jft[n-1]=0.0f;}
}

// Similarity 2ab/(a^2+b^2), bounded [0,1], continuous and scale invariant.
#if defined(__AVX2__)
static inline __m256 sim8(__m256 a,__m256 b){
    const __m256 two=_mm256_set1_ps(2.0f),eps=_mm256_set1_ps(EPS);
    return _mm256_div_ps(_mm256_mul_ps(two,_mm256_mul_ps(a,b)),
                         _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(a,a),_mm256_mul_ps(b,b)),eps));
}
#endif




static inline void analyse_features(const Static& st,State& state,Scratch& s,int n){
    const float* a=s.a.data();
    gradients_avx2(a,state.prev.data(),n,s);

    // Prefix amplitude for asymmetric masking context.  Serial dependency is
    // only 427 additions/block and is much cheaper than another transform.
    s.prefix[0]=0.0f; for(int i=0;i<n;++i)s.prefix[i+1]=s.prefix[i]+a[i];

    // Edges are copied continuously from their nearest interior cell later.
    int i=2;
#if defined(__AVX2__)
    const __m256 eps=_mm256_set1_ps(EPS),two=_mm256_set1_ps(2.0f),half=_mm256_set1_ps(0.5f);
    const __m256 c1=_mm256_set1_ps(1.0f/16.0f),c4=_mm256_set1_ps(4.0f/16.0f),c6=_mm256_set1_ps(6.0f/16.0f);
    for(;i+8<=n-2;i+=8){
        const __m256 am2=_mm256_loadu_ps(a+i-2),am1=_mm256_loadu_ps(a+i-1),ac=_mm256_loadu_ps(a+i),ap1=_mm256_loadu_ps(a+i+1),ap2=_mm256_loadu_ps(a+i+2);
        const __m256 sm3=_mm256_mul_ps(_mm256_set1_ps(0.25f),_mm256_add_ps(_mm256_add_ps(am1,ap1),_mm256_mul_ps(two,ac)));
        const __m256 sm5=_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(c1,_mm256_add_ps(am2,ap2)),_mm256_mul_ps(c4,_mm256_add_ps(am1,ap1))),_mm256_mul_ps(c6,ac));
        const __m256 dog=_mm256_div_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.0f),_mm256_sub_ps(sm3,sm5)),_mm256_add_ps(sm5,eps));
        const __m256 curv=_mm256_div_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.0f),_mm256_add_ps(_mm256_sub_ps(am1,_mm256_mul_ps(two,ac)),ap1)),_mm256_add_ps(_mm256_add_ps(am1,ap1),_mm256_add_ps(_mm256_mul_ps(two,ac),eps)));

        const __m256 jf0=_mm256_loadu_ps(s.jff.data()+i), jt0=_mm256_loadu_ps(s.jtt.data()+i), jx0=_mm256_loadu_ps(s.jft.data()+i);
        const __m256 jfm=_mm256_loadu_ps(s.jff.data()+i-1), jfp=_mm256_loadu_ps(s.jff.data()+i+1);
        const __m256 jtm=_mm256_loadu_ps(s.jtt.data()+i-1), jtp=_mm256_loadu_ps(s.jtt.data()+i+1);
        const __m256 jxm=_mm256_loadu_ps(s.jft.data()+i-1), jxp=_mm256_loadu_ps(s.jft.data()+i+1);
        const __m256 jff=_mm256_mul_ps(_mm256_set1_ps(0.25f),_mm256_add_ps(_mm256_add_ps(jfm,jfp),_mm256_mul_ps(two,jf0)));
        const __m256 jtt=_mm256_mul_ps(_mm256_set1_ps(0.25f),_mm256_add_ps(_mm256_add_ps(jtm,jtp),_mm256_mul_ps(two,jt0)));
        const __m256 jft=_mm256_mul_ps(_mm256_set1_ps(0.25f),_mm256_add_ps(_mm256_add_ps(jxm,jxp),_mm256_mul_ps(two,jx0)));
        const __m256 d=_mm256_sub_ps(jff,jtt);
        const __m256 coh=_mm256_div_ps(_mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(d,d),_mm256_mul_ps(_mm256_set1_ps(4.0f),_mm256_mul_ps(jft,jft)))),_mm256_add_ps(_mm256_add_ps(jff,jtt),eps));

        // Wide AST from +/-2 finite differences, independent of narrow tensor.
        const __m256 gw=_mm256_sub_ps(ap2,am2), tw=_mm256_loadu_ps(s.gt.data()+i);
        const __m256 wff=_mm256_mul_ps(gw,gw),wtt=_mm256_mul_ps(tw,tw),wft=_mm256_mul_ps(gw,tw),wd=_mm256_sub_ps(wff,wtt);
        const __m256 wcoh=_mm256_div_ps(_mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(wd,wd),_mm256_mul_ps(_mm256_set1_ps(4.0f),_mm256_mul_ps(wft,wft)))),_mm256_add_ps(_mm256_add_ps(wff,wtt),eps));

        // Local coefficient of variation on the same normalized 5-tap kernel.
        const __m256 q2=_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(c1,_mm256_add_ps(_mm256_mul_ps(am2,am2),_mm256_mul_ps(ap2,ap2))),_mm256_mul_ps(c4,_mm256_add_ps(_mm256_mul_ps(am1,am1),_mm256_mul_ps(ap1,ap1)))),_mm256_mul_ps(c6,_mm256_mul_ps(ac,ac)));
        const __m256 var=_mm256_max_ps(_mm256_sub_ps(q2,_mm256_mul_ps(sm5,sm5)),_mm256_setzero_ps());
        const __m256 cv=_mm256_div_ps(_mm256_sqrt_ps(var),_mm256_add_ps(sm5,eps));
        const __m256 ridge=_mm256_div_ps(_mm256_mul_ps(two,ac),_mm256_add_ps(_mm256_add_ps(ac,sm5),eps));
        const __m256 geom=_mm256_mul_ps(_mm256_set1_ps(1.0f/6.0f),_mm256_add_ps(_mm256_add_ps(coh,wcoh),_mm256_add_ps(_mm256_add_ps(dog,curv),_mm256_add_ps(cv,ridge))));
        _mm256_storeu_ps(s.geom.data()+i,geom);
        const __m256 orient=_mm256_div_ps(d,_mm256_add_ps(_mm256_add_ps(jff,jtt),eps));
        const __m256 horiz=_mm256_mul_ps(half,_mm256_add_ps(_mm256_set1_ps(1.0f),orient));
        _mm256_storeu_ps(s.ridge_tonal.data()+i,_mm256_mul_ps(_mm256_mul_ps(coh,coh),_mm256_mul_ps(horiz,horiz)));

        const __m256 pv=_mm256_loadu_ps(state.prev.data()+i),p2=_mm256_loadu_ps(state.prev2.data()+i);
        const __m256 persist=sim8(ac,pv);
        const __m256 pred=_mm256_sub_ps(_mm256_mul_ps(two,pv),p2);
        const __m256 ae=_mm256_andnot_ps(_mm256_set1_ps(-0.0f),_mm256_sub_ps(ac,pred));
        const __m256 apred=_mm256_andnot_ps(_mm256_set1_ps(-0.0f),pred);
        const __m256 predictable=_mm256_sub_ps(_mm256_set1_ps(1.0f),_mm256_min_ps(_mm256_set1_ps(1.0f),_mm256_div_ps(ae,_mm256_add_ps(_mm256_add_ps(ac,apred),eps))));
        _mm256_storeu_ps(s.temp.data()+i,_mm256_mul_ps(half,_mm256_add_ps(persist,predictable)));

        const __m256 el=_mm256_loadu_ps(s.l2.data()+i),er=_mm256_loadu_ps(s.r2.data()+i),ec=_mm256_loadu_ps(s.lr.data()+i);
        const __m256 cohLR=_mm256_div_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.0f),ec),_mm256_add_ps(_mm256_sqrt_ps(_mm256_mul_ps(el,er)),eps));
        const __m256 bal=_mm256_div_ps(_mm256_mul_ps(two,_mm256_sqrt_ps(_mm256_mul_ps(el,er))),_mm256_add_ps(_mm256_add_ps(el,er),eps));
        _mm256_storeu_ps(s.stereo.data()+i,_mm256_mul_ps(half,_mm256_add_ps(cohLR,bal)));
    }
#endif
    for(;i<n-2;++i){
        const float sm3=0.25f*(a[i-1]+2*a[i]+a[i+1]);
        const float sm5=(a[i-2]+4*a[i-1]+6*a[i]+4*a[i+1]+a[i+2])*(1.0f/16.0f);
        const float dog=std::fabs(sm3-sm5)/(sm5+EPS);
        const float curv=std::fabs(a[i-1]-2*a[i]+a[i+1])/(a[i-1]+2*a[i]+a[i+1]+EPS);
        const float jf=.25f*(s.jff[i-1]+2*s.jff[i]+s.jff[i+1]),jt=.25f*(s.jtt[i-1]+2*s.jtt[i]+s.jtt[i+1]),jx=.25f*(s.jft[i-1]+2*s.jft[i]+s.jft[i+1]);
        const float d=jf-jt,coh=std::sqrt(d*d+4*jx*jx)/(jf+jt+EPS);
        const float gw=a[i+2]-a[i-2],tw=s.gt[i],wff=gw*gw,wtt=tw*tw,wft=gw*tw,wd=wff-wtt;
        const float wcoh=std::sqrt(wd*wd+4*wft*wft)/(wff+wtt+EPS);
        const float q2=(a[i-2]*a[i-2]+4*a[i-1]*a[i-1]+6*a[i]*a[i]+4*a[i+1]*a[i+1]+a[i+2]*a[i+2])*(1.0f/16.0f);
        const float cv=std::sqrt(std::max(q2-sm5*sm5,0.0f))/(sm5+EPS),ridge=2*a[i]/(a[i]+sm5+EPS);
        s.geom[i]=(coh+wcoh+dog+curv+cv+ridge)*(1.0f/6.0f);
        const float persist=2*a[i]*state.prev[i]/(a[i]*a[i]+state.prev[i]*state.prev[i]+EPS);
        const float pred=2*state.prev[i]-state.prev2[i];
        const float predictable=1.0f-std::min(1.0f,std::fabs(a[i]-pred)/(a[i]+std::fabs(pred)+EPS));
        s.temp[i]=0.5f*(persist+predictable);
        const float den=std::sqrt(s.l2[i]*s.r2[i])+EPS;
        const float cohLR=std::fabs(s.lr[i])/den,bal=2*std::sqrt(s.l2[i]*s.r2[i])/(s.l2[i]+s.r2[i]+EPS);
        s.stereo[i]=0.5f*(cohLR+bal);
    }

    // Constant geometry edge extension; no signal decision.
    if(n>4){for(int e=0;e<2;++e){s.geom[e]=s.geom[2];s.temp[e]=s.temp[2];s.stereo[e]=s.stereo[2];s.geom[n-1-e]=s.geom[n-3];s.temp[n-1-e]=s.temp[n-3];s.stereo[n-1-e]=s.stereo[n-3];}}

    // Harmonic and log-frequency-periodic context.  The source indices are
    // irregular, and on current x86 the AVX2 gather sequence is more expensive
    // than the ten scalar indexed loads per cell.  Keep identical equations but
    // let OoO execution overlap independent scalar loads.
#if defined(__AVX2__) && !defined(LX_FEAR_DISABLE_HARM_SIMD)
    // Lanes = adjacent spectral cells (not five neighbors of the same cell).
    // Gather the discontinuous references, then do vector arithmetic per cell.
    int g=0;
    const __m256 eps_h=_mm256_set1_ps(EPS), tw=_mm256_set1_ps(2.0f);
    for(;g+8<=n;g+=8){
        const __m256 ag=_mm256_loadu_ps(a+g), ag2=_mm256_mul_ps(ag,ag);
        __m256 hs=_mm256_setzero_ps(),ps=_mm256_setzero_ps();
        for(int q=0;q<5;++q){
            const __m256i hi=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(st.harmonic[q].idx.data()+g));
            const __m256i pi=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(st.periodic[q].idx.data()+g));
            const __m256 h=_mm256_i32gather_ps(a,hi,4), p=_mm256_i32gather_ps(a,pi,4);
            const __m256 hn=_mm256_mul_ps(tw,_mm256_mul_ps(ag,h));
            const __m256 pn=_mm256_mul_ps(tw,_mm256_mul_ps(ag,p));
            const __m256 hd=_mm256_add_ps(_mm256_add_ps(ag2,_mm256_mul_ps(h,h)),eps_h);
            const __m256 pd=_mm256_add_ps(_mm256_add_ps(ag2,_mm256_mul_ps(p,p)),eps_h);
            hs=_mm256_add_ps(hs,_mm256_mul_ps(_mm256_loadu_ps(st.harmonic[q].valid.data()+g),_mm256_div_ps(hn,hd)));
            ps=_mm256_add_ps(ps,_mm256_mul_ps(_mm256_loadu_ps(st.periodic[q].valid.data()+g),_mm256_div_ps(pn,pd)));
        }
        _mm256_storeu_ps(s.harm.data()+g,_mm256_div_ps(hs,_mm256_add_ps(_mm256_loadu_ps(st.harmonic_count.data()+g),eps_h)));
        _mm256_storeu_ps(s.periodic.data()+g,_mm256_div_ps(ps,_mm256_add_ps(_mm256_loadu_ps(st.periodic_count.data()+g),eps_h)));
    }
    for(;g<n;++g){
        const float ag=a[g],ag2=ag*ag;
        float hs=0,ps=0;
        for(int q=0;q<5;++q){
            const float h=a[st.harmonic[q].idx[g]],p=a[st.periodic[q].idx[g]];
            hs+=st.harmonic[q].valid[g]*(2*ag*h/(ag2+h*h+EPS));
            ps+=st.periodic[q].valid[g]*(2*ag*p/(ag2+p*p+EPS));
        }
        s.harm[g]=hs/(st.harmonic_count[g]+EPS);
        s.periodic[g]=ps/(st.periodic_count[g]+EPS);
    }
#else
    for(int g=0;g<n;++g){
        const float ag=a[g], ag2=ag*ag;
#if defined(__AVX2__)
        // Five irregular references fit in one vector. Scalar loads are faster
        // than gathers here, while one packed vector divide replaces five
        // dependent scalar divides for each sensor family.
        const __m256 hval=_mm256_set_ps(0.0f,0.0f,0.0f,
            a[st.harmonic[4].idx[g]],a[st.harmonic[3].idx[g]],a[st.harmonic[2].idx[g]],
            a[st.harmonic[1].idx[g]],a[st.harmonic[0].idx[g]]);
        const __m256 hmask=_mm256_set_ps(0.0f,0.0f,0.0f,
            st.harmonic[4].valid[g],st.harmonic[3].valid[g],st.harmonic[2].valid[g],
            st.harmonic[1].valid[g],st.harmonic[0].valid[g]);
        const __m256 pval=_mm256_set_ps(0.0f,0.0f,0.0f,
            a[st.periodic[4].idx[g]],a[st.periodic[3].idx[g]],a[st.periodic[2].idx[g]],
            a[st.periodic[1].idx[g]],a[st.periodic[0].idx[g]]);
        const __m256 pmask=_mm256_set_ps(0.0f,0.0f,0.0f,
            st.periodic[4].valid[g],st.periodic[3].valid[g],st.periodic[2].valid[g],
            st.periodic[1].valid[g],st.periodic[0].valid[g]);
        const __m256 A=_mm256_set1_ps(ag), A2=_mm256_set1_ps(ag2);
        const __m256 numerator=_mm256_set1_ps(2.0f*ag);
        const __m256 epsv=_mm256_set1_ps(EPS);
        const __m256 hs_v=_mm256_mul_ps(hmask,_mm256_div_ps(_mm256_mul_ps(numerator,hval),_mm256_add_ps(_mm256_add_ps(A2,_mm256_mul_ps(hval,hval)),epsv)));
        const __m256 ps_v=_mm256_mul_ps(pmask,_mm256_div_ps(_mm256_mul_ps(numerator,pval),_mm256_add_ps(_mm256_add_ps(A2,_mm256_mul_ps(pval,pval)),epsv)));
        alignas(32) float hv[8],pv[8];
        _mm256_store_ps(hv,hs_v); _mm256_store_ps(pv,ps_v);
        const float hs=hv[0]+hv[1]+hv[2]+hv[3]+hv[4];
        const float ps= pv[0]+pv[1]+pv[2]+pv[3]+pv[4];
#else
        float hs=0.0f,ps=0.0f;
        for(int q=0;q<5;++q){
            const float hm=st.harmonic[q].valid[g],hv=a[st.harmonic[q].idx[g]];
            hs+=hm*(2.0f*ag*hv/(ag2+hv*hv+EPS));
            const float pm=st.periodic[q].valid[g],pv=a[st.periodic[q].idx[g]];
            ps+=pm*(2.0f*ag*pv/(ag2+pv*pv+EPS));
        }
#endif
        s.harm[g]=hs/(st.harmonic_count[g]+EPS);
        s.periodic[g]=ps/(st.periodic_count[g]+EPS);
    }

#endif

    // Asymmetric masking context from equal-width physical-Hz neighborhoods.
    // The window radius is one ERB-scale analysis neighborhood on this coarse
    // lattice (8 cells = 375 Hz), chosen as geometry rather than signal state.
    constexpr int R=8;
#if defined(__AVX2__) && !defined(LX_FEAR_DISABLE_MASK_SIMD)
    // Only 16 endpoint cells need variable window lengths; full windows
    // have the same 1/8 normalization, with no bounds checks in the hot loop.
    const int left=std::min(n,8),right=std::max(left,n-8);
    for(int g=0;g<left;++g){
        const int l0=std::max(0,g-R),r1=std::min(n,g+1+R);
        const float ml=(s.prefix[g]-s.prefix[l0])/float(std::max(1,g-l0));
        const float mr=(s.prefix[r1]-s.prefix[g+1])/float(std::max(1,r1-g-1));
        const float ctx=std::sqrt((ml+EPS)*(mr+EPS));
        s.mask[g]=0.5f*(2*a[g]/(a[g]+ctx+EPS)+std::fabs(ml-mr)/(ml+mr+EPS));
        const float ratio=a[g]/(ctx+EPS);s.tmr[g]=std::log1p(ratio*ratio);
    }
    const __m256 epsm=_mm256_set1_ps(EPS), twom=_mm256_set1_ps(2.0f);
    const __m256 eighth=_mm256_set1_ps(0.125f),halfm=_mm256_set1_ps(0.5f),sign=_mm256_set1_ps(-0.0f);
    int gm=left;
    for(;gm+8<=right;gm+=8){
        const __m256 ml=_mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(s.prefix.data()+gm),_mm256_loadu_ps(s.prefix.data()+gm-8)),eighth);
        const __m256 mr=_mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(s.prefix.data()+gm+9),_mm256_loadu_ps(s.prefix.data()+gm+1)),eighth);
        const __m256 c=_mm256_sqrt_ps(_mm256_mul_ps(_mm256_add_ps(ml,epsm),_mm256_add_ps(mr,epsm)));
        const __m256 ag=_mm256_loadu_ps(a+gm);
        const __m256 aud=_mm256_div_ps(_mm256_mul_ps(twom,ag),_mm256_add_ps(_mm256_add_ps(ag,c),epsm));
        const __m256 asym=_mm256_div_ps(_mm256_andnot_ps(sign,_mm256_sub_ps(ml,mr)),_mm256_add_ps(_mm256_add_ps(ml,mr),epsm));
        _mm256_storeu_ps(s.mask.data()+gm,_mm256_mul_ps(halfm,_mm256_add_ps(aud,asym)));
        const __m256 ratio=_mm256_div_ps(ag,_mm256_add_ps(c,epsm));
        // Leave the normative log1p unchanged. SIMD only for its arguments.
        alignas(32) float tmp[8];_mm256_store_ps(tmp,_mm256_mul_ps(ratio,ratio));
        for(int j=0;j<8;++j)s.tmr[gm+j]=std::log1p(tmp[j]);
    }
    for(;gm<n;++gm){
        const int l0=std::max(0,gm-R),r1=std::min(n,gm+1+R);
        const float ml=(s.prefix[gm]-s.prefix[l0])/float(std::max(1,gm-l0));
        const float mr=(s.prefix[r1]-s.prefix[gm+1])/float(std::max(1,r1-gm-1));
        const float ctx=std::sqrt((ml+EPS)*(mr+EPS));
        s.mask[gm]=0.5f*(2*a[gm]/(a[gm]+ctx+EPS)+std::fabs(ml-mr)/(ml+mr+EPS));
        const float ratio=a[gm]/(ctx+EPS);s.tmr[gm]=std::log1p(ratio*ratio);
    }
#else
    for(int g=0;g<n;++g){
        const int l0=std::max(0,g-R),l1=g,r0=g+1,r1=std::min(n,g+1+R);
        const float ml=(s.prefix[l1]-s.prefix[l0])/float(std::max(1,l1-l0));
        const float mr=(s.prefix[r1]-s.prefix[r0])/float(std::max(1,r1-r0));
        const float context=std::sqrt((ml+EPS)*(mr+EPS));
        const float aud=2.0f*a[g]/(a[g]+context+EPS);
        const float asym=std::fabs(ml-mr)/(ml+mr+EPS);
        s.mask[g]=0.5f*(aud+asym);
        const float ratio=a[g]/(context+EPS); s.tmr[g]=std::log1p(ratio*ratio);
    }

#endif

    state.prev2=state.prev; state.prev=s.a; state.primed=true;
}

// Research fusion for the six independent sensor families.  Runtime controls are
// read once per process, outside all signal-value hot loops.  This lets us fit
// the analytic FEAR utility without recompiling hundreds of candidates; the
// production constants are selected after validation.
struct Fusion {
    float wg=0.5f, wh=1.0f, wp=1.5f, wm=1.5f, wt=1.0f, ws=0.1f;
    float wgh=0.0f, wgp=0.0f, whp=0.0f, whm=0.0f, wpm=0.0f, wht=0.0f, wpt=0.0f;
    float wgmt=0.0f, wpmt=0.0f;
    float wg3=0.0f, wh3=1.0f, wp3=0.0f;
    float scale=0.08f; float tonal_gain=1.0f; float tonal_gamma=1.0f;
};

static inline float envf(const char* n,float d){
    const char* p=std::getenv(n); return p?std::strtof(p,nullptr):d;
}
static inline const Fusion& fusion_params(){
    static const Fusion f=[](){
        Fusion x;
        x.wg=envf("LX_FEAR_WG",x.wg); x.wh=envf("LX_FEAR_WH",x.wh);
        x.wp=envf("LX_FEAR_WP",x.wp); x.wm=envf("LX_FEAR_WM",x.wm);
        x.wt=envf("LX_FEAR_WT",x.wt); x.ws=envf("LX_FEAR_WS",x.ws);
        x.wgh=envf("LX_FEAR_WGH",0); x.wgp=envf("LX_FEAR_WGP",0); x.whp=envf("LX_FEAR_WHP",0);
        x.whm=envf("LX_FEAR_WHM",0); x.wpm=envf("LX_FEAR_WPM",0);
        x.wht=envf("LX_FEAR_WHT",0); x.wpt=envf("LX_FEAR_WPT",0);
        x.wgmt=envf("LX_FEAR_WGMT",0); x.wpmt=envf("LX_FEAR_WPMT",0);
        x.wg3=envf("LX_FEAR_WG3",0); x.wh3=envf("LX_FEAR_WH3",0); x.wp3=envf("LX_FEAR_WP3",0);
        x.scale=envf("LX_FEAR_FUSION_SCALE",x.scale); x.tonal_gain=envf("LX_FEAR_TONAL_GAIN",0.0f); x.tonal_gamma=envf("LX_FEAR_TONAL_GAMMA",0.0f);
        return x;
    }();
    return f;
}

// Reduce independently normalized sensor evidence to a zero-common-mode ERB
// pressure. z_q = 2*x/(x+mean)-1 is scale-invariant and bounded [-1,+1].
// Cross terms encode the actual psychoacoustic question: structure that is
// simultaneously audible and persistent deserves more marginal rate than any
// one sensor alone.  Polynomial fusion is branchless and SIMD-friendly.
static inline void reduce_to_erb(const Static& st,const Scratch& s,int n,std::array<float,48>& out,std::array<float,48>* tonal_out=nullptr){
    float mean[FAMILIES]={0,0,0,0,0,0};
    for(int g=0;g<n;++g){mean[0]+=s.geom[g];mean[1]+=s.harm[g];mean[2]+=s.periodic[g];mean[3]+=s.mask[g];mean[4]+=s.temp[g];mean[5]+=s.stereo[g];}
    const float invn=1.0f/float(std::max(n,1)); for(float& x:mean)x=x*invn+EPS;
    const Fusion& f=fusion_params();
    // Production/experimental mode is frame-invariant.  Decide once, never once per
    // lattice cell.  The production polynomial is also evaluated eight cells at
    // a time on AVX2; the ERB scatter remains scalar and tiny.
    const bool prod = f.wgh==0.0f && f.wgp==0.0f && f.whp==0.0f && f.whm==0.0f && f.wpm==0.0f
                   && f.wht==0.0f && f.wpt==0.0f && f.wgmt==0.0f && f.wpmt==0.0f
                   && f.wg3==0.0f && f.wp3==0.0f;
    out.fill(0.0f); if(tonal_out) tonal_out->fill(0.0f);
#if defined(__AVX2__)
    if(prod){
        alignas(64) std::array<float,GRID> fused{};
        const __m256 two=_mm256_set1_ps(2.0f), one=_mm256_set1_ps(1.0f), eps=_mm256_set1_ps(EPS);
        const __m256 mg=_mm256_set1_ps(mean[0]), mh=_mm256_set1_ps(mean[1]), mp=_mm256_set1_ps(mean[2]);
        const __m256 mm=_mm256_set1_ps(mean[3]), mt=_mm256_set1_ps(mean[4]), ms=_mm256_set1_ps(mean[5]);
        const __m256 wg=_mm256_set1_ps(f.wg), wh=_mm256_set1_ps(f.wh), wp=_mm256_set1_ps(f.wp);
        const __m256 wm=_mm256_set1_ps(f.wm), wt=_mm256_set1_ps(f.wt), ws=_mm256_set1_ps(f.ws), wh3=_mm256_set1_ps(f.wh3);
        int g=0;
        for(;g+8<=n;g+=8){
            const __m256 G=_mm256_loadu_ps(s.geom.data()+g), H=_mm256_loadu_ps(s.harm.data()+g);
            const __m256 P=_mm256_loadu_ps(s.periodic.data()+g), M=_mm256_loadu_ps(s.mask.data()+g);
            const __m256 T=_mm256_loadu_ps(s.temp.data()+g), S=_mm256_loadu_ps(s.stereo.data()+g);
            const __m256 zg=_mm256_sub_ps(_mm256_div_ps(_mm256_mul_ps(two,G),_mm256_add_ps(_mm256_add_ps(G,mg),eps)),one);
            const __m256 zh=_mm256_sub_ps(_mm256_div_ps(_mm256_mul_ps(two,H),_mm256_add_ps(_mm256_add_ps(H,mh),eps)),one);
            const __m256 zp=_mm256_sub_ps(_mm256_div_ps(_mm256_mul_ps(two,P),_mm256_add_ps(_mm256_add_ps(P,mp),eps)),one);
            const __m256 zm=_mm256_sub_ps(_mm256_div_ps(_mm256_mul_ps(two,M),_mm256_add_ps(_mm256_add_ps(M,mm),eps)),one);
            const __m256 zt=_mm256_sub_ps(_mm256_div_ps(_mm256_mul_ps(two,T),_mm256_add_ps(_mm256_add_ps(T,mt),eps)),one);
            const __m256 zs=_mm256_sub_ps(_mm256_div_ps(_mm256_mul_ps(two,S),_mm256_add_ps(_mm256_add_ps(S,ms),eps)),one);
            __m256 e=_mm256_mul_ps(wg,zg);
            e=_mm256_fmadd_ps(wh,zh,e); e=_mm256_fmadd_ps(wp,zp,e); e=_mm256_fmadd_ps(wm,zm,e);
            e=_mm256_fmadd_ps(wt,zt,e); e=_mm256_fmadd_ps(ws,zs,e);
            e=_mm256_fmadd_ps(wh3,_mm256_mul_ps(_mm256_mul_ps(zh,zh),zh),e);
            _mm256_store_ps(fused.data()+g,e);
        }
        for(;g<n;++g){
            const float zg=2.0f*s.geom[g]/(s.geom[g]+mean[0]+EPS)-1.0f;
            const float zh=2.0f*s.harm[g]/(s.harm[g]+mean[1]+EPS)-1.0f;
            const float zp=2.0f*s.periodic[g]/(s.periodic[g]+mean[2]+EPS)-1.0f;
            const float zm=2.0f*s.mask[g]/(s.mask[g]+mean[3]+EPS)-1.0f;
            const float zt=2.0f*s.temp[g]/(s.temp[g]+mean[4]+EPS)-1.0f;
            const float zs=2.0f*s.stereo[g]/(s.stereo[g]+mean[5]+EPS)-1.0f;
            fused[g]=f.wg*zg + f.wh*zh + f.wp*zp + f.wm*zm + f.wt*zt + f.ws*zs + f.wh3*zh*zh*zh;
        }
        for(int k=0;k<n;++k) out[st.erb[k]] += f.scale*fused[k];
    } else
#endif
    {
        for(int g=0;g<n;++g){
            const float zg=2.0f*s.geom[g]/(s.geom[g]+mean[0]+EPS)-1.0f;
            const float zh=2.0f*s.harm[g]/(s.harm[g]+mean[1]+EPS)-1.0f;
            const float zp=2.0f*s.periodic[g]/(s.periodic[g]+mean[2]+EPS)-1.0f;
            const float zm=2.0f*s.mask[g]/(s.mask[g]+mean[3]+EPS)-1.0f;
            const float zt=2.0f*s.temp[g]/(s.temp[g]+mean[4]+EPS)-1.0f;
            const float zs=2.0f*s.stereo[g]/(s.stereo[g]+mean[5]+EPS)-1.0f;
            float e = f.wg*zg + f.wh*zh + f.wp*zp + f.wm*zm + f.wt*zt + f.ws*zs + f.wh3*zh*zh*zh;
            e += f.wgh*zg*zh + f.wgp*zg*zp + f.whp*zh*zp + f.whm*zh*zm + f.wpm*zp*zm
               + f.wht*zh*zt + f.wpt*zp*zt + f.wgmt*zg*zm*zt + f.wpmt*zp*zm*zt
               + f.wg3*zg*zg*zg + f.wp3*zp*zp*zp;
            out[st.erb[g]] += f.scale*e;
        }
    }
    for(int b=0;b<48;++b) out[b]*=st.inv_count[b];
    // v6 aggressive tonal-object utility.  A coherent ridge is a sparse
    // auditory object, so arithmetic ERB averaging is the wrong reduction:
    // it attenuates one narrow line in proportion to ERB width.  Use the L4
    // generalized mean of positive ridge utility instead.  L4 is continuous,
    // branchless on signal values and approaches peak sensitivity while still
    // pooling evidence from multiple coherent partials.
    if(f.tonal_gain!=0.0f || tonal_out){
        float q4sum[48]={0};
        for(int g=0;g<n;++g){
            const float support=std::sqrt((1.0f+s.harm[g])*(1.0f+s.periodic[g]));
            const float q=s.ridge_tonal[g]*s.temp[g]*s.tmr[g]*support;
            const float q2=q*q;
            q4sum[st.erb[g]] += q2*q2;
        }
        float p[48],psum=0.0f,pdim=0.0f;
        for(int b=0;b<48;++b){
            const float invb=st.inv_count[b];
            const float obj=std::sqrt(std::sqrt(q4sum[b]*invb + EPS));
            // Squared object strength makes high-confidence tonal ridges win
            // rate decisively; weak/noisy evidence falls quadratically.
            // gamma=0 is the frozen production law: x^0 == 1 exactly.
            // Avoid a libm pow() call per ERB while retaining configurable gamma.
            const float fw = (f.tonal_gamma==0.0f) ? 1.0f
                           : (f.tonal_gamma==1.0f) ? (st.norm_freq[b] + EPS)
                           : std::pow(st.norm_freq[b] + EPS, f.tonal_gamma);
            const float objw=obj*obj*fw;
            if(tonal_out) (*tonal_out)[b]=objw;
            p[b]=f.tonal_gain*objw;
            const float d=invb>0.0f?1.0f/invb:0.0f;
            psum+=d*p[b];pdim+=d;
        }
        if(f.tonal_gain!=0.0f){
            const float pc=psum/(pdim+EPS); for(int b=0;b<48;++b) out[b]+=p[b]-pc;
        }
    }
    // c.loga is a per-coordinate utility.  Preserve the frame budget in the
    // same measure used by the allocator: degrees of freedom / physical
    // bandwidth, not an unweighted mean over unequal ERB bands.
    float m=0.0f, md=0.0f;
    for(int b=0;b<48;++b){
        const float d = st.inv_count[b]>0.0f ? 1.0f/st.inv_count[b] : 0.0f;
        m += d*out[b]; md += d;
    }
    m /= (md+EPS); for(float& x:out)x-=m;
}

} // namespace lx_fear_v4
