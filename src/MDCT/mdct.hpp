
#pragma once

#if defined(_MSC_VER)
#define LX_ALWAYS_INLINE __forceinline
#else
#define LX_ALWAYS_INLINE __attribute__((always_inline)) inline
#endif
#include <array>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace lx_nsmdct {

constexpr float PI = 3.14159265358979323846f;
constexpr int OVERLAP = 128;
constexpr int HALF_OVERLAP = OVERLAP/2;

// Power-of-two complex FFT used by the DCT-IV core.
//
// The existing hot path is kept as radix-4 whenever F=4^m.  The two new
// transform sizes require F=512 and F=128 (=2*4^m); for those sizes one
// radix-2 split is followed by two radix-4 FFTs.  This is an exact mixed-radix
// factorization, not a fallback DFT, and preserves the old 2048/512 path.
constexpr bool is_pow4_constexpr(int x) {
    if (x < 1) return false;
    while ((x & 3) == 0) x >>= 2;
    return x == 1;
}

template<int F>
struct FFTPow4Core {
    static constexpr int stages=[](){int x=F,s=0;while(x>1){x>>=2;++s;}return s;}();
    alignas(64) std::array<float,F> re{},im{},tr{},ti{},wr{},wi{};
    // Contiguous per-stage twiddles for the AVX2 radix-4 butterflies.
    // This removes three gathers and all index-vector construction from the hot loop.
    static constexpr int TWN = (F-1)/3; // sum(F/4 + F/16 + ...)
    alignas(64) std::array<float,TWN> sw1r{},sw1i{},sw2r{},sw2i{},sw3r{},sw3i{};
    std::array<uint16_t,F> perm{};
    alignas(64) std::array<int32_t,F> perm32{};
    FFTPow4Core(){
        static_assert(is_pow4_constexpr(F));
        for(int k=0;k<F;++k){
            const float a=-2.0f*PI*k/F;
            wr[k]=std::cos(a);wi[k]=std::sin(a);
        }
        for(int i=0;i<F;++i){
            int x=i,y=0;
            for(int s=0;s<stages;++s){y=(y<<2)|(x&3);x>>=2;}
            perm[i]=(uint16_t)y; perm32[i]=y;
        }
        int to=0;
        for(int m=F;m>=4;m>>=2){
            const int q=m>>2, step=F/m;
            for(int j=0;j<q;++j){
                const int k=j*step;
                sw1r[to+j]=wr[k]; sw1i[to+j]=wi[k];
                sw2r[to+j]=wr[(2*k)&(F-1)]; sw2i[to+j]=wi[(2*k)&(F-1)];
                sw3r[to+j]=wr[(3*k)&(F-1)]; sw3i[to+j]=wi[(3*k)&(F-1)];
            }
            to += q;
        }
    }
    static inline void cmul(float ar,float ai,float br,float bi,float&rr,float&ri){
        rr=ar*br-ai*bi;ri=ar*bi+ai*br;
    }
    LX_ALWAYS_INLINE void butterflies(){
        int to=0;
        for(int m=F;m>=4;m>>=2){
            const int q=m>>2,step=F/m;
            for(int base=0;base<F;base+=m){
                int j=0;
#if defined(__AVX2__)
                for(;j+8<=q;j+=8){
                    const int p0=base+j,p1=p0+q,p2=p1+q,p3=p2+q;
                    const __m256 a0r=_mm256_loadu_ps(re.data()+p0), a0i=_mm256_loadu_ps(im.data()+p0);
                    const __m256 a1r=_mm256_loadu_ps(re.data()+p1), a1i=_mm256_loadu_ps(im.data()+p1);
                    const __m256 a2r=_mm256_loadu_ps(re.data()+p2), a2i=_mm256_loadu_ps(im.data()+p2);
                    const __m256 a3r=_mm256_loadu_ps(re.data()+p3), a3i=_mm256_loadu_ps(im.data()+p3);
                    const __m256 s02r=_mm256_add_ps(a0r,a2r), s02i=_mm256_add_ps(a0i,a2i);
                    const __m256 d02r=_mm256_sub_ps(a0r,a2r), d02i=_mm256_sub_ps(a0i,a2i);
                    const __m256 s13r=_mm256_add_ps(a1r,a3r), s13i=_mm256_add_ps(a1i,a3i);
                    const __m256 d13r=_mm256_sub_ps(a1r,a3r), d13i=_mm256_sub_ps(a1i,a3i);
                    _mm256_storeu_ps(re.data()+p0,_mm256_add_ps(s02r,s13r));
                    _mm256_storeu_ps(im.data()+p0,_mm256_add_ps(s02i,s13i));
                    const __m256 y1r=_mm256_add_ps(d02r,d13i), y1i=_mm256_sub_ps(d02i,d13r);
                    const __m256 y2r=_mm256_sub_ps(s02r,s13r), y2i=_mm256_sub_ps(s02i,s13i);
                    const __m256 y3r=_mm256_sub_ps(d02r,d13i), y3i=_mm256_add_ps(d02i,d13r);
                    const __m256 w1r=_mm256_loadu_ps(sw1r.data()+to+j), w1i=_mm256_loadu_ps(sw1i.data()+to+j);
                    const __m256 w2r=_mm256_loadu_ps(sw2r.data()+to+j), w2i=_mm256_loadu_ps(sw2i.data()+to+j);
                    const __m256 w3r=_mm256_loadu_ps(sw3r.data()+to+j), w3i=_mm256_loadu_ps(sw3i.data()+to+j);
#if defined(__FMA__)
                    _mm256_storeu_ps(re.data()+p1,_mm256_fmsub_ps(y1r,w1r,_mm256_mul_ps(y1i,w1i)));
                    _mm256_storeu_ps(im.data()+p1,_mm256_fmadd_ps(y1r,w1i,_mm256_mul_ps(y1i,w1r)));
                    _mm256_storeu_ps(re.data()+p2,_mm256_fmsub_ps(y2r,w2r,_mm256_mul_ps(y2i,w2i)));
                    _mm256_storeu_ps(im.data()+p2,_mm256_fmadd_ps(y2r,w2i,_mm256_mul_ps(y2i,w2r)));
                    _mm256_storeu_ps(re.data()+p3,_mm256_fmsub_ps(y3r,w3r,_mm256_mul_ps(y3i,w3i)));
                    _mm256_storeu_ps(im.data()+p3,_mm256_fmadd_ps(y3r,w3i,_mm256_mul_ps(y3i,w3r)));
#else
                    _mm256_storeu_ps(re.data()+p1,_mm256_sub_ps(_mm256_mul_ps(y1r,w1r),_mm256_mul_ps(y1i,w1i)));
                    _mm256_storeu_ps(im.data()+p1,_mm256_add_ps(_mm256_mul_ps(y1r,w1i),_mm256_mul_ps(y1i,w1r)));
                    _mm256_storeu_ps(re.data()+p2,_mm256_sub_ps(_mm256_mul_ps(y2r,w2r),_mm256_mul_ps(y2i,w2i)));
                    _mm256_storeu_ps(im.data()+p2,_mm256_add_ps(_mm256_mul_ps(y2r,w2i),_mm256_mul_ps(y2i,w2r)));
                    _mm256_storeu_ps(re.data()+p3,_mm256_sub_ps(_mm256_mul_ps(y3r,w3r),_mm256_mul_ps(y3i,w3i)));
                    _mm256_storeu_ps(im.data()+p3,_mm256_add_ps(_mm256_mul_ps(y3r,w3i),_mm256_mul_ps(y3i,w3r)));
#endif
                }
#endif
                for(;j<q;++j){
                    const int p0=base+j,p1=p0+q,p2=p1+q,p3=p2+q;
                    const float a0r=re[p0],a0i=im[p0],a1r=re[p1],a1i=im[p1];
                    const float a2r=re[p2],a2i=im[p2],a3r=re[p3],a3i=im[p3];
                    const float s02r=a0r+a2r,s02i=a0i+a2i,d02r=a0r-a2r,d02i=a0i-a2i;
                    const float s13r=a1r+a3r,s13i=a1i+a3i,d13r=a1r-a3r,d13i=a1i-a3i;
                    re[p0]=s02r+s13r;im[p0]=s02i+s13i;
                    const float y1r=d02r+d13i,y1i=d02i-d13r;
                    const float y2r=s02r-s13r,y2i=s02i-s13i;
                    const float y3r=d02r-d13i,y3i=d02i+d13r;
                    const int k=j*step;
                    cmul(y1r,y1i,wr[k],wi[k],re[p1],im[p1]);
                    cmul(y2r,y2i,wr[(2*k)&(F-1)],wi[(2*k)&(F-1)],re[p2],im[p2]);
                    cmul(y3r,y3i,wr[(3*k)&(F-1)],wi[(3*k)&(F-1)],re[p3],im[p3]);
                }
            }
            to += q;
        }
    }
    LX_ALWAYS_INLINE void forward_unordered(){ butterflies(); }
    inline float natural_re(int k) const { return re[perm[k]]; }
    inline float natural_im(int k) const { return im[perm[k]]; }
    void forward(){
        butterflies();
        for(int i=0;i<F;++i){tr[i]=re[perm[i]];ti[i]=im[perm[i]];}
        re.swap(tr);im.swap(ti);
    }
};

template<int F, bool PURE_RADIX4=is_pow4_constexpr(F)>
struct FFTPow2DImpl;

// No regression on the legacy FFT sizes: same radix-4 implementation and
// storage layout as before.
template<int F>
struct FFTPow2DImpl<F,true> : FFTPow4Core<F> {};

// F=2*4^m: even/odd radix-2 decomposition, each half handled by radix-4.
template<int F>
struct FFTPow2DImpl<F,false> {
    static_assert((F & (F-1)) == 0 && is_pow4_constexpr(F/2),
                  "mixed-radix FFT supports F=4^m or 2*4^m");
    static constexpr int H=F/2;
    alignas(64) std::array<float,F> re{},im{};
    alignas(64) std::array<float,H> wr{},wi{};
    FFTPow4Core<H> even{},odd{};

    FFTPow2DImpl(){
        for(int k=0;k<H;++k){
            const float a=-2.0f*PI*k/F;
            wr[k]=std::cos(a);wi[k]=std::sin(a);
        }
    }
    LX_ALWAYS_INLINE void forward_preloaded(){
        even.forward_unordered(); odd.forward_unordered();
        int k=0;
#if defined(__AVX2__)
        for(;k+8<=H;k+=8){
            const __m256i ip=_mm256_loadu_si256((const __m256i*)(even.perm32.data()+k));
            const __m256 er=_mm256_i32gather_ps(even.re.data(),ip,4);
            const __m256 ei=_mm256_i32gather_ps(even.im.data(),ip,4);
            const __m256 orr=_mm256_i32gather_ps(odd.re.data(),ip,4);
            const __m256 oii=_mm256_i32gather_ps(odd.im.data(),ip,4);
            const __m256 vr=_mm256_loadu_ps(wr.data()+k), vi=_mm256_loadu_ps(wi.data()+k);
#if defined(__FMA__)
            const __m256 tr=_mm256_fmsub_ps(orr,vr,_mm256_mul_ps(oii,vi));
            const __m256 ti=_mm256_fmadd_ps(orr,vi,_mm256_mul_ps(oii,vr));
#else
            const __m256 tr=_mm256_sub_ps(_mm256_mul_ps(orr,vr),_mm256_mul_ps(oii,vi));
            const __m256 ti=_mm256_add_ps(_mm256_mul_ps(orr,vi),_mm256_mul_ps(oii,vr));
#endif
            _mm256_storeu_ps(re.data()+k,_mm256_add_ps(er,tr));
            _mm256_storeu_ps(im.data()+k,_mm256_add_ps(ei,ti));
            _mm256_storeu_ps(re.data()+k+H,_mm256_sub_ps(er,tr));
            _mm256_storeu_ps(im.data()+k+H,_mm256_sub_ps(ei,ti));
        }
#endif
        for(;k<H;++k){
            const float orr=odd.natural_re(k), oii=odd.natural_im(k);
            const float tr=orr*wr[k]-oii*wi[k];
            const float ti=orr*wi[k]+oii*wr[k];
            const float er=even.natural_re(k), ei=even.natural_im(k);
            re[k]=er+tr; im[k]=ei+ti;
            re[k+H]=er-tr; im[k+H]=ei-ti;
        }
    }
    void forward(){
        int n=0;
#if defined(__AVX2__)
        const __m256i eoff=_mm256_setr_epi32(0,2,4,6,8,10,12,14);
        const __m256i ooff=_mm256_setr_epi32(1,3,5,7,9,11,13,15);
        for(;n+8<=H;n+=8){
            const __m256i base=_mm256_set1_epi32(2*n);
            const __m256i ie=_mm256_add_epi32(base,eoff);
            const __m256i io=_mm256_add_epi32(base,ooff);
            _mm256_storeu_ps(even.re.data()+n,_mm256_i32gather_ps(re.data(),ie,4));
            _mm256_storeu_ps(even.im.data()+n,_mm256_i32gather_ps(im.data(),ie,4));
            _mm256_storeu_ps(odd.re.data()+n, _mm256_i32gather_ps(re.data(),io,4));
            _mm256_storeu_ps(odd.im.data()+n, _mm256_i32gather_ps(im.data(),io,4));
        }
#endif
        for(;n<H;++n){
            even.re[n]=re[2*n]; even.im[n]=im[2*n];
            odd.re[n]=re[2*n+1]; odd.im[n]=im[2*n+1];
        }
        forward_preloaded();
    }
};


template<int F>
struct FFT3Pow2D {
    static_assert(F % 3 == 0);
    static constexpr int P=F/3;
    FFTPow2DImpl<P> a0{},a1{},a2{};
    alignas(64) std::array<float,F> re{},im{};
    alignas(64) std::array<float,P> w1r{},w1i{},w2r{},w2i{};
    FFT3Pow2D(){
        for(int q=0;q<P;++q){
            const float a=-2.0f*PI*float(q)/float(F);
            w1r[q]=std::cos(a); w1i[q]=std::sin(a);
            w2r[q]=std::cos(2*a); w2i[q]=std::sin(2*a);
        }
    }
    void forward(){
        for(int n=0;n<P;++n){
            a0.re[n]=re[3*n];   a0.im[n]=im[3*n];
            a1.re[n]=re[3*n+1]; a1.im[n]=im[3*n+1];
            a2.re[n]=re[3*n+2]; a2.im[n]=im[3*n+2];
        }
        a0.forward(); a1.forward(); a2.forward();
        constexpr float c=-0.5f;
        constexpr float ss=0.86602540378443864676f;
        int q=0;
#if defined(__AVX2__)
        const __m256 vc=_mm256_set1_ps(c), vs=_mm256_set1_ps(ss);
        for(;q+8<=P;q+=8){
            const __m256 a1r=_mm256_loadu_ps(a1.re.data()+q), a1i=_mm256_loadu_ps(a1.im.data()+q);
            const __m256 a2r=_mm256_loadu_ps(a2.re.data()+q), a2i=_mm256_loadu_ps(a2.im.data()+q);
            const __m256 x1r=_mm256_loadu_ps(w1r.data()+q), x1i=_mm256_loadu_ps(w1i.data()+q);
            const __m256 x2r=_mm256_loadu_ps(w2r.data()+q), x2i=_mm256_loadu_ps(w2i.data()+q);
#if defined(__FMA__)
            const __m256 b1r=_mm256_fmsub_ps(a1r,x1r,_mm256_mul_ps(a1i,x1i));
            const __m256 b1i=_mm256_fmadd_ps(a1r,x1i,_mm256_mul_ps(a1i,x1r));
            const __m256 b2r=_mm256_fmsub_ps(a2r,x2r,_mm256_mul_ps(a2i,x2i));
            const __m256 b2i=_mm256_fmadd_ps(a2r,x2i,_mm256_mul_ps(a2i,x2r));
#else
            const __m256 b1r=_mm256_sub_ps(_mm256_mul_ps(a1r,x1r),_mm256_mul_ps(a1i,x1i));
            const __m256 b1i=_mm256_add_ps(_mm256_mul_ps(a1r,x1i),_mm256_mul_ps(a1i,x1r));
            const __m256 b2r=_mm256_sub_ps(_mm256_mul_ps(a2r,x2r),_mm256_mul_ps(a2i,x2i));
            const __m256 b2i=_mm256_add_ps(_mm256_mul_ps(a2r,x2i),_mm256_mul_ps(a2i,x2r));
#endif
            const __m256 u0r=_mm256_loadu_ps(a0.re.data()+q), u0i=_mm256_loadu_ps(a0.im.data()+q);
            const __m256 sr=_mm256_add_ps(b1r,b2r), si=_mm256_add_ps(b1i,b2i);
            _mm256_storeu_ps(re.data()+q,_mm256_add_ps(u0r,sr));
            _mm256_storeu_ps(im.data()+q,_mm256_add_ps(u0i,si));
            const __m256 d12i=_mm256_sub_ps(b1i,b2i), d21r=_mm256_sub_ps(b2r,b1r);
            const __m256 d21i=_mm256_sub_ps(b2i,b1i), d12r=_mm256_sub_ps(b1r,b2r);
#if defined(__FMA__)
            _mm256_storeu_ps(re.data()+q+P,_mm256_fmadd_ps(vs,d12i,_mm256_fmadd_ps(vc,sr,u0r)));
            _mm256_storeu_ps(im.data()+q+P,_mm256_fmadd_ps(vs,d21r,_mm256_fmadd_ps(vc,si,u0i)));
            _mm256_storeu_ps(re.data()+q+2*P,_mm256_fmadd_ps(vs,d21i,_mm256_fmadd_ps(vc,sr,u0r)));
            _mm256_storeu_ps(im.data()+q+2*P,_mm256_fmadd_ps(vs,d12r,_mm256_fmadd_ps(vc,si,u0i)));
#else
            _mm256_storeu_ps(re.data()+q+P,_mm256_add_ps(u0r,_mm256_add_ps(_mm256_mul_ps(vc,sr),_mm256_mul_ps(vs,d12i))));
            _mm256_storeu_ps(im.data()+q+P,_mm256_add_ps(u0i,_mm256_add_ps(_mm256_mul_ps(vc,si),_mm256_mul_ps(vs,d21r))));
            _mm256_storeu_ps(re.data()+q+2*P,_mm256_add_ps(u0r,_mm256_add_ps(_mm256_mul_ps(vc,sr),_mm256_mul_ps(vs,d21i))));
            _mm256_storeu_ps(im.data()+q+2*P,_mm256_add_ps(u0i,_mm256_add_ps(_mm256_mul_ps(vc,si),_mm256_mul_ps(vs,d12r))));
#endif
        }
#endif
        for(;q<P;++q){
            float b1r=a1.re[q]*w1r[q]-a1.im[q]*w1i[q];
            float b1i=a1.re[q]*w1i[q]+a1.im[q]*w1r[q];
            float b2r=a2.re[q]*w2r[q]-a2.im[q]*w2i[q];
            float b2i=a2.re[q]*w2i[q]+a2.im[q]*w2r[q];
            const float u0r=a0.re[q],u0i=a0.im[q];
            re[q]=u0r+b1r+b2r; im[q]=u0i+b1i+b2i;
            re[q+P]=u0r + c*(b1r+b2r) + ss*(b1i-b2i);
            im[q+P]=u0i + c*(b1i+b2i) + ss*(b2r-b1r);
            re[q+2*P]=u0r + c*(b1r+b2r) + ss*(b2i-b1i);
            im[q+2*P]=u0i + c*(b1i+b2i) + ss*(b1r-b2r);
        }
    }
};

template<> struct FFTPow2DImpl<192,false> : FFT3Pow2D<192> {};
template<> struct FFTPow2DImpl<384,false> : FFT3Pow2D<384> {};

template<int F>
using FFTPow2D = FFTPow2DImpl<F>;

// Orthonormal DCT-IV via one N/2-point complex FFT.
// For N=2M, pair x[2m] with x[N-1-2m].  One M-point FFT then yields
// both the even and odd DCT-IV coefficients by conjugate symmetry.
// This is 4x smaller than the former zero-padded 2N FFT while computing
// exactly the same orthonormal DCT-IV.
template<int N>
struct DCT4Ortho {
    static constexpr int M=N/2;
    FFTPow2D<M> fft;
    alignas(64) std::array<float,M> pre_c{},pre_s{},even_c{},even_s{},odd_c{},odd_s{};
    // VOXEL low-overlap MDCT: sparse 2N analysis window and N-sample folded vector.
    // The window satisfies w[j]^2 + w[j+N]^2 = 1 exactly (up to float roundoff).
    // Only OVERLAP samples around each frame boundary are non-trivial/non-zero.
    alignas(64) std::array<float,2*N> mdct_w{};
    alignas(64) std::array<float,N> mdct_fold{};
    DCT4Ortho(){
        static_assert(N==256 || N==384 || N==512 || N==768 || N==1024 || N==2048);
        for(int m=0;m<M;++m){
            const float a=PI*float(m)/(2.0f*M);
            pre_c[m]=std::cos(a);pre_s[m]=std::sin(a);
        }
        const float post_g=std::sqrt(2.0f/float(N));
        for(int r=0;r<M;++r){
            const float ae=PI*float(r)/(2.0f*M)+PI/(8.0f*M);
            const float ao=PI*float(r)/(2.0f*M)+3.0f*PI/(8.0f*M);
            even_c[r]=post_g*std::cos(ae);even_s[r]=post_g*std::sin(ae);
            odd_c[r]=post_g*std::cos(ao);odd_s[r]=post_g*std::sin(ao);
        }

        // Sparse low-overlap Princen-Bradley window in the canonical 2N MDCT domain.
        // Frame support is [pos-OVERLAP/2, pos+N+OVERLAP/2), while hop remains N.
        // This is a true MDCT folding window, not a post-DCT boundary rotation.
        static_assert((OVERLAP & 1) == 0 && OVERLAP <= N);
        constexpr int H=N/2;
        constexpr int OH=OVERLAP/2;
        constexpr int lo=H-OH;
        constexpr int hi=H+OH;
        for(int j=0;j<N;++j){
            float a;
            if(j<lo) a=0.f;
            else if(j>=hi) a=1.f;
            else {
                const int q=j-lo;
                // Half-sample sine crossfade. Reverse partners are complementary.
                a=std::sin((float(q)+0.5f)*PI/(2.0f*float(OVERLAP)));
            }
            mdct_w[j]=a;
            const float b2=std::max(0.f,1.f-a*a);
            mdct_w[N+j]=std::sqrt(b2);
        }
    }
    void apply(const float* x,float* y){
        int m=0;
#if defined(__AVX2__)
        const __m256i take_e=_mm256_setr_epi32(0,2,4,6,0,2,4,6);
        const __m256i take_o=_mm256_setr_epi32(1,3,5,7,1,3,5,7);
        const __m256i rev8=_mm256_setr_epi32(7,6,5,4,3,2,1,0);
        for(;m+8<=M;m+=8){
            // The pre-rotation uses x[2m+2i] and x[N-1-2m-2i].  Both
            // sequences live in contiguous 16-float windows; AVX2 gathers are
            // therefore unnecessary.  Deinterleave two regular loads and
            // reverse the backward sequence.
            const float* fa=x+2*m;
            const __m256 fa0=_mm256_loadu_ps(fa);
            const __m256 fa1=_mm256_loadu_ps(fa+8);
            const __m256 ae=_mm256_shuffle_ps(fa0,fa1,_MM_SHUFFLE(2,0,2,0));
            const __m256 a=_mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(ae),_MM_SHUFFLE(3,1,2,0)));

            const float* fb=x+(N-16-2*m);
            const __m256 fb0=_mm256_loadu_ps(fb);
            const __m256 fb1=_mm256_loadu_ps(fb+8);
            const __m256 bo=_mm256_shuffle_ps(fb0,fb1,_MM_SHUFFLE(3,1,3,1));
            const __m256 b_asc=_mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(bo),_MM_SHUFFLE(3,1,2,0)));
            const __m256 b=_mm256_permutevar8x32_ps(b_asc,rev8);
            const __m256 c=_mm256_loadu_ps(pre_c.data()+m), ss=_mm256_loadu_ps(pre_s.data()+m);
#if defined(__FMA__)
            const __m256 vr=_mm256_fmadd_ps(b,ss,_mm256_mul_ps(a,c));
            const __m256 vi=_mm256_fnmadd_ps(b,c,_mm256_mul_ps(a,ss));
#else
            const __m256 vr=_mm256_add_ps(_mm256_mul_ps(a,c),_mm256_mul_ps(b,ss));
            const __m256 vi=_mm256_sub_ps(_mm256_mul_ps(a,ss),_mm256_mul_ps(b,c));
#endif
            if constexpr (M==512 || M==128) {
                const __m256 er=_mm256_permutevar8x32_ps(vr,take_e), ei=_mm256_permutevar8x32_ps(vi,take_e);
                const __m256 orr=_mm256_permutevar8x32_ps(vr,take_o), oi=_mm256_permutevar8x32_ps(vi,take_o);
                _mm_storeu_ps(fft.even.re.data()+m/2,_mm256_castps256_ps128(er));
                _mm_storeu_ps(fft.even.im.data()+m/2,_mm256_castps256_ps128(ei));
                _mm_storeu_ps(fft.odd.re.data()+m/2,_mm256_castps256_ps128(orr));
                _mm_storeu_ps(fft.odd.im.data()+m/2,_mm256_castps256_ps128(oi));
            } else {
                _mm256_storeu_ps(fft.re.data()+m,vr);
                _mm256_storeu_ps(fft.im.data()+m,vi);
            }
        }
#endif
        for(;m<M;++m){
            const float a=x[2*m], b=x[N-1-2*m], c=pre_c[m],ss=pre_s[m];
            const float vr=a*c+b*ss, vi=a*ss-b*c;
            if constexpr (M==512 || M==128) {
                const int q=m>>1;
                if((m&1)==0){ fft.even.re[q]=vr; fft.even.im[q]=vi; }
                else { fft.odd.re[q]=vr; fft.odd.im[q]=vi; }
            } else { fft.re[m]=vr; fft.im[m]=vi; }
        }
        if constexpr (is_pow4_constexpr(M) && N>=1024) {
            // DIF radix-4 already computes every FFT value exactly once.  The old
            // path copied the whole complex spectrum through a digit-reversal buffer
            // only for DCT-IV to read it once.  Read natural bins through perm[]
            // directly and eliminate that O(M) copy without changing arithmetic.
            fft.forward_unordered();
            int r=0;
            // Boundary bins have wrapped indices; handle them scalar, vectorize the interior.
            {
                const int je=0, jo=(M==1)?0:1;
                y[0]=fft.natural_re(je)*even_c[0]-fft.natural_im(je)*even_s[0];
                y[1]=fft.natural_re(jo)*odd_c[0]+fft.natural_im(jo)*odd_s[0];
                r=1;
            }
#if defined(__AVX2__)
            const __m256i revlane=_mm256_setr_epi32(0,-1,-2,-3,-4,-5,-6,-7);
            for(;r+8<=M-1;r+=8){
                const __m256i je_nat=_mm256_add_epi32(_mm256_set1_epi32(M-r),revlane);
                const __m256i jo_nat=_mm256_setr_epi32(r+1,r+2,r+3,r+4,r+5,r+6,r+7,r+8);
                // perm32 maps natural FFT bins to DIF storage positions.
                const __m256i je_store=_mm256_i32gather_epi32(fft.perm32.data(),je_nat,4);
                const __m256i jo_store=_mm256_i32gather_epi32(fft.perm32.data(),jo_nat,4);
                const __m256 er=_mm256_i32gather_ps(fft.re.data(),je_store,4);
                const __m256 ei=_mm256_i32gather_ps(fft.im.data(),je_store,4);
                const __m256 orr=_mm256_i32gather_ps(fft.re.data(),jo_store,4);
                const __m256 oii=_mm256_i32gather_ps(fft.im.data(),jo_store,4);
                const __m256 ec=_mm256_loadu_ps(even_c.data()+r), es=_mm256_loadu_ps(even_s.data()+r);
                const __m256 oc=_mm256_loadu_ps(odd_c.data()+r), os=_mm256_loadu_ps(odd_s.data()+r);
#if defined(__FMA__)
                const __m256 ye=_mm256_fmsub_ps(er,ec,_mm256_mul_ps(ei,es));
                const __m256 yo=_mm256_fmadd_ps(orr,oc,_mm256_mul_ps(oii,os));
#else
                const __m256 ye=_mm256_sub_ps(_mm256_mul_ps(er,ec),_mm256_mul_ps(ei,es));
                const __m256 yo=_mm256_add_ps(_mm256_mul_ps(orr,oc),_mm256_mul_ps(oii,os));
#endif
                const __m256 lo=_mm256_unpacklo_ps(ye,yo), hi=_mm256_unpackhi_ps(ye,yo);
                _mm256_storeu_ps(y+2*r,_mm256_permute2f128_ps(lo,hi,0x20));
                _mm256_storeu_ps(y+2*r+8,_mm256_permute2f128_ps(lo,hi,0x31));
            }
#endif
            for(;r<M;++r){
                const int je=(r==0)?0:(M-r);
                const int jo=(r+1==M)?0:(r+1);
                y[2*r]=fft.natural_re(je)*even_c[r]-fft.natural_im(je)*even_s[r];
                y[2*r+1]=fft.natural_re(jo)*odd_c[r]+fft.natural_im(jo)*odd_s[r];
            }
        } else {
            if constexpr (M==512 || M==128) fft.forward_preloaded();
            else fft.forward();
            int r=0;
#if defined(__AVX2__)
            // je walks backwards, jo forwards. Gather the former and load/gather the latter.
            const __m256i revlane=_mm256_setr_epi32(0,-1,-2,-3,-4,-5,-6,-7);
            for(;r+8<=M && r>0;r+=8){} // keeps scalar r=0 boundary below
#endif
            // Scalar boundary first.
            {
                const int je=0, jo=(M==1)?0:1;
                y[0]=fft.re[je]*even_c[0]-fft.im[je]*even_s[0];
                y[1]=fft.re[jo]*odd_c[0]+fft.im[jo]*odd_s[0];
                r=1;
            }
#if defined(__AVX2__)
            for(;r+8<=M-1;r+=8){
                const __m256i je=_mm256_add_epi32(_mm256_set1_epi32(M-r),revlane);
                const __m256 er=_mm256_i32gather_ps(fft.re.data(),je,4);
                const __m256 ei=_mm256_i32gather_ps(fft.im.data(),je,4);
                const __m256 orr=_mm256_loadu_ps(fft.re.data()+r+1);
                const __m256 oii=_mm256_loadu_ps(fft.im.data()+r+1);
                const __m256 ec=_mm256_loadu_ps(even_c.data()+r), es=_mm256_loadu_ps(even_s.data()+r);
                const __m256 oc=_mm256_loadu_ps(odd_c.data()+r), os=_mm256_loadu_ps(odd_s.data()+r);
#if defined(__FMA__)
                const __m256 ye=_mm256_fmsub_ps(er,ec,_mm256_mul_ps(ei,es));
                const __m256 yo=_mm256_fmadd_ps(orr,oc,_mm256_mul_ps(oii,os));
#else
                const __m256 ye=_mm256_sub_ps(_mm256_mul_ps(er,ec),_mm256_mul_ps(ei,es));
                const __m256 yo=_mm256_add_ps(_mm256_mul_ps(orr,oc),_mm256_mul_ps(oii,os));
#endif
                const __m256 lo=_mm256_unpacklo_ps(ye,yo), hi=_mm256_unpackhi_ps(ye,yo);
                _mm256_storeu_ps(y+2*r,_mm256_permute2f128_ps(lo,hi,0x20));
                _mm256_storeu_ps(y+2*r+8,_mm256_permute2f128_ps(lo,hi,0x31));
            }
#endif
            for(;r<M;++r){
                const int je=(r==0)?0:(M-r);
                const int jo=(r+1==M)?0:(r+1);
                y[2*r]=fft.re[je]*even_c[r]-fft.im[je]*even_s[r];
                y[2*r+1]=fft.re[jo]*odd_c[r]+fft.im[jo]*odd_s[r];
            }
        }
    }

    // Frozen LX-VOXEL v0.2.3 Fold-MDCT analysis path.
    // The conceptual transform is a sparse 2N Princen-Bradley MDCT.
    // Zero/one window plateaus are eliminated analytically, so only the
    // 128 boundary samples require window multiplies before the N-point fold.
    void apply_lapped(const float* x,bool has_left,bool has_right,
                      const float* /*lc*/,const float* /*ls*/,float* y){
        // VOXEL Fold-MDCT. Algebraically identical to the sparse 2N PB-window MDCT,
        // but the zero/one plateaus are eliminated analytically: only OVERLAP edge
        // samples need multiplies. The N-OVERLAP interior is pure signed folding.
        constexpr int H=N/2;
        constexpr int OH=OVERLAP/2;
        constexpr int LO=H-OH;

        // Right boundary alias: current tail + next-frame lookahead.
        if(has_right){
            for(int n=0;n<OH;++n){
                const int ja=N+(H-1-n);
                const int jb=N+(H+n);
                mdct_fold[n]=-x[N-1-n]*mdct_w[ja]-x[N+n]*mdct_w[jb];
            }
        }else{
            // Exposed stream boundary: there is no complementary neighbour
            // contribution.  Pre-compensate the synthesis PB weight so the
            // boundary remains perfect-reconstruction rather than w^2-scaled.
            for(int n=0;n<OH;++n){
                const int ja=N+(H-1-n);
                mdct_fold[n]=-x[N-1-n]/mdct_w[ja];
            }
        }
        for(int n=OH;n<H;++n) mdct_fold[n]=-x[N-1-n];

        // Interior first half is another signed reverse copy.
        for(int n=0;n<LO;++n) mdct_fold[H+n]=-x[H-1-n];

        // Left boundary alias: previous-frame lookback + current head.
        if(has_left){
            for(int r=0;r<OH;++r){
                const int n=LO+r;
                const int jc=H-OH+r;
                const int jd=H+OH-1-r;
                mdct_fold[H+n]=x[-OH+r]*mdct_w[jc]-x[OH-1-r]*mdct_w[jd];
            }
        }else{
            // Same exposed-boundary compensation at stream start.  The
            // relevant PB weights are safely bounded away from zero here.
            for(int r=0;r<OH;++r){
                const int n=LO+r;
                const int jd=H+OH-1-r;
                mdct_fold[H+n]=-x[OH-1-r]/mdct_w[jd];
            }
        }
        apply(mdct_fold.data(),y);
    }

    // Decoder-only preparation: pre-rotate directly into the FFT representation.
    // Kept separate from apply() so synthesis can fuse post-rotation with OLA and
    // avoid materializing the N-float inverse DCT scratch vector.
    LX_ALWAYS_INLINE void prepare_inverse_fft(const float* x){
        int m=0;
#if defined(__AVX2__)
        const __m256i take_e=_mm256_setr_epi32(0,2,4,6,0,2,4,6);
        const __m256i take_r=_mm256_setr_epi32(7,5,3,1,7,5,3,1);
        const __m256i take_o=_mm256_setr_epi32(1,3,5,7,1,3,5,7);
        for(;m+8<=M;m+=8){
            const float* pf=x+2*m;
            const __m256 f0=_mm256_loadu_ps(pf), f1=_mm256_loadu_ps(pf+8);
            const __m256 pe0=_mm256_permutevar8x32_ps(f0,take_e), pe1=_mm256_permutevar8x32_ps(f1,take_e);
            const __m256 a=_mm256_permute2f128_ps(pe0,pe1,0x20);
            const float* pr=x+(N-16-2*m);
            const __m256 r0=_mm256_loadu_ps(pr), r1=_mm256_loadu_ps(pr+8);
            const __m256 po0=_mm256_permutevar8x32_ps(r0,take_r), po1=_mm256_permutevar8x32_ps(r1,take_r);
            const __m256 b=_mm256_permute2f128_ps(po1,po0,0x20);
            const __m256 c=_mm256_loadu_ps(pre_c.data()+m), ss=_mm256_loadu_ps(pre_s.data()+m);
#if defined(__FMA__)
            const __m256 vr=_mm256_fmadd_ps(b,ss,_mm256_mul_ps(a,c));
            const __m256 vi=_mm256_fnmadd_ps(b,c,_mm256_mul_ps(a,ss));
#else
            const __m256 vr=_mm256_add_ps(_mm256_mul_ps(a,c),_mm256_mul_ps(b,ss));
            const __m256 vi=_mm256_sub_ps(_mm256_mul_ps(a,ss),_mm256_mul_ps(b,c));
#endif
            if constexpr (M==512 || M==128) {
                const __m256 er=_mm256_permutevar8x32_ps(vr,take_e), ei=_mm256_permutevar8x32_ps(vi,take_e);
                const __m256 orr=_mm256_permutevar8x32_ps(vr,take_o), oi=_mm256_permutevar8x32_ps(vi,take_o);
                _mm_storeu_ps(fft.even.re.data()+m/2,_mm256_castps256_ps128(er));
                _mm_storeu_ps(fft.even.im.data()+m/2,_mm256_castps256_ps128(ei));
                _mm_storeu_ps(fft.odd.re.data()+m/2,_mm256_castps256_ps128(orr));
                _mm_storeu_ps(fft.odd.im.data()+m/2,_mm256_castps256_ps128(oi));
            } else {
                _mm256_storeu_ps(fft.re.data()+m,vr);
                _mm256_storeu_ps(fft.im.data()+m,vi);
            }
        }
#endif
        for(;m<M;++m){
            const float a=x[2*m], b=x[N-1-2*m], c=pre_c[m],ss=pre_s[m];
            const float vr=a*c+b*ss, vi=a*ss-b*c;
            if constexpr (M==512 || M==128) {
                const int q=m>>1;
                if((m&1)==0){ fft.even.re[q]=vr; fft.even.im[q]=vi; }
                else { fft.odd.re[q]=vr; fft.odd.im[q]=vi; }
            } else { fft.re[m]=vr; fft.im[m]=vi; }
        }
        if constexpr (is_pow4_constexpr(M) && N>=1024) fft.forward_unordered();
        else if constexpr (M==512 || M==128) fft.forward_preloaded();
        else fft.forward();
    }

    void inverse_overlap_add(const float* X,size_t pos,size_t ns,float* out){
        constexpr int H=N/2;
        constexpr int OH=OVERLAP/2;
        constexpr int LO=H-OH;

#if defined(__AVX2__)
        // All decoder transform sizes used by VariableLowOverlapMDCT end in a
        // natural-order FFT. Fuse DCT-IV post-rotation with OLA: only the 128
        // PB-boundary samples need scalar/windowed treatment; the N-128 interior
        // goes straight from FFT registers to its final reversed PCM location.
        if constexpr (!(is_pow4_constexpr(M) && N>=1024)) {
            if(pos >= size_t(OH) && pos + size_t(N + OH) <= ns){
                prepare_inverse_fft(X);
                auto post_pair_scalar = [&](int r,float &ye,float &yo){
                    const int je=(r==0)?0:(M-r);
                    const int jo=(r+1==M)?0:(r+1);
                    ye=fft.re[je]*even_c[r]-fft.im[je]*even_s[r];
                    yo=fft.re[jo]*odd_c[r]+fft.im[jo]*odd_s[r];
                };
                // First OH DCT samples: right PB boundary of the current frame.
                for(int r=0;r<OH/2;++r){
                    float ye,yo; post_pair_scalar(r,ye,yo);
                    const int n0=2*r, n1=n0+1;
                    out[pos+size_t(N-1-n0)] += -ye*mdct_w[N+H-1-n0];
                    out[pos+size_t(N+n0)]   += -ye*mdct_w[N+H+n0];
                    out[pos+size_t(N-1-n1)] += -yo*mdct_w[N+H-1-n1];
                    out[pos+size_t(N+n1)]   += -yo*mdct_w[N+H+n1];
                }
                // Interior: direct reversed store, 16 DCT samples per iteration.
                const __m256i ridx=_mm256_setr_epi32(7,6,5,4,3,2,1,0);
                const __m256 sign=_mm256_set1_ps(-0.0f);
                int r=OH/2;
                const int rend=(N-OH)/2;
                for(;r+8<=rend;r+=8){
                    // Natural-order FFT bins M-r ... M-r-7 are contiguous in
                    // reverse order.  Load once and reverse lanes instead of two
                    // expensive AVX2 gathers.
                    const __m256 er=_mm256_permutevar8x32_ps(_mm256_loadu_ps(fft.re.data()+M-r-7),ridx);
                    const __m256 ei=_mm256_permutevar8x32_ps(_mm256_loadu_ps(fft.im.data()+M-r-7),ridx);
                    const __m256 orr=_mm256_loadu_ps(fft.re.data()+r+1);
                    const __m256 oii=_mm256_loadu_ps(fft.im.data()+r+1);
                    const __m256 ec=_mm256_loadu_ps(even_c.data()+r), es=_mm256_loadu_ps(even_s.data()+r);
                    const __m256 oc=_mm256_loadu_ps(odd_c.data()+r), os=_mm256_loadu_ps(odd_s.data()+r);
#if defined(__FMA__)
                    const __m256 ye=_mm256_fmsub_ps(er,ec,_mm256_mul_ps(ei,es));
                    const __m256 yo=_mm256_fmadd_ps(orr,oc,_mm256_mul_ps(oii,os));
#else
                    const __m256 ye=_mm256_sub_ps(_mm256_mul_ps(er,ec),_mm256_mul_ps(ei,es));
                    const __m256 yo=_mm256_add_ps(_mm256_mul_ps(orr,oc),_mm256_mul_ps(oii,os));
#endif
                    const __m256 lo=_mm256_unpacklo_ps(ye,yo), hi=_mm256_unpackhi_ps(ye,yo);
                    __m256 y0=_mm256_permute2f128_ps(lo,hi,0x20);
                    __m256 y1=_mm256_permute2f128_ps(lo,hi,0x31);
                    y0=_mm256_xor_ps(y0,sign); y1=_mm256_xor_ps(y1,sign);
                    y0=_mm256_permutevar8x32_ps(y0,ridx);
                    y1=_mm256_permutevar8x32_ps(y1,ridx);
                    const int n0=2*r;
                    float* dst=out+pos+size_t(N-(n0+16));
                    _mm256_storeu_ps(dst,y1);
                    _mm256_storeu_ps(dst+8,y0);
                }
                // Last OH DCT samples: left PB boundary contribution.
                for(;r<M;++r){
                    float ye,yo; post_pair_scalar(r,ye,yo);
                    const int n0=2*r, n1=n0+1;
                    const int rr0=n0-(N-OH), rr1=n1-(N-OH);
                    out[pos-OH+rr0] += ye*mdct_w[H-OH+rr0];
                    out[pos+OH-1-rr0] += -ye*mdct_w[H+OH-1-rr0];
                    out[pos-OH+rr1] += yo*mdct_w[H-OH+rr1];
                    out[pos+OH-1-rr1] += -yo*mdct_w[H+OH-1-rr1];
                }
                return;
            }
        }
#endif
        // Stream boundaries and the unused large pure-radix transform retain the
        // conservative scratch-vector implementation.
        apply(X,mdct_fold.data());

#if defined(__AVX2__)
        // All normal interior frames satisfy this.  Separating stream-boundary
        // clipping from the hot synthesis path removes four bounds tests per
        // output sample and lets OLA run as straight AVX2 loads/stores.
        if(pos >= size_t(OH) && pos + size_t(N + OH) <= ns){
            const __m256i rev = _mm256_setr_epi32(7,6,5,4,3,2,1,0);
            int n=OH;
            for(; n+8<=H; n+=8){
                __m256 v=_mm256_loadu_ps(mdct_fold.data()+n);
                v=_mm256_permutevar8x32_ps(v,rev);
                v=_mm256_sub_ps(_mm256_setzero_ps(),v);
                _mm256_storeu_ps(out + pos + size_t(N-1-(n+7)),v);
            }
            for(;n<H;++n) out[pos+size_t(N-1-n)]=-mdct_fold[n];

            n=0;
            for(; n+8<=LO; n+=8){
                __m256 v=_mm256_loadu_ps(mdct_fold.data()+H+n);
                v=_mm256_permutevar8x32_ps(v,rev);
                v=_mm256_sub_ps(_mm256_setzero_ps(),v);
                _mm256_storeu_ps(out + pos + size_t(H-1-(n+7)),v);
            }
            for(;n<LO;++n) out[pos+size_t(H-1-n)]=-mdct_fold[H+n];

            int r=0;
            for(;r+8<=OH;r+=8){
                const __m256 b=_mm256_loadu_ps(mdct_fold.data()+H+LO+r);
                const __m256 wc=_mm256_loadu_ps(mdct_w.data()+H-OH+r);
                __m256 left=_mm256_mul_ps(b,wc);
                __m256 wd=_mm256_loadu_ps(mdct_w.data()+H+OH-1-(r+7));
                wd=_mm256_permutevar8x32_ps(wd,rev);
                __m256 right=_mm256_sub_ps(_mm256_setzero_ps(),_mm256_mul_ps(b,wd));
                right=_mm256_permutevar8x32_ps(right,rev);
                _mm256_storeu_ps(out + pos - OH + r,
                                 _mm256_add_ps(_mm256_loadu_ps(out + pos - OH + r),left));
                _mm256_storeu_ps(out + pos + OH - (r+8),
                                 _mm256_add_ps(_mm256_loadu_ps(out + pos + OH - (r+8)),right));
            }
            for(;r<OH;++r){
                const int nn=LO+r; const float b=mdct_fold[H+nn];
                out[pos-OH+r]+=b*mdct_w[H-OH+r];
                out[pos+OH-1-r]+=-b*mdct_w[H+OH-1-r];
            }

            n=0;
            for(;n+8<=OH;n+=8){
                const __m256 a=_mm256_loadu_ps(mdct_fold.data()+n);
                __m256 wa=_mm256_loadu_ps(mdct_w.data()+N+H-1-(n+7));
                wa=_mm256_permutevar8x32_ps(wa,rev);
                __m256 left=_mm256_sub_ps(_mm256_setzero_ps(),_mm256_mul_ps(a,wa));
                left=_mm256_permutevar8x32_ps(left,rev);
                const __m256 wb=_mm256_loadu_ps(mdct_w.data()+N+H+n);
                const __m256 right=_mm256_sub_ps(_mm256_setzero_ps(),_mm256_mul_ps(a,wb));
                _mm256_storeu_ps(out + pos + N - (n+8),
                                 _mm256_add_ps(_mm256_loadu_ps(out + pos + N - (n+8)),left));
                _mm256_storeu_ps(out + pos + N + n,
                                 _mm256_add_ps(_mm256_loadu_ps(out + pos + N + n),right));
            }
            for(;n<OH;++n){
                const float a=mdct_fold[n];
                out[pos+size_t(N-1-n)]+=-a*mdct_w[N+H-1-n];
                out[pos+size_t(N+n)]+=-a*mdct_w[N+H+n];
            }
            return;
        }
#endif

        // Boundary fallback (first/last frame only).
        for(int n=OH;n<H;++n){
            const size_t g=pos+size_t(N-1-n);
            if(g<ns) out[g]=-mdct_fold[n];
        }
        for(int n=0;n<LO;++n){
            const size_t g=pos+size_t(H-1-n);
            if(g<ns) out[g]=-mdct_fold[H+n];
        }
        for(int r=0;r<OH;++r){
            const int n=LO+r; const float b=mdct_fold[H+n];
            const int jc=H-OH+r, jd=H+OH-1-r;
            const ptrdiff_t gl=ptrdiff_t(pos)-OH+r;
            if(gl>=0 && size_t(gl)<ns) out[gl]+=b*mdct_w[jc];
            const size_t gr=pos+size_t(OH-1-r);
            if(gr<ns) out[gr]+=-b*mdct_w[jd];
        }
        for(int n=0;n<OH;++n){
            const float a=mdct_fold[n];
            const int ja=N+(H-1-n), jb=N+(H+n);
            const size_t gl=pos+size_t(N-1-n);
            if(gl<ns) out[gl]+=-a*mdct_w[ja];
            const size_t gr=pos+size_t(N+n);
            if(gr<ns) out[gr]+=-a*mdct_w[jb];
        }
    }

    // Single-thread stereo synthesis: transform both channels into one interleaved
    // destination.  This removes two full-size planar output buffers plus the
    // final planar->interleaved copy from the decoder hot path.
    void inverse_overlap_add_stereo(DCT4Ortho& ro,const float* XL,const float* XR,
                                    size_t pos,size_t ns,float* out){
        constexpr int H=N/2, OH=OVERLAP/2, LO=H-OH;
#if defined(__AVX2__)
        if(pos>=size_t(OH) && pos+size_t(N+OH)<=ns){
            prepare_inverse_fft(XL); ro.prepare_inverse_fft(XR);
            auto post_scalar=[&](DCT4Ortho& d,int r,float& ye,float& yo){
                const int je=(r==0)?0:(M-r), jo=(r+1==M)?0:(r+1);
                ye=d.fft.re[je]*d.even_c[r]-d.fft.im[je]*d.even_s[r];
                yo=d.fft.re[jo]*d.odd_c[r]+d.fft.im[jo]*d.odd_s[r];
            };
            for(int r=0;r<OH/2;++r){
                float le,lo,re,rr;post_scalar(*this,r,le,lo);post_scalar(ro,r,re,rr);
                const int n0=2*r,n1=n0+1;
                auto add=[&](size_t g,float l,float rv){out[2*g]+=l;out[2*g+1]+=rv;};
                add(pos+size_t(N-1-n0),-le*mdct_w[N+H-1-n0],-re*ro.mdct_w[N+H-1-n0]);
                add(pos+size_t(N+n0),-le*mdct_w[N+H+n0],-re*ro.mdct_w[N+H+n0]);
                add(pos+size_t(N-1-n1),-lo*mdct_w[N+H-1-n1],-rr*ro.mdct_w[N+H-1-n1]);
                add(pos+size_t(N+n1),-lo*mdct_w[N+H+n1],-rr*ro.mdct_w[N+H+n1]);
            }
            const __m256i ridx=_mm256_setr_epi32(7,6,5,4,3,2,1,0);
            const __m256 sign=_mm256_set1_ps(-0.0f);
            int r=OH/2;const int rend=(N-OH)/2;
            for(;r+8<=rend;r+=8){
#define LX_POST16(D, Y0, Y1) do { \
                const __m256 er=_mm256_permutevar8x32_ps(_mm256_loadu_ps((D).fft.re.data()+M-r-7),ridx); \
                const __m256 ei=_mm256_permutevar8x32_ps(_mm256_loadu_ps((D).fft.im.data()+M-r-7),ridx); \
                const __m256 orr=_mm256_loadu_ps((D).fft.re.data()+r+1); \
                const __m256 oii=_mm256_loadu_ps((D).fft.im.data()+r+1); \
                const __m256 ec=_mm256_loadu_ps((D).even_c.data()+r), es=_mm256_loadu_ps((D).even_s.data()+r); \
                const __m256 oc=_mm256_loadu_ps((D).odd_c.data()+r), os=_mm256_loadu_ps((D).odd_s.data()+r); \
                __m256 ye,yo; \
                /* FMA preserves the same R188 transform algebra. */ \
                ye=_mm256_fmsub_ps(er,ec,_mm256_mul_ps(ei,es)); \
                yo=_mm256_fmadd_ps(orr,oc,_mm256_mul_ps(oii,os)); \
                const __m256 ul=_mm256_unpacklo_ps(ye,yo), uh=_mm256_unpackhi_ps(ye,yo); \
                (Y0)=_mm256_permute2f128_ps(ul,uh,0x20); (Y1)=_mm256_permute2f128_ps(ul,uh,0x31); \
                (Y0)=_mm256_permutevar8x32_ps(_mm256_xor_ps((Y0),sign),ridx); \
                (Y1)=_mm256_permutevar8x32_ps(_mm256_xor_ps((Y1),sign),ridx); \
            } while(0)
                __m256 l0,l1,r0,r1;LX_POST16((*this),l0,l1);LX_POST16(ro,r0,r1);
#undef LX_POST16
                const int n0=2*r;float* dst=out+2*(pos+size_t(N-(n0+16)));
                auto store8=[&](float* d,__m256 lv,__m256 rv){
                    const __m256 a=_mm256_unpacklo_ps(lv,rv),b=_mm256_unpackhi_ps(lv,rv);
                    _mm256_storeu_ps(d,_mm256_permute2f128_ps(a,b,0x20));
                    _mm256_storeu_ps(d+8,_mm256_permute2f128_ps(a,b,0x31));
                };
                store8(dst,l1,r1);store8(dst+16,l0,r0);
            }
            for(;r<M;++r){
                float le,lo,re,rr;post_scalar(*this,r,le,lo);post_scalar(ro,r,re,rr);
                const int n0=2*r,n1=n0+1,rr0=n0-(N-OH),rr1=n1-(N-OH);
                auto add=[&](size_t g,float l,float rv){out[2*g]+=l;out[2*g+1]+=rv;};
                add(pos-OH+rr0,le*mdct_w[H-OH+rr0],re*ro.mdct_w[H-OH+rr0]);
                add(pos+OH-1-rr0,-le*mdct_w[H+OH-1-rr0],-re*ro.mdct_w[H+OH-1-rr0]);
                add(pos-OH+rr1,lo*mdct_w[H-OH+rr1],rr*ro.mdct_w[H-OH+rr1]);
                add(pos+OH-1-rr1,-lo*mdct_w[H+OH-1-rr1],-rr*ro.mdct_w[H+OH-1-rr1]);
            }
            return;
        }
#endif
        // Stream-boundary fallback.  Only first/last frames use this path.
        apply(XL,mdct_fold.data()); ro.apply(XR,ro.mdct_fold.data());
        auto setv=[&](size_t g,float l,float rv){if(g<ns){out[2*g]=l;out[2*g+1]=rv;}};
        auto addv=[&](ptrdiff_t gg,float l,float rv){if(gg>=0&&size_t(gg)<ns){const size_t g=size_t(gg);out[2*g]+=l;out[2*g+1]+=rv;}};
        for(int n=OH;n<H;++n){const size_t g=pos+size_t(N-1-n);setv(g,-mdct_fold[n],-ro.mdct_fold[n]);}
        for(int n=0;n<LO;++n){const size_t g=pos+size_t(H-1-n);setv(g,-mdct_fold[H+n],-ro.mdct_fold[H+n]);}
        for(int r=0;r<OH;++r){const int n=LO+r,jc=H-OH+r,jd=H+OH-1-r;addv(ptrdiff_t(pos)-OH+r,mdct_fold[H+n]*mdct_w[jc],ro.mdct_fold[H+n]*ro.mdct_w[jc]);addv(ptrdiff_t(pos)+OH-1-r,-mdct_fold[H+n]*mdct_w[jd],-ro.mdct_fold[H+n]*ro.mdct_w[jd]);}
        for(int n=0;n<OH;++n){const int ja=N+(H-1-n),jb=N+(H+n);addv(ptrdiff_t(pos)+N-1-n,-mdct_fold[n]*mdct_w[ja],-ro.mdct_fold[n]*ro.mdct_w[ja]);addv(ptrdiff_t(pos)+N+n,-mdct_fold[n]*mdct_w[jb],-ro.mdct_fold[n]*ro.mdct_w[jb]);}
    }
};

struct Frame {
    size_t pos=0;
    int n=2048;
    size_t off=0;
};

// Frozen nonstationary 128-sample low-overlap Fold-MDCT.
//
// Each frame is a critically sampled sparse 2N Princen-Bradley MDCT:
// PB window -> signed MDCT fold -> orthonormal N-point DCT-IV.
// Decoder synthesis applies DCT-IV -> F^T -> identical PB window -> OLA.
//
// The overlap operator is independent of block length, so arbitrary
// 2048/1024/512/256 sequences remain exactly reconstructible without transition syntax.
struct VariableLowOverlapMDCT {
    DCT4Ortho<1024> d2048;
    DCT4Ortho<768>  d1024;
    DCT4Ortho<512>  d512;
    DCT4Ortho<384>  d256;
    std::array<float,HALF_OVERLAP> c{},s{};

    VariableLowOverlapMDCT(){
        // Same Princen-Bradley sine family as LX-VOXEL's 128-sample window.
        // Using complementary entries makes c^2+s^2=1 analytically.
        for(int i=0;i<HALF_OVERLAP;++i){
            const float u=(float(i)+0.5f)/float(OVERLAP);
            s[i]=std::sin(0.5f*PI*u);
            c[i]=std::cos(0.5f*PI*u);
        }
    }

    void analysis(const float* input,size_t ns,
                  const std::vector<Frame>& frames,
                  std::vector<float>& coeff){
        size_t total=0;
        for(const auto&f:frames) total=std::max(total,f.off+size_t(f.n));
        coeff.assign(total,0.0f);
        for(size_t fi=0;fi<frames.size();++fi){
            const auto&f=frames[fi];
            if(f.pos+size_t(f.n)>ns) break;
            const bool left=fi!=0;
            const bool right=fi+1<frames.size();
            switch(f.n){
                case 1024: d2048.apply_lapped(input+f.pos,left,right,nullptr,nullptr,coeff.data()+f.off); break;
                case 768: d1024.apply_lapped(input+f.pos,left,right,nullptr,nullptr,coeff.data()+f.off); break;
                case  512: d512 .apply_lapped(input+f.pos,left,right,nullptr,nullptr,coeff.data()+f.off); break;
                case  384: d256 .apply_lapped(input+f.pos,left,right,nullptr,nullptr,coeff.data()+f.off); break;
                default: throw std::runtime_error("unsupported Fold-MDCT block size");
            }
        }
    }

    void synthesis(const float* coeff,size_t ns,
                   const std::vector<Frame>& frames,
                   std::vector<float>& output){
        output.assign(ns,0.0f);
        for(const auto&f:frames){
            // The final transform may extend beyond the original PCM length;
            // its missing input samples are zero-padded by the encoder.  OLA
            // already clips every write to ns, so decoding it is required to
            // reconstruct the real tail samples.
            switch(f.n){
                case 1024: d2048.inverse_overlap_add(coeff+f.off,f.pos,ns,output.data()); break;
                case 768: d1024.inverse_overlap_add(coeff+f.off,f.pos,ns,output.data()); break;
                case  512: d512 .inverse_overlap_add(coeff+f.off,f.pos,ns,output.data()); break;
                case  384: d256 .inverse_overlap_add(coeff+f.off,f.pos,ns,output.data()); break;
                default: throw std::runtime_error("unsupported Fold-MDCT block size");
            }
        }
    }

    void synthesis_stereo(const float* coeffL,const float* coeffR,size_t ns,
                          const std::vector<Frame>& frames,std::vector<float>& output,
                          VariableLowOverlapMDCT& right){
        output.assign(ns*2,0.0f);
        for(const auto&f:frames){
            switch(f.n){
                case 1024: d2048.inverse_overlap_add_stereo(right.d2048,coeffL+f.off,coeffR+f.off,f.pos,ns,output.data());break;
                case 768: d1024.inverse_overlap_add_stereo(right.d1024,coeffL+f.off,coeffR+f.off,f.pos,ns,output.data());break;
                case 512: d512.inverse_overlap_add_stereo(right.d512,coeffL+f.off,coeffR+f.off,f.pos,ns,output.data());break;
                case 384: d256.inverse_overlap_add_stereo(right.d256,coeffL+f.off,coeffR+f.off,f.pos,ns,output.data());break;
                default: throw std::runtime_error("unsupported Fold-MDCT block size");
            }
        }
    }

};

} // namespace lx_nsmdct
