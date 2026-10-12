#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#if defined(__AVX2__) || defined(__SSE4_1__)
#include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#include "rANS/rans.hpp"
namespace lx_pitch {
// LX-VOXEL Pitch Predictor — VOXEL original design.
// The pre/post-comb concept is general transform-codec practice; this implementation
// is independently designed for VOXEL: whitening-benefit group gating, log-lag
// continuity, fundamental/doubling rejection, 4-bit gain, and track-adaptive rANS
// delta-lag coding. It is not a CELT source-code transplant.
constexpr int PITCH_FRAME=960,PITCH_GROUP=32,DS=4,LAG_MIN=24,LAG_MAX=960;
constexpr float COMB_C0=.7998046875f,COMB_C1=.10009765625f,GAIN_MAX=.075f;
constexpr double CONT_LAMBDA=9.5e-4,WHITEN_GATE=.033;
constexpr int GAIN_LEVELS=15,DELTA_ALPHA=33;
struct Schedule{std::vector<uint16_t> lag;std::vector<uint8_t> gain_q,group_on;};
struct BitWriter{std::vector<uint8_t> out;uint64_t acc=0;unsigned bits=0;void put(uint32_t v,unsigned n){if(!n)return;acc|=(uint64_t(v)&((uint64_t(1)<<n)-1u))<<bits;bits+=n;while(bits>=8){out.push_back(uint8_t(acc));acc>>=8;bits-=8;}}void finish(){if(bits){out.push_back(uint8_t(acc));acc=0;bits=0;}}};
struct BitReader{const uint8_t*p,*e;uint64_t acc=0;unsigned bits=0;BitReader(const uint8_t*a,const uint8_t*b):p(a),e(b){}bool get(unsigned n,uint32_t&v){while(bits<n){if(p>=e)return false;acc|=uint64_t(*p++)<<bits;bits+=8;}v=uint32_t(acc&((uint64_t(1)<<n)-1u));acc>>=n;bits-=n;return true;}};
inline uint32_t zz_enc(int x){return x>=0?uint32_t(2*x):uint32_t(-2*x-1);}inline int zz_dec(uint32_t z){return(z&1u)?-int((z+1u)>>1):int(z>>1);}

// The pitch search keeps its double-precision correlation statistic. Convert
// four input float samples to four FP64 lanes per iteration; a scalar tail
// handles incomplete frames. No signal-dependent branches in the SIMD loop.
inline double corr_dot(const float* a,const float* b,int n) noexcept {
#if defined(__AVX2__)
  __m256 z0=_mm256_setzero_ps(),z1=_mm256_setzero_ps(),z2=_mm256_setzero_ps(),z3=_mm256_setzero_ps();
  int i=0;
  for(;i+32<=n;i+=32){
    z0=_mm256_fmadd_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i),z0);
    z1=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+8),_mm256_loadu_ps(b+i+8),z1);
    z2=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+16),_mm256_loadu_ps(b+i+16),z2);
    z3=_mm256_fmadd_ps(_mm256_loadu_ps(a+i+24),_mm256_loadu_ps(b+i+24),z3);
  }
  alignas(32) float acc[8];_mm256_store_ps(acc,_mm256_add_ps(_mm256_add_ps(z0,z1),_mm256_add_ps(z2,z3)));
  double dot=0;for(int k=0;k<8;++k)dot+=double(acc[k]);
  for(;i<n;++i)dot+=double(a[i])*b[i];return dot;
#else
  double dot=0;for(int i=0;i<n;++i)dot+=double(a[i])*b[i];return dot;
#endif
}

inline void corr_dot2(const float* a,const float* b0,const float* b1,int n,double& d0,double& d1) noexcept {
#if defined(__AVX2__)
  __m256 a00=_mm256_setzero_ps(),a01=_mm256_setzero_ps(),a02=_mm256_setzero_ps(),a03=_mm256_setzero_ps();
  __m256 a10=_mm256_setzero_ps(),a11=_mm256_setzero_ps(),a12=_mm256_setzero_ps(),a13=_mm256_setzero_ps();
  int i=0;
  for(;i+32<=n;i+=32){
    const __m256 x0=_mm256_loadu_ps(a+i),x1=_mm256_loadu_ps(a+i+8),x2=_mm256_loadu_ps(a+i+16),x3=_mm256_loadu_ps(a+i+24);
    a00=_mm256_fmadd_ps(x0,_mm256_loadu_ps(b0+i),a00);
    a01=_mm256_fmadd_ps(x1,_mm256_loadu_ps(b0+i+8),a01);
    a02=_mm256_fmadd_ps(x2,_mm256_loadu_ps(b0+i+16),a02);
    a03=_mm256_fmadd_ps(x3,_mm256_loadu_ps(b0+i+24),a03);
    a10=_mm256_fmadd_ps(x0,_mm256_loadu_ps(b1+i),a10);
    a11=_mm256_fmadd_ps(x1,_mm256_loadu_ps(b1+i+8),a11);
    a12=_mm256_fmadd_ps(x2,_mm256_loadu_ps(b1+i+16),a12);
    a13=_mm256_fmadd_ps(x3,_mm256_loadu_ps(b1+i+24),a13);
  }
  alignas(32) float t0[8],t1[8];
  _mm256_store_ps(t0,_mm256_add_ps(_mm256_add_ps(a00,a01),_mm256_add_ps(a02,a03)));
  _mm256_store_ps(t1,_mm256_add_ps(_mm256_add_ps(a10,a11),_mm256_add_ps(a12,a13)));
  d0=0.0;d1=0.0;for(int k=0;k<8;++k){d0+=double(t0[k]);d1+=double(t1[k]);}
  for(;i<n;++i){const double x=double(a[i]);d0+=x*b0[i];d1+=x*b1[i];}
#else
  d0=corr_dot(a,b0,n);d1=corr_dot(a,b1,n);
#endif
}
inline Schedule analyse(const float*x,size_t ns,std::vector<float>* filter_output=nullptr){
 const size_t nd=(ns+DS-1)/DS;std::vector<float>md(nd);for(size_t j=0;j<nd;++j){size_t a=j*DS,b=std::min(ns,a+size_t(DS));double v=0;for(size_t n=a;n<b;++n)v+=.5*double(x[2*n]+x[2*n+1]);md[j]=float(v/double(b-a));}
 const size_t nf=(ns+PITCH_FRAME-1)/PITCH_FRAME;Schedule sc;sc.lag.resize(nf,LAG_MIN);sc.gain_q.resize(nf);sc.group_on.resize((nf+PITCH_GROUP-1)/PITCH_GROUP);std::vector<float>gain(nf),eta(nf);const int fd=PITCH_FRAME/DS,lmin=LAG_MIN/DS,lmax=LAG_MAX/DS;static constexpr int second[16]={0,0,3,2,3,2,5,2,3,2,3,2,5,2,3,2};int prev=std::max(lmin,120/DS);double prev_gain=0;std::array<double,lmax+1> log_lags{};for(int L=lmin;L<=lmax;++L)log_lags[L]=std::log(double(L)+1e-12);
 for(size_t f=0;f<nf;++f){int s=int(f)*fd,e=std::min<int>(int(nd),s+fd);double E=1e-30;for(int n=s;n<e;++n)E+=double(md[n])*md[n];
  double best=-1e300,bestdot=0,bestEp=1;int T0d=prev;
  // For fixed frame [s,e), successive lag windows differ by one sample:
  // Ep(L+1) = Ep(L) + md[s-L-1]^2 - md[e-L-1]^2.
  // There is no reason to rescan 240 values for every trial lag.
  const int first=lmin;
  if(s>=first){
    double Ep=1e-30;
    for(int n=s;n<e;++n){double v=md[n-first];Ep+=v*v;}
    const int last=std::min(lmax,s);
    const double lp=std::log(double(prev)+1e-12);
    for(int L=first;L<=last;L+=2){
        if(L!=first){
            const double into=md[s-L],out=md[e-L];
            Ep=std::max(1e-30,Ep+into*into-out*out);
        }
        if(L+1<=last){
            double dot0=0.0,dot1=0.0;
            corr_dot2(md.data()+s,md.data()+s-L,md.data()+s-(L+1),e-s,dot0,dot1);
            {
                const double rr=dot0/std::sqrt(E*Ep);
                const double d=log_lags[L]-lp;
                const double score=rr-CONT_LAMBDA*d*d;
                const bool take=score>best;best=take?score:best;T0d=take?L:T0d;bestdot=take?dot0:bestdot;bestEp=take?Ep:bestEp;
            }
            {
                const int L1=L+1;
                const double into=md[s-L1],out=md[e-L1];
                Ep=std::max(1e-30,Ep+into*into-out*out);
                const double rr=dot1/std::sqrt(E*Ep);
                const double d=log_lags[L1]-lp;
                const double score=rr-CONT_LAMBDA*d*d;
                const bool take=score>best;best=take?score:best;T0d=take?L1:T0d;bestdot=take?dot1:bestdot;bestEp=take?Ep:bestEp;
            }
        }else{
            const double dot=corr_dot(md.data()+s,md.data()+s-L,e-s);
            const double rr=dot/std::sqrt(E*Ep);
            const double d=log_lags[L]-lp;
            const double score=rr-CONT_LAMBDA*d*d;
            const bool take=score>best;best=take?score:best;T0d=take?L:T0d;bestdot=take?dot:bestdot;bestEp=take?Ep:bestEp;
        }
    }
  }
  double g0=std::max(0.0,bestdot/std::sqrt(1.0+E*bestEp));int T=T0d;double gb=g0;for(int kk=2;kk<16;++kk){int T1d=(2*T0d+kk)/(2*kk);if(T1d<lmin)break;int T1b=(kk==2)?((T1d+T0d>lmax)?T0d:T0d+T1d):(2*second[kk]*T0d+kk)/(2*kk);if(s-T1d<0||s-T1b<0)continue;double d1=0,d2=0,e1=1e-30,e2=1e-30;for(int n=s;n<e;++n){double a=md[n-T1d],b=md[n-T1b];d1+=double(md[n])*a;d2+=double(md[n])*b;e1+=a*a;e2+=b*b;}double g1=.5*(d1+d2)/std::sqrt(1.0+E*.5*(e1+e2)),cont=0;if(std::abs(T1d-prev)<=1)cont=prev_gain;else if(std::abs(T1d-prev)<=2&&5*kk*kk<T0d)cont=.5*prev_gain;double th=std::max(.3,.7*g0-cont);if(T1d<3*lmin)th=std::max(.4,.85*g0-cont);else if(T1d<2*lmin)th=std::max(.5,.9*g0-cont);if(g1>th){T=T1d;gb=g1;}}
  if(T<2*lmin){int a=T*5/8,b=T*6/8;if(a>=lmin&&b>=lmin&&s-b>=0){double d1=0,d2=0,e1=1e-30,e2=1e-30;for(int n=s;n<e;++n){double u=md[n-a],v=md[n-b];d1+=double(md[n])*u;d2+=double(md[n])*v;e1+=u*u;e2+=v*v;}if(d1/std::sqrt(1.0+E*e1)>=gb||d2/std::sqrt(1.0+E*e2)>=gb)gb=0;}}
  double Ep=1e-30,dot=0;if(s-T>=0)for(int n=s;n<e;++n){double v=md[n-T];Ep+=v*v;dot+=double(md[n])*v;}double rr=std::max(0.0,dot/std::sqrt(E*Ep)),a=std::max(0.0,dot/Ep),u=a*rr*rr,gg=GAIN_MAX*u/(1.0+u);int q=std::clamp(int(std::lround(gg/GAIN_MAX*GAIN_LEVELS)),0,GAIN_LEVELS);sc.lag[f]=uint16_t(T*DS);sc.gain_q[f]=uint8_t(q);gain[f]=GAIN_MAX*float(q)/GAIN_LEVELS;prev=T;prev_gain=rr;}
 if(filter_output)filter_output->resize(2*ns);
 for(size_t f=0;f<nf;++f){size_t a=f*PITCH_FRAME,b=std::min(ns,(f+1)*size_t(PITCH_FRAME));int T=sc.lag[f];float gg=gain[f];double E=1e-30,Ey=0;for(size_t n=a;n<b;++n)for(int ch=0;ch<2;++ch){float xv=x[2*n+ch];E+=double(xv)*xv;float p=0;if(n>=size_t(T+1))p=COMB_C0*x[2*(n-T)+ch]+COMB_C1*(x[2*(n-T-1)+ch]+x[2*(n-T+1)+ch]);float y=xv-gg*p;Ey+=double(y)*y;if(filter_output)(*filter_output)[2*n+ch]=y;}eta[f]=float(1.0-Ey/E);}for(size_t g=0;g<sc.group_on.size();++g){size_t a=g*PITCH_GROUP,b=std::min(nf,a+size_t(PITCH_GROUP));double m=0;for(size_t f=a;f<b;++f)m+=eta[f];sc.group_on[g]=uint8_t(m/double(b-a)>=WHITEN_GATE);
  if(filter_output && !sc.group_on[g]){
    const size_t begin=a*size_t(PITCH_FRAME),end=std::min(ns,b*size_t(PITCH_FRAME));
    std::memcpy(filter_output->data()+2*begin,x+2*begin,2*(end-begin)*sizeof(float));
  }
 }return sc;}
inline void postfilter(std::vector<float>&y,size_t ns,const Schedule&sc){
 for(size_t f=0;f<sc.lag.size();++f){
  if(!sc.group_on[f/PITCH_GROUP])continue;
  const int T=sc.lag[f];const float gg=GAIN_MAX*float(sc.gain_q[f])/GAIN_LEVELS;
  size_t a=f*PITCH_FRAME,b=std::min(ns,(f+1)*size_t(PITCH_FRAME));
  size_t n=std::max(a,size_t(T+1));
#if defined(__AVX2__)
  const __m256 vc0=_mm256_set1_ps(COMB_C0),vc1=_mm256_set1_ps(COMB_C1),vg=_mm256_set1_ps(gg);
  for(;n+4<=b;n+=4){
   const __m256 d0=_mm256_loadu_ps(y.data()+2*(n-size_t(T)));
   const __m256 dm=_mm256_loadu_ps(y.data()+2*(n-size_t(T)-1));
   const __m256 dp=_mm256_loadu_ps(y.data()+2*(n-size_t(T)+1));
   const __m256 p=_mm256_add_ps(_mm256_mul_ps(vc0,d0),_mm256_mul_ps(vc1,_mm256_add_ps(dm,dp)));
   const __m256 cur=_mm256_loadu_ps(y.data()+2*n);
   _mm256_storeu_ps(y.data()+2*n,_mm256_add_ps(cur,_mm256_mul_ps(vg,p)));
  }
#elif defined(__SSE4_1__)
  const __m128 vc0=_mm_set1_ps(COMB_C0),vc1=_mm_set1_ps(COMB_C1),vg=_mm_set1_ps(gg);
  for(;n+2<=b;n+=2){
   const __m128 d0=_mm_loadu_ps(y.data()+2*(n-size_t(T)));
   const __m128 dm=_mm_loadu_ps(y.data()+2*(n-size_t(T)-1));
   const __m128 dp=_mm_loadu_ps(y.data()+2*(n-size_t(T)+1));
   const __m128 p=_mm_add_ps(_mm_mul_ps(vc0,d0),_mm_mul_ps(vc1,_mm_add_ps(dm,dp)));
   const __m128 cur=_mm_loadu_ps(y.data()+2*n);
   _mm_storeu_ps(y.data()+2*n,_mm_add_ps(cur,_mm_mul_ps(vg,p)));
  }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  const float32x4_t vc0=vdupq_n_f32(COMB_C0),vc1=vdupq_n_f32(COMB_C1),vg=vdupq_n_f32(gg);
  for(;n+2<=b;n+=2){
   const float32x4_t d0=vld1q_f32(y.data()+2*(n-size_t(T)));
   const float32x4_t dm=vld1q_f32(y.data()+2*(n-size_t(T)-1));
   const float32x4_t dp=vld1q_f32(y.data()+2*(n-size_t(T)+1));
   const float32x4_t p=vaddq_f32(vmulq_f32(vc0,d0),vmulq_f32(vc1,vaddq_f32(dm,dp)));
   const float32x4_t cur=vld1q_f32(y.data()+2*n);
   vst1q_f32(y.data()+2*n,vaddq_f32(cur,vmulq_f32(vg,p)));
  }
#endif
  for(;n<b;++n){
   const size_t d=n-size_t(T);
   for(int ch=0;ch<2;++ch){const float p=COMB_C0*y[2*d+ch]+COMB_C1*(y[2*(d-1)+ch]+y[2*(d+1)+ch]);y[2*n+ch]+=gg*p;}
  }
 }
}
inline std::vector<uint8_t> encode_payload(const Schedule&sc){std::vector<uint8_t>sym;std::array<uint32_t,DELTA_ALPHA>hist{};for(size_t g=0;g<sc.group_on.size();++g){if(!sc.group_on[g])continue;size_t a=g*PITCH_GROUP,b=std::min(sc.lag.size(),a+size_t(PITCH_GROUP));for(size_t f=a+1;f<b;++f){uint32_t z=zz_enc((int(sc.lag[f])-int(sc.lag[f-1]))/DS);uint8_t s=uint8_t(z<32?z:32);sym.push_back(s);++hist[s];}}lx_rans4_fast::Model<DELTA_ALPHA>m;m.build(hist);lx_rans4_fast::Scratch scratch;auto enc=lx_rans4_fast::encode4(sym.data(),sym.size(),m,scratch);BitWriter bw;size_t si=0;for(size_t g=0;g<sc.group_on.size();++g){bool on=sc.group_on[g];bw.put(on,1);if(!on)continue;size_t a=g*PITCH_GROUP,b=std::min(sc.lag.size(),a+size_t(PITCH_GROUP));bw.put(sc.lag[a]/DS,8);for(size_t f=a;f<b;++f)bw.put(sc.gain_q[f],4);for(size_t f=a+1;f<b;++f){uint8_t s=sym[si++];if(s==32)bw.put(zz_enc((int(sc.lag[f])-int(sc.lag[f-1]))/DS),9);}}bw.finish();std::vector<uint8_t>out;auto p16=[&](uint16_t v){out.push_back(uint8_t(v));out.push_back(uint8_t(v>>8));};auto p32=[&](uint32_t v){for(int i=0;i<4;++i)out.push_back(uint8_t(v>>(8*i)));};p16(uint16_t(sc.lag.size()));p16(uint16_t(sc.group_on.size()));p16(uint16_t(sym.size()));p16(0);p32(uint32_t(enc.bytes));for(int s=0;s<DELTA_ALPHA;++s)p16(uint16_t(std::min<uint32_t>(hist[s],65535)));out.insert(out.end(),enc.data,enc.data+enc.bytes);out.insert(out.end(),bw.out.begin(),bw.out.end());return out;}
inline Schedule decode_payload(const std::vector<uint8_t>&in){if(in.size()<12+2*DELTA_ALPHA)throw std::runtime_error("short pitch payload");size_t o=0;auto g16=[&](){uint16_t v=uint16_t(in[o])|(uint16_t(in[o+1])<<8);o+=2;return v;};auto g32=[&](){uint32_t v=0;for(int i=0;i<4;++i)v|=uint32_t(in[o++])<<(8*i);return v;};uint16_t nf=g16(),ng=g16(),nd=g16();(void)g16();uint32_t rb=g32();std::array<uint32_t,DELTA_ALPHA>hist{};for(int s=0;s<DELTA_ALPHA;++s)hist[s]=g16();if(o+rb>in.size())throw std::runtime_error("bad pitch rANS size");lx_rans4_fast::Model<DELTA_ALPHA>m;m.build(hist);std::vector<uint8_t>sym(nd);if(nd&&!lx_rans4_fast::decode4(in.data()+o,rb,sym.data(),nd,m))throw std::runtime_error("bad pitch rANS");o+=rb;BitReader br(in.data()+o,in.data()+in.size());Schedule sc;sc.lag.resize(nf,LAG_MIN);sc.gain_q.resize(nf);sc.group_on.resize(ng);size_t si=0;for(size_t g=0;g<ng;++g){uint32_t v;if(!br.get(1,v))throw std::runtime_error("pitch flag underflow");sc.group_on[g]=uint8_t(v);size_t a=g*PITCH_GROUP,b=std::min<size_t>(nf,a+PITCH_GROUP);if(!v)continue;if(!br.get(8,v))throw std::runtime_error("pitch anchor underflow");sc.lag[a]=uint16_t(v*DS);for(size_t f=a;f<b;++f){if(!br.get(4,v))throw std::runtime_error("pitch gain underflow");sc.gain_q[f]=uint8_t(v);}for(size_t f=a+1;f<b;++f){uint32_t z=sym.at(si++);if(z==32&&!br.get(9,z))throw std::runtime_error("pitch escape underflow");sc.lag[f]=uint16_t(std::clamp(int(sc.lag[f-1])+zz_dec(z)*DS,LAG_MIN,LAG_MAX));}}if(si!=sym.size())throw std::runtime_error("pitch delta count mismatch");return sc;}
} // namespace lx_pitch
