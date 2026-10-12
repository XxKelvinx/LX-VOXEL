#pragma once
#include "MDCT/mdct.hpp"
#include <array>
#include <cmath>
#include <algorithm>

namespace lx_cpm {

// Canonical persistent-line score. z5[] is local stereo energy around the candidate line,
// a0/a1 are current/previous line amplitudes. All terms are dimensionless and bounded.

// Pure analytic 5-D line-shape projector. This is a cheap baseline independent of learned tables.
// w = normalize([1,1,3,1,1]); pred = w*(w^T prev).

}

namespace lx_cpm {

template<int N>
static inline void physical_line_templates(lx_nsmdct::DCT4Ortho<N>& d,
                                           float freq_hz, size_t pos,
                                           bool has_left, bool has_right,
                                           int ip, float* c5, float* s5){
    constexpr int OH=lx_nsmdct::OVERLAP/2;
    alignas(64) std::array<float,N+2*OH> xc{}, xs{};
    alignas(64) std::array<float,N> out{};
    const double w=2.0*3.1415926535897932384626433832795*double(freq_hz)/48000.0;
    const double cw=std::cos(w), sw=std::sin(w);
    double c=std::cos(w*(double(pos)-double(OH))), s=std::sin(w*(double(pos)-double(OH)));
    for(int n=-OH;n<N+OH;++n){
        const int j=n+OH; xc[j]=float(c); xs[j]=float(s);
        const double cn=c*cw-s*sw, sn=s*cw+c*sw; c=cn; s=sn;
    }
    d.apply_lapped(xc.data()+OH,has_left,has_right,nullptr,nullptr,out.data());
    for(int t=0;t<5;++t)c5[t]=out[ip+t-2];
    d.apply_lapped(xs.data()+OH,has_left,has_right,nullptr,nullptr,out.data());
    for(int t=0;t<5;++t)s5[t]=out[ip+t-2];
}

static inline void fit_complex_line5(const float* x,const float* c,const float* s,float& a,float& b){
    double cc=0.0,ss=0.0,cs=0.0,cx=0.0,sx=0.0;
    for(int t=0;t<5;++t){cc+=double(c[t])*c[t];ss+=double(s[t])*s[t];cs+=double(c[t])*s[t];cx+=double(c[t])*x[t];sx+=double(s[t])*x[t];}
    const double det=cc*ss-cs*cs;
    const double inv=1.0/std::max(det,double(std::numeric_limits<float>::min()));
    a=float((cx*ss-sx*cs)*inv);
    b=float((sx*cc-cx*cs)*inv);
}

static inline void synth_complex_line5(float a,float b,const float* c,const float* s,float* x){
    for(int t=0;t<5;++t)x[t]=a*c[t]+b*s[t];
}

}
