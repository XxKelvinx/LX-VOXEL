#pragma once
#include <cmath>
#include <cstdlib>
#include <algorithm>

static inline int lx_warp_mode_cfg(){ static const int v=[](){const char*p=::lxvoxel_release_env("LX_WARP_MODE");return p?std::atoi(p):0;}(); return v; }
static inline float lx_warp_strength_cfg(){ static const float v=[](){const char*p=::lxvoxel_release_env("LX_WARP_STRENGTH");return p?std::atof(p):0.0f;}(); return v; }
static inline int lx_warp_start_cfg(){ static const int v=[](){const char*p=::lxvoxel_release_env("LX_WARP_START_BAND");return p?std::atoi(p):30;}(); return v; }
static inline float lx_warp_shape_raw(int mode,float u){
    constexpr float PI=3.14159265358979323846f; const float x=u-0.5f;
    switch(mode){
        case 1: return x;
        case 2: return std::exp(-8.0f*x*x);
        case 3: return x*x;
        case 4: return std::cos(2.0f*PI*u);
        case 5: return std::cos(4.0f*PI*u);
        case 6: return std::tanh(4.0f*x);
        case 7: return std::exp(-32.0f*x*x);
        case 8: return std::cos(6.0f*PI*u);
        default:return 0.0f;
    }
}
static inline float lx_warp_mean(int mode,int n){
    if(n<=1) return 0.0f;
    if(mode==3) return float((double(n)*double(n)-1.0)/(12.0*double(n)*double(n)));
    if(mode==2) return 0.598144f; /* integral exp(-8x^2), x in [-.5,.5] */
    if(mode==7) return 0.313309f; /* integral exp(-32x^2) */
    return 0.0f; // exact by symmetry/discrete roots for modes 1,4,5,6,8
}
static inline float lx_warp_weight(int band,int j,int n){
    const int mode=lx_warp_mode_cfg(); const float a=lx_warp_strength_cfg();
    if(mode<=0 || band<lx_warp_start_cfg() || n<=1 || a==0.0f) return 1.0f;
    const float u=(float(j)+0.5f)/float(n);
    const float z=lx_warp_shape_raw(mode,u)-lx_warp_mean(mode,n);
    return std::exp(a*z);
}
