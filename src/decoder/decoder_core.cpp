#include <array>
#include <bit>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif


#include "bitstream/format.hpp"
#include "decoder/decoder_core.hpp"
#include "rANS/rans.hpp"
#include "rANS/sideinfo_rans.hpp"
#include "PitchPredictor/pitch_predictor.hpp"
#include "quant/conditional_gain.hpp"
#include "quant/warp.hpp"
#include "synthesis/cpm.hpp"
#include "synthesis/sfc.hpp"
#include "synthesis/gain_predictor.hpp"
#include "MDCT/block_switch.hpp"

#ifndef LX_PREC_RES_MODE
#define LX_PREC_RES_MODE 0
#endif
#ifndef LX_CPM_PREC_SHIFT
#define LX_CPM_PREC_SHIFT 4
#endif

using namespace pulsar;
using namespace pulsar_bs;
using namespace lx_nsmdct;

namespace {
constexpr int MODEL_SUPERFRAME=32;
constexpr int CLASS_FREQ_CTX = ERB_BANDS/2;
constexpr int CLASS_CTX = CLASS_FREQ_CTX*2;
constexpr int CLASS_TEMP_STATES = 4;
constexpr int CLASS_TUPLE_CTX = CLASS_FREQ_CTX*3*CLASS_TEMP_STATES;
constexpr int MAG_BASE_CTX = 11;
constexpr int MAG_PRED_STATES = 4;
constexpr int MAG_LOCAL_STATES = 3;
constexpr int MAG_CTX = MAG_BASE_CTX * MAG_PRED_STATES * MAG_LOCAL_STATES;
constexpr std::array<uint8_t,ERB_BANDS> MAG_CTX_MAP = {0,0,0,1,1,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,10,10,10,10,10,10};
constexpr int MAG_ESC=31;
constexpr int MAG_ALPHA=MAG_ESC+1;
constexpr float INV_SQRT2=0.7071067811865475244f;
constexpr int IS_MAX_BANDS=8;
constexpr int IS_FIRST_BAND=ERB_BANDS-IS_MAX_BANDS;
constexpr float LX_PI=3.14159265358979323846f;
constexpr int IS_ANGLE_LEVELS=256; // storage for runtime codebooks
struct EgvRange { uint8_t b0,b1; };
inline constexpr std::array<EgvRange,16> EGV_LONG = {{{0,19},{19,26},{26,30},{30,33},{33,35},{35,37},{37,39},{39,40},{40,41},{41,42},{42,43},{43,44},{44,45},{45,46},{46,47},{47,48}}};
inline constexpr std::array<EgvRange,8> EGV_1024 = {{{0,26},{26,33},{33,37},{37,40},{40,42},{42,44},{44,46},{46,48}}};
inline constexpr std::array<EgvRange,4> EGV_SHORT = {{{0,34},{34,41},{41,45},{45,48}}};
inline constexpr std::array<EgvRange,2> EGV_256 = {{{0,30},{30,48}}};
inline constexpr std::array<float,16> LX_DELTA_FRAC16 = {{
    1.000000000f, 1.044273782f, 1.090507733f, 1.138788635f,
    1.189207115f, 1.241857812f, 1.296839555f, 1.354255547f,
    1.414213562f, 1.476826146f, 1.542210825f, 1.610490332f,
    1.681792831f, 1.756252160f, 1.834008086f, 1.915206561f
}};

static inline int is_style_cfg(){return 10;}
static inline int is_cfg_dec(){return 2;}
static inline int is_bits_cfg(){return 6;}
static inline int is_code_count_cfg(){const int st=is_style_cfg();return (st==2||st==4||st==6)?1:((st==9||st==10)?(IS_MAX_BANDS/2):IS_MAX_BANDS);}
static inline float decode_is_angle(uint8_t q){const int bits=is_bits_cfg();if(bits==0)return 0.25f*LX_PI;if(bits==1)return q?0.25f*LX_PI:-0.25f*LX_PI;const int L=1<<bits;return float(int(q)-(L/2-1))*(LX_PI/float(L));}
static inline uint8_t unpack_is_code(uint32_t packed,int b){const int st=is_style_cfg(),bits=is_bits_cfg(),j=b-IS_FIRST_BAND;if(bits==0)return 0;const uint32_t mask=(uint32_t(1)<<bits)-1u;if(st==2||st==4||st==6)return uint8_t(packed&mask);if(st==9||st==10)return uint8_t((packed>>(bits*(j>>1)))&mask);return uint8_t((packed>>(bits*j))&mask);}

static inline int32_t unzigzag_u32(uint32_t z){return (z&1u)?-int32_t((z+1u)>>1):int32_t(z>>1);}
static inline int scale_predict(const std::vector<uint16_t>& s,size_t f,int b){
    if(f==0&&b==0)return 0; if(f==0)return int(s[size_t(b-1)]); if(b==0)return int(s[(f-1)*ERB_BANDS]);
    const int left=int(s[f*ERB_BANDS+size_t(b-1)]),top=int(s[(f-1)*ERB_BANDS+size_t(b)]),tl=int(s[(f-1)*ERB_BANDS+size_t(b-1)]);
    if(tl>=std::max(left,top))return std::min(left,top); if(tl<=std::min(left,top))return std::max(left,top); return left+top-tl;
}

static inline int lr_scale_predict(const std::vector<int16_t>& s,size_t f,int b){
    if(f==0&&b==0)return 0; if(f==0)return int(s[size_t(b-1)]); if(b==0)return int(s[(f-1)*ERB_BANDS]);
    const int left=int(s[f*ERB_BANDS+size_t(b-1)]),top=int(s[(f-1)*ERB_BANDS+size_t(b)]),tl=int(s[(f-1)*ERB_BANDS+size_t(b-1)]);
    if(tl>=std::max(left,top))return std::min(left,top); if(tl<=std::min(left,top))return std::max(left,top); return left+top-tl;
}


struct Model9 {
    std::array<uint16_t,9> freq{},start{};
    alignas(16) std::array<int16_t,8> endm1{};
    void build(const std::array<uint32_t,9>& hist){
        uint64_t total=0;for(int s=0;s<9;++s)total+=2ull*hist[s]+1ull;
        constexpr uint64_t rem=uint64_t(lx_rans4_fast::M)-9;uint64_t cum=0,prev=0;uint32_t c=0;
        for(int s=0;s<9;++s){cum+=2ull*hist[s]+1ull;const uint64_t q=(cum*rem)/total;freq[s]=uint16_t(1u+uint32_t(q-prev));prev=q;start[s]=uint16_t(c);c+=freq[s];if(s<8)endm1[s]=int16_t(c-1u);}
    }
};
struct Models9 {
    std::array<Model9,CLASS_TUPLE_CTX> m{};
    Models9(){std::array<uint32_t,9> h{};h.fill(1);for(auto&x:m)x.build(h);}
    void rebuild(const std::array<std::array<uint32_t,9>,CLASS_TUPLE_CTX>&h){for(int c=0;c<CLASS_TUPLE_CTX;++c)m[c].build(h[c]);}
};
template<int A,int C> struct Models {
    std::array<lx_rans4_fast::Model<A>,C> m{};
    Models(){std::array<uint32_t,A> h{};h.fill(1);for(auto& x:m)x.build_decoder_only(h);}
    void rebuild(const std::array<std::array<uint32_t,A>,C>& h){for(int c=0;c<C;++c)m[c].build_decoder_only(h[c]);}
};

struct Rans4 {
    uint32_t x[4]{}; const uint8_t* p=nullptr; const uint8_t* end=nullptr; size_t index=0;
    bool init(const std::vector<uint8_t>& v){if(v.size()<16)return false;std::memcpy(x,v.data(),16);p=v.data()+16;end=v.data()+v.size();index=0;return true;}
    template<int A> bool get(const lx_rans4_fast::Model<A>& m,uint8_t& sym){
        unsigned lane=unsigned(index++)&3u;uint32_t v=x[lane];const uint32_t slot=v&lx_rans4_fast::MASK;sym=m.dec[slot];
        v=uint32_t(m.freq[sym])*(v>>lx_rans4_fast::SCALE_BITS)+slot-m.start[sym];
        const uint32_t need=uint32_t(v<lx_rans4_fast::L); uint16_t w=0;
        if(size_t(end-p)>=2u) std::memcpy(&w,p,2); else if(need) return false;
        const uint32_t nv=(v<<16)|uint32_t(w), mask=0u-need;
        v=(v&~mask)|(nv&mask); p+=size_t(need)*2u; x[lane]=v;return true;
    }
    bool get9(const Model9& m,uint8_t& sym){
        const unsigned lane=unsigned(index++)&3u;uint32_t v=x[lane];const uint32_t slot=v&lx_rans4_fast::MASK;
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
        const __m128i e=_mm_load_si128(reinterpret_cast<const __m128i*>(m.endm1.data()));
        const __m128i sv=_mm_set1_epi16(static_cast<int16_t>(slot));
        const unsigned mask=unsigned(_mm_movemask_epi8(_mm_cmpgt_epi16(sv,e)));
        sym=uint8_t(std::popcount(mask)>>1);
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(__ARM_NEON__)
        const int16x8_t e=vld1q_s16(m.endm1.data());
        const int16x8_t sv=vdupq_n_s16(static_cast<int16_t>(slot));
        const uint16x8_t gt=vcgtq_s16(sv,e);
        const uint16x4_t lo=vget_low_u16(gt), hi=vget_high_u16(gt);
        const uint32x4_t s32=vaddl_u16(lo,hi);
        const uint64x2_t s64=vpaddlq_u32(s32);
        const uint64_t total=vgetq_lane_u64(s64,0)+vgetq_lane_u64(s64,1);
        sym=uint8_t(total/0xffffu);
#else
        unsigned count=0; for(int k=0;k<8;++k) count+=slot>uint16_t(m.endm1[k]);
        sym=uint8_t(count);
#endif
        v=uint32_t(m.freq[sym])*(v>>lx_rans4_fast::SCALE_BITS)+slot-m.start[sym];
        {const uint32_t need=uint32_t(v<lx_rans4_fast::L);uint16_t w=0;if(size_t(end-p)>=2u)std::memcpy(&w,p,2);else if(need)return false;const uint32_t nv=(v<<16)|uint32_t(w),mask=0u-need;v=(v&~mask)|(nv&mask);p+=size_t(need)*2u;}x[lane]=v;return true;
    }
    // Decode four consecutive symbols from one model. The four rANS states are
    // independent, so exposing all four updates together gives the OoO core four
    // dependency chains without AVX gather/shuffle overhead. Renormalization stays
    // in exact bitstream order.
    template<int A> bool get4_same(const lx_rans4_fast::Model<A>& m,uint8_t out[4]){
        const unsigned l0=unsigned(index)&3u,l1=(l0+1u)&3u,l2=(l0+2u)&3u,l3=(l0+3u)&3u;
        uint32_t v0=x[l0],v1=x[l1],v2=x[l2],v3=x[l3];
        const uint32_t s0=v0&lx_rans4_fast::MASK,s1=v1&lx_rans4_fast::MASK,s2=v2&lx_rans4_fast::MASK,s3=v3&lx_rans4_fast::MASK;
        const uint8_t y0=m.dec[s0],y1=m.dec[s1],y2=m.dec[s2],y3=m.dec[s3];
        out[0]=y0;out[1]=y1;out[2]=y2;out[3]=y3;
        v0=uint32_t(m.freq[y0])*(v0>>lx_rans4_fast::SCALE_BITS)+s0-m.start[y0];
        v1=uint32_t(m.freq[y1])*(v1>>lx_rans4_fast::SCALE_BITS)+s1-m.start[y1];
        v2=uint32_t(m.freq[y2])*(v2>>lx_rans4_fast::SCALE_BITS)+s2-m.start[y2];
        v3=uint32_t(m.freq[y3])*(v3>>lx_rans4_fast::SCALE_BITS)+s3-m.start[y3];
#define LX_DRENORM(V) do{if((V)<lx_rans4_fast::L){if(size_t(end-p)<2u)return false;uint16_t w;std::memcpy(&w,p,2);p+=2;(V)=((V)<<16)|w;}}while(0)
        LX_DRENORM(v0);LX_DRENORM(v1);LX_DRENORM(v2);LX_DRENORM(v3);
#undef LX_DRENORM
        x[l0]=v0;x[l1]=v1;x[l2]=v2;x[l3]=v3;index+=4;return true;
    }
    bool done()const{return p==end;}
};

struct PacketRead {vox_beta::SuperframeHeader h{};std::vector<uint8_t> side,block_hi,sfc,sr,lrr,cr,mr,sraw,lrraw,mraw;};
struct DecSfcDecision { uint8_t mask=0; std::array<uint8_t,lx_sfc::SFC_BANDS> gain{},ar_k1{},ar_k2{},tilt{},mode{},ifc_on{},ifc_mode{},ifc_pos{},ifc_amp{}; };
struct DecCpmDecision { uint8_t on=0,anchor=0,band=0,pos=0; uint16_t signs=0; };
struct CpmTrackState { bool valid=false; uint8_t band=0,pos=0; size_t frame=0; float freq_hz=0.f; float al=0.f,bl=0.f,ar=0.f,br=0.f; };
static bool read_exact(std::istream& f,void* p,size_t n){return bool(f.read((char*)p,std::streamsize(n)));}
static inline void read_into(std::istream& f,std::vector<uint8_t>& v,size_t n){v.resize(n);if(n&&!read_exact(f,v.data(),n))throw std::runtime_error("truncated .vox");}

} // namespace

static inline uint64_t sgc_choose(int n,int k){if(k<0||k>n)return 0;if(k>n-k)k=n-k;uint64_t r=1;for(int i=1;i<=k;++i)r=(r*uint64_t(n-k+i))/uint64_t(i);return r;}
static inline unsigned sgc_rank_bits(int n,int k){uint64_t c=sgc_choose(n,k);unsigned b=0;while((uint64_t(1)<<b)<c&&b<63)++b;return b;}
static inline uint64_t sgc_unrank_mask_n(uint64_t rank,int n,int k){uint64_t mask=0;int x=n-1;for(int left=k;left>0;--left){while(x>=0&&sgc_choose(x,left)>rank)--x;if(x<0)throw std::runtime_error("bad SGC rank");mask|=uint64_t(1)<<x;rank-=sgc_choose(x,left);--x;}return mask;}


namespace lxvoxel {
#ifndef LXVOXEL_DECODER_ENTRY
#define LXVOXEL_DECODER_ENTRY decode_stream_variant
#endif
int LXVOXEL_DECODER_ENTRY(std::istream& f, DecodedAudio& decoded){
    vox_beta::Header vh{};if(!read_exact(f,&vh,sizeof(vh)))throw std::runtime_error("short VOX header");
    if(vh.magic!=vox_beta::MAGIC || vh.major!=vox_beta::MAJOR || vh.minor!=vox_beta::MINOR ||
       vh.sample_rate!=vox_beta::SAMPLE_RATE || vh.channels!=vox_beta::CHANNELS || vh.erb_bands!=vox_beta::ERB_BANDS)
        throw std::runtime_error("unsupported VOX profile: decoder accepts current bitstream only");
    if(vh.header_bytes!=sizeof(vox_beta::Header)) throw std::runtime_error("unsupported VOX header size");
    if(vh.superframe_frames!=vox_beta::SUPERFRAME_FRAMES) throw std::runtime_error("unsupported VOX superframe geometry");
    if((vh.flags & 0xff00u)!=0) throw std::runtime_error("unsupported VOX extension flags");
    if(vh.frame_count==0 || vh.total_samples_per_channel==0) throw std::runtime_error("empty VOX stream");
    // Each frame advances by one normative block of at most K_LONG and at
    // least K_MIN samples. These bounds prevent forged headers from causing
    // unbounded allocations before the frame syntax is parsed.
    const uint64_t max_samples_from_frames=uint64_t(vh.frame_count)*uint64_t(K_LONG);
    if(vh.total_samples_per_channel>max_samples_from_frames) throw std::runtime_error("VOX sample/frame geometry is inconsistent");
    const uint64_t max_frames_from_samples=(vh.total_samples_per_channel+uint64_t(K_MIN)-1u)/uint64_t(K_MIN)+1u;
    if(uint64_t(vh.frame_count)>max_frames_from_samples) throw std::runtime_error("VOX frame/sample geometry is inconsistent");

    ErbLayoutN<K_LONG> eL;eL.init(48000.0);ErbLayoutN<K_1024> e1024;e1024.init(48000.0);ErbLayoutN<K_SHORT> eS;eS.init(48000.0);ErbLayoutN<K_256> e256;e256.init(48000.0);
    DCT4Ortho<1024> cpmLong;DCT4Ortho<768> cpm1024;DCT4Ortho<512> cpmShort;DCT4Ortho<384> cpm256;
    const int cbL=eL.coded_bins(),cb1024=e1024.coded_bins(),cbS=eS.coded_bins(),cb256=e256.coded_bins();
    // Decoder qstep LUT: scale codes are 4-bit fractional octaves.  Avoid
    // thousands of scalar ldexp() calls in the per-frame/per-band hot path.
    alignas(64) std::array<float,481> delta_code_f{};
    for(int sc=0;sc<=480;++sc){
        const int octave=sc>>4, fraction=sc&15;
        delta_code_f[sc]=std::ldexp(vh.base_delta*LX_DELTA_FRAC16[fraction],octave);
    }
    constexpr int min_block=K_MIN;
    auto edges_for=[&](BlockSize bs)->const std::array<int,ERB_BANDS+1>&{
        switch(bs){case BlockSize::Long2048:return eL.edge;case BlockSize::Medium1024:return e1024.edge;case BlockSize::Short512:return eS.edge;case BlockSize::Micro256:return e256.edge;}
        return eL.edge;
    };
    Models9 classTupleModel;Models<MAG_ALPHA,MAG_CTX> magModel;
    // R206: decoder-synchronous slow entropy memory (5/8 EWMA).
    std::array<std::array<uint64_t,9>,CLASS_TUPLE_CTX> class_hist_mem{};
    std::array<std::array<uint64_t,MAG_ALPHA>,MAG_CTX> mag_hist_mem{};
    std::array<uint64_t,256> scale_hist_mem{},lr_scale_hist_mem{};
    // R204 lossless temporal magnitude-context predictor.
    std::array<std::array<uint64_t,ERB_BANDS>,2> magpred_prev_q1{};
    std::array<std::array<uint32_t,ERB_BANDS>,2> magpred_prev_nz{};
    std::array<std::array<uint8_t,ERB_BANDS>,2> magpred_prev_valid{};
    const bool warp_off=(lx_warp_mode_cfg()<=0 || lx_warp_strength_cfg()==0.0f);
    lx_rans4_fast::Model<256> scaleModel,lrScaleModel;{std::array<uint32_t,256> h{};h.fill(1);scaleModel.build_decoder_only(h);lrScaleModel.build_decoder_only(h);}

    std::vector<uint16_t> scales;scales.resize(size_t(vh.frame_count)*ERB_BANDS);
    std::vector<int16_t> lr_balance_q(size_t(vh.frame_count)*ERB_BANDS,0);
    std::array<std::array<int16_t,16>,4> lr_pred_prev{};
    std::array<std::array<uint8_t,16>,4> lr_pred_valid{};
    // Research metadata for decoder-known reliability-adaptive HF filtering.
    // qstep is the actual residual/direct step seen by the decoder; support is
    // the fraction of non-zero quantizer coordinates; tool identifies the
    // reconstruction family without adding any side information.
    alignas(64) std::array<float,K_LONG> cL{}, cR{};
    // R231: synthesize each completed frame while its coefficients are still hot
    // in cache.  This preserves frame order and OLA arithmetic exactly while
    // removing the cold full-song coefficient reread at end of stream.
    VariableLowOverlapMDCT invL; VariableLowOverlapMDCT invR=invL;
    decoded.sample_rate = vh.sample_rate;
    decoded.channels = vh.channels;
    decoded.samples_per_channel = vh.total_samples_per_channel;
    decoded.pcm.assign(size_t(vh.total_samples_per_channel)*2u,0.0f);
    size_t globalFrame=0,pos=0;
    CpmTrackState cpm_state{};
    struct CpmTemplateCache { bool valid=false; float freq_hz=0.f; size_t pos=0; float c[5]{},s[5]{}; };
    std::array<std::array<CpmTemplateCache,K_LONG>,4> cpm_tpl_cache{};
    // v0.13: decoder-synchronous consecutive SFC exposure, measured in
    // longest-block support units for the zero-bit Diffuse Guard.
    std::array<double,lx_sfc::SFC_BANDS> sfc_diffuse_exposure{};
    alignas(64) std::array<int32_t,K_LONG> qa{}, qb{};
    std::array<float,IS_ANGLE_LEVELS> is_ct{}, is_st{};
    for(unsigned q=0;q<IS_ANGLE_LEVELS;++q){
        const float th=decode_is_angle(uint8_t(q));
        is_ct[q]=std::cos(th); is_st[q]=std::sin(th);
    }

    PacketRead pk;
    while(globalFrame<vh.frame_count){
        uint32_t compact_payload_bytes=0, compact_sfc_bytes=0;
        pk.h.frame_count=uint16_t(std::min<uint64_t>(MODEL_SUPERFRAME,uint64_t(vh.frame_count)-uint64_t(globalFrame)));
        struct HeaderBitReader {
            std::istream& f; uint64_t acc=0; unsigned bits=0;
            uint32_t get(unsigned n){
                while(bits<n){uint8_t b=0;if(!read_exact(f,&b,1))throw std::runtime_error("truncated compact superframe header");acc|=uint64_t(b)<<bits;bits+=8;}
                const uint64_t mask=(uint64_t(1)<<n)-1u;const uint32_t v=uint32_t(acc&mask);acc>>=n;bits-=n;return v;
            }
            uint32_t uleb(){uint32_t v=0;unsigned sh=0;while(sh<=28){const uint8_t b=uint8_t(get(8));if(sh==28u && (b&0x70u))throw std::runtime_error("overflow in compact superframe ULEB128");v|=uint32_t(b&0x7fu)<<sh;if(!(b&0x80u))return v;sh+=7;}throw std::runtime_error("invalid compact superframe ULEB128");}
            void align(){acc=0;bits=0;}
        } hb{f};
        compact_payload_bytes=hb.get(17u);
        const uint32_t sf_future_bits=hb.get(3u);
        if(sf_future_bits!=0u) throw std::runtime_error("unsupported superframe extension flags");
        compact_sfc_bytes=hb.uleb();
        pk.h.scale_rans_bytes=hb.uleb();
        pk.h.lr_rans_bytes=hb.uleb();
        pk.h.class_rans_bytes=hb.uleb();
        pk.h.mag_rans_bytes=hb.uleb();
        pk.h.scale_raw_bytes=hb.uleb();
        pk.h.lr_raw_bytes=hb.uleb();
        hb.align();
        pk.h.reserved=uint16_t(std::min<uint32_t>(compact_sfc_bytes,65535u));
        if(pk.h.frame_count==0||pk.h.frame_count>MODEL_SUPERFRAME)throw std::runtime_error("bad superframe frame count");
        read_into(f,pk.side,size_t(pk.h.frame_count));
        read_into(f,pk.block_hi,(size_t(pk.h.frame_count)+7u)>>3);
        auto block_at=[&](unsigned lf)->BlockSize{
            const unsigned lo=unsigned(pk.side[lf]&1u);
            const unsigned hi=unsigned((pk.block_hi[lf>>3]>>(lf&7u))&1u);
            return BlockSize(uint8_t(lo|(hi<<1)));
        };

        const size_t side_bytes=pk.side.size(), block_bytes=pk.block_hi.size();
        const size_t sfc_bytes=size_t(compact_sfc_bytes);
        const uint64_t known=uint64_t(side_bytes)+uint64_t(block_bytes)+uint64_t(sfc_bytes)
            +uint64_t(pk.h.scale_rans_bytes)+uint64_t(pk.h.lr_rans_bytes)+uint64_t(pk.h.class_rans_bytes)+uint64_t(pk.h.mag_rans_bytes)+uint64_t(pk.h.scale_raw_bytes)+uint64_t(pk.h.lr_raw_bytes);
        if(known>compact_payload_bytes) throw std::runtime_error("bad compact superframe lengths");
        pk.h.mag_raw_bytes=uint32_t(uint64_t(compact_payload_bytes)-known);
        read_into(f,pk.sfc,sfc_bytes);
        read_into(f,pk.sr,pk.h.scale_rans_bytes);read_into(f,pk.lrr,pk.h.lr_rans_bytes);read_into(f,pk.cr,pk.h.class_rans_bytes);read_into(f,pk.mr,pk.h.mag_rans_bytes);read_into(f,pk.sraw,pk.h.scale_raw_bytes);read_into(f,pk.lrraw,pk.h.lr_raw_bytes);read_into(f,pk.mraw,pk.h.mag_raw_bytes);
        if(pk.sfc.size()<7) throw std::runtime_error("short SFC v0.36 payload");
        size_t so=0; auto g16s=[&](){uint16_t v=uint16_t(pk.sfc[so])|(uint16_t(pk.sfc[so+1])<<8);so+=2;return v;};
        const uint16_t ml=g16s(), al=g16s();
        const unsigned side_k=pk.sfc[so++];
        const uint16_t side_n=g16s();
        if(side_k>7u || so+size_t(ml)+size_t(al)!=pk.sfc.size()) throw std::runtime_error("bad SFC v0.36 envelope");
        Rans4 ds; if(!ds.init(pk.sr))throw std::runtime_error("bad scale/side rANS stream");
        std::vector<uint8_t> side_syms(side_n); auto sm=lx_sideinfo_rans::model(side_k);
        for(size_t i=0;i<side_syms.size();++i) if(!ds.get(sm,side_syms[i]))throw std::runtime_error("bad SFC side rANS");
        const uint8_t* sfc_begin=pk.sfc.data()+so; const uint8_t* sfc_end=sfc_begin+ml; so+=ml;
        const uint8_t* ac_begin=pk.sfc.data()+so;  const uint8_t* ac_end=ac_begin+al;
        vox_beta::SfcBitReader sfcr(sfc_begin,sfc_end);
        vox_beta::SfcBitReader acr(ac_begin,ac_end);
        size_t side_si=0;
        uint64_t gain_k=0,ar1_k=0,sgc_k=0,ac_gain_k=0;
        if(!sfcr.get(3,gain_k))throw std::runtime_error("SFC gain Rice header underflow");
        if(!sfcr.get(3,ar1_k))throw std::runtime_error("SFC AR1 Rice header underflow");
        if(!sfcr.get(2,sgc_k) || sgc_k>2u)throw std::runtime_error("SGC Rice header underflow");
        if(ac_begin!=ac_end && !acr.get(3,ac_gain_k))throw std::runtime_error("AC gain Rice header underflow");
        lx_gainpred::State gain_predictor, ar1_predictor, ar2_predictor, tilt_predictor, ac_gain_predictor;
        constexpr int SGC_NG=lx_sfc::SFC_FIRST_BAND/2;
        struct SgcPredState {
            std::array<int,SGC_NG> last{},prior{},current{};
            SgcPredState(){last.fill(-1);prior.fill(-1);current.fill(-1);}
            void begin_frame(){prior=last;current.fill(-1);}
            int predict(int g) const {const int t=prior[g],a=g?current[g-1]:-1,c=g?prior[g-1]:-1;if(t>=0&&a>=0&&c>=0){const int z=t+a-c;return std::clamp(z,std::min(t,a),std::max(t,a));}if(t>=0)return t;if(a>=0)return a;return -1;}
            void observe(int g,int q){current[g]=q;last[g]=q;}
        } sgc_predictor;
        auto sgc_unzz=[](unsigned v)->int{return (v&1u)?-int((v+1u)>>1):int(v>>1);};
        auto get_sgc_rice=[&](int pr,uint8_t& q)->bool{
            uint64_t v=0;if(pr<0){if(!sfcr.get(3,v))return false;q=uint8_t(v);return true;}
            unsigned n=0;for(;n<3u;++n){if(!sfcr.get(1,v))return false;if(!v)break;}
            if(n==3u){if(!sfcr.get(3,v))return false;q=uint8_t(v);return true;}
            uint64_t low=0;if(sgc_k&&!sfcr.get(unsigned(sgc_k),low))return false;const unsigned z=(n<<unsigned(sgc_k))|unsigned(low);const int d=pr+sgc_unzz(z);if(d<0||d>7)return false;q=uint8_t(d);return true;
        };
        std::array<DecSfcDecision,MODEL_SUPERFRAME> sfc_dec{};
        std::array<std::array<uint8_t,ERB_BANDS>,MODEL_SUPERFRAME> sgc_on{}, sgc_q{};
        std::array<DecCpmDecision,MODEL_SUPERFRAME> cpm_dec{};
        std::array<uint32_t,MODEL_SUPERFRAME> is_ang_packed{};
        std::array<uint8_t,MODEL_SUPERFRAME> is_start_dec{};
        std::array<std::array<uint8_t,ERB_BANDS>,MODEL_SUPERFRAME> is_q_dec{};
        is_start_dec.fill(uint8_t(ERB_BANDS));
        for(unsigned lf=0;lf<pk.h.frame_count;++lf){
            gain_predictor.begin_frame(int(lf)); sgc_predictor.begin_frame();
            ar1_predictor.begin_frame(int(lf)); ar2_predictor.begin_frame(int(lf)); tilt_predictor.begin_frame(int(lf));
            const unsigned pc=(pk.side[lf]>>3)&7u;
            if(pc>unsigned(lx_sfc::SFC_MAX_ACTIVE)) throw std::runtime_error("bad SFC cardinality");
            uint64_t v=0;
            if((pk.side[lf]&2u)!=0){
                if(!sfcr.get(1,v)) throw std::runtime_error("SGC present underflow");
                if(v){
                    constexpr int NG=lx_sfc::SFC_FIRST_BAND/2; int ksgc=0;
                    for(;;){if(!sfcr.get(1,v))throw std::runtime_error("SGC unary count underflow"); if(!v)break; if(++ksgc>NG)throw std::runtime_error("bad SGC group count");}
                    if(ksgc<1)throw std::runtime_error("bad SGC group count");
                    const unsigned rb=sgc_rank_bits(NG,ksgc); uint64_t rank=0; if(rb&&!sfcr.get(rb,rank))throw std::runtime_error("SGC rank underflow");
                    const uint64_t mask=sgc_unrank_mask_n(rank,NG,ksgc);
                    for(int g=0;g<NG;++g) if(mask&(uint64_t(1)<<g)){
                        sgc_on[lf][g]=1;
                        const int pr=sgc_predictor.predict(g);uint8_t q=0;if(!get_sgc_rice(pr,q))throw std::runtime_error("SGC angle Rice underflow");sgc_q[lf][g]=q;sgc_predictor.observe(g,int(q));
                    }
                }
            }
            {
                uint32_t is_ang=0;
                if((pk.side[lf]&4u)!=0){
                    if(is_cfg_dec()==4){
                        if(!sfcr.get(2,v)) throw std::runtime_error("IS suffix underflow");
                        const unsigned nib=unsigned(v)+1u;
                        if(nib<1u||nib>unsigned(IS_MAX_BANDS)) throw std::runtime_error("bad IS suffix");
                        const int ist=ERB_BANDS-int(nib); is_start_dec[lf]=uint8_t(ist);
                        if(is_bits_cfg()>0 && !sfcr.get(unsigned(is_bits_cfg()),v)) throw std::runtime_error("IS pooled angle underflow");
                        const uint8_t q=uint8_t(v); is_ang=uint32_t(q);
                        for(int b=ist;b<ERB_BANDS;++b) is_q_dec[lf][b]=q;
                    }else{
                        is_start_dec[lf]=uint8_t(IS_FIRST_BAND);
                        const unsigned ib=unsigned(is_code_count_cfg()*is_bits_cfg());
                        if(ib && !sfcr.get(ib,v)) throw std::runtime_error("IS angle payload underflow");
                        is_ang=uint32_t(v);
                        for(int b=IS_FIRST_BAND;b<ERB_BANDS;++b) is_q_dec[lf][b]=unpack_is_code(is_ang,b);
                    }
                }
                is_ang_packed[lf]=is_ang;
            }
            if(pc){
                const unsigned mb=unsigned(lx_sfc::sfc_subset_bits(int(pc)));
                if(!sfcr.get(mb,v)) throw std::runtime_error("SFC mask underflow");
                const uint32_t rank=uint32_t(v);
                uint8_t mask=0; uint32_t seen=0; bool found=false;
                for(unsigned m=1;m<(1u<<lx_sfc::SFC_BANDS);++m){if(std::popcount(m)!=int(pc))continue;if(seen==rank){mask=uint8_t(m);found=true;break;}++seen;}
                if(!found) throw std::runtime_error("bad SFC mask rank");
                sfc_dec[lf].mask=mask;
                for(int j=0;j<lx_sfc::SFC_BANDS;++j) if(mask&(1u<<j)){
                    uint8_t gq=0;
                    if(!lx_gainpred::get(sfcr,gain_predictor.predict(j),unsigned(gain_k),gq))throw std::runtime_error("SFC predicted gain underflow");
                    gain_predictor.observe(j,gq);sfc_dec[lf].gain[j]=gq;
                    if(!sfcr.get(1,v)) throw std::runtime_error("SFC mode underflow"); sfc_dec[lf].mode[j]=uint8_t(v);
                    if(!sfc_dec[lf].mode[j]){
                        uint8_t q=0;
                        if(!lx_gainpred::get(sfcr,ar1_predictor.predict(j),unsigned(ar1_k),q)) throw std::runtime_error("SFC AR1 underflow"); sfc_dec[lf].ar_k1[j]=q; ar1_predictor.observe(j,q);
                        if(!lx_sideinfo_rans::get(sfcr,side_syms,side_si,ar2_predictor.predict(j),lx_sfc::SFC_AR_BITS,q)) throw std::runtime_error("SFC AR2 underflow"); sfc_dec[lf].ar_k2[j]=q; ar2_predictor.observe(j,q);
                        if(!lx_sideinfo_rans::get(sfcr,side_syms,side_si,tilt_predictor.predict(j),lx_sfc::SFC_TILT_BITS,q)) throw std::runtime_error("SFC tilt underflow"); sfc_dec[lf].tilt[j]=q; tilt_predictor.observe(j,q);
                    }
                    const bool frame_ifc_syntax = (pk.side[lf]&0x40u)!=0;
                    if(frame_ifc_syntax){
                        if(!sfcr.get(1,v)) throw std::runtime_error("SFC IFC flag underflow"); sfc_dec[lf].ifc_on[j]=uint8_t(v);
                        if(v){
                            if(!sfcr.get(lx_sfc::SFC_IFC_FILTER_BITS,v)) throw std::runtime_error("SFC IFC mode underflow"); sfc_dec[lf].ifc_mode[j]=uint8_t(v);
                            if(!sfcr.get(lx_sfc::SFC_IFC_POS_BITS,v)) throw std::runtime_error("SFC IFC position underflow"); sfc_dec[lf].ifc_pos[j]=uint8_t(v);
                            if(!sfcr.get(lx_sfc::SFC_IFC_AMP_BITS,v)) throw std::runtime_error("SFC IFC amplitude underflow"); sfc_dec[lf].ifc_amp[j]=uint8_t(v);
                        }
                    }
                }
            }
            const uint8_t pctrl=pk.side[lf];const bool pshort=(pctrl&1u)!=0,pms=(pctrl&2u)!=0,pisen=(pctrl&4u)!=0;
            const bool cpm_on=(pctrl&0x80u)!=0;
            if(cpm_on){
                auto& cd=cpm_dec[lf];cd.on=1;
                if(!sfcr.get(1,v))throw std::runtime_error("CPM anchor flag underflow");cd.anchor=uint8_t(v);
                if(cd.anchor){
                    if(!sfcr.get(3,v))throw std::runtime_error("CPM band underflow");cd.band=uint8_t(v);
                    if(!sfcr.get(8,v))throw std::runtime_error("CPM position underflow");cd.pos=uint8_t(v);
                    if(cd.band>=lx_sfc::SFC_BANDS)throw std::runtime_error("bad CPM band");
                }
            }
        }

        if(is_cfg_dec()==4){
            uint64_t expected=0;
            for(unsigned lf=0;lf<pk.h.frame_count;++lf){
                const auto& ed=edges_for(block_at(lf)); const int coded=ed[ERB_BANDS];
                uint64_t n=uint64_t(2*coded);
                const int ist=int(is_start_dec[lf]); if(ist<ERB_BANDS)n-=uint64_t(coded-ed[ist]);
                expected+=n;
            }
        }

        Rans4 dlr,dc,dm;if(!dlr.init(pk.lrr)||!dc.init(pk.cr)||!dm.init(pk.mr))throw std::runtime_error("bad rANS stream");
        const uint8_t* psr=pk.sraw.data();const uint8_t* esr=psr+pk.sraw.size();
        const uint8_t* plrr=pk.lrraw.data();const uint8_t* elrr=plrr+pk.lrraw.size();
        const uint8_t* pmr=pk.mraw.data();const uint8_t* emr=pmr+pk.mraw.size();
        std::array<std::array<uint32_t,9>,CLASS_TUPLE_CTX> hClassTuple{};std::array<std::array<uint32_t,MAG_ALPHA>,MAG_CTX> hMag{};std::array<uint32_t,256> hScale{},hLrScale{};
        // Magnitude rANS is a separate stream. Batch four non-zero magnitudes
        // whenever a band uses one magnitude context, while class prediction remains scalar/causal.
        auto decode_mag4 = [&](int32_t* q,const int* pos,const int* sign,int count,uint8_t mctx,lx_conditional_gain::DnMoments* mom)->bool{
            uint8_t sy[4];
            if(count==4){
                if(!dm.get4_same(magModel.m[mctx],sy))return false;
                for(int j=0;j<4;++j){++hMag[mctx][sy[j]];uint32_t u=sy[j];if(sy[j]==MAG_ESC&&!vox_beta::get_uleb128(pmr,emr,u))return false;if(u>=uint32_t(INT32_MAX))return false;const uint32_t a=u+1u;q[pos[j]]=sign[j]*int32_t(a);if(mom){mom->q1+=double(a);mom->q2+=double(a)*double(a);mom->nz+=1.0;}}
                return true;
            }
            for(int j=0;j<count;++j){uint8_t ms;if(!dm.get(magModel.m[mctx],ms))return false;++hMag[mctx][ms];uint32_t u=ms;if(ms==MAG_ESC&&!vox_beta::get_uleb128(pmr,emr,u))return false;if(u>=uint32_t(INT32_MAX))return false;const uint32_t a=u+1u;q[pos[j]]=sign[j]*int32_t(a);if(mom){mom->q1+=double(a);mom->q2+=double(a)*double(a);mom->nz+=1.0;}}
            return true;
        };

        auto decode_class_band = [&](int32_t* q,int lo,int hi,int b,int ch,uint8_t mbase,lx_conditional_gain::DnMoments* mom)->bool{
            uint8_t prev_mag_state=0;
            uint64_t mq1=0,mq2=0,mnz=0;
            auto push_nonzero=[&](int qi,int sign)->bool{
                const uint16_t mctx=uint16_t(unsigned(mbase)*MAG_LOCAL_STATES + prev_mag_state);
                uint8_t ms; if(!dm.get(magModel.m[mctx],ms))return false;
                ++hMag[mctx][ms];
                uint32_t u=ms; if(ms==MAG_ESC&&!vox_beta::get_uleb128(pmr,emr,u))return false;
                if(u>=uint32_t(INT32_MAX))return false;const uint32_t a=u+1u; q[size_t(qi)]=sign*int32_t(a);
                mq1+=uint64_t(a);mq2+=uint64_t(a)*uint64_t(a);++mnz;
                prev_mag_state=uint8_t(1u + uint32_t(a>1u));
                return true;
            };
            uint8_t prev_count=0;
            const uint8_t fctx=uint8_t(b>>1);
            uint8_t ts=0;
            if(magpred_prev_valid[ch][b]){
                const uint32_t den=uint32_t(std::max(1,hi-lo));
                ts=uint8_t(1u+std::min(2u,(3u*magpred_prev_nz[ch][b])/den));
            }
            int qi=0;
            static constexpr uint8_t kTupleUnpack[9]={0x0,0x11,0x12,0x14,0x25,0x26,0x18,0x29,0x2a};
            for(int ii=lo;ii<hi;ii+=2,qi+=2){
                const uint16_t cctx=uint16_t((unsigned(fctx)*3u+prev_count)*CLASS_TEMP_STATES+ts);
                uint8_t tup; if(!dc.get9(classTupleModel.m[cctx],tup) || tup>=9)return false;
                ++hClassTuple[cctx][tup];
                const uint8_t up=kTupleUnpack[tup];
                const uint8_t c0=up&3u,c1=(up>>2)&3u;
                q[size_t(qi)]=0;
                if(c0 && !push_nonzero(qi,3-2*int(c0)))return false;
                if(ii+1<hi){
                    q[size_t(qi+1)]=0;
                    if(c1 && !push_nonzero(qi+1,3-2*int(c1)))return false;
                }
                prev_count=uint8_t(up>>4);
            }
            if(mom){mom->q1=double(mq1);mom->q2=double(mq2);mom->nz=double(mnz);}
            return true;
        };

        // Scale surface is decoded first, exactly as in the encoder grammar.
        for(unsigned lf=0;lf<pk.h.frame_count;++lf){
            size_t fi=globalFrame+lf;
            for(int b=0;b<ERB_BANDS;++b){uint8_t ss;if(!ds.get(scaleModel,ss))throw std::runtime_error("scale rANS underflow");++hScale[ss];uint32_t zz=ss;if(ss==255u&&!vox_beta::get_uleb128(psr,esr,zz))throw std::runtime_error("scale raw underflow");int pred=scale_predict(scales,fi,b);const int64_t sk=int64_t(pred)+int64_t(unzigzag_u32(zz));if(sk<0||sk>480)throw std::runtime_error("scale out of range");scales[fi*ERB_BANDS+b]=uint16_t(sk);}
        }
        if(psr!=esr||!ds.done())throw std::runtime_error("scale stream length mismatch");

        // R332 true-independent L/R geometry, one signed scale offset per
        // active D_n EGV. Decoder reconstructs the same per-band q surface.
        for(unsigned lf=0;lf<pk.h.frame_count;++lf){
            const size_t fi=globalFrame+lf;
            const bool lr_frame=(pk.side[lf]&2u)==0;
            const int istart=int(is_start_dec[lf]);
            for(int b=0;b<ERB_BANDS;++b)lr_balance_q[fi*ERB_BANDS+size_t(b)]=0;
            if(!lr_frame)continue;
            const int bt=int(block_at(lf));
            auto decode_groups=[&](const auto& groups){
                int gi=0,left_q=0;
                for(const auto g:groups){
                    const int b0=int(g.b0),b1=std::min<int>(int(g.b1),istart);
                    if(b0>=b1){++gi;continue;}
                    uint8_t ss;if(!dlr.get(lrScaleModel,ss))throw std::runtime_error("LR scale rANS underflow");++hLrScale[ss];
                    uint32_t zz=ss;if(ss==255u&&!vox_beta::get_uleb128(plrr,elrr,zz))throw std::runtime_error("LR scale raw underflow");
                    int pred=left_q;
                    if(lr_pred_valid[bt][gi]){
                        const int top=int(lr_pred_prev[bt][gi]);
                        pred=(gi==0)?top:int(std::nearbyint(0.5*double(left_q+top)));
                    }
                    const int64_t q=int64_t(pred)+int64_t(unzigzag_u32(zz));
                    int qlo=-480,qhi=480;
                    for(int b=b0;b<b1;++b){
                        const int common=int(scales[fi*ERB_BANDS+size_t(b)]);
                        qlo=std::max(qlo,std::max(-common,common-480));
                        qhi=std::min(qhi,std::min(480-common,common));
                    }
                    if(q<qlo||q>qhi)throw std::runtime_error("LR scale out of range");
                    for(int b=b0;b<b1;++b)lr_balance_q[fi*ERB_BANDS+size_t(b)]=int16_t(q);
                    lr_pred_prev[bt][gi]=int16_t(q);lr_pred_valid[bt][gi]=1u;
                    left_q=q;++gi;
                }
            };
            switch(block_at(lf)){
                case BlockSize::Long2048:decode_groups(EGV_LONG);break;
                case BlockSize::Medium1024:decode_groups(EGV_1024);break;
                case BlockSize::Short512:decode_groups(EGV_SHORT);break;
                case BlockSize::Micro256:decode_groups(EGV_256);break;
            }
        }
        if(plrr!=elrr||!dlr.done())throw std::runtime_error("LR scale stream length mismatch");

        for(unsigned lf=0;lf<pk.h.frame_count;++lf){
            ac_gain_predictor.begin_frame(int(lf));
            const size_t fi=globalFrame+lf;const uint8_t ctrl=pk.side[lf];const uint32_t ang=is_ang_packed[lf];
            const int istart_dec=int(is_start_dec[lf]);
            const auto& sfd=sfc_dec[lf];const auto& cfd=cpm_dec[lf];
            const auto sfc_guard_exposure=sfc_diffuse_exposure;
            const BlockSize bs=block_at(lf);
            const bool ms=(ctrl&2u)!=0;
            const bool isen=(ctrl&4u)!=0;
            const int k=block_samples(bs);
            const auto& edge=edges_for(bs);const int coded=edge[ERB_BANDS];
            const double cpm_df=48000.0/(2.0*double(k));
            int cpm_pred_ip=-1,cpm_pred_band=-1;
            std::array<float,ERB_BANDS> direct_debias_gain_l{},direct_debias_gain_r{};
            direct_debias_gain_l.fill(1.0f);direct_debias_gain_r.fill(1.0f);
            if(cpm_state.valid){
                cpm_pred_ip=int(std::floor(double(cpm_state.freq_hz)/cpm_df));
                for(int bb=lx_sfc::SFC_FIRST_BAND;bb<ERB_BANDS;++bb)if(cpm_pred_ip>=edge[bb]+2&&cpm_pred_ip<=edge[bb+1]-3){cpm_pred_band=bb;break;}
            }
            bool cpm_tpl_valid=false; int cpm_tpl_ip=-1; float cpm_tc[5]{},cpm_ts[5]{};
            auto ensure_cpm_templates=[&](int ip){
                if(cpm_tpl_valid && cpm_tpl_ip==ip)return;
                const bool hl=pos!=0,hr=pos+size_t(k)+size_t(min_block)+size_t(lx_nsmdct::OVERLAP)<=vh.total_samples_per_channel;
                const int ci=(bs==BlockSize::Long2048)?0:(bs==BlockSize::Medium1024)?1:(bs==BlockSize::Short512)?2:3;
                auto &cc=cpm_tpl_cache[size_t(ci)][size_t(std::clamp(ip,0,K_LONG-1))];
                if(hl&&hr&&cc.valid&&cc.freq_hz==cpm_state.freq_hz){
                    const double ww=2.0*3.1415926535897932384626433832795*double(cpm_state.freq_hz)/48000.0;
                    const double ph=ww*double(ptrdiff_t(pos)-ptrdiff_t(cc.pos));
                    const float cp=float(std::cos(ph)),sp=float(std::sin(ph));
                    for(int t=0;t<5;++t){const float C=cc.c[t],S=cc.s[t];cpm_tc[t]=cp*C-sp*S;cpm_ts[t]=sp*C+cp*S;}
                }else{
                    switch(bs){case BlockSize::Long2048:lx_cpm::physical_line_templates(cpmLong,cpm_state.freq_hz,pos,hl,hr,ip,cpm_tc,cpm_ts);break;case BlockSize::Medium1024:lx_cpm::physical_line_templates(cpm1024,cpm_state.freq_hz,pos,hl,hr,ip,cpm_tc,cpm_ts);break;case BlockSize::Short512:lx_cpm::physical_line_templates(cpmShort,cpm_state.freq_hz,pos,hl,hr,ip,cpm_tc,cpm_ts);break;case BlockSize::Micro256:lx_cpm::physical_line_templates(cpm256,cpm_state.freq_hz,pos,hl,hr,ip,cpm_tc,cpm_ts);break;}
                    if(hl&&hr){cc.valid=true;cc.freq_hz=cpm_state.freq_hz;cc.pos=pos;for(int t=0;t<5;++t){cc.c[t]=cpm_tc[t];cc.s[t]=cpm_ts[t];}}
                }
                cpm_tpl_valid=true;cpm_tpl_ip=ip;
            };
            for(int b=0;b<ERB_BANDS;++b){
                if(b==lx_sfc::SFC_FIRST_BAND && ms){
                    constexpr int NG=lx_sfc::SFC_FIRST_BAND/2;
                    for(int g=0;g<NG;++g) if(sgc_on[lf][g]){
                        const int b0=2*g,b1=b0+1,slo=edge[b0],shi=edge[b1+1];
                        double qE=0.0,d2sum=0.0;
                        for(int sb=b0;sb<=b1;++sb){
                            const int ssk=int(scales[fi*ERB_BANDS+sb]),soct=ssk>>4,sfrac=ssk&15;
                            const double sd=double(delta_code_f[std::clamp((soct<<4)|sfrac,0,480)]);
                            d2sum+=double(edge[sb+1]-edge[sb])*sd*sd;
                        }
                        for(int ii=slo;ii<shi;++ii)qE+=double(cL[ii])*cL[ii]+double(cR[ii])*cR[ii];
                        const double esum=qE+d2sum;
                        if(esum>0.0){
                            // R348 3-bit SGC: midpoint codebook
                            // +/-{1,3,5,7}/8 * theta.  Build theta/8 once with
                            // half-angle identities, then synthesize odd multiples
                            // algebraically; no atan2/sin/cos in the hot path.
                            const double ct=std::sqrt(std::clamp(qE/esum,0.0,1.0));
                            const double st=std::sqrt(std::clamp(d2sum/esum,0.0,1.0));
                            const double c2=std::sqrt(0.5*(1.0+ct));
                            const double s2=std::sqrt(0.5*(1.0-ct));
                            const double c4=std::sqrt(0.5*(1.0+c2));
                            const double s4=std::sqrt(0.5*(1.0-c2));
                            const double c8=std::sqrt(0.5*(1.0+c4));
                            const double s8=std::sqrt(0.5*(1.0-c4));
                            const double c3=c4*c8-s4*s8, s3=s4*c8+c4*s8;
                            const double c5=c2*c8-s2*s8, s5=s2*c8+c2*s8;
                            const double c7=ct*c8+st*s8, s7=st*c8-ct*s8;
                            const unsigned q=sgc_q[lf][g];const unsigned mag=(q<4u)?(7u-2u*q):(2u*q-7u);
                            double cc=c8,ss=s8;if(mag==3u){cc=c3;ss=s3;}else if(mag==5u){cc=c5;ss=s5;}else if(mag==7u){cc=c7;ss=s7;}
                            ss=std::copysign(ss,q<4u?-1.0:1.0);
                            for(int ii=slo;ii<shi;++ii){const float qx=cL[ii],qy=cR[ii];cL[ii]=float(cc*qx-ss*qy);cR[ii]=float(ss*qx+cc*qy);}
                        }
                    }
                }
                const int lo=edge[b],hi=edge[b+1];
                const bool sfc_active=b>=lx_sfc::SFC_FIRST_BAND && (sfd.mask&(1u<<(b-lx_sfc::SFC_FIRST_BAND)));
                const int nch=(isen&&b>=istart_dec)?1:2;
                uint8_t magctx_ch[2]={0,0};
                for(int ch=0;ch<nch;++ch){
                    uint8_t ps=0;
                    if(magpred_prev_valid[ch][b] && magpred_prev_nz[ch][b]){
                        const uint64_t q1=magpred_prev_q1[ch][b];
                        const uint64_t nz=magpred_prev_nz[ch][b];
                        ps=uint8_t(1u + uint32_t(q1>=2u*nz) + uint32_t(q1>=4u*nz));
                    }
                    magctx_ch[ch]=uint8_t(unsigned(MAG_CTX_MAP[b])*MAG_PRED_STATES + ps);
                }
                const int sk=int(scales[fi*ERB_BANDS+b]);const float delta=delta_code_f[std::clamp(sk,0,480)];
                const int lrq=(!ms && nch==2)?int(lr_balance_q[fi*ERB_BANDS+size_t(b)]):0;
                const float lrg=std::exp2(float(lrq)*(1.0f/16.0f));
                const float deltaA=delta*lrg,deltaB=delta/lrg;
                float ct=1.0f,st=0.0f;
                if(isen&&b>=istart_dec){ const unsigned aq=is_q_dec[lf][b]; ct=is_ct[aq]; st=is_st[aq]; }
                // Fast path for ordinary non-SFC/non-CPM bands: decode the second
                // channel directly into the final LR spectrum, eliminating the old
                // qa/qb + second dequantization pass. Arithmetic order is preserved.
                const uint8_t cb_fast=cfd.anchor?cfd.band:uint8_t(std::max(0,cpm_pred_band-lx_sfc::SFC_FIRST_BAND));
                const bool cpm_here_fast=(cfd.on && ((cfd.anchor && int(cb_fast)==b-lx_sfc::SFC_FIRST_BAND) || (!cfd.anchor && cpm_pred_band==b)));
                if(!sfc_active && !cpm_here_fast){
                    lx_conditional_gain::DnMoments ma{},mb{};
                    if(!decode_class_band(qa.data(),lo,hi,b,0,magctx_ch[0],&ma))throw std::runtime_error("class/mag rANS underflow");
                    if(nch==1){
                        int ii=lo;
#if defined(__AVX2__)
                        if(warp_off){
                            const __m256 vd=_mm256_set1_ps(delta),vct=_mm256_set1_ps(ct),vst=_mm256_set1_ps(st);
                            for(;ii+8<=hi;ii+=8){
                                const __m256i qv=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(qa.data()+size_t(ii-lo)));
                                const __m256 a=_mm256_mul_ps(_mm256_cvtepi32_ps(qv),vd);
                                _mm256_storeu_ps(cL.data()+ii,_mm256_mul_ps(vct,a));
                                _mm256_storeu_ps(cR.data()+ii,_mm256_mul_ps(vst,a));
                            }
                        }
#elif defined(__SSE4_1__)
                        if(warp_off){
                            const __m128 vd=_mm_set1_ps(delta),vct=_mm_set1_ps(ct),vst=_mm_set1_ps(st);
                            for(;ii+4<=hi;ii+=4){
                                const __m128i qv=_mm_loadu_si128(reinterpret_cast<const __m128i*>(qa.data()+size_t(ii-lo)));
                                const __m128 a=_mm_mul_ps(_mm_cvtepi32_ps(qv),vd);
                                _mm_storeu_ps(cL.data()+ii,_mm_mul_ps(vct,a));
                                _mm_storeu_ps(cR.data()+ii,_mm_mul_ps(vst,a));
                            }
                        }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
                        if(warp_off){
                            const float32x4_t vd=vdupq_n_f32(delta),vct=vdupq_n_f32(ct),vst=vdupq_n_f32(st);
                            for(;ii+4<=hi;ii+=4){
                                const int32x4_t qv=vld1q_s32(qa.data()+size_t(ii-lo));
                                const float32x4_t a=vmulq_f32(vcvtq_f32_s32(qv),vd);
                                vst1q_f32(cL.data()+ii,vmulq_f32(vct,a));
                                vst1q_f32(cR.data()+ii,vmulq_f32(vst,a));
                            }
                        }
#endif
                        for(;ii<hi;++ii){const float ww=warp_off?1.0f:lx_warp_weight(b,ii-lo,hi-lo);const float a=float(qa[size_t(ii-lo)])*(delta/ww);cL[ii]=ct*a;cR[ii]=st*a;}
                    }else{
                        if(!decode_class_band(qb.data(),lo,hi,b,1,magctx_ch[1],&mb))throw std::runtime_error("class/mag rANS underflow");
                        int ii=lo;
#if defined(__AVX2__)
                        if(warp_off){
                            const __m256 vdA=_mm256_set1_ps(deltaA),vdB=_mm256_set1_ps(deltaB),vd=_mm256_set1_ps(delta),vg=_mm256_set1_ps(INV_SQRT2);
                            for(;ii+8<=hi;ii+=8){
                                const size_t qi=size_t(ii-lo);
                                const __m256 qa_v=_mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(qa.data()+qi)));
                                const __m256 qb_v=_mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(qb.data()+qi)));
                                const __m256 a=_mm256_mul_ps(qa_v,ms?vd:vdA);
                                const __m256 bb=_mm256_mul_ps(qb_v,ms?vd:vdB);
                                if(!ms){_mm256_storeu_ps(cL.data()+ii,a);_mm256_storeu_ps(cR.data()+ii,bb);}
                                else{_mm256_storeu_ps(cL.data()+ii,_mm256_mul_ps(_mm256_add_ps(a,bb),vg));_mm256_storeu_ps(cR.data()+ii,_mm256_mul_ps(_mm256_sub_ps(a,bb),vg));}
                            }
                        }
#elif defined(__SSE4_1__)
                        if(warp_off){
                            const __m128 vdA=_mm_set1_ps(deltaA),vdB=_mm_set1_ps(deltaB),vd=_mm_set1_ps(delta),vg=_mm_set1_ps(INV_SQRT2);
                            for(;ii+4<=hi;ii+=4){
                                const size_t qi=size_t(ii-lo);
                                const __m128 qa_v=_mm_cvtepi32_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(qa.data()+qi)));
                                const __m128 qb_v=_mm_cvtepi32_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(qb.data()+qi)));
                                const __m128 a=_mm_mul_ps(qa_v,ms?vd:vdA);
                                const __m128 bb=_mm_mul_ps(qb_v,ms?vd:vdB);
                                if(!ms){_mm_storeu_ps(cL.data()+ii,a);_mm_storeu_ps(cR.data()+ii,bb);}
                                else{_mm_storeu_ps(cL.data()+ii,_mm_mul_ps(_mm_add_ps(a,bb),vg));_mm_storeu_ps(cR.data()+ii,_mm_mul_ps(_mm_sub_ps(a,bb),vg));}
                            }
                        }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
                        if(warp_off){
                            const float32x4_t vdA=vdupq_n_f32(deltaA),vdB=vdupq_n_f32(deltaB),vd=vdupq_n_f32(delta),vg=vdupq_n_f32(INV_SQRT2);
                            for(;ii+4<=hi;ii+=4){
                                const size_t qi=size_t(ii-lo);
                                const float32x4_t qa_v=vcvtq_f32_s32(vld1q_s32(qa.data()+qi));
                                const float32x4_t qb_v=vcvtq_f32_s32(vld1q_s32(qb.data()+qi));
                                const float32x4_t a=vmulq_f32(qa_v,ms?vd:vdA);
                                const float32x4_t bb=vmulq_f32(qb_v,ms?vd:vdB);
                                if(!ms){vst1q_f32(cL.data()+ii,a);vst1q_f32(cR.data()+ii,bb);}
                                else{vst1q_f32(cL.data()+ii,vmulq_f32(vaddq_f32(a,bb),vg));vst1q_f32(cR.data()+ii,vmulq_f32(vsubq_f32(a,bb),vg));}
                            }
                        }
#endif
                        for(;ii<hi;++ii){const int qi=ii-lo;const int32_t qbi=qb[size_t(qi)];const float ww=warp_off?1.0f:lx_warp_weight(b,qi,hi-lo);const float a=float(qa[size_t(qi)])*((ms?delta:deltaA)/ww),bb=float(qbi)*((ms?delta:deltaB)/ww);if(!ms){cL[ii]=a;cR[ii]=bb;}else{cL[ii]=(a+bb)*INV_SQRT2;cR[ii]=(a-bb)*INV_SQRT2;}}
                    }
                    {
                            if(!ms && nch==2){
                                direct_debias_gain_l[b]=lx_conditional_gain::conditional_dn_gain_from_moments(ma.q1,ma.q2,ma.nz,uint32_t(hi-lo));
                                direct_debias_gain_r[b]=lx_conditional_gain::conditional_dn_gain_from_moments(mb.q1,mb.q2,mb.nz,uint32_t(hi-lo));
                            }else if(warp_off){
                                const double q1=ma.q1+(nch==2?mb.q1:0.0), q2=ma.q2+(nch==2?mb.q2:0.0), nz=ma.nz+(nch==2?mb.nz:0.0);
                                const float gg=lx_conditional_gain::conditional_dn_gain_from_moments(q1,q2,nz,uint32_t((hi-lo)*nch));
                                direct_debias_gain_l[b]=gg;direct_debias_gain_r[b]=gg;
                            }else{
                                const float gg=lx_conditional_gain::conditional_dn_gain(qa.data(),qb.data(),hi-lo,nch,b,delta);
                                direct_debias_gain_l[b]=gg;direct_debias_gain_r[b]=gg;
                            }
                    }
                    if(b>=lx_sfc::SFC_FIRST_BAND){
                        const uint32_t nz=uint32_t(ma.nz + (nch==2?mb.nz:0.0));
                        if(!nz){
                            uint8_t acq=0;
                            if(!lx_gainpred::get(acr,ac_gain_predictor.predict(b-lx_sfc::SFC_FIRST_BAND),unsigned(ac_gain_k),acq)) throw std::runtime_error("anti-collapse predicted gain underflow");
                            ac_gain_predictor.observe(b-lx_sfc::SFC_FIRST_BAND,acq);
                            const float target_l2=lx_sfc::decode_gain32(acq)*std::sqrt(2.0f*float(k));
                            const int sb=std::max(0,b-1),slo=edge[sb],shi=edge[sb+1];
                            lx_sfc::anti_collapse_stereo_band(cL.data(),cR.data(),lo,hi,target_l2,uint32_t(fi),nch==1,ct,st,ms);
                        }
                    }
                    magpred_prev_q1[0][b]=uint64_t(ma.q1); magpred_prev_nz[0][b]=uint32_t(ma.nz); magpred_prev_valid[0][b]=1u;
                    if(nch==2){magpred_prev_q1[1][b]=uint64_t(mb.q1);magpred_prev_nz[1][b]=uint32_t(mb.nz);magpred_prev_valid[1][b]=1u;}
                    else {magpred_prev_q1[1][b]=0;magpred_prev_nz[1][b]=0;magpred_prev_valid[1][b]=0;}
                    continue;
                }
                lx_conditional_gain::DnMoments predmom[2]{};
                for(int ch=0;ch<nch;++ch){int32_t* q=ch?qb.data():qa.data();if(!decode_class_band(q,lo,hi,b,ch,magctx_ch[ch],&predmom[ch]))throw std::runtime_error("class/mag rANS underflow");
                    magpred_prev_q1[ch][b]=uint64_t(predmom[ch].q1); magpred_prev_nz[ch][b]=uint32_t(predmom[ch].nz); magpred_prev_valid[ch][b]=1u;}
                for(int ch=nch;ch<2;++ch){magpred_prev_q1[ch][b]=0;magpred_prev_nz[ch][b]=0;magpred_prev_valid[ch][b]=0;}
                const float target_l2=sfc_active?lx_sfc::decode_gain32(sfd.gain[b-lx_sfc::SFC_FIRST_BAND])*std::sqrt(2.0f*float(k)):0.0f;
                if(sfc_active){
                    const int sj=b-lx_sfc::SFC_FIRST_BAND;
                    if(sfd.mode[sj]){
                        const int sb=b-1; const int slo=edge[sb],shi=edge[sb+1];
                        lx_sfc::fold_stereo_band(cL.data(),cR.data(),slo,shi,lo,hi,target_l2);
                    }else lx_sfc::continue_stereo_band_guided(cL.data(),cR.data(),lo,hi,target_l2,uint32_t(fi),sfd.ar_k1[sj],sfd.ar_k2[sj]);
                    if(!sfd.mode[sj]){
                        lx_sfc::apply_spectral_tilt3(cL.data(),cR.data(),lo,hi,target_l2,sfd.tilt[sj]);
                        lx_sfc::apply_sfc_diffuse_guard(cL.data(),cR.data(),lo,hi,target_l2,sfd.tilt[sj],sfc_guard_exposure[sj]);
                    }
                    if(sfd.ifc_on[sj]) lx_sfc::apply_ifc_band(cL.data(),cR.data(),lo,hi,target_l2,sfd.ifc_mode[sj],sfd.ifc_pos[sj],sfd.ifc_amp[sj]);
                }
#if LX_PREC_RES_MODE == 2 || LX_PREC_RES_MODE == 6
                const float qstep=sfc_active?0.5f*delta:delta;
#elif LX_PREC_RES_MODE == 3
                const float qstep=sfc_active?0.25f*delta:delta;
#elif LX_PREC_RES_MODE == 4
                const float qstep=sfc_active?2.0f*delta:delta;
#else
                const float qstep=delta;
#endif
                const float cpm_qstep=std::ldexp(qstep,-LX_CPM_PREC_SHIFT);
                const uint8_t cb_resolved=cfd.anchor?cfd.band:uint8_t(std::max(0,cpm_pred_band-lx_sfc::SFC_FIRST_BAND));
                const uint8_t cp_resolved=cfd.anchor?cfd.pos:uint8_t(cpm_pred_ip>=0?cpm_pred_ip-lo:0);
                const bool cpm_here_any=(cfd.on && ((cfd.anchor && int(cb_resolved)==b-lx_sfc::SFC_FIRST_BAND) || (!cfd.anchor && cpm_pred_band==b)));
                const bool cpm_cont_any=cpm_here_any && !cfd.anchor && cpm_state.valid;
                if(sfc_active){
                    if(!cpm_here_any){
#if LX_PREC_RES_MODE > 0
                        lx_sfc::guide_stereo_band_with_precision(cL.data(),cR.data(),lo,hi,target_l2,
                            qa.data(),qb.data(),nch,ms,ct,st,qstep);
#endif
                    }else{
                        const int cip=std::clamp(lo+int(cp_resolved),lo+2,hi-3);
                        float predAbsA[5]{},predAbsB[5]{};
                        if(cpm_cont_any){
                            float pl[5],pr[5];ensure_cpm_templates(cip);
                            lx_cpm::synth_complex_line5(cpm_state.al,cpm_state.bl,cpm_tc,cpm_ts,pl);lx_cpm::synth_complex_line5(cpm_state.ar,cpm_state.br,cpm_tc,cpm_ts,pr);
                            for(int t=0;t<5;++t){if(nch==1){predAbsA[t]=ct*pl[t]+st*pr[t];predAbsB[t]=0.f;}else if(!ms){predAbsA[t]=pl[t];predAbsB[t]=pr[t];}else{predAbsA[t]=(pl[t]+pr[t])*INV_SQRT2;predAbsB[t]=(pl[t]-pr[t])*INV_SQRT2;}}
                        }
                        for(int ii=cip-2;ii<=cip+2;++ii){
                            const int t=ii-(cip-2);const int32_t qai=qa[size_t(ii-lo)],qbi=qb[size_t(ii-lo)];float a=0.f,bb=0.f;
                            if(cpm_cont_any){
                                a=predAbsA[t]+float(qai)*cpm_qstep;if(nch==2)bb=predAbsB[t]+float(qbi)*cpm_qstep;
                            }else{a=float(qai)*cpm_qstep;if(nch==2)bb=float(qbi)*cpm_qstep;}
                            if(isen&&b>=istart_dec){cL[ii]=ct*a;cR[ii]=st*a;}else if(!ms){cL[ii]=a;cR[ii]=bb;}else{cL[ii]=(a+bb)*INV_SQRT2;cR[ii]=(a-bb)*INV_SQRT2;}
                        }
                    }
                }else{
                    const uint8_t cb=cfd.anchor?cfd.band:uint8_t(std::max(0,cpm_pred_band-lx_sfc::SFC_FIRST_BAND));
                    const uint8_t cp=cfd.anchor?cfd.pos:uint8_t(cpm_pred_ip>=0?cpm_pred_ip-lo:0);
                    const bool cpm_here=(cfd.on && ((cfd.anchor && int(cb)==b-lx_sfc::SFC_FIRST_BAND) || (!cfd.anchor && cpm_pred_band==b)));
                    const bool cpm_cont=cpm_here && !cfd.anchor && cpm_state.valid;
                    int cip=0;if(cpm_here)cip=std::clamp(lo+int(cp),lo+2,hi-3);
                    float predAbsA[5]{},predAbsB[5]{};
                    if(cpm_cont){
                        float pl[5],pr[5];ensure_cpm_templates(cip);
                        lx_cpm::synth_complex_line5(cpm_state.al,cpm_state.bl,cpm_tc,cpm_ts,pl);lx_cpm::synth_complex_line5(cpm_state.ar,cpm_state.br,cpm_tc,cpm_ts,pr);
                        for(int t=0;t<5;++t){if(nch==1){predAbsA[t]=ct*pl[t]+st*pr[t];predAbsB[t]=0.f;}else if(!ms){predAbsA[t]=pl[t];predAbsB[t]=pr[t];}else{predAbsA[t]=(pl[t]+pr[t])*INV_SQRT2;predAbsB[t]=(pl[t]-pr[t])*INV_SQRT2;}}
                    }
                    const float qstepA=(!ms&&nch==2)?deltaA:qstep;
                    const float qstepB=(!ms&&nch==2)?deltaB:qstep;
                    for(int ii=lo;ii<hi;++ii){
                        const int32_t qai=qa[size_t(ii-lo)],qbi=qb[size_t(ii-lo)];float a=float(qai)*qstepA,bb=float(qbi)*qstepB;
                        if(cpm_cont && ii>=cip-2 && ii<=cip+2){
                            const int t=ii-(cip-2);
                            a=predAbsA[t]+float(qai)*cpm_qstep;if(nch==2)bb=predAbsB[t]+float(qbi)*cpm_qstep;else bb=0.f;
                        }
                        if(isen&&b>=istart_dec){cL[ii]=ct*a;cR[ii]=st*a;}else if(!ms){cL[ii]=a;cR[ii]=bb;}else{cL[ii]=(a+bb)*INV_SQRT2;cR[ii]=(a-bb)*INV_SQRT2;}
                    }
                }
            }
            if(cfd.on){
                const int b=cfd.anchor?(lx_sfc::SFC_FIRST_BAND+int(cfd.band)):cpm_pred_band;
                if(b<lx_sfc::SFC_FIRST_BAND||b>=ERB_BANDS)throw std::runtime_error("CPM continuation frequency out of range");
                const int lo=edge[b],hi=edge[b+1];
                const int ip=cfd.anchor?std::clamp(lo+int(cfd.pos),lo+2,hi-3):cpm_pred_ip;
                if(cfd.anchor || !cpm_state.valid)cpm_state.freq_hz=float((double(ip)+0.5)*cpm_df);
                float xl[5],xr[5];ensure_cpm_templates(ip);
                for(int t=0;t<5;++t){const int ii=ip+t-2;xl[t]=cL[ii];xr[t]=cR[ii];}
                lx_cpm::fit_complex_line5(xl,cpm_tc,cpm_ts,cpm_state.al,cpm_state.bl);lx_cpm::fit_complex_line5(xr,cpm_tc,cpm_ts,cpm_state.ar,cpm_state.br);
                cpm_state.valid=true;cpm_state.band=uint8_t(b-lx_sfc::SFC_FIRST_BAND);cpm_state.pos=uint8_t(ip-lo);cpm_state.frame=fi;
            }
            for(int b=0;b<ERB_BANDS;++b){
                const int rlo=edge[b],rhi=edge[b+1];
                const float gL=direct_debias_gain_l[b],gR=direct_debias_gain_r[b]; int ii=rlo;
#if defined(__AVX2__)
                const __m256 vgL=_mm256_set1_ps(gL),vgR=_mm256_set1_ps(gR);
                for(;ii+8<=rhi;ii+=8){
                    float* pl=cL.data()+size_t(ii); float* pr=cR.data()+size_t(ii);
                    _mm256_storeu_ps(pl,_mm256_mul_ps(_mm256_loadu_ps(pl),vgL));
                    _mm256_storeu_ps(pr,_mm256_mul_ps(_mm256_loadu_ps(pr),vgR));
                }
#elif defined(__SSE4_1__)
                const __m128 vgL=_mm_set1_ps(gL),vgR=_mm_set1_ps(gR);
                for(;ii+4<=rhi;ii+=4){
                    float* pl=cL.data()+size_t(ii); float* pr=cR.data()+size_t(ii);
                    _mm_storeu_ps(pl,_mm_mul_ps(_mm_loadu_ps(pl),vgL));
                    _mm_storeu_ps(pr,_mm_mul_ps(_mm_loadu_ps(pr),vgR));
                }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
                const float32x4_t vgL=vdupq_n_f32(gL),vgR=vdupq_n_f32(gR);
                for(;ii+4<=rhi;ii+=4){
                    float* pl=cL.data()+size_t(ii); float* pr=cR.data()+size_t(ii);
                    vst1q_f32(pl,vmulq_f32(vld1q_f32(pl),vgL));
                    vst1q_f32(pr,vmulq_f32(vld1q_f32(pr),vgR));
                }
#endif
                for(;ii<rhi;++ii){cL[size_t(ii)]*=gL;cR[size_t(ii)]*=gR;}
            }
            std::fill(cL.begin()+coded,cL.begin()+k,0.f);std::fill(cR.begin()+coded,cR.begin()+k,0.f);
            const double sfc_frame_support=double(k)/double(K_LONG);
            for(int gj=0;gj<lx_sfc::SFC_BANDS;++gj)
                sfc_diffuse_exposure[gj]=(sfd.mask&(1u<<gj))?(sfc_guard_exposure[gj]+sfc_frame_support):0.0;

            // Same synthesis order as the former final pass, but coefficients
            // are consumed immediately instead of being reread cold later.
            switch(k){
                case 1024: invL.d2048.inverse_overlap_add_stereo(invR.d2048,cL.data(),cR.data(),pos,vh.total_samples_per_channel,decoded.pcm.data()); break;
                case  768: invL.d1024.inverse_overlap_add_stereo(invR.d1024,cL.data(),cR.data(),pos,vh.total_samples_per_channel,decoded.pcm.data()); break;
                case  512: invL.d512 .inverse_overlap_add_stereo(invR.d512 ,cL.data(),cR.data(),pos,vh.total_samples_per_channel,decoded.pcm.data()); break;
                case  384: invL.d256 .inverse_overlap_add_stereo(invR.d256 ,cL.data(),cR.data(),pos,vh.total_samples_per_channel,decoded.pcm.data()); break;
                default: throw std::runtime_error("unsupported Fold-MDCT block size");
            }
            pos+=size_t(k);
        }
        if(!sfcr.done())throw std::runtime_error("SFC stream length mismatch");
        if(!acr.done())throw std::runtime_error("anti-collapse stream length mismatch");
        if(side_si!=side_syms.size())throw std::runtime_error("SFC side rANS symbol count mismatch");
        if(pmr!=emr||!dc.done()||!dm.done())throw std::runtime_error("symbol stream length mismatch");
        {
            constexpr uint32_t MEM_NUM=5;
            auto hClassEff=hClassTuple;
            auto hMagEff=hMag;
            for(int c=0;c<CLASS_TUPLE_CTX;++c)for(int a=0;a<9;++a){
                class_hist_mem[c][a]=(class_hist_mem[c][a]*MEM_NUM+4u)>>3;
                class_hist_mem[c][a]+=hClassTuple[c][a];
                hClassEff[c][a]=uint32_t(std::min<uint64_t>(class_hist_mem[c][a],0xffffffffu));
            }
            for(int c=0;c<MAG_CTX;++c)for(int a=0;a<MAG_ALPHA;++a){
                mag_hist_mem[c][a]=(mag_hist_mem[c][a]*MEM_NUM+4u)>>3;
                mag_hist_mem[c][a]+=hMag[c][a];
                hMagEff[c][a]=uint32_t(std::min<uint64_t>(mag_hist_mem[c][a],0xffffffffu));
            }
            // Match encoder sign-symmetric regularization exactly.
            for(int c=0;c<CLASS_TUPLE_CTX;++c){
                auto &h=hClassEff[c];
                const uint32_t s10=h[1]+h[2]; const uint32_t a10=(s10+1u)>>1; h[1]=a10; h[2]=s10-a10;
                const uint32_t s01=h[3]+h[6]; const uint32_t a01=(s01+1u)>>1; h[3]=a01; h[6]=s01-a01;
                const uint32_t s11=h[4]+h[5]+h[7]+h[8]; const uint32_t q=s11>>2, r=s11&3u;
                h[4]=q+uint32_t(r>0); h[5]=q+uint32_t(r>1); h[7]=q+uint32_t(r>2); h[8]=q;
            }
            auto hc=hClassEff;
            constexpr uint32_t CLASS_BACKOFF=8;
            for(int parent=0; parent<CLASS_FREQ_CTX*3; ++parent){
                std::array<uint32_t,9> pooled{};
                for(int t=0;t<CLASS_TEMP_STATES;++t)for(int a=0;a<9;++a)pooled[a]+=hClassEff[parent*CLASS_TEMP_STATES+t][a];
                for(int t=0;t<CLASS_TEMP_STATES;++t){const int c=parent*CLASS_TEMP_STATES+t;for(int a=0;a<9;++a)hc[c][a]=CLASS_BACKOFF*hClassEff[c][a]+pooled[a];}
            }
            auto hm=hMagEff;
            constexpr uint32_t MAG_BACKOFF=16;
            constexpr int NST=MAG_PRED_STATES*MAG_LOCAL_STATES;
            for(int parent=0; parent<MAG_BASE_CTX; ++parent){
                std::array<uint32_t,MAG_ALPHA> pooled{};
                for(int t=0;t<NST;++t)for(int a=0;a<MAG_ALPHA;++a)pooled[a]+=hMagEff[parent*NST+t][a];
                for(int t=0;t<NST;++t){const int c=parent*NST+t;for(int a=0;a<MAG_ALPHA;++a)hm[c][a]=MAG_BACKOFF*hMagEff[c][a]+pooled[a];}
            }
            classTupleModel.rebuild(hc); magModel.rebuild(hm);
        }
        {
            std::array<uint32_t,256> hs=hScale;
            constexpr uint32_t SCALE_MEM_NUM=5;
            for(int a=0;a<256;++a){
                scale_hist_mem[a]=(scale_hist_mem[a]*SCALE_MEM_NUM+4u)>>3;
                scale_hist_mem[a]+=hScale[a];
                hs[a]=uint32_t(std::min<uint64_t>(scale_hist_mem[a],0xffffffffu));
            }
            scaleModel.build_decoder_only(hs);
        }
        {
            std::array<uint32_t,256> hs=hLrScale;
            constexpr uint32_t LR_SCALE_MEM_NUM=5;
            for(int a=0;a<256;++a){
                lr_scale_hist_mem[a]=(lr_scale_hist_mem[a]*LR_SCALE_MEM_NUM+4u)>>3;
                lr_scale_hist_mem[a]+=hLrScale[a];
                hs[a]=uint32_t(std::min<uint64_t>(lr_scale_hist_mem[a],0xffffffffu));
            }
            lrScaleModel.build_decoder_only(hs);
        }
        globalFrame+=pk.h.frame_count;
    }

    lx_pitch::Schedule pitch_schedule;
    if(vh.reserved[0]){
        if(vh.reserved[0]>(1u<<24)) throw std::runtime_error("pitch payload too large");
        std::vector<uint8_t> pitch_payload(size_t(vh.reserved[0]));
        if(!read_exact(f,pitch_payload.data(),pitch_payload.size())) throw std::runtime_error("truncated pitch payload");
        pitch_schedule=lx_pitch::decode_payload(pitch_payload);
    }

    // Disabled R203 HF postfilter research path removed from production decoder.

    // v0.24 reconstruction bias fused into per-frame decode.

    if(vh.reserved[0]) lx_pitch::postfilter(decoded.pcm,size_t(vh.total_samples_per_channel),pitch_schedule);
    return 0;
}
} // namespace lxvoxel

