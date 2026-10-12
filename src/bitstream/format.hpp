#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "metadata/metadata.hpp"

namespace vox_beta {
constexpr uint32_t MAGIC = 0x00584f56u; // "VOX\0" little-endian
constexpr uint16_t MAJOR = 0;
constexpr uint16_t MINOR = 36;
constexpr uint32_t SAMPLE_RATE = 48000;
constexpr uint16_t CHANNELS = 2;
constexpr uint16_t ERB_BANDS = 48;
constexpr uint16_t SUPERFRAME_FRAMES = 32;

#pragma pack(push,1)
struct Header {
    uint32_t magic = MAGIC;
    uint16_t major = MAJOR;
    uint16_t minor = MINOR;
    uint32_t header_bytes = sizeof(Header);
    uint32_t sample_rate = SAMPLE_RATE;
    uint16_t channels = CHANNELS;
    // v0.31: bits 8..15 are the global future-extension byte. Current
    // encoders MUST write zero and v0.31 decoders reject nonzero values. A
    // future bitstream revision may assign semantics explicitly.
    uint16_t flags = 1; // bit0 = beta/unstable, bits8..15 reserved
    uint64_t total_samples_per_channel = 0;
    uint32_t frame_count = 0;
    uint16_t erb_bands = ERB_BANDS;
    uint16_t superframe_frames = SUPERFRAME_FRAMES;
    float base_delta = 0.0f;
    uint32_t reserved[4]{};
};
struct SuperframeHeader {
    uint16_t frame_count = 0;
    uint16_t reserved = 0;
    uint32_t scale_symbols = 0;
    uint32_t lr_symbols = 0;
    uint32_t class_symbols = 0;
    uint32_t mag_symbols = 0;
    uint32_t scale_rans_bytes = 0;
    uint32_t lr_rans_bytes = 0;
    uint32_t class_rans_bytes = 0;
    uint32_t mag_rans_bytes = 0;
    uint32_t scale_raw_bytes = 0;
    uint32_t lr_raw_bytes = 0;
    uint32_t mag_raw_bytes = 0;
};
#pragma pack(pop)

struct SuperframePacket {
    SuperframeHeader h{};
    std::vector<uint8_t> side;      // 2 bytes/frame: control + packed IS angles
    std::vector<uint8_t> block_hi;  // v0.11+: one additional block-size bit/frame, packed LSB-first
    // SFC v0.5 payload is bit-packed across the superframe. control[5:3]
    // carries k, the number of active top-8 SFC bands. Each active band carries
    // 7-bit gain + shared stereo AR(2). v0.12 adds 3-bit intraband spectral tilt per active band. v0.13 adds the zero-bit decoder-synchronous SFC Diffuse Guard behind that tilt. v0.14 adds one mode bit per active SFC band: 0 = existing SFC continuation, 1 = decoder-synchronous folding from the immediately preceding decoded ERB band. v0.15 makes this a true pSFC/fSFC syntax split: gain is always coded, then the mode bit; pSFC (mode=0) additionally codes AR(2)+tilt, while fSFC (mode=1) omits those 11 unused bits. The mode payload is followed by one IFC-present bit. If IFC
    // is present, 2-bit Gabor mode + 4-bit position + 3-bit amplitude follow.
    // SuperframeHeader::reserved stores the SFC payload byte count for v0.5.
    std::vector<uint8_t> sfc;
    std::vector<uint8_t> scale_rans, lr_rans, class_rans, mag_rans;
    std::vector<uint8_t> scale_raw, lr_raw, mag_raw;
};

static inline void put_uleb128(std::vector<uint8_t>& out, uint32_t v){
    do { uint8_t b=uint8_t(v&0x7fu); v>>=7; if(v)b|=0x80u; out.push_back(b); } while(v);
}
static inline bool get_uleb128(const uint8_t*& p,const uint8_t* end,uint32_t& v){
    uint32_t x=0; unsigned sh=0;
    while(p<end && sh<=28){uint8_t b=*p++;if(sh==28 && (b&0x70u))return false;x|=uint32_t(b&0x7f)<<sh;if(!(b&0x80)){v=x;return true;}sh+=7;}
    return false;
}


class SfcBitWriter {
    std::vector<uint8_t>& out_;
    uint64_t acc_ = 0;
    unsigned bits_ = 0;
public:
    explicit SfcBitWriter(std::vector<uint8_t>& out):out_(out){out_.clear();}
    void put(uint64_t v,unsigned n){
        if(n>56) throw std::runtime_error("SFC payload too wide");
        if(n==0) return;
        const uint64_t mask=(uint64_t(1)<<n)-1u;
        acc_|=(uint64_t(v)&mask)<<bits_;
        bits_+=n;
        while(bits_>=8){out_.push_back(uint8_t(acc_));acc_>>=8;bits_-=8;}
    }
    void finish(){if(bits_){out_.push_back(uint8_t(acc_));acc_=0;bits_=0;}}
};

class SfcBitReader {
    const uint8_t* p_ = nullptr;
    const uint8_t* end_ = nullptr;
    uint64_t acc_ = 0;
    unsigned bits_ = 0;
public:
    SfcBitReader(const uint8_t* p,const uint8_t* end):p_(p),end_(end){}
    bool get(unsigned n,uint64_t& v){
        if(n>56) return false;
        if(n==0){v=0;return true;}
        while(bits_<n){if(p_>=end_)return false;acc_|=uint64_t(*p_++)<<bits_;bits_+=8;}
        const uint64_t mask=(uint64_t(1)<<n)-1u;
        v=uint64_t(acc_&mask);acc_>>=n;bits_-=n;return true;
    }
    bool done() const { return p_==end_; }
};

class Writer {
    Header hdr_{};
    std::vector<SuperframePacket> packets_;
    lxvoxel::metadata::Metadata metadata_;
    std::vector<uint8_t> pitch_payload_;
public:
    explicit Writer(uint64_t ns,float base_delta,uint32_t frames){hdr_.total_samples_per_channel=ns;hdr_.base_delta=base_delta;hdr_.frame_count=frames;}
    void set_base_delta(float v){hdr_.base_delta=v;}
    void set_metadata(lxvoxel::metadata::Metadata metadata){metadata_=std::move(metadata);}
    void set_pitch_payload(std::vector<uint8_t> payload){pitch_payload_=std::move(payload);}
    void add(SuperframePacket&& p){packets_.push_back(std::move(p));}
    uint64_t write(const std::filesystem::path& path) const {
        // Reuse one OS-output buffer per writing thread. Avoid a per-file
        // allocation and collapse small superframe fwrite() calls into
        // substantially fewer large buffered writes.
        constexpr size_t OUTPUT_BUFFER_BYTES = 256u * 1024u;
        static thread_local std::unique_ptr<char[]> output_buffer =
            std::make_unique_for_overwrite<char[]>(OUTPUT_BUFFER_BYTES);
#if defined(_WIN32)
        FILE* f=nullptr;
        _wfopen_s(&f,path.c_str(),L"wb");
#else
        FILE* f=std::fopen(path.c_str(),"wb");
#endif
        if(!f)throw std::runtime_error("cannot open .vox for write");
        if(std::setvbuf(f,output_buffer.get(),_IOFBF,OUTPUT_BUFFER_BYTES)!=0){
            std::fclose(f);
            throw std::runtime_error("cannot buffer .vox output");
        }
        uint64_t bytes=0; auto wr=[&](const void* p,size_t n){if(n&&std::fwrite(p,1,n,f)!=n){std::fclose(f);throw std::runtime_error(".vox write failed");}bytes+=n;};
        Header hh=hdr_; if(hh.frame_count==0){uint32_t fc=0;for(const auto& p:packets_)fc+=p.h.frame_count;hh.frame_count=fc;}
        hh.reserved[0]=uint32_t(pitch_payload_.size());
        wr(&hh,sizeof(hh));
        for(const auto& p:packets_){
            // v0.31 compact superframe framing. Global frame count and the
            // normative 32-frame superframe size make per-superframe frame_count
            // derivable. Symbol counts are also derivable and are not serialized.
            std::vector<uint8_t> ch; ch.reserve(16);
            const uint64_t payload64=uint64_t(p.side.size())+uint64_t(p.block_hi.size())+uint64_t(p.sfc.size())
                +uint64_t(p.scale_rans.size())+uint64_t(p.lr_rans.size())+uint64_t(p.class_rans.size())+uint64_t(p.mag_rans.size())
                +uint64_t(p.scale_raw.size())+uint64_t(p.lr_raw.size())+uint64_t(p.mag_raw.size());
            constexpr uint32_t PAYLOAD_BITS=17u;
            constexpr uint32_t PAYLOAD_LIMIT=uint32_t(1u)<<PAYLOAD_BITS;
            if(payload64>=PAYLOAD_LIMIT){std::fclose(f);throw std::runtime_error("VOX superframe payload exceeds v0.31 17-bit limit");}
            constexpr uint32_t sf_future_bits=0u; // 3 reserved bits
            SfcBitWriter hb(ch);
            hb.put(uint32_t(payload64),PAYLOAD_BITS);
            hb.put(sf_future_bits,3u);
            auto put_h_uleb=[&](uint32_t v){
                do{uint8_t b=uint8_t(v&0x7fu);v>>=7;if(v)b|=0x80u;hb.put(b,8u);}while(v);
            };
            put_h_uleb(uint32_t(p.sfc.size()));
            put_h_uleb(uint32_t(p.scale_rans.size()));
            put_h_uleb(uint32_t(p.lr_rans.size()));
            put_h_uleb(uint32_t(p.class_rans.size()));
            put_h_uleb(uint32_t(p.mag_rans.size()));
            put_h_uleb(uint32_t(p.scale_raw.size()));
            put_h_uleb(uint32_t(p.lr_raw.size()));
            hb.finish();
            wr(ch.data(),ch.size());
            wr(p.side.data(),p.side.size());
            wr(p.block_hi.data(),p.block_hi.size());
            wr(p.sfc.data(),p.sfc.size());
            wr(p.scale_rans.data(),p.scale_rans.size()); wr(p.lr_rans.data(),p.lr_rans.size()); wr(p.class_rans.data(),p.class_rans.size()); wr(p.mag_rans.data(),p.mag_rans.size());
            wr(p.scale_raw.data(),p.scale_raw.size()); wr(p.lr_raw.data(),p.lr_raw.size()); wr(p.mag_raw.data(),p.mag_raw.size());
        }
        wr(pitch_payload_.data(),pitch_payload_.size());
        const uint64_t audio_bytes=bytes;
        try { bytes+=lxvoxel::metadata::write_trailer(f,audio_bytes,metadata_); }
        catch(...){std::fclose(f);throw;}
        if(std::fclose(f)!=0) throw std::runtime_error(".vox final flush/close failed");
        // Rate control is defined on the encoded audio bitstream, not on cover
        // art or tags. Returning audio_bytes keeps R165 Fast-FBR calibration exact.
        return audio_bytes;
    }
};

} // namespace vox_beta
