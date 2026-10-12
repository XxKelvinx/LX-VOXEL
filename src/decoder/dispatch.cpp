#include "decoder/decoder_core.hpp"
#include "dsp/cpu_dispatch.hpp"

#include <fstream>
#include <stdexcept>

namespace lxvoxel {

#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
int decode_stream_sse41(std::istream&, DecodedAudio&);
int decode_stream_avx2(std::istream&, DecodedAudio&);
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
int decode_stream_neon(std::istream&, DecodedAudio&);
#endif

using DecodeStreamFn = int(*)(std::istream&, DecodedAudio&);

static DecodeStreamFn select_decoder() noexcept {
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    return lx_cpu::best_path()==lx_cpu::Path::AVX2_FMA
        ? &decode_stream_avx2 : &decode_stream_sse41;
#else
    return &decode_stream_neon;
#endif
}

int decode_stream(std::istream& input, DecodedAudio& decoded){
    static const DecodeStreamFn fn=select_decoder();
    return fn(input,decoded);
}

int decode_file(const std::filesystem::path& input_path, DecodedAudio& decoded){
    std::ifstream input(input_path,std::ios::binary);
    if(!input) throw std::runtime_error("cannot open input");
    const int result=decode_stream(input,decoded);
    decoded.metadata=metadata::read_vox(input_path,decoded.samples_per_channel,decoded.sample_rate);
    return result;
}

const char* decoder_cpu_path() noexcept {
    static const lx_cpu::Path path=lx_cpu::best_path();
    return lx_cpu::name(path);
}

} // namespace lxvoxel
