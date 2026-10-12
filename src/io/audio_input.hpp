#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace lxio {

struct AudioInput {
    uint32_t sample_rate = 48000;
    uint16_t channels = 2;
    std::vector<float> interleaved;
};

bool is_flac_path(const std::filesystem::path& path);

// FLAC is decoded and converted to the codec's fixed 48-kHz stereo float
// contract. Raw .f32 input retains the existing interleaved format.
AudioInput read_audio_input(const std::filesystem::path& path);

} // namespace lxio
