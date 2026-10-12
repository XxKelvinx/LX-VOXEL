#pragma once
#include <cstdint>
#include <filesystem>
#include <istream>
#include <vector>

#include "metadata/metadata.hpp"

namespace lxvoxel {
struct DecodedAudio {
    uint32_t sample_rate = 0;
    uint16_t channels = 0;
    uint64_t samples_per_channel = 0;
    std::vector<float> pcm;
    metadata::Metadata metadata;
};
int decode_file(const std::filesystem::path& input_path, DecodedAudio& decoded);
int decode_stream(std::istream& input, DecodedAudio& decoded);
const char* decoder_cpu_path() noexcept;
}

