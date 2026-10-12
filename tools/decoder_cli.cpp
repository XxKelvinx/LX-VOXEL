#include <fstream>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include "decoder/decoder_core.hpp"

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--cpu-info") {
        std::cout << lxvoxel::decoder_cpu_path() << "\n";
        return 0;
    }
    if (argc < 3) {
        std::cerr << "usage: lxvoxel-dec input.vox output.f32\n";
        return 2;
    }
    try {
        lxvoxel::DecodedAudio decoded;
        lxvoxel::decode_file(
            std::filesystem::path(std::u8string(reinterpret_cast<const char8_t*>(argv[1]))),
            decoded);
        std::ofstream out(argv[2], std::ios::binary);
        if (!out) throw std::runtime_error("cannot open PCM output");
        out.write(reinterpret_cast<const char*>(decoded.pcm.data()),
                  std::streamsize(decoded.pcm.size() * sizeof(float)));
        if (!out) throw std::runtime_error("short PCM output write");
        std::cout << "decoded samples=" << decoded.samples_per_channel
                  << " channels=" << decoded.channels
                  << " pcm_f32_bytes=" << (decoded.pcm.size() * sizeof(float)) << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    } catch (...) {
        std::cerr << "error: unknown fatal exception\n";
        return 3;
    }
}

