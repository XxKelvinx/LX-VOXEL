#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace lxvoxel::metadata {

enum class Field : uint16_t {
    title = 1,
    artist = 2,
    album = 3,
    album_artist = 4,
    genre = 5,
    date = 6,
    track_number = 7,
    track_total = 8,
    disc_number = 9,
    disc_total = 10,
    composer = 11,
    comment = 12,
    copyright = 13,
    encoder = 14,
    cover_art = 0x100
};

struct TextEntry {
    Field field{};
    std::string value;
};

struct CoverArt {
    uint32_t picture_type = 3;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 0;
    uint32_t colors = 0;
    std::string mime_type;
    std::string description;
    std::vector<uint8_t> data;
};

struct Metadata {
    std::vector<TextEntry> text;
    CoverArt cover;
    bool has_cover = false;
    uint64_t audio_bytes = 0;
    uint32_t average_bitrate_bps = 0;

    const std::string* find(Field field) const noexcept;
    void set(Field field, std::string value);
};

// Writes a metadata block and its fixed EOF locator. The locator is written
// even for empty metadata so the exact encoded-audio size remains available.
uint64_t write_trailer(FILE* file, uint64_t audio_bytes, const Metadata& metadata);

// Invalid or absent optional metadata never makes otherwise valid audio fail.
Metadata read_vox(const std::filesystem::path& path,
                  uint64_t samples_per_channel,
                  uint32_t sample_rate) noexcept;

// Imports native FLAC VORBIS_COMMENT and PICTURE blocks without transcoding or
// interpreting image bytes.
Metadata read_flac(const std::filesystem::path& path);

} // namespace lxvoxel::metadata
