#include "lxvoxel/lxvoxel.h"
#include "decoder/decoder_core.hpp"
#include "lxvoxel/version.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>

struct lxvoxel_decoder {
    lxvoxel::DecodedAudio audio;
    uint64_t position = 0;
    std::string error;
};

extern "C" {

lxvoxel_decoder* lxvoxel_decoder_open_file(const char* path) {
    if (!path) return nullptr;
    std::unique_ptr<lxvoxel_decoder> d(new lxvoxel_decoder());
    try {
        lxvoxel::decode_file(
            std::filesystem::path(std::u8string(reinterpret_cast<const char8_t*>(path))),
            d->audio);
        return d.release();
    } catch (const std::exception& e) {
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

#if defined(_WIN32)
lxvoxel_decoder* lxvoxel_decoder_open_file_w(const wchar_t* path) {
    if (!path) return nullptr;
    std::unique_ptr<lxvoxel_decoder> d(new lxvoxel_decoder());
    try {
        lxvoxel::decode_file(std::filesystem::path(path), d->audio);
        return d.release();
    } catch (const std::exception&) {
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}
#endif

void lxvoxel_decoder_close(lxvoxel_decoder* decoder) { delete decoder; }

int lxvoxel_decoder_get_info(const lxvoxel_decoder* d, lxvoxel_stream_info* info) {
    if (!d || !info) return -1;
    info->sample_rate = d->audio.sample_rate;
    info->channels = d->audio.channels;
    info->reserved = 0;
    info->samples_per_channel = d->audio.samples_per_channel;
    return 0;
}

const char* lxvoxel_decoder_get_metadata_text(
        const lxvoxel_decoder* d, lxvoxel_metadata_field field) {
    if (!d) return nullptr;
    const auto* value = d->audio.metadata.find(
        static_cast<lxvoxel::metadata::Field>(field));
    return value ? value->c_str() : nullptr;
}

int lxvoxel_decoder_get_cover_art(const lxvoxel_decoder* d, lxvoxel_cover_art* cover) {
    if (!d || !cover || !d->audio.metadata.has_cover) return -1;
    const auto& source = d->audio.metadata.cover;
    cover->data = source.data.data();
    cover->size = source.data.size();
    cover->mime_type = source.mime_type.c_str();
    cover->description = source.description.c_str();
    cover->picture_type = source.picture_type;
    cover->width = source.width;
    cover->height = source.height;
    return 0;
}

uint32_t lxvoxel_decoder_get_average_bitrate(const lxvoxel_decoder* d) {
    return d ? d->audio.metadata.average_bitrate_bps : 0;
}

size_t lxvoxel_decoder_read_f32(lxvoxel_decoder* d, float* out, size_t frames) {
    if (!d || !out || frames == 0 || d->audio.channels == 0) return 0;
    const uint64_t remaining = d->audio.samples_per_channel > d->position
        ? d->audio.samples_per_channel - d->position : 0;
    const size_t n = size_t(std::min<uint64_t>(remaining, frames));
    const size_t ch = d->audio.channels;
    const size_t offset = size_t(d->position) * ch;
    std::memcpy(out, d->audio.pcm.data() + offset, n * ch * sizeof(float));
    d->position += n;
    return n;
}

int lxvoxel_decoder_seek(lxvoxel_decoder* d, uint64_t frame_index) {
    if (!d || frame_index > d->audio.samples_per_channel) return -1;
    d->position = frame_index;
    return 0;
}

const char* lxvoxel_decoder_last_error(const lxvoxel_decoder* d) {
    return d ? d->error.c_str() : "decoder is null";
}

const char* lxvoxel_version_string(void) { return LXVOXEL_VERSION_STRING; }

} // extern "C"

