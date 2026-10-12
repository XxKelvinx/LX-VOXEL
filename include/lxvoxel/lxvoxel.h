#pragma once
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(LXVOXEL_SHARED)
#  if defined(LXVOXEL_BUILDING_DLL)
#    define LXVOXEL_API __declspec(dllexport)
#  else
#    define LXVOXEL_API __declspec(dllimport)
#  endif
#else
#  define LXVOXEL_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lxvoxel_decoder lxvoxel_decoder;

typedef struct lxvoxel_stream_info {
    uint32_t sample_rate;
    uint16_t channels;
    uint16_t reserved;
    uint64_t samples_per_channel;
} lxvoxel_stream_info;

typedef enum lxvoxel_metadata_field {
    LXVOXEL_METADATA_TITLE = 1,
    LXVOXEL_METADATA_ARTIST = 2,
    LXVOXEL_METADATA_ALBUM = 3,
    LXVOXEL_METADATA_ALBUM_ARTIST = 4,
    LXVOXEL_METADATA_GENRE = 5,
    LXVOXEL_METADATA_DATE = 6,
    LXVOXEL_METADATA_TRACK_NUMBER = 7,
    LXVOXEL_METADATA_TRACK_TOTAL = 8,
    LXVOXEL_METADATA_DISC_NUMBER = 9,
    LXVOXEL_METADATA_DISC_TOTAL = 10,
    LXVOXEL_METADATA_COMPOSER = 11,
    LXVOXEL_METADATA_COMMENT = 12,
    LXVOXEL_METADATA_COPYRIGHT = 13,
    LXVOXEL_METADATA_ENCODER = 14
} lxvoxel_metadata_field;

typedef struct lxvoxel_cover_art {
    const uint8_t* data;
    size_t size;
    const char* mime_type;
    const char* description;
    uint32_t picture_type;
    uint32_t width;
    uint32_t height;
} lxvoxel_cover_art;

/* Opens and decodes a .vox file into an in-process pull decoder.
   v0.1.0 compatibility API: decoding is currently eager; callers can pull PCM
   in small chunks and never create an intermediate raw PCM file. */
LXVOXEL_API lxvoxel_decoder* lxvoxel_decoder_open_file(const char* path);
#if defined(_WIN32)
/* Wide-path variant for Windows hosts and packaged app services. */
LXVOXEL_API lxvoxel_decoder* lxvoxel_decoder_open_file_w(const wchar_t* path);
#endif
LXVOXEL_API void lxvoxel_decoder_close(lxvoxel_decoder* decoder);
LXVOXEL_API int lxvoxel_decoder_get_info(const lxvoxel_decoder* decoder, lxvoxel_stream_info* info);
/* Returned text and cover pointers remain valid until decoder_close. */
LXVOXEL_API const char* lxvoxel_decoder_get_metadata_text(
    const lxvoxel_decoder* decoder, lxvoxel_metadata_field field);
LXVOXEL_API int lxvoxel_decoder_get_cover_art(
    const lxvoxel_decoder* decoder, lxvoxel_cover_art* cover);
LXVOXEL_API uint32_t lxvoxel_decoder_get_average_bitrate(const lxvoxel_decoder* decoder);
LXVOXEL_API size_t lxvoxel_decoder_read_f32(lxvoxel_decoder* decoder, float* interleaved_pcm, size_t frames);
LXVOXEL_API int lxvoxel_decoder_seek(lxvoxel_decoder* decoder, uint64_t frame_index);
LXVOXEL_API const char* lxvoxel_decoder_last_error(const lxvoxel_decoder* decoder);
LXVOXEL_API const char* lxvoxel_version_string(void);

#ifdef __cplusplus
}
#endif

