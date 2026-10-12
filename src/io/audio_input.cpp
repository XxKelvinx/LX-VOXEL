#include "io/audio_input.hpp"
#include "io/fast_io.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <objbase.h>
#endif

namespace lxio {
namespace {

std::string lower_extension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](unsigned char value) { return char(std::tolower(value)); });
    return extension;
}

#if defined(_WIN32)

void check_hresult(HRESULT result, const char* operation) {
    if (FAILED(result)) {
        char buffer[160]{};
        std::snprintf(buffer, sizeof(buffer), "%s failed (HRESULT 0x%08lx)",
            operation, static_cast<unsigned long>(result));
        throw std::runtime_error(buffer);
    }
}

template<class T>
class ComPtr {
public:
    ComPtr() = default;
    ComPtr(const ComPtr&) = delete;
    ComPtr& operator=(const ComPtr&) = delete;
    ~ComPtr() { reset(); }

    T* get() const noexcept { return value_; }
    T** put() noexcept { reset(); return &value_; }
    T* operator->() const noexcept { return value_; }
    explicit operator bool() const noexcept { return value_ != nullptr; }
    void reset() noexcept { if (value_) value_->Release(); value_ = nullptr; }

private:
    T* value_ = nullptr;
};

class MediaFoundationLifetime {
public:
    MediaFoundationLifetime() {
        const HRESULT com_result = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (SUCCEEDED(com_result)) uninitialize_com_ = true;
        else if (com_result != RPC_E_CHANGED_MODE) check_hresult(com_result, "CoInitializeEx");
        check_hresult(MFStartup(MF_VERSION, MFSTARTUP_FULL), "MFStartup");
        started_ = true;
    }
    ~MediaFoundationLifetime() {
        if (started_) MFShutdown();
        if (uninitialize_com_) CoUninitialize();
    }
    MediaFoundationLifetime(const MediaFoundationLifetime&) = delete;
    MediaFoundationLifetime& operator=(const MediaFoundationLifetime&) = delete;

private:
    bool started_ = false;
    bool uninitialize_com_ = false;
};

AudioInput read_flac_windows(const std::filesystem::path& path) {
    MediaFoundationLifetime lifetime;
    ComPtr<IMFAttributes> attributes;
    check_hresult(MFCreateAttributes(attributes.put(), 1), "MFCreateAttributes");
    check_hresult(attributes->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, FALSE),
        "select deterministic software transforms");

    ComPtr<IMFSourceReader> reader;
    check_hresult(MFCreateSourceReaderFromURL(path.c_str(), attributes.get(), reader.put()),
        "open FLAC with Media Foundation");
    check_hresult(reader->SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS, FALSE),
        "disable unused media streams");
    check_hresult(reader->SetStreamSelection(MF_SOURCE_READER_FIRST_AUDIO_STREAM, TRUE),
        "select FLAC audio stream");

    ComPtr<IMFMediaType> requested;
    check_hresult(MFCreateMediaType(requested.put()), "MFCreateMediaType");
    check_hresult(requested->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio), "set audio major type");
    check_hresult(requested->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_Float), "set float PCM subtype");
    check_hresult(requested->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, 48000), "set 48-kHz output");
    check_hresult(requested->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, 2), "set stereo output");
    check_hresult(requested->SetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, 32), "set float width");
    check_hresult(requested->SetUINT32(MF_MT_AUDIO_BLOCK_ALIGNMENT, 8), "set audio block alignment");
    check_hresult(requested->SetUINT32(MF_MT_AUDIO_AVG_BYTES_PER_SECOND, 48000U * 8U),
        "set float PCM byte rate");
    check_hresult(reader->SetCurrentMediaType(
        MF_SOURCE_READER_FIRST_AUDIO_STREAM, nullptr, requested.get()),
        "configure FLAC decoder/resampler");

    ComPtr<IMFMediaType> actual;
    check_hresult(reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_AUDIO_STREAM, actual.put()),
        "read decoded FLAC format");
    GUID subtype{};
    UINT32 sample_rate = 0, channels = 0, bits = 0;
    check_hresult(actual->GetGUID(MF_MT_SUBTYPE, &subtype), "read decoded subtype");
    check_hresult(actual->GetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, &sample_rate),
        "read decoded sample rate");
    check_hresult(actual->GetUINT32(MF_MT_AUDIO_NUM_CHANNELS, &channels),
        "read decoded channel count");
    check_hresult(actual->GetUINT32(MF_MT_AUDIO_BITS_PER_SAMPLE, &bits),
        "read decoded sample width");
    if (subtype != MFAudioFormat_Float || sample_rate != 48000 || channels != 2 || bits != 32)
        throw std::runtime_error("Media Foundation did not produce 48-kHz stereo float PCM");

    AudioInput output;
    for (;;) {
        DWORD stream_index = 0;
        DWORD flags = 0;
        LONGLONG timestamp = 0;
        ComPtr<IMFSample> sample;
        check_hresult(reader->ReadSample(MF_SOURCE_READER_FIRST_AUDIO_STREAM, 0,
            &stream_index, &flags, &timestamp, sample.put()), "decode FLAC sample");
        if ((flags & MF_SOURCE_READERF_ERROR) != 0)
            throw std::runtime_error("Media Foundation reported a FLAC decoding error");
        if ((flags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED) != 0)
            throw std::runtime_error("FLAC output format changed during decoding");
        if (sample) {
            ComPtr<IMFMediaBuffer> buffer;
            check_hresult(sample->ConvertToContiguousBuffer(buffer.put()),
                "coalesce decoded FLAC sample");
            BYTE* bytes = nullptr;
            DWORD maximum_bytes = 0;
            DWORD current_bytes = 0;
            check_hresult(buffer->Lock(&bytes, &maximum_bytes, &current_bytes),
                "lock decoded FLAC sample");
            try {
                if ((current_bytes % (2U * sizeof(float))) != 0)
                    throw std::runtime_error("decoded FLAC sample is not stereo-frame aligned");
                const size_t float_count = current_bytes / sizeof(float);
                if (float_count > output.interleaved.max_size() - output.interleaved.size())
                    throw std::runtime_error("decoded FLAC is too large for memory");
                const size_t old_size = output.interleaved.size();
                output.interleaved.resize(old_size + float_count);
                std::memcpy(output.interleaved.data() + old_size, bytes, current_bytes);
            } catch (...) {
                buffer->Unlock();
                throw;
            }
            check_hresult(buffer->Unlock(), "unlock decoded FLAC sample");
        }
        if ((flags & MF_SOURCE_READERF_ENDOFSTREAM) != 0) break;
    }
    if (output.interleaved.empty()) throw std::runtime_error("FLAC contains no decodable audio");
    return output;
}
#endif

} // namespace

bool is_flac_path(const std::filesystem::path& path) {
    return lower_extension(path) == ".flac";
}

AudioInput read_audio_input(const std::filesystem::path& path) {
    if (is_flac_path(path)) {
#if defined(_WIN32)
        return read_flac_windows(path);
#else
        throw std::runtime_error("FLAC input currently requires the Windows Media Foundation build");
#endif
    }
    AudioInput input;
    const auto path_u8 = path.u8string();
    const std::string utf8(path_u8.begin(), path_u8.end());
    input.interleaved = read_f32_file(utf8.c_str());
    if ((input.interleaved.size() & 1U) != 0)
        throw std::runtime_error("raw float input has an incomplete stereo frame");
    return input;
}

} // namespace lxio
