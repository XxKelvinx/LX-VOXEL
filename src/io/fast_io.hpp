#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace lxio {

#if defined(__linux__)
// Zero-copy read-only view of a float32 input file.  mmap removes the explicit
// file->heap copy; pages are faulted by the kernel as the deinterleaver consumes
// them.  This is appropriate for benchmark/raw-input paths and large PCM files.
class MappedF32File {
public:
    explicit MappedF32File(const char* path) {
        fd_ = ::open(path, O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("open(mmap) failed");
        struct stat st{};
        if (::fstat(fd_, &st) != 0) { close_fd(); throw std::runtime_error("fstat failed"); }
        if (st.st_size < 0 || static_cast<uintmax_t>(st.st_size) > SIZE_MAX) { close_fd(); throw std::runtime_error("invalid mmap input size"); }
        bytes_ = static_cast<std::size_t>(st.st_size);
        if ((bytes_ % sizeof(float)) != 0) { close_fd(); throw std::runtime_error("input is not float32-aligned"); }
        if (bytes_) {
            void* p = ::mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
            if (p == MAP_FAILED) { close_fd(); throw std::runtime_error("mmap failed"); }
            data_ = static_cast<const float*>(p);
            ::madvise(const_cast<float*>(data_), bytes_, MADV_SEQUENTIAL);
        }
    }
    MappedF32File(const MappedF32File&) = delete;
    MappedF32File& operator=(const MappedF32File&) = delete;
    ~MappedF32File() {
        if (data_) ::munmap(const_cast<float*>(data_), bytes_);
        close_fd();
    }
    const float* data() const { return data_; }
    std::size_t size() const { return bytes_ / sizeof(float); }
private:
    void close_fd() { if (fd_ >= 0) { ::close(fd_); fd_ = -1; } }
    int fd_ = -1;
    std::size_t bytes_ = 0;
    const float* data_ = nullptr;
};
#endif

// Large sequential stdio buffer.  The codec emits many small entropy packets,
// but the OS should see large writes.  One buffer also avoids per-packet flushes.
class BufferedFileWriter {
public:
    explicit BufferedFileWriter(const char* path, std::size_t buffer_bytes = 1u << 20) {
        f_ = std::fopen(path, "wb");
        if (!f_) throw std::runtime_error("fopen(write) failed: " + std::string(std::strerror(errno)));
        buffer_.resize(buffer_bytes);
        std::setvbuf(f_, reinterpret_cast<char*>(buffer_.data()), _IOFBF, buffer_.size());
    }

    BufferedFileWriter(const BufferedFileWriter&) = delete;
    BufferedFileWriter& operator=(const BufferedFileWriter&) = delete;

    ~BufferedFileWriter() {
        if (f_) std::fclose(f_);
    }

    void write(const void* src, std::size_t bytes) {
        if (!bytes) return;
        if (std::fwrite(src, 1, bytes, f_) != bytes)
            throw std::runtime_error("fwrite failed");
    }

    template<class T>
    void write_pod(const T& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        write(&v, sizeof(v));
    }

    void flush() {
        if (std::fflush(f_) != 0) throw std::runtime_error("fflush failed");
    }

private:
    std::FILE* f_ = nullptr;
    std::vector<std::uint8_t> buffer_;
};

// Read an interleaved float32 stereo file in one sequential transfer.
// posix_fadvise is only a hint; correctness never depends on it.
inline std::vector<float> read_f32_file(const char* path) {
#if defined(__linux__)
    const int fd = ::open(path, O_RDONLY);
    if (fd < 0) throw std::runtime_error("open(read) failed");
    struct stat st{};
    if (::fstat(fd, &st) != 0) { ::close(fd); throw std::runtime_error("fstat failed"); }
    if (st.st_size < 0 || static_cast<uintmax_t>(st.st_size) > SIZE_MAX || (st.st_size % sizeof(float)) != 0) { ::close(fd); throw std::runtime_error("invalid float32 input size"); }
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);

    std::vector<float> out(static_cast<std::size_t>(st.st_size) / sizeof(float));
    std::byte* dst = reinterpret_cast<std::byte*>(out.data());
    std::size_t left = static_cast<std::size_t>(st.st_size);
    while (left) {
        const ssize_t n = ::read(fd, dst, left);
        if (n < 0) {
            if (errno == EINTR) continue;
            ::close(fd);
            throw std::runtime_error("read failed");
        }
        if (n == 0) break;
        dst += n;
        left -= static_cast<std::size_t>(n);
    }
    ::close(fd);
    if (left) throw std::runtime_error("short read");
    return out;
#else
    std::FILE* f = std::fopen(path, "rb");
    if (!f) throw std::runtime_error("fopen(read) failed");
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::rewind(f);
    if (bytes < 0 || (bytes % long(sizeof(float))) != 0) { std::fclose(f); throw std::runtime_error("invalid float32 file"); }
    std::vector<float> out(static_cast<std::size_t>(bytes) / sizeof(float));
    if (std::fread(out.data(), 1, static_cast<std::size_t>(bytes), f) != static_cast<std::size_t>(bytes)) {
        std::fclose(f); throw std::runtime_error("fread failed");
    }
    std::fclose(f);
    return out;
#endif
}

// Deinterleave once, outside the transform hot loop.  This gives each channel a
// contiguous buffer so MDCT frames can be read directly without a per-frame copy.
inline void deinterleave_stereo(const float* interleaved, std::size_t float_count,
                                std::vector<float>& left,
                                std::vector<float>& right,
                                std::size_t pad_samples = 0) {
    if (float_count & 1u) throw std::runtime_error("stereo input has odd float count");
    const std::size_t n = float_count / 2;
    left.resize(n + pad_samples);
    right.resize(n + pad_samples);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
    for (std::size_t i = 0; i < n; ++i) {
        left[i] = interleaved[2 * i];
        right[i] = interleaved[2 * i + 1];
    }
    if (pad_samples) {
        std::memset(left.data() + n, 0, pad_samples * sizeof(float));
        std::memset(right.data() + n, 0, pad_samples * sizeof(float));
    }
}

inline void deinterleave_stereo(const std::vector<float>& interleaved,
                                std::vector<float>& left,
                                std::vector<float>& right,
                                std::size_t pad_samples = 0) {
    deinterleave_stereo(interleaved.data(), interleaved.size(), left, right, pad_samples);
}

} // namespace lxio
