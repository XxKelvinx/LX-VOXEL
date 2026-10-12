#include "metadata/metadata.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace lxvoxel::metadata {
namespace {

constexpr uint32_t block_magic = 0x444d584cU;  // "LXMD"
constexpr uint32_t footer_magic = 0x464d584cU; // "LXMF"
constexpr uint16_t metadata_version = 1;
constexpr uint64_t max_metadata_bytes = 128ULL * 1024ULL * 1024ULL;
constexpr uint32_t max_entries = 1024;

#pragma pack(push, 1)
struct BlockHeader {
    uint32_t magic = block_magic;
    uint16_t version = metadata_version;
    uint16_t header_bytes = sizeof(BlockHeader);
    uint32_t entry_count = 0;
    uint32_t reserved = 0;
};

struct EntryHeader {
    uint16_t type = 0;
    uint16_t flags = 0;
    uint32_t byte_count = 0;
};

struct CoverHeader {
    uint32_t picture_type = 3;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 0;
    uint32_t colors = 0;
    uint32_t mime_bytes = 0;
    uint32_t description_bytes = 0;
    uint32_t image_bytes = 0;
};

struct Footer {
    uint32_t magic = footer_magic;
    uint16_t version = metadata_version;
    uint16_t footer_bytes = sizeof(Footer);
    uint64_t metadata_offset = 0;
    uint64_t metadata_bytes = 0;
    uint32_t crc32 = 0;
    uint32_t reserved = 0;
};
#pragma pack(pop)

static_assert(sizeof(BlockHeader) == 16);
static_assert(sizeof(EntryHeader) == 8);
static_assert(sizeof(CoverHeader) == 32);
static_assert(sizeof(Footer) == 32);

uint32_t crc32(const uint8_t* data, size_t size) noexcept {
    uint32_t crc = 0xffffffffU;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (unsigned bit = 0; bit < 8; ++bit) {
            const uint32_t mask = uint32_t(0) - (crc & 1U);
            crc = (crc >> 1U) ^ (0xedb88320U & mask);
        }
    }
    return ~crc;
}

template<class T>
void append_pod(std::vector<uint8_t>& out, const T& value) {
    const auto* first = reinterpret_cast<const uint8_t*>(&value);
    out.insert(out.end(), first, first + sizeof(value));
}

template<class T>
bool read_pod(const uint8_t*& cursor, const uint8_t* end, T& value) noexcept {
    if (size_t(end - cursor) < sizeof(value)) return false;
    std::memcpy(&value, cursor, sizeof(value));
    cursor += sizeof(value);
    return true;
}

uint32_t read_le32(const uint8_t*& cursor, const uint8_t* end) {
    if (size_t(end - cursor) < 4) throw std::runtime_error("truncated FLAC metadata");
    const uint32_t value = uint32_t(cursor[0]) | (uint32_t(cursor[1]) << 8U)
        | (uint32_t(cursor[2]) << 16U) | (uint32_t(cursor[3]) << 24U);
    cursor += 4;
    return value;
}

uint32_t read_be32(const uint8_t*& cursor, const uint8_t* end) {
    if (size_t(end - cursor) < 4) throw std::runtime_error("truncated FLAC picture");
    const uint32_t value = (uint32_t(cursor[0]) << 24U) | (uint32_t(cursor[1]) << 16U)
        | (uint32_t(cursor[2]) << 8U) | uint32_t(cursor[3]);
    cursor += 4;
    return value;
}

std::string ascii_upper(std::string value) {
    for (char& ch : value) ch = char(std::toupper(static_cast<unsigned char>(ch)));
    return value;
}

void import_comment(Metadata& metadata, std::string entry) {
    const size_t equals = entry.find('=');
    if (equals == std::string::npos || equals == 0) return;
    const std::string key = ascii_upper(entry.substr(0, equals));
    std::string value = entry.substr(equals + 1);
    if (value.empty()) return;

    Field field{};
    bool known = true;
    if (key == "TITLE") field = Field::title;
    else if (key == "ARTIST") field = Field::artist;
    else if (key == "ALBUM") field = Field::album;
    else if (key == "ALBUMARTIST" || key == "ALBUM ARTIST") field = Field::album_artist;
    else if (key == "GENRE") field = Field::genre;
    else if (key == "DATE" || key == "YEAR") field = Field::date;
    else if (key == "TRACKNUMBER") field = Field::track_number;
    else if (key == "TRACKTOTAL" || key == "TOTALTRACKS") field = Field::track_total;
    else if (key == "DISCNUMBER") field = Field::disc_number;
    else if (key == "DISCTOTAL" || key == "TOTALDISCS") field = Field::disc_total;
    else if (key == "COMPOSER") field = Field::composer;
    else if (key == "COMMENT" || key == "DESCRIPTION") field = Field::comment;
    else if (key == "COPYRIGHT") field = Field::copyright;
    else if (key == "ENCODER") field = Field::encoder;
    else known = false;

    // Keep the first value. It is deterministic and matches the single-value
    // fields exposed by Windows MediaStreamSource.
    if (known && metadata.find(field) == nullptr) metadata.set(field, std::move(value));
}

CoverArt parse_flac_picture(const uint8_t* data, size_t size) {
    const uint8_t* cursor = data;
    const uint8_t* end = data + size;
    CoverArt cover;
    cover.picture_type = read_be32(cursor, end);
    const uint32_t mime_size = read_be32(cursor, end);
    if (mime_size > size_t(end - cursor)) throw std::runtime_error("invalid FLAC picture MIME length");
    cover.mime_type.assign(reinterpret_cast<const char*>(cursor), mime_size);
    cursor += mime_size;
    const uint32_t description_size = read_be32(cursor, end);
    if (description_size > size_t(end - cursor)) throw std::runtime_error("invalid FLAC picture description length");
    cover.description.assign(reinterpret_cast<const char*>(cursor), description_size);
    cursor += description_size;
    cover.width = read_be32(cursor, end);
    cover.height = read_be32(cursor, end);
    cover.depth = read_be32(cursor, end);
    cover.colors = read_be32(cursor, end);
    const uint32_t image_size = read_be32(cursor, end);
    if (image_size > size_t(end - cursor) || image_size != size_t(end - cursor))
        throw std::runtime_error("invalid FLAC picture data length");
    if (cover.mime_type == "-->") throw std::runtime_error("linked FLAC pictures are not embedded cover art");
    cover.data.assign(cursor, end);
    return cover;
}

uint32_t calculate_bitrate(uint64_t audio_bytes,
                           uint64_t samples_per_channel,
                           uint32_t sample_rate) noexcept {
    if (samples_per_channel == 0 || sample_rate == 0) return 0;
    const long double value = (static_cast<long double>(audio_bytes) * 8.0L
        * static_cast<long double>(sample_rate))
        / static_cast<long double>(samples_per_channel);
    if (!(value > 0.0L)) return 0;
    if (value >= static_cast<long double>(std::numeric_limits<uint32_t>::max()))
        return std::numeric_limits<uint32_t>::max();
    return uint32_t(std::llround(value));
}

} // namespace

const std::string* Metadata::find(Field field) const noexcept {
    const auto it = std::find_if(text.begin(), text.end(),
        [field](const TextEntry& entry) { return entry.field == field; });
    return it == text.end() ? nullptr : &it->value;
}

void Metadata::set(Field field, std::string value) {
    const auto it = std::find_if(text.begin(), text.end(),
        [field](const TextEntry& entry) { return entry.field == field; });
    if (it == text.end()) text.push_back({field, std::move(value)});
    else it->value = std::move(value);
}

uint64_t write_trailer(FILE* file, uint64_t audio_bytes, const Metadata& metadata) {
    if (!file) throw std::runtime_error("null metadata output file");
    if (metadata.text.size() > max_entries) throw std::runtime_error("too many metadata fields");

    BlockHeader block_header;
    block_header.entry_count = uint32_t(metadata.text.size()) + (metadata.has_cover ? 1U : 0U);
    std::vector<uint8_t> block;
    block.reserve(sizeof(BlockHeader) + metadata.text.size() * 32U
        + (metadata.has_cover ? metadata.cover.data.size() + sizeof(CoverHeader) + 64U : 0U));
    append_pod(block, block_header);

    for (const auto& text : metadata.text) {
        if (text.value.size() > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("metadata text is too large");
        EntryHeader entry;
        entry.type = uint16_t(text.field);
        entry.flags = 1; // UTF-8 text
        entry.byte_count = uint32_t(text.value.size());
        append_pod(block, entry);
        block.insert(block.end(), text.value.begin(), text.value.end());
    }

    if (metadata.has_cover) {
        const auto& cover = metadata.cover;
        if (cover.mime_type.size() > std::numeric_limits<uint32_t>::max()
            || cover.description.size() > std::numeric_limits<uint32_t>::max()
            || cover.data.size() > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("cover art is too large");
        const uint64_t payload_size = sizeof(CoverHeader) + cover.mime_type.size()
            + cover.description.size() + cover.data.size();
        if (payload_size > std::numeric_limits<uint32_t>::max())
            throw std::runtime_error("cover art payload is too large");
        EntryHeader entry;
        entry.type = uint16_t(Field::cover_art);
        entry.byte_count = uint32_t(payload_size);
        append_pod(block, entry);
        CoverHeader cover_header;
        cover_header.picture_type = cover.picture_type;
        cover_header.width = cover.width;
        cover_header.height = cover.height;
        cover_header.depth = cover.depth;
        cover_header.colors = cover.colors;
        cover_header.mime_bytes = uint32_t(cover.mime_type.size());
        cover_header.description_bytes = uint32_t(cover.description.size());
        cover_header.image_bytes = uint32_t(cover.data.size());
        append_pod(block, cover_header);
        block.insert(block.end(), cover.mime_type.begin(), cover.mime_type.end());
        block.insert(block.end(), cover.description.begin(), cover.description.end());
        block.insert(block.end(), cover.data.begin(), cover.data.end());
    }

    if (block.size() > max_metadata_bytes) throw std::runtime_error("metadata block exceeds 128 MiB");
    Footer footer;
    footer.metadata_offset = audio_bytes;
    footer.metadata_bytes = block.size();
    footer.crc32 = crc32(block.data(), block.size());
    if (std::fwrite(block.data(), 1, block.size(), file) != block.size()
        || std::fwrite(&footer, 1, sizeof(footer), file) != sizeof(footer))
        throw std::runtime_error("metadata trailer write failed");
    return uint64_t(block.size()) + sizeof(footer);
}

Metadata read_vox(const std::filesystem::path& path,
                  uint64_t samples_per_channel,
                  uint32_t sample_rate) noexcept {
    Metadata metadata;
    try {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input) return metadata;
        const std::streamoff end_position = input.tellg();
        if (end_position < 0) return metadata;
        const uint64_t file_bytes = uint64_t(end_position);
        metadata.audio_bytes = file_bytes;
        if (file_bytes < sizeof(Footer)) {
            metadata.average_bitrate_bps = calculate_bitrate(file_bytes, samples_per_channel, sample_rate);
            return metadata;
        }

        Footer footer{};
        input.seekg(std::streamoff(file_bytes - sizeof(Footer)), std::ios::beg);
        input.read(reinterpret_cast<char*>(&footer), sizeof(footer));
        if (!input || footer.magic != footer_magic || footer.version != metadata_version
            || footer.footer_bytes != sizeof(Footer) || footer.metadata_bytes < sizeof(BlockHeader)
            || footer.metadata_bytes > max_metadata_bytes
            || footer.metadata_offset > file_bytes - sizeof(Footer)
            || footer.metadata_bytes != file_bytes - sizeof(Footer) - footer.metadata_offset) {
            metadata.average_bitrate_bps = calculate_bitrate(file_bytes, samples_per_channel, sample_rate);
            return metadata;
        }

        std::vector<uint8_t> block(size_t(footer.metadata_bytes));
        input.seekg(std::streamoff(footer.metadata_offset), std::ios::beg);
        input.read(reinterpret_cast<char*>(block.data()), std::streamsize(block.size()));
        if (!input || crc32(block.data(), block.size()) != footer.crc32) {
            metadata.average_bitrate_bps = calculate_bitrate(file_bytes, samples_per_channel, sample_rate);
            return metadata;
        }

        const uint8_t* cursor = block.data();
        const uint8_t* end = cursor + block.size();
        BlockHeader block_header{};
        if (!read_pod(cursor, end, block_header) || block_header.magic != block_magic
            || block_header.version != metadata_version || block_header.header_bytes != sizeof(BlockHeader)
            || block_header.entry_count > max_entries) {
            metadata.average_bitrate_bps = calculate_bitrate(file_bytes, samples_per_channel, sample_rate);
            return metadata;
        }

        Metadata parsed;
        parsed.audio_bytes = footer.metadata_offset;
        for (uint32_t index = 0; index < block_header.entry_count; ++index) {
            EntryHeader entry{};
            if (!read_pod(cursor, end, entry) || entry.byte_count > size_t(end - cursor))
                throw std::runtime_error("invalid LX-VOXEL metadata entry");
            const uint8_t* entry_end = cursor + entry.byte_count;
            const Field field = Field(entry.type);
            if ((entry.flags & 1U) != 0U && field != Field::cover_art) {
                parsed.text.push_back({field,
                    std::string(reinterpret_cast<const char*>(cursor), entry.byte_count)});
            } else if (field == Field::cover_art) {
                CoverHeader cover_header{};
                if (!read_pod(cursor, entry_end, cover_header))
                    throw std::runtime_error("invalid LX-VOXEL cover header");
                const uint64_t variable_bytes = uint64_t(cover_header.mime_bytes)
                    + cover_header.description_bytes + cover_header.image_bytes;
                if (variable_bytes != uint64_t(entry_end - cursor))
                    throw std::runtime_error("invalid LX-VOXEL cover lengths");
                parsed.cover.picture_type = cover_header.picture_type;
                parsed.cover.width = cover_header.width;
                parsed.cover.height = cover_header.height;
                parsed.cover.depth = cover_header.depth;
                parsed.cover.colors = cover_header.colors;
                parsed.cover.mime_type.assign(reinterpret_cast<const char*>(cursor), cover_header.mime_bytes);
                cursor += cover_header.mime_bytes;
                parsed.cover.description.assign(reinterpret_cast<const char*>(cursor), cover_header.description_bytes);
                cursor += cover_header.description_bytes;
                parsed.cover.data.assign(cursor, cursor + cover_header.image_bytes);
                parsed.has_cover = !parsed.cover.data.empty();
            }
            cursor = entry_end;
        }
        if (cursor != end) throw std::runtime_error("LX-VOXEL metadata block has trailing bytes");
        parsed.average_bitrate_bps = calculate_bitrate(
            parsed.audio_bytes, samples_per_channel, sample_rate);
        return parsed;
    } catch (...) {
        metadata.average_bitrate_bps = calculate_bitrate(
            metadata.audio_bytes, samples_per_channel, sample_rate);
        return metadata;
    }
}

Metadata read_flac(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open FLAC input");
    std::array<uint8_t, 4> signature{};
    input.read(reinterpret_cast<char*>(signature.data()), signature.size());
    if (!input || signature != std::array<uint8_t, 4>{'f', 'L', 'a', 'C'})
        throw std::runtime_error("input is not a native FLAC stream");

    Metadata metadata;
    bool last = false;
    while (!last) {
        std::array<uint8_t, 4> header{};
        input.read(reinterpret_cast<char*>(header.data()), header.size());
        if (!input) throw std::runtime_error("truncated FLAC metadata header");
        last = (header[0] & 0x80U) != 0;
        const uint8_t type = header[0] & 0x7fU;
        const uint32_t size = (uint32_t(header[1]) << 16U)
            | (uint32_t(header[2]) << 8U) | uint32_t(header[3]);
        std::vector<uint8_t> block(size);
        input.read(reinterpret_cast<char*>(block.data()), std::streamsize(block.size()));
        if (!input) throw std::runtime_error("truncated FLAC metadata block");

        if (type == 4) {
            const uint8_t* cursor = block.data();
            const uint8_t* end = cursor + block.size();
            const uint32_t vendor_size = read_le32(cursor, end);
            if (vendor_size > size_t(end - cursor)) throw std::runtime_error("invalid FLAC vendor length");
            cursor += vendor_size;
            const uint32_t count = read_le32(cursor, end);
            if (count > max_entries * 16U) throw std::runtime_error("too many FLAC comments");
            for (uint32_t index = 0; index < count; ++index) {
                const uint32_t comment_size = read_le32(cursor, end);
                if (comment_size > size_t(end - cursor)) throw std::runtime_error("invalid FLAC comment length");
                import_comment(metadata,
                    std::string(reinterpret_cast<const char*>(cursor), comment_size));
                cursor += comment_size;
            }
        } else if (type == 6) {
            try {
                CoverArt candidate = parse_flac_picture(block.data(), block.size());
                const bool preferred = candidate.picture_type == 3;
                if (!metadata.has_cover || (preferred && metadata.cover.picture_type != 3)) {
                    metadata.cover = std::move(candidate);
                    metadata.has_cover = true;
                }
            } catch (const std::exception&) {
                // A malformed optional picture must not discard valid audio/tags.
            }
        }
    }
    return metadata;
}

} // namespace lxvoxel::metadata
