// SAVA executable (.sbin) container format.
//
// Layout (little-endian), see docs/hardware/sava-v0-isa.md#executable-format:
//   Header (32 bytes)
//     0  char[4] magic "SAVA"
//     4  u16     version_major
//     6  u16     version_minor
//     8  u32     code_offset     (8-byte aligned)
//    12  u32     code_count      (number of 64-bit instructions)
//    16  u32     args_offset
//    20  u32     args_count
//    24  u32     name_offset     (NUL-terminated kernel name)
//    28  u32     flags           (reserved, must be 0)
//   Arg record (48 bytes)
//     0  char[24] name (NUL padded)
//    24  u8       kind  (0 = in, 1 = out, 2 = inout)
//    25  u8       dtype (0 = f32, 1 = bf16, 2 = i32, 3 = i8)
//    26  u8       rank  (0..4)
//    27  u8       reserved
//    28  u32[4]   dims
//    44  u32      reserved
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sava {

inline constexpr uint16_t kSbinVersionMajor = 0;
inline constexpr uint16_t kSbinVersionMinor = 1;
inline constexpr size_t kSbinHeaderSize = 32;
inline constexpr size_t kSbinArgRecordSize = 48;
inline constexpr size_t kSbinMaxRank = 4;

enum class ArgKind : uint8_t { In = 0, Out = 1, InOut = 2 };
enum class DType : uint8_t { F32 = 0, BF16 = 1, I32 = 2, I8 = 3 };

size_t dtypeSize(DType dtype);
const char* dtypeName(DType dtype);
const char* argKindName(ArgKind kind);

struct ArgInfo {
  std::string name;
  ArgKind kind = ArgKind::In;
  DType dtype = DType::F32;
  std::vector<uint32_t> dims;

  uint64_t numElements() const;
  uint64_t bytes() const { return numElements() * dtypeSize(dtype); }
};

struct Executable {
  uint16_t version_major = 0;
  uint16_t version_minor = 0;
  std::string name;
  std::vector<uint64_t> code;
  std::vector<ArgInfo> args;
};

// Parses and validates an .sbin image. Returns false and sets *error on
// malformed input.
bool parseExecutable(const uint8_t* data, size_t size, Executable* out,
                     std::string* error);

}  // namespace sava
