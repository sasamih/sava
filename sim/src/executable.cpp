#include "executable.h"

#include <cstring>

namespace sava {

size_t dtypeSize(DType dtype) {
  switch (dtype) {
    case DType::F32: return 4;
    case DType::BF16: return 2;
    case DType::I32: return 4;
    case DType::I8: return 1;
  }
  return 0;
}

const char* dtypeName(DType dtype) {
  switch (dtype) {
    case DType::F32: return "f32";
    case DType::BF16: return "bf16";
    case DType::I32: return "i32";
    case DType::I8: return "i8";
  }
  return "?";
}

const char* argKindName(ArgKind kind) {
  switch (kind) {
    case ArgKind::In: return "in";
    case ArgKind::Out: return "out";
    case ArgKind::InOut: return "inout";
  }
  return "?";
}

uint64_t ArgInfo::numElements() const {
  uint64_t n = 1;
  for (uint32_t d : dims) n *= d;
  return n;
}

namespace {

uint16_t readU16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }
uint32_t readU32(const uint8_t* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
         (uint32_t(p[3]) << 24);
}
uint64_t readU64(const uint8_t* p) {
  return uint64_t(readU32(p)) | (uint64_t(readU32(p + 4)) << 32);
}

bool fail(std::string* error, const std::string& msg) {
  if (error) *error = "bad executable: " + msg;
  return false;
}

}  // namespace

bool parseExecutable(const uint8_t* data, size_t size, Executable* out,
                     std::string* error) {
  if (size < kSbinHeaderSize) return fail(error, "file too small");
  if (std::memcmp(data, "SAVA", 4) != 0) return fail(error, "bad magic");

  Executable exe;
  exe.version_major = readU16(data + 4);
  exe.version_minor = readU16(data + 6);
  if (exe.version_major != kSbinVersionMajor)
    return fail(error, "unsupported version " +
                           std::to_string(exe.version_major) + "." +
                           std::to_string(exe.version_minor));

  uint32_t code_offset = readU32(data + 8);
  uint32_t code_count = readU32(data + 12);
  uint32_t args_offset = readU32(data + 16);
  uint32_t args_count = readU32(data + 20);
  uint32_t name_offset = readU32(data + 24);
  uint32_t flags = readU32(data + 28);
  if (flags != 0) return fail(error, "unknown flags");

  if (code_offset % 8 != 0) return fail(error, "misaligned code section");
  if (uint64_t(code_offset) + uint64_t(code_count) * 8 > size)
    return fail(error, "code section out of bounds");
  if (code_count == 0) return fail(error, "empty code section");
  exe.code.resize(code_count);
  for (uint32_t i = 0; i < code_count; ++i)
    exe.code[i] = readU64(data + code_offset + 8ull * i);

  if (uint64_t(args_offset) + uint64_t(args_count) * kSbinArgRecordSize > size)
    return fail(error, "arg table out of bounds");
  if (args_count > 8) return fail(error, "more than 8 kernel arguments");
  for (uint32_t i = 0; i < args_count; ++i) {
    const uint8_t* rec = data + args_offset + kSbinArgRecordSize * i;
    ArgInfo arg;
    arg.name.assign(reinterpret_cast<const char*>(rec),
                    strnlen(reinterpret_cast<const char*>(rec), 24));
    if (rec[24] > 2) return fail(error, "bad arg kind");
    if (rec[25] > 3) return fail(error, "bad arg dtype");
    if (rec[26] > kSbinMaxRank) return fail(error, "bad arg rank");
    arg.kind = ArgKind(rec[24]);
    arg.dtype = DType(rec[25]);
    for (uint8_t d = 0; d < rec[26]; ++d)
      arg.dims.push_back(readU32(rec + 28 + 4 * d));
    exe.args.push_back(std::move(arg));
  }

  if (name_offset >= size) return fail(error, "name out of bounds");
  exe.name.assign(reinterpret_cast<const char*>(data + name_offset),
                  strnlen(reinterpret_cast<const char*>(data + name_offset),
                          size - name_offset));

  *out = std::move(exe);
  return true;
}

}  // namespace sava
