// Number format helpers shared by the MXU and VPU models.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace sava {

inline float bitsToF32(uint32_t bits) {
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

inline uint32_t f32ToBits(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  return bits;
}

inline float bf16ToF32(uint16_t h) { return bitsToF32(uint32_t(h) << 16); }

// Round-to-nearest-even, NaN stays NaN (quiet).
inline uint16_t f32ToBf16(float f) {
  uint32_t u = f32ToBits(f);
  if (std::isnan(f)) return 0x7fc0;
  uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return uint16_t(u >> 16);
}

}  // namespace sava
