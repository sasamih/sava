// Minimal reader/writer for NumPy .npy files (format version 1.0/2.0,
// little-endian, C order). Enough for exchanging tensors with the simulator.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sava::npy {

struct Array {
  std::string descr;  // e.g. "<f4", "<u2", "<i4", "|i1"
  std::vector<uint64_t> shape;
  std::vector<uint8_t> data;
};

bool load(const std::string& path, Array* out, std::string* error);
bool save(const std::string& path, const Array& array, std::string* error);

}  // namespace sava::npy
