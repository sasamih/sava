#include "npy.h"

#include <cstring>
#include <fstream>
#include <sstream>

namespace sava::npy {

namespace {

bool fail(std::string* error, const std::string& msg) {
  if (error) *error = msg;
  return false;
}

// Extracts the value following `'key':` in the header dict.
std::string dictValue(const std::string& header, const std::string& key) {
  size_t k = header.find("'" + key + "'");
  if (k == std::string::npos) return "";
  size_t colon = header.find(':', k);
  if (colon == std::string::npos) return "";
  size_t start = header.find_first_not_of(' ', colon + 1);
  if (start == std::string::npos) return "";
  if (header[start] == '(') return header.substr(start, header.find(')', start) - start + 1);
  if (header[start] == '\'')
    return header.substr(start + 1, header.find('\'', start + 1) - start - 1);
  size_t end = header.find_first_of(",}", start);
  return header.substr(start, end - start);
}

}  // namespace

bool load(const std::string& path, Array* out, std::string* error) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return fail(error, "cannot open " + path);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), {});
  if (bytes.size() < 10 || std::memcmp(bytes.data(), "\x93NUMPY", 6) != 0)
    return fail(error, path + ": not a .npy file");
  uint8_t major = bytes[6];
  size_t header_len, header_start;
  if (major == 1) {
    header_len = bytes[8] | (bytes[9] << 8);
    header_start = 10;
  } else {
    if (bytes.size() < 12) return fail(error, path + ": truncated");
    header_len = bytes[8] | (bytes[9] << 8) | (bytes[10] << 16) | (size_t(bytes[11]) << 24);
    header_start = 12;
  }
  if (header_start + header_len > bytes.size()) return fail(error, path + ": truncated");
  std::string header(reinterpret_cast<const char*>(bytes.data() + header_start), header_len);

  Array a;
  a.descr = dictValue(header, "descr");
  if (dictValue(header, "fortran_order").find("True") != std::string::npos)
    return fail(error, path + ": Fortran order is not supported");
  std::string shape = dictValue(header, "shape");
  std::stringstream ss(shape.substr(1, shape.size() - 2));
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.find_first_not_of(' ') == std::string::npos) continue;
    a.shape.push_back(std::stoull(item));
  }
  a.data.assign(bytes.begin() + header_start + header_len, bytes.end());
  *out = std::move(a);
  return true;
}

bool save(const std::string& path, const Array& array, std::string* error) {
  std::string shape = "(";
  for (size_t i = 0; i < array.shape.size(); ++i) {
    shape += std::to_string(array.shape[i]);
    if (array.shape.size() == 1 || i + 1 < array.shape.size()) shape += ",";
    if (i + 1 < array.shape.size()) shape += " ";
  }
  shape += ")";
  std::string header = "{'descr': '" + array.descr +
                       "', 'fortran_order': False, 'shape': " + shape + ", }";
  // Pad so that magic(6) + version(2) + len(2) + header is a multiple of 64.
  size_t total = 10 + header.size() + 1;
  header.append((64 - total % 64) % 64, ' ');
  header += '\n';

  std::ofstream f(path, std::ios::binary);
  if (!f) return fail(error, "cannot write " + path);
  f.write("\x93NUMPY\x01\x00", 8);
  uint16_t len = uint16_t(header.size());
  char len_bytes[2] = {char(len & 0xff), char(len >> 8)};
  f.write(len_bytes, 2);
  f.write(header.data(), std::streamsize(header.size()));
  f.write(reinterpret_cast<const char*>(array.data.data()),
          std::streamsize(array.data.size()));
  return bool(f);
}

}  // namespace sava::npy
