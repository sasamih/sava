// C API wrapper around sava::Device. Converts exceptions into status codes.
#include <new>
#include <string>

#include "device.h"
#include "sava/sava_sim.h"

struct sava_device {
  explicit sava_device(const sava_config& config) : impl(config) {}
  sava::Device impl;
  std::string last_error;
};

namespace {

template <typename Fn>
sava_status guarded(sava_device* device, Fn&& fn) {
  if (!device) return SAVA_ERR_INVALID_ARGUMENT;
  device->last_error.clear();
  try {
    fn();
    return SAVA_OK;
  } catch (const sava::SimError& e) {
    device->last_error = e.what();
    return e.status;
  } catch (const std::exception& e) {
    device->last_error = e.what();
    return SAVA_ERR_EXECUTION;
  }
}

}  // namespace

extern "C" {

void sava_config_init_default(sava_config* config) {
  config->dram_bytes = sava::defaults::kDramBytes;
  config->sram_bytes = sava::defaults::kSramBytes;
  config->strict_hazards = 1;
  config->max_instructions = 0;
  config->trace_path = nullptr;
}

sava_device* sava_device_create(const sava_config* config) {
  sava_config cfg;
  if (config) cfg = *config;
  else sava_config_init_default(&cfg);
  try {
    return new sava_device(cfg);
  } catch (...) {
    return nullptr;
  }
}

void sava_device_destroy(sava_device* device) { delete device; }

const char* sava_device_last_error(const sava_device* device) {
  return device ? device->last_error.c_str() : "null device";
}

sava_status sava_dram_write(sava_device* device, uint64_t addr, const void* src,
                            uint64_t size) {
  return guarded(device, [&] { device->impl.dramWrite(addr, src, size); });
}

sava_status sava_dram_read(sava_device* device, uint64_t addr, void* dst,
                           uint64_t size) {
  return guarded(device, [&] { device->impl.dramRead(addr, dst, size); });
}

sava_status sava_load_executable(sava_device* device, const void* data,
                                 uint64_t size) {
  return guarded(device, [&] {
    device->impl.load(static_cast<const uint8_t*>(data), size);
  });
}

sava_status sava_run(sava_device* device, const uint64_t* args,
                     uint32_t num_args) {
  return guarded(device, [&] { device->impl.run(args, num_args); });
}

sava_status sava_get_stats(const sava_device* device, sava_stats* stats) {
  if (!device || !stats) return SAVA_ERR_INVALID_ARGUMENT;
  *stats = device->impl.stats();
  return SAVA_OK;
}

}  // extern "C"
