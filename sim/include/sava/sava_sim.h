/* SAVA simulator C API.
 *
 * This is the "driver-level" interface of the simulated device: it is what a
 * host runtime (the Python runtime today, an IREE HAL driver later) programs
 * against. It deliberately mirrors what a real PCIe accelerator driver
 * exposes: device memory read/write, program load, launch, counters.
 */
#ifndef SAVA_SIM_H_
#define SAVA_SIM_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SAVA_SIM_API_VERSION 1

typedef enum sava_status {
  SAVA_OK = 0,
  SAVA_ERR_INVALID_ARGUMENT = 1,
  SAVA_ERR_OUT_OF_RANGE = 2,
  SAVA_ERR_BAD_EXECUTABLE = 3,
  SAVA_ERR_EXECUTION = 4,   /* trap: illegal instruction, bad address, ... */
  SAVA_ERR_HAZARD = 5,      /* unsynchronized access detected (strict mode) */
  SAVA_ERR_TIMEOUT = 6,     /* instruction budget exhausted */
} sava_status;

typedef struct sava_config {
  uint64_t dram_bytes;
  uint32_t sram_bytes;
  /* Report unsynchronized SRAM accesses between units as errors. */
  int strict_hazards;
  /* Abort after this many retired instructions (0 = unlimited). */
  uint64_t max_instructions;
  /* Write an instruction trace to this file ("-" = stderr, NULL = off). */
  const char* trace_path;
} sava_config;

typedef struct sava_stats {
  uint64_t cycles;               /* total kernel cycles */
  uint64_t instructions;         /* retired instructions (all units) */
  uint64_t scalar_instructions;
  uint64_t wait_stall_cycles;    /* cycles the scalar core spent in wait/halt */
  uint64_t queue_stall_cycles;   /* cycles stalled on full unit queues */
  uint64_t dma_ops, dma_busy_cycles, dma_bytes_in, dma_bytes_out;
  uint64_t mxu_ops, mxu_busy_cycles, mxu_macs;
  uint64_t vpu_ops, vpu_busy_cycles, vpu_elements;
} sava_stats;

typedef struct sava_device sava_device;

void sava_config_init_default(sava_config* config);

sava_device* sava_device_create(const sava_config* config);
void sava_device_destroy(sava_device* device);

/* Last error message for this device (empty string if none). */
const char* sava_device_last_error(const sava_device* device);

sava_status sava_dram_write(sava_device* device, uint64_t addr,
                            const void* src, uint64_t size);
sava_status sava_dram_read(sava_device* device, uint64_t addr, void* dst,
                           uint64_t size);

/* Loads a .sbin executable (see docs/hardware/sava-v0-isa.md). */
sava_status sava_load_executable(sava_device* device, const void* data,
                                 uint64_t size);

/* Runs the loaded kernel. args[i] is placed in register a<i> (x1 + i). */
sava_status sava_run(sava_device* device, const uint64_t* args,
                     uint32_t num_args);

sava_status sava_get_stats(const sava_device* device, sava_stats* stats);

#ifdef __cplusplus
}
#endif

#endif /* SAVA_SIM_H_ */
