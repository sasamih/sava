// SAVA v0 core model.
//
// Functional + approximately-timed simulator of one SAVA core:
//   * an in-order scalar control core that fetches every instruction,
//   * three asynchronous units (DMA, MXU, VPU), each with an in-order command
//     queue,
//   * DRAM (device memory) and a software-managed SRAM scratchpad.
//
// Functional effects of async commands are applied at issue time, in program
// order. That is only equivalent to real hardware if the program synchronizes
// correctly, so the model tracks every in-flight command's SRAM footprint and
// (in strict mode) reports any unsynchronized cross-unit access as a hazard.
// See docs/hardware/sava-v0-architecture.md for the full timing model.
#pragma once

#include <cstdint>
#include <cstdio>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

#include "executable.h"
#include "sava/isa_gen.h"
#include "sava/sava_sim.h"

namespace sava {

class SimError : public std::runtime_error {
 public:
  SimError(sava_status status, const std::string& message)
      : std::runtime_error(message), status(status) {}
  sava_status status;
};

// Disassembles one instruction word. `pc` is used to print branch targets.
std::string disassemble(uint64_t word, uint64_t pc);

class Device {
 public:
  explicit Device(const sava_config& config);
  ~Device();

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  void dramWrite(uint64_t addr, const void* src, uint64_t size);
  void dramRead(uint64_t addr, void* dst, uint64_t size);
  void load(const uint8_t* data, uint64_t size);
  void run(const uint64_t* args, uint32_t num_args);

  const sava_stats& stats() const { return stats_; }
  const Executable* executable() const { return loaded_ ? &exe_ : nullptr; }
  const sava_config& config() const { return config_; }

 private:
  struct Range {
    uint64_t lo, hi;  // [lo, hi) SRAM byte addresses
  };
  struct InFlight {
    isa::Unit unit;
    uint32_t tag;
    uint64_t done;
    uint64_t pc;
    std::vector<Range> reads, writes;
  };
  struct Queue {
    uint32_t depth = 1;
    uint64_t free_at = 0;
    std::deque<uint64_t> done_times;
  };

  void reset(const uint64_t* args, uint32_t num_args);
  void step();
  [[noreturn]] void trap(sava_status status, const std::string& message) const;

  uint64_t reg(unsigned r) const { return regs_[r]; }
  void setReg(unsigned r, uint64_t v) {
    if (r != 0) regs_[r] = v;
  }
  uint64_t csr(unsigned id) const;
  void setCsr(unsigned id, uint64_t value);

  uint8_t* sramPtr(uint64_t addr, uint64_t size, const char* what);
  uint8_t* dramPtr(uint64_t addr, uint64_t size, const char* what);
  void checkAlign(uint64_t addr, uint64_t align, const char* what) const;

  // Issues an async command. Returns the cycle at which it completes.
  uint64_t issue(isa::Unit unit, uint32_t tag, uint64_t duration,
                 uint64_t extra_latency, std::vector<Range> reads,
                 std::vector<Range> writes);
  void checkHazards(isa::Unit unit, const std::vector<Range>& reads,
                    const std::vector<Range>& writes) const;
  void waitMask(uint32_t mask);
  void waitAll();

  void execDma(bool to_sram, uint64_t sram, uint64_t dram, uint32_t tag);
  void execMxu(uint64_t c, uint64_t a, uint64_t b, uint32_t flags);
  void execVpu(isa::Opcode op, uint64_t dst, uint64_t src0, uint64_t src1,
               uint32_t imm);

  void traceLine(uint64_t word, const char* suffix);

  sava_config config_;
  std::vector<uint8_t> dram_;
  std::vector<uint8_t> sram_;
  Executable exe_;
  bool loaded_ = false;

  uint64_t regs_[32] = {};
  uint64_t csrs_[isa::kNumCsrs] = {};
  uint64_t pc_ = 0;
  uint64_t now_ = 0;
  bool halted_ = false;

  Queue queues_[4];  // indexed by isa::Unit (Scalar entry unused)
  std::vector<InFlight> inflight_;

  sava_stats stats_ = {};
  FILE* trace_ = nullptr;
  bool own_trace_ = false;
  char trace_suffix_[160] = {};
};

}  // namespace sava
