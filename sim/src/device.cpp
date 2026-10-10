#include "device.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>

#include "numerics.h"

namespace sava {

using isa::Opcode;
using isa::Unit;

namespace {

struct Decoded {
  uint8_t opcode, rd, rs1, rs2, rs3;
  int32_t imm;
};

uint64_t bitsAt(uint64_t word, unsigned lsb, unsigned bits) {
  return (word >> lsb) & ((1ull << bits) - 1);
}

Decoded decode(uint64_t w) {
  Decoded d;
  d.opcode = uint8_t(bitsAt(w, isa::kOpcodeLsb, isa::kOpcodeBits));
  d.rd = uint8_t(bitsAt(w, isa::kRdLsb, isa::kRdBits));
  d.rs1 = uint8_t(bitsAt(w, isa::kRs1Lsb, isa::kRs1Bits));
  d.rs2 = uint8_t(bitsAt(w, isa::kRs2Lsb, isa::kRs2Bits));
  d.rs3 = uint8_t(bitsAt(w, isa::kRs3Lsb, isa::kRs3Bits));
  d.imm = int32_t(uint32_t(bitsAt(w, isa::kImmLsb, isa::kImmBits)));
  return d;
}

uint32_t fieldValue(const Decoded& d, isa::Field field) {
  switch (field) {
    case isa::Field::Rd: return d.rd;
    case isa::Field::Rs1: return d.rs1;
    case isa::Field::Rs2: return d.rs2;
    case isa::Field::Rs3: return d.rs3;
    case isa::Field::Imm: return uint32_t(d.imm);
    case isa::Field::None: break;
  }
  return 0;
}

std::string regName(unsigned r) {
  if (r == 0) return "zero";
  if (r >= 1 && r <= 8) return "a" + std::to_string(r - 1);
  return "x" + std::to_string(r);
}

const char* unitName(Unit unit) {
  switch (unit) {
    case Unit::Scalar: return "scalar";
    case Unit::Dma: return "dma";
    case Unit::Mxu: return "mxu";
    case Unit::Vpu: return "vpu";
  }
  return "?";
}

uint64_t ceilDiv(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

constexpr uint32_t kWaitMxuBit = 1u << 16;
constexpr uint32_t kWaitVpuBit = 1u << 17;
constexpr uint32_t kWaitAll = 0x3ffffu;

}  // namespace

std::string disassemble(uint64_t word, uint64_t pc) {
  Decoded d = decode(word);
  const isa::InstrInfo* info = isa::lookup(d.opcode);
  if (!info) {
    char buf[48];
    std::snprintf(buf, sizeof(buf), "<illegal 0x%016" PRIx64 ">", word);
    return buf;
  }
  std::string s = info->mnemonic;
  for (unsigned i = 0; i < info->num_operands; ++i) {
    s += (i == 0) ? " " : ", ";
    const isa::OperandInfo& op = info->operands[i];
    uint32_t raw = fieldValue(d, op.field);
    char buf[48];
    switch (op.kind) {
      case isa::OperandKind::Reg: s += regName(raw); break;
      case isa::OperandKind::Csr: {
        const char* name = isa::csrName(raw);
        s += name ? name : std::to_string(raw);
        break;
      }
      case isa::OperandKind::Int: s += std::to_string(int32_t(raw)); break;
      case isa::OperandKind::F32:
        std::snprintf(buf, sizeof(buf), "%g", double(bitsToF32(raw)));
        s += buf;
        break;
      case isa::OperandKind::PcRel:
        s += "@" + std::to_string(int64_t(pc) + int32_t(raw));
        break;
      case isa::OperandKind::Mask:
        std::snprintf(buf, sizeof(buf), "0x%x", raw);
        s += buf;
        break;
      case isa::OperandKind::None: break;
    }
  }
  return s;
}

Device::Device(const sava_config& config) : config_(config) {
  dram_.resize(config_.dram_bytes);
  sram_.resize(config_.sram_bytes);
  if (config_.trace_path) {
    if (std::strcmp(config_.trace_path, "-") == 0) {
      trace_ = stderr;
    } else {
      trace_ = std::fopen(config_.trace_path, "w");
      if (!trace_)
        throw SimError(SAVA_ERR_INVALID_ARGUMENT,
                       std::string("cannot open trace file ") +
                           config_.trace_path);
      own_trace_ = true;
    }
  }
  // The config's trace_path pointer is not owned; don't keep it around.
  config_.trace_path = nullptr;
}

Device::~Device() {
  if (own_trace_) std::fclose(trace_);
}

void Device::dramWrite(uint64_t addr, const void* src, uint64_t size) {
  if (addr > dram_.size() || size > dram_.size() - addr)
    throw SimError(SAVA_ERR_OUT_OF_RANGE, "dram write out of range");
  std::memcpy(dram_.data() + addr, src, size);
}

void Device::dramRead(uint64_t addr, void* dst, uint64_t size) {
  if (addr > dram_.size() || size > dram_.size() - addr)
    throw SimError(SAVA_ERR_OUT_OF_RANGE, "dram read out of range");
  std::memcpy(dst, dram_.data() + addr, size);
}

void Device::load(const uint8_t* data, uint64_t size) {
  std::string error;
  Executable exe;
  if (!parseExecutable(data, size_t(size), &exe, &error))
    throw SimError(SAVA_ERR_BAD_EXECUTABLE, error);
  exe_ = std::move(exe);
  loaded_ = true;
}

void Device::reset(const uint64_t* args, uint32_t num_args) {
  std::memset(regs_, 0, sizeof(regs_));
  std::memset(csrs_, 0, sizeof(csrs_));
  for (uint32_t i = 0; i < num_args; ++i) regs_[1 + i] = args[i];
  pc_ = 0;
  now_ = 0;
  halted_ = false;
  for (Queue& q : queues_) q = Queue();
  queues_[int(Unit::Dma)].depth = defaults::kDmaQueueDepth;
  queues_[int(Unit::Mxu)].depth = defaults::kMxuQueueDepth;
  queues_[int(Unit::Vpu)].depth = defaults::kVpuQueueDepth;
  inflight_.clear();
  stats_ = sava_stats{};
}

void Device::run(const uint64_t* args, uint32_t num_args) {
  if (!loaded_) throw SimError(SAVA_ERR_INVALID_ARGUMENT, "no executable loaded");
  if (num_args != exe_.args.size())
    throw SimError(SAVA_ERR_INVALID_ARGUMENT,
                   "kernel '" + exe_.name + "' expects " +
                       std::to_string(exe_.args.size()) + " arguments, got " +
                       std::to_string(num_args));
  reset(args, num_args);
  if (trace_)
    std::fprintf(trace_, "# kernel %s: %zu instructions\n", exe_.name.c_str(),
                 exe_.code.size());
  while (!halted_) {
    if (config_.max_instructions && stats_.instructions >= config_.max_instructions)
      throw SimError(SAVA_ERR_TIMEOUT,
                     "instruction budget of " +
                         std::to_string(config_.max_instructions) +
                         " exhausted at pc=" + std::to_string(pc_));
    step();
  }
  stats_.cycles = now_;
  if (trace_) {
    std::fprintf(trace_, "# halted at cycle %" PRIu64 "\n", now_);
    std::fflush(trace_);
  }
}

void Device::trap(sava_status status, const std::string& message) const {
  std::string where = "pc=" + std::to_string(pc_);
  if (pc_ < exe_.code.size())
    where += " (" + disassemble(exe_.code[pc_], pc_) + ")";
  throw SimError(status, where + ": " + message);
}

uint64_t Device::csr(unsigned id) const {
  if (id >= isa::kNumCsrs || !isa::csrName(id))
    trap(SAVA_ERR_EXECUTION, "read of unknown CSR " + std::to_string(id));
  switch (id) {
    case isa::CSR_CYCLE: return now_;
    case isa::CSR_CORE_ID: return 0;
    case isa::CSR_NUM_CORES: return 1;
    default: return csrs_[id];
  }
}

void Device::setCsr(unsigned id, uint64_t value) {
  if (id >= isa::kNumCsrs || !isa::csrName(id))
    trap(SAVA_ERR_EXECUTION, "write of unknown CSR " + std::to_string(id));
  if (isa::csrIsReadOnly(id))
    trap(SAVA_ERR_EXECUTION, std::string("write of read-only CSR ") + isa::csrName(id));
  csrs_[id] = value;
}

uint8_t* Device::sramPtr(uint64_t addr, uint64_t size, const char* what) {
  if (addr > sram_.size() || size > sram_.size() - addr) {
    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  "%s: SRAM access [0x%" PRIx64 ", 0x%" PRIx64
                  ") outside of SRAM (size 0x%zx)",
                  what, addr, addr + size, sram_.size());
    trap(SAVA_ERR_EXECUTION, buf);
  }
  return sram_.data() + addr;
}

uint8_t* Device::dramPtr(uint64_t addr, uint64_t size, const char* what) {
  if (addr > dram_.size() || size > dram_.size() - addr) {
    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  "%s: DRAM access [0x%" PRIx64 ", 0x%" PRIx64
                  ") outside of DRAM (size 0x%zx)",
                  what, addr, addr + size, dram_.size());
    trap(SAVA_ERR_EXECUTION, buf);
  }
  return dram_.data() + addr;
}

void Device::checkAlign(uint64_t addr, uint64_t align, const char* what) const {
  if (addr % align != 0)
    trap(SAVA_ERR_EXECUTION, std::string(what) + " must be " +
                                 std::to_string(align) + "-byte aligned");
}

void Device::checkHazards(Unit unit, const std::vector<Range>& reads,
                          const std::vector<Range>& writes) const {
  if (!config_.strict_hazards) return;
  auto overlaps = [](const std::vector<Range>& xs, const std::vector<Range>& ys) {
    for (const Range& x : xs)
      for (const Range& y : ys)
        if (x.lo < y.hi && y.lo < x.hi) return true;
    return false;
  };
  for (const InFlight& op : inflight_) {
    if (op.unit == unit) continue;  // same queue: executes in order
    const char* kind = nullptr;
    if (overlaps(reads, op.writes)) kind = "read-after-write";
    else if (overlaps(writes, op.writes)) kind = "write-after-write";
    else if (overlaps(writes, op.reads)) kind = "write-after-read";
    if (!kind) continue;
    trap(SAVA_ERR_HAZARD,
         std::string(kind) + " hazard on SRAM with " + unitName(op.unit) +
             " command at pc=" + std::to_string(op.pc) + " (" +
             disassemble(exe_.code[op.pc], op.pc) +
             ") that has not been waited for; insert a `wait`");
  }
}

uint64_t Device::issue(Unit unit, uint32_t tag, uint64_t duration,
                       uint64_t extra_latency, std::vector<Range> reads,
                       std::vector<Range> writes) {
  checkHazards(unit, reads, writes);

  Queue& q = queues_[int(unit)];
  while (!q.done_times.empty() && q.done_times.front() <= now_)
    q.done_times.pop_front();
  if (q.done_times.size() >= q.depth) {
    uint64_t t = q.done_times.front();
    stats_.queue_stall_cycles += t - now_;
    now_ = t;
    q.done_times.pop_front();
  }

  uint64_t start = std::max(now_, q.free_at);
  q.free_at = start + duration;
  uint64_t done = q.free_at + extra_latency;
  q.done_times.push_back(done);
  inflight_.push_back({unit, tag, done, pc_, std::move(reads), std::move(writes)});

  switch (unit) {
    case Unit::Dma: stats_.dma_ops++; stats_.dma_busy_cycles += duration; break;
    case Unit::Mxu: stats_.mxu_ops++; stats_.mxu_busy_cycles += duration; break;
    case Unit::Vpu: stats_.vpu_ops++; stats_.vpu_busy_cycles += duration; break;
    case Unit::Scalar: break;
  }
  if (trace_)
    std::snprintf(trace_suffix_, sizeof(trace_suffix_),
                  "  ; %s start=%" PRIu64 " done=%" PRIu64, unitName(unit), start,
                  done);
  return done;
}

void Device::waitMask(uint32_t mask) {
  uint64_t t = now_;
  auto selected = [mask](const InFlight& op) {
    switch (op.unit) {
      case Unit::Dma: return (mask & (1u << op.tag)) != 0;
      case Unit::Mxu: return (mask & kWaitMxuBit) != 0;
      case Unit::Vpu: return (mask & kWaitVpuBit) != 0;
      case Unit::Scalar: return false;
    }
    return false;
  };
  for (const InFlight& op : inflight_)
    if (selected(op)) t = std::max(t, op.done);
  inflight_.erase(std::remove_if(inflight_.begin(), inflight_.end(), selected),
                  inflight_.end());
  stats_.wait_stall_cycles += t - now_;
  if (trace_ && t != now_)
    std::snprintf(trace_suffix_, sizeof(trace_suffix_),
                  "  ; stalled %" PRIu64 " cycles", t - now_);
  now_ = t;
}

void Device::waitAll() { waitMask(kWaitAll); }

void Device::execDma(bool to_sram, uint64_t sram, uint64_t dram, uint32_t tag) {
  if (tag >= defaults::kDmaTags)
    trap(SAVA_ERR_EXECUTION, "DMA tag must be < " + std::to_string(defaults::kDmaTags));
  uint64_t rows = csrs_[isa::CSR_DMA_ROWS];
  uint64_t row_bytes = csrs_[isa::CSR_DMA_ROW_BYTES];
  uint64_t sram_stride = csrs_[isa::CSR_DMA_SRAM_STRIDE];
  uint64_t dram_stride = csrs_[isa::CSR_DMA_DRAM_STRIDE];
  if (rows == 0 || row_bytes == 0)
    trap(SAVA_ERR_EXECUTION, "DMA_ROWS and DMA_ROW_BYTES must be non-zero");

  uint64_t sram_extent = (rows - 1) * sram_stride + row_bytes;
  uint64_t dram_extent = (rows - 1) * dram_stride + row_bytes;
  uint8_t* s = sramPtr(sram, sram_extent, to_sram ? "dma.ld" : "dma.st");
  uint8_t* d = dramPtr(dram, dram_extent, to_sram ? "dma.ld" : "dma.st");

  uint64_t bytes = rows * row_bytes;
  uint64_t duration = std::max<uint64_t>(1, ceilDiv(bytes, defaults::kDramBytesPerCycle));
  std::vector<Range> footprint = {{sram, sram + sram_extent}};
  if (to_sram)
    issue(Unit::Dma, tag, duration, defaults::kDramLatency, {}, footprint);
  else
    issue(Unit::Dma, tag, duration, defaults::kDramLatency, footprint, {});

  for (uint64_t r = 0; r < rows; ++r) {
    if (to_sram)
      std::memmove(s + r * sram_stride, d + r * dram_stride, row_bytes);
    else
      std::memmove(d + r * dram_stride, s + r * sram_stride, row_bytes);
  }
  if (to_sram) stats_.dma_bytes_in += bytes;
  else stats_.dma_bytes_out += bytes;
}

void Device::execMxu(uint64_t c, uint64_t a, uint64_t b, uint32_t flags) {
  uint64_t M = csrs_[isa::CSR_MXU_M];
  uint64_t N = csrs_[isa::CSR_MXU_N];
  uint64_t K = csrs_[isa::CSR_MXU_K];
  uint64_t lda = csrs_[isa::CSR_MXU_LDA];
  uint64_t ldb = csrs_[isa::CSR_MXU_LDB];
  uint64_t ldc = csrs_[isa::CSR_MXU_LDC];
  uint64_t dtype = csrs_[isa::CSR_MXU_DTYPE];
  if (M < 1 || M > defaults::kMxuMaxM || N < 1 || N > defaults::kMxuMaxN ||
      K < 1 || K > defaults::kMxuMaxK)
    trap(SAVA_ERR_EXECUTION,
         "MXU shape " + std::to_string(M) + "x" + std::to_string(N) + "x" +
             std::to_string(K) + " outside of hardware limits (max " +
             std::to_string(defaults::kMxuMaxM) + "x" +
             std::to_string(defaults::kMxuMaxN) + "x" +
             std::to_string(defaults::kMxuMaxK) + ")");
  if (dtype > 1) trap(SAVA_ERR_EXECUTION, "MXU_DTYPE must be 0 (f32) or 1 (bf16)");
  if (flags & ~1u) trap(SAVA_ERR_EXECUTION, "unknown mxu.mm flags");
  const bool bf16 = dtype == 1;
  const uint64_t esz = bf16 ? 2 : 4;
  checkAlign(a, esz, "mxu.mm A");
  checkAlign(b, esz, "mxu.mm B");
  checkAlign(c, 4, "mxu.mm C");
  checkAlign(lda, esz, "MXU_LDA");
  checkAlign(ldb, esz, "MXU_LDB");
  checkAlign(ldc, 4, "MXU_LDC");
  if (lda < K * esz || ldb < N * esz || ldc < N * 4)
    trap(SAVA_ERR_EXECUTION, "MXU row stride smaller than row size");

  uint64_t a_ext = (M - 1) * lda + K * esz;
  uint64_t b_ext = (K - 1) * ldb + N * esz;
  uint64_t c_ext = (M - 1) * ldc + N * 4;
  const uint8_t* pa = sramPtr(a, a_ext, "mxu.mm A");
  const uint8_t* pb = sramPtr(b, b_ext, "mxu.mm B");
  uint8_t* pc = sramPtr(c, c_ext, "mxu.mm C");

  const bool accumulate = flags & 1u;
  std::vector<Range> reads = {{a, a + a_ext}, {b, b + b_ext}};
  if (accumulate) reads.push_back({c, c + c_ext});

  // Weight-stationary array of D x D PEs: every D x D block of B is loaded
  // (D cycles) and the M rows of A are streamed through it (M cycles), plus
  // one pipeline fill/drain. f32 operands are processed in multiple passes.
  const uint64_t D = defaults::kMxuArrayDim;
  uint64_t duration = ceilDiv(K, D) * ceilDiv(N, D) * (D + M) + D;
  if (!bf16) duration *= defaults::kMxuF32Slowdown;
  issue(Unit::Mxu, 0, duration, 0, std::move(reads), {{c, c + c_ext}});

  auto load = [&](const uint8_t* base, uint64_t ld, uint64_t rows, uint64_t cols) {
    std::vector<float> m(rows * cols);
    for (uint64_t r = 0; r < rows; ++r)
      for (uint64_t col = 0; col < cols; ++col) {
        const uint8_t* p = base + r * ld + col * esz;
        if (bf16) {
          uint16_t h;
          std::memcpy(&h, p, 2);
          m[r * cols + col] = bf16ToF32(h);
        } else {
          std::memcpy(&m[r * cols + col], p, 4);
        }
      }
    return m;
  };
  std::vector<float> A = load(pa, lda, M, K);
  std::vector<float> B = load(pb, ldb, K, N);
  std::vector<float> acc(N);
  for (uint64_t i = 0; i < M; ++i) {
    float* crow = reinterpret_cast<float*>(pc + i * ldc);
    for (uint64_t j = 0; j < N; ++j) {
      float v = 0.0f;
      if (accumulate) std::memcpy(&v, crow + j, 4);
      acc[j] = v;
    }
    for (uint64_t k = 0; k < K; ++k) {
      const float aik = A[i * K + k];
      const float* brow = &B[k * N];
      for (uint64_t j = 0; j < N; ++j) acc[j] += aik * brow[j];
    }
    std::memcpy(crow, acc.data(), N * 4);
  }
  stats_.mxu_macs += M * N * K;
}

void Device::execVpu(Opcode op, uint64_t dst, uint64_t src0, uint64_t src1,
                     uint32_t imm) {
  const uint64_t n = csrs_[isa::CSR_VPU_LEN];
  if (n == 0) trap(SAVA_ERR_EXECUTION, "VPU_LEN must be non-zero");

  int num_srcs = 1;
  uint64_t dst_esz = 4, src_esz = 4, dst_elems = n;
  bool transcendental = false;
  switch (op) {
    case Opcode::V_FILL: num_srcs = 0; break;
    case Opcode::V_ADD: case Opcode::V_SUB: case Opcode::V_MUL: case Opcode::V_MAX:
      num_srcs = 2;
      break;
    case Opcode::V_RSUM: case Opcode::V_RMAX: dst_elems = 1; break;
    case Opcode::V_CVT_BF16: dst_esz = 2; break;
    case Opcode::V_CVT_F32: src_esz = 2; break;
    case Opcode::V_EXP: case Opcode::V_RSQRT: transcendental = true; break;
    default: break;
  }

  checkAlign(dst, dst_esz, "VPU destination");
  uint8_t* pd = sramPtr(dst, dst_elems * dst_esz, "VPU destination");
  std::vector<Range> reads;
  const uint8_t* ps[2] = {nullptr, nullptr};
  const uint64_t srcs[2] = {src0, src1};
  for (int i = 0; i < num_srcs; ++i) {
    checkAlign(srcs[i], src_esz, "VPU source");
    ps[i] = sramPtr(srcs[i], n * src_esz, "VPU source");
    reads.push_back({srcs[i], srcs[i] + n * src_esz});
  }

  uint64_t duration = ceilDiv(n, defaults::kVpuLanes) *
                          (transcendental ? defaults::kVpuTranscendentalSlowdown : 1) +
                      defaults::kVpuLatency;
  issue(Unit::Vpu, 0, duration, 0, std::move(reads),
        {{dst, dst + dst_elems * dst_esz}});

  // Read all sources before writing so in-place operations are well defined.
  auto readF32 = [n](const uint8_t* p) {
    std::vector<float> v(n);
    std::memcpy(v.data(), p, n * 4);
    return v;
  };
  std::vector<float> out(dst_elems);
  const float s = bitsToF32(imm);
  switch (op) {
    case Opcode::V_FILL: std::fill(out.begin(), out.end(), s); break;
    case Opcode::V_ADD: case Opcode::V_SUB: case Opcode::V_MUL: case Opcode::V_MAX: {
      std::vector<float> x = readF32(ps[0]), y = readF32(ps[1]);
      for (uint64_t i = 0; i < n; ++i) {
        switch (op) {
          case Opcode::V_ADD: out[i] = x[i] + y[i]; break;
          case Opcode::V_SUB: out[i] = x[i] - y[i]; break;
          case Opcode::V_MUL: out[i] = x[i] * y[i]; break;
          default: out[i] = std::max(x[i], y[i]); break;
        }
      }
      break;
    }
    case Opcode::V_ADDS: case Opcode::V_MULS: case Opcode::V_EXP: case Opcode::V_RELU:
    case Opcode::V_RSQRT: case Opcode::V_COPY: {
      std::vector<float> x = readF32(ps[0]);
      for (uint64_t i = 0; i < n; ++i) {
        switch (op) {
          case Opcode::V_ADDS: out[i] = x[i] + s; break;
          case Opcode::V_MULS: out[i] = x[i] * s; break;
          case Opcode::V_EXP: out[i] = std::exp(x[i]); break;
          case Opcode::V_RELU: out[i] = std::max(x[i], 0.0f); break;
          case Opcode::V_RSQRT: out[i] = 1.0f / std::sqrt(x[i]); break;
          default: out[i] = x[i]; break;
        }
      }
      break;
    }
    case Opcode::V_RSUM: case Opcode::V_RMAX: {
      std::vector<float> x = readF32(ps[0]);
      float r = x[0];
      for (uint64_t i = 1; i < n; ++i)
        r = (op == Opcode::V_RSUM) ? r + x[i] : std::max(r, x[i]);
      out[0] = r;
      break;
    }
    case Opcode::V_CVT_BF16: {
      std::vector<float> x = readF32(ps[0]);
      std::vector<uint16_t> h(n);
      for (uint64_t i = 0; i < n; ++i) h[i] = f32ToBf16(x[i]);
      std::memcpy(pd, h.data(), n * 2);
      stats_.vpu_elements += n;
      return;
    }
    case Opcode::V_CVT_F32: {
      std::vector<uint16_t> h(n);
      std::memcpy(h.data(), ps[0], n * 2);
      for (uint64_t i = 0; i < n; ++i) out[i] = bf16ToF32(h[i]);
      break;
    }
    default: trap(SAVA_ERR_EXECUTION, "not a VPU instruction");
  }
  std::memcpy(pd, out.data(), dst_elems * 4);
  stats_.vpu_elements += n;
}

void Device::traceLine(uint64_t word, const char* suffix) {
  std::fprintf(trace_, "%10" PRIu64 "  %6" PRIu64 ": %-44s%s\n", now_, pc_,
               disassemble(word, pc_).c_str(), suffix);
}

void Device::step() {
  if (pc_ >= exe_.code.size())
    throw SimError(SAVA_ERR_EXECUTION,
                   "pc=" + std::to_string(pc_) +
                       ": fell off the end of the program (missing halt?)");
  const uint64_t word = exe_.code[pc_];
  const Decoded d = decode(word);
  const isa::InstrInfo* info = isa::lookup(d.opcode);
  if (!info) trap(SAVA_ERR_EXECUTION, "illegal instruction");

  const uint64_t issue_cycle = now_;
  trace_suffix_[0] = '\0';
  uint64_t next_pc = pc_ + 1;
  const int64_t s1 = int64_t(reg(d.rs1)), s2 = int64_t(reg(d.rs2));
  const int64_t imm = d.imm;

  switch (Opcode(d.opcode)) {
    case Opcode::NOP: break;
    case Opcode::HALT:
      waitAll();
      halted_ = true;
      break;
    case Opcode::LI: setReg(d.rd, uint64_t(imm)); break;

    case Opcode::ADD: setReg(d.rd, uint64_t(s1) + uint64_t(s2)); break;
    case Opcode::SUB: setReg(d.rd, uint64_t(s1) - uint64_t(s2)); break;
    case Opcode::MUL: setReg(d.rd, uint64_t(s1) * uint64_t(s2)); break;
    case Opcode::AND: setReg(d.rd, uint64_t(s1 & s2)); break;
    case Opcode::OR: setReg(d.rd, uint64_t(s1 | s2)); break;
    case Opcode::XOR: setReg(d.rd, uint64_t(s1 ^ s2)); break;
    case Opcode::SLL: setReg(d.rd, uint64_t(s1) << (s2 & 63)); break;
    case Opcode::SRL: setReg(d.rd, uint64_t(s1) >> (s2 & 63)); break;
    case Opcode::SLT: setReg(d.rd, s1 < s2 ? 1 : 0); break;
    case Opcode::DIV:
    case Opcode::REM: {
      if (s2 == 0) trap(SAVA_ERR_EXECUTION, "division by zero");
      if (s1 == INT64_MIN && s2 == -1) {
        setReg(d.rd, Opcode(d.opcode) == Opcode::DIV ? uint64_t(s1) : 0);
      } else {
        setReg(d.rd, uint64_t(Opcode(d.opcode) == Opcode::DIV ? s1 / s2 : s1 % s2));
      }
      break;
    }

    case Opcode::ADDI: setReg(d.rd, uint64_t(s1) + uint64_t(imm)); break;
    case Opcode::MULI: setReg(d.rd, uint64_t(s1) * uint64_t(imm)); break;
    case Opcode::SLLI: setReg(d.rd, uint64_t(s1) << (imm & 63)); break;
    case Opcode::SRLI: setReg(d.rd, uint64_t(s1) >> (imm & 63)); break;
    case Opcode::ANDI: setReg(d.rd, uint64_t(s1 & imm)); break;

    case Opcode::BEQ: if (s1 == s2) next_pc = pc_ + imm; break;
    case Opcode::BNE: if (s1 != s2) next_pc = pc_ + imm; break;
    case Opcode::BLT: if (s1 < s2) next_pc = pc_ + imm; break;
    case Opcode::BGE: if (s1 >= s2) next_pc = pc_ + imm; break;
    case Opcode::JMP: next_pc = pc_ + imm; break;

    case Opcode::LW: {
      uint64_t addr = uint64_t(s1 + imm);
      checkAlign(addr, 4, "lw address");
      const uint8_t* p = sramPtr(addr, 4, "lw");
      checkHazards(Unit::Scalar, {{addr, addr + 4}}, {});
      int32_t v;
      std::memcpy(&v, p, 4);
      setReg(d.rd, uint64_t(int64_t(v)));
      break;
    }
    case Opcode::SW: {
      uint64_t addr = uint64_t(s1 + imm);
      checkAlign(addr, 4, "sw address");
      uint8_t* p = sramPtr(addr, 4, "sw");
      checkHazards(Unit::Scalar, {}, {{addr, addr + 4}});
      uint32_t v = uint32_t(s2);
      std::memcpy(p, &v, 4);
      break;
    }

    case Opcode::CSRW: setCsr(d.rd, uint64_t(s1)); break;
    case Opcode::CSRWI: setCsr(d.rd, uint64_t(imm)); break;
    case Opcode::CSRR: setReg(d.rd, csr(d.rs1)); break;

    case Opcode::DMA_LD: execDma(true, uint64_t(s1), uint64_t(s2), uint32_t(d.imm)); break;
    case Opcode::DMA_ST: execDma(false, uint64_t(s1), uint64_t(s2), uint32_t(d.imm)); break;

    case Opcode::MXU_MM:
      execMxu(uint64_t(s1), uint64_t(s2), reg(d.rs3), uint32_t(d.imm));
      break;

    case Opcode::V_FILL: case Opcode::V_ADD: case Opcode::V_SUB: case Opcode::V_MUL:
    case Opcode::V_MAX: case Opcode::V_ADDS: case Opcode::V_MULS: case Opcode::V_EXP:
    case Opcode::V_RELU: case Opcode::V_RSQRT: case Opcode::V_RSUM: case Opcode::V_RMAX:
    case Opcode::V_CVT_BF16: case Opcode::V_CVT_F32: case Opcode::V_COPY:
      execVpu(Opcode(d.opcode), uint64_t(s1), uint64_t(s2), reg(d.rs3), uint32_t(d.imm));
      break;

    case Opcode::WAIT:
      if (uint32_t(d.imm) & ~kWaitAll) trap(SAVA_ERR_EXECUTION, "unknown wait mask bits");
      waitMask(uint32_t(d.imm));
      break;
  }

  stats_.instructions++;
  if (info->unit == Unit::Scalar) stats_.scalar_instructions++;
  if (trace_) {
    const uint64_t saved = now_;
    now_ = issue_cycle;
    traceLine(word, trace_suffix_);
    now_ = saved;
  }
  // Every instruction occupies the scalar core's issue slot for one cycle.
  now_ += 1;
  pc_ = next_pc;
}

}  // namespace sava
