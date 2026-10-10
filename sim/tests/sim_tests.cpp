// Unit tests for the SAVA simulator, written against the public C API.
// Programs are encoded directly here so the tests do not depend on the
// Python assembler.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "../src/numerics.h"
#include "sava/isa_gen.h"
#include "sava/sava_sim.h"

using sava::isa::Opcode;

namespace {

int g_failures = 0;

#define EXPECT(cond)                                                     \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::fprintf(stderr, "  %s:%d: EXPECT(%s) failed\n", __FILE__,     \
                   __LINE__, #cond);                                     \
      ++g_failures;                                                      \
    }                                                                    \
  } while (0)

uint64_t enc(Opcode op, unsigned rd = 0, unsigned rs1 = 0, unsigned rs2 = 0,
             unsigned rs3 = 0, int32_t imm = 0) {
  using namespace sava::isa;
  return (uint64_t(op) << kOpcodeLsb) | (uint64_t(rd) << kRdLsb) |
         (uint64_t(rs1) << kRs1Lsb) | (uint64_t(rs2) << kRs2Lsb) |
         (uint64_t(rs3) << kRs3Lsb) | uint64_t(uint32_t(imm));
}

uint64_t csrwi(unsigned csr, int32_t v) { return enc(Opcode::CSRWI, csr, 0, 0, 0, v); }
uint64_t li(unsigned rd, int32_t v) { return enc(Opcode::LI, rd, 0, 0, 0, v); }

struct Arg {
  uint8_t kind, dtype;
  std::vector<uint32_t> dims;
};

void put32(std::vector<uint8_t>& b, size_t off, uint32_t v) {
  for (int i = 0; i < 4; ++i) b[off + i] = uint8_t(v >> (8 * i));
}

// Builds an .sbin image (see sim/src/executable.h for the layout).
std::vector<uint8_t> makeSbin(const std::vector<uint64_t>& code,
                              const std::vector<Arg>& args) {
  const size_t args_off = 32, code_off = args_off + 48 * args.size();
  const size_t name_off = code_off + 8 * code.size();
  std::vector<uint8_t> b(name_off + 5, 0);
  std::memcpy(b.data(), "SAVA", 4);
  b[4] = 0; b[6] = 1;  // v0.1
  put32(b, 8, uint32_t(code_off));
  put32(b, 12, uint32_t(code.size()));
  put32(b, 16, uint32_t(args_off));
  put32(b, 20, uint32_t(args.size()));
  put32(b, 24, uint32_t(name_off));
  for (size_t i = 0; i < args.size(); ++i) {
    uint8_t* rec = &b[args_off + 48 * i];
    std::snprintf(reinterpret_cast<char*>(rec), 24, "arg%zu", i);
    rec[24] = args[i].kind;
    rec[25] = args[i].dtype;
    rec[26] = uint8_t(args[i].dims.size());
    for (size_t d = 0; d < args[i].dims.size(); ++d) put32(b, args_off + 48 * i + 28 + 4 * d, args[i].dims[d]);
  }
  for (size_t i = 0; i < code.size(); ++i)
    for (int k = 0; k < 8; ++k) b[code_off + 8 * i + k] = uint8_t(code[i] >> (8 * k));
  std::memcpy(&b[name_off], "test", 4);
  return b;
}

struct Fixture {
  Fixture() {
    sava_config cfg;
    sava_config_init_default(&cfg);
    cfg.dram_bytes = 1 << 20;
    cfg.max_instructions = 1000000;
    dev = sava_device_create(&cfg);
  }
  ~Fixture() { sava_device_destroy(dev); }
  sava_status run(const std::vector<uint64_t>& code, const std::vector<Arg>& args,
                  const std::vector<uint64_t>& addrs) {
    auto image = makeSbin(code, args);
    sava_status st = sava_load_executable(dev, image.data(), image.size());
    if (st != SAVA_OK) return st;
    return sava_run(dev, addrs.data(), uint32_t(addrs.size()));
  }
  sava_stats stats() {
    sava_stats s;
    sava_get_stats(dev, &s);
    return s;
  }
  sava_device* dev;
};

// Sum 1..10 in a loop, store through SRAM, DMA the word out.
void testScalarLoop() {
  Fixture fx;
  // x10 = i, x11 = acc, x12 = limit, x13 = sram addr 0
  std::vector<uint64_t> code = {
      li(10, 1), li(11, 0), li(12, 11), li(13, 0),
      enc(Opcode::ADD, 11, 11, 10),        // 4: acc += i
      enc(Opcode::ADDI, 10, 10, 0, 0, 1),  // 5: i += 1
      enc(Opcode::BLT, 0, 10, 12, 0, -2),  // 6: if i < 11 goto 4
      enc(Opcode::SW, 0, 13, 11, 0, 0),    // SRAM[0] = acc
      csrwi(sava::isa::CSR_DMA_ROWS, 1), csrwi(sava::isa::CSR_DMA_ROW_BYTES, 4),
      enc(Opcode::DMA_ST, 0, 13, 1, 0, 0), // DRAM[a0] = SRAM[0]
      enc(Opcode::HALT)};
  EXPECT(fx.run(code, {{1, 2, {1}}}, {0x100}) == SAVA_OK);
  int32_t v = 0;
  sava_dram_read(fx.dev, 0x100, &v, 4);
  EXPECT(v == 55);
}

// 2D strided DMA: copy a 4x3 window of an 8x8 i32 matrix in and back out.
void testDma2d() {
  Fixture fx;
  std::vector<int32_t> src(64), dst(64, -1);
  for (int i = 0; i < 64; ++i) src[i] = i;
  sava_dram_write(fx.dev, 0x1000, src.data(), 256);
  sava_dram_write(fx.dev, 0x2000, dst.data(), 256);
  std::vector<uint64_t> code = {
      csrwi(sava::isa::CSR_DMA_ROWS, 4), csrwi(sava::isa::CSR_DMA_ROW_BYTES, 12),
      csrwi(sava::isa::CSR_DMA_DRAM_STRIDE, 32), csrwi(sava::isa::CSR_DMA_SRAM_STRIDE, 12),
      li(10, 64), enc(Opcode::ADDI, 11, 1, 0, 0, 2 * 32 + 1 * 4),  // window at (2,1)
      enc(Opcode::DMA_LD, 0, 10, 11, 0, 3),
      enc(Opcode::WAIT, 0, 0, 0, 0, 1 << 3),
      enc(Opcode::DMA_ST, 0, 10, 2, 0, 0),
      enc(Opcode::HALT)};
  EXPECT(fx.run(code, {{0, 2, {8, 8}}, {1, 2, {8, 8}}}, {0x1000, 0x2000}) == SAVA_OK);
  sava_dram_read(fx.dev, 0x2000, dst.data(), 256);
  // dst rows 0..3 have stride 32 bytes as well: (DMA_DRAM_STRIDE is shared)
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 3; ++c) EXPECT(dst[r * 8 + c] == src[(2 + r) * 8 + 1 + c]);
  EXPECT(dst[3] == -1);
  auto s = fx.stats();
  EXPECT(s.dma_bytes_in == 48 && s.dma_bytes_out == 48);
}

// C = A x B on the MXU for f32 and bf16 inputs, compared with a reference.
void testMxu(bool bf16) {
  Fixture fx;
  const int M = 5, N = 7, K = 40;  // K > array dim: multiple passes
  const int esz = bf16 ? 2 : 4;
  std::vector<float> A(M * K), B(K * N), C(M * N, 1.0f);
  for (int i = 0; i < M * K; ++i) A[i] = float((i * 7) % 11) - 5.0f;
  for (int i = 0; i < K * N; ++i) B[i] = float((i * 5) % 13) * 0.25f;
  std::vector<uint8_t> a_bytes(M * K * esz), b_bytes(K * N * esz);
  for (int i = 0; i < M * K; ++i) {
    if (bf16) { uint16_t h = sava::f32ToBf16(A[i]); std::memcpy(&a_bytes[2 * i], &h, 2); }
    else std::memcpy(&a_bytes[4 * i], &A[i], 4);
  }
  for (int i = 0; i < K * N; ++i) {
    if (bf16) { uint16_t h = sava::f32ToBf16(B[i]); std::memcpy(&b_bytes[2 * i], &h, 2); }
    else std::memcpy(&b_bytes[4 * i], &B[i], 4);
  }
  sava_dram_write(fx.dev, 0x1000, a_bytes.data(), a_bytes.size());
  sava_dram_write(fx.dev, 0x2000, b_bytes.data(), b_bytes.size());
  sava_dram_write(fx.dev, 0x3000, C.data(), C.size() * 4);

  using namespace sava::isa;
  const int sa = 0, sb = 1024, sc = 4096;
  std::vector<uint64_t> code = {
      li(10, sa), li(11, sb), li(12, sc),
      csrwi(CSR_DMA_ROWS, 1),
      csrwi(CSR_DMA_ROW_BYTES, M * K * esz), enc(Opcode::DMA_LD, 0, 10, 1, 0, 0),
      csrwi(CSR_DMA_ROW_BYTES, K * N * esz), enc(Opcode::DMA_LD, 0, 11, 2, 0, 1),
      csrwi(CSR_DMA_ROW_BYTES, M * N * 4), enc(Opcode::DMA_LD, 0, 12, 3, 0, 2),
      enc(Opcode::WAIT, 0, 0, 0, 0, 0x7),
      csrwi(CSR_MXU_M, M), csrwi(CSR_MXU_N, N), csrwi(CSR_MXU_K, K),
      csrwi(CSR_MXU_LDA, K * esz), csrwi(CSR_MXU_LDB, N * esz), csrwi(CSR_MXU_LDC, N * 4),
      csrwi(CSR_MXU_DTYPE, bf16 ? 1 : 0),
      enc(Opcode::MXU_MM, 0, 12, 10, 11, 1),  // accumulate into C
      enc(Opcode::WAIT, 0, 0, 0, 0, 1 << 16),
      enc(Opcode::DMA_ST, 0, 12, 3, 0, 0),
      enc(Opcode::HALT)};
  sava_status st = fx.run(code, {{0, uint8_t(bf16), {M, K}}, {0, uint8_t(bf16), {K, N}}, {2, 0, {M, N}}},
                          {0x1000, 0x2000, 0x3000});
  if (st != SAVA_OK) std::fprintf(stderr, "  %s\n", sava_device_last_error(fx.dev));
  EXPECT(st == SAVA_OK);
  std::vector<float> out(M * N);
  sava_dram_read(fx.dev, 0x3000, out.data(), out.size() * 4);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      float ref = 1.0f;
      for (int k = 0; k < K; ++k) ref += A[i * K + k] * B[k * N + j];  // values exact in bf16
      EXPECT(std::fabs(out[i * N + j] - ref) < 1e-3f);
    }
  auto s = fx.stats();
  EXPECT(s.mxu_macs == uint64_t(M * N * K));
  // 2 K-passes x 1 N-pass x (32 + M) + 32, times 4 for f32.
  EXPECT(s.mxu_busy_cycles == uint64_t((2 * (32 + M) + 32) * (bf16 ? 1 : 4)));
}

// Reading SRAM that an in-flight DMA writes must be flagged in strict mode.
void testHazardDetection() {
  using namespace sava::isa;
  std::vector<uint64_t> code = {
      li(10, 0), csrwi(CSR_DMA_ROWS, 1), csrwi(CSR_DMA_ROW_BYTES, 64),
      enc(Opcode::DMA_LD, 0, 10, 1, 0, 0),
      csrwi(CSR_VPU_LEN, 16), li(11, 128),
      enc(Opcode::V_COPY, 0, 11, 10),  // reads SRAM[0..64) without waiting
      enc(Opcode::HALT)};
  {
    Fixture fx;
    EXPECT(fx.run(code, {{0, 0, {16}}}, {0x100}) == SAVA_ERR_HAZARD);
    EXPECT(std::strstr(sava_device_last_error(fx.dev), "read-after-write") != nullptr);
  }
  {
    // Same program with the wait inserted is fine.
    code.insert(code.begin() + 4, enc(Opcode::WAIT, 0, 0, 0, 0, 1));
    Fixture fx;
    EXPECT(fx.run(code, {{0, 0, {16}}}, {0x100}) == SAVA_OK);
  }
}

void testVpu() {
  using namespace sava::isa;
  Fixture fx;
  std::vector<float> x = {1, -2, 3, -4, 0.5f, 9, -1, 2};
  sava_dram_write(fx.dev, 0x100, x.data(), 32);
  std::vector<uint64_t> code = {
      li(10, 0), li(11, 64), li(12, 128),
      csrwi(CSR_DMA_ROWS, 1), csrwi(CSR_DMA_ROW_BYTES, 32),
      enc(Opcode::DMA_LD, 0, 10, 1, 0, 0), enc(Opcode::WAIT, 0, 0, 0, 0, 1),
      csrwi(CSR_VPU_LEN, 8),
      enc(Opcode::V_RELU, 0, 11, 10),                                    // y = relu(x)
      enc(Opcode::V_MULS, 0, 11, 11, 0, int32_t(sava::f32ToBits(2.0f))), // y *= 2
      enc(Opcode::V_RSUM, 0, 12, 11),                                    // s = sum(y)
      enc(Opcode::WAIT, 0, 0, 0, 0, 1 << 17),
      csrwi(CSR_DMA_ROW_BYTES, 4), enc(Opcode::DMA_ST, 0, 12, 2, 0, 0),
      enc(Opcode::HALT)};
  EXPECT(fx.run(code, {{0, 0, {8}}, {1, 0, {1}}}, {0x100, 0x200}) == SAVA_OK);
  float s = 0;
  sava_dram_read(fx.dev, 0x200, &s, 4);
  EXPECT(s == 2.0f * (1 + 3 + 0.5f + 9 + 2));
}

void testTiming() {
  using namespace sava::isa;
  Fixture fx;
  std::vector<uint64_t> code = {
      li(10, 0), csrwi(CSR_DMA_ROWS, 64), csrwi(CSR_DMA_ROW_BYTES, 256),
      enc(Opcode::DMA_LD, 0, 10, 1, 0, 0), enc(Opcode::HALT)};
  EXPECT(fx.run(code, {{0, 0, {64, 64}}}, {0x1000}) == SAVA_OK);
  // dma issued at cycle 3: 16 KiB / 64 B per cycle + 200 latency; halt waits.
  auto s = fx.stats();
  EXPECT(s.dma_busy_cycles == 256);
  EXPECT(s.cycles == 3 + 256 + 200 + 1);
}

void testErrors() {
  Fixture fx;
  uint8_t junk[40] = {'N', 'O', 'P', 'E'};
  EXPECT(sava_load_executable(fx.dev, junk, sizeof(junk)) == SAVA_ERR_BAD_EXECUTABLE);
  // Out-of-range SRAM access traps with the pc in the message.
  std::vector<uint64_t> code = {li(10, 1 << 30), enc(Opcode::LW, 11, 10), enc(Opcode::HALT)};
  EXPECT(fx.run(code, {}, {}) == SAVA_ERR_EXECUTION);
  EXPECT(std::strstr(sava_device_last_error(fx.dev), "pc=1") != nullptr);
  // Missing halt.
  EXPECT(fx.run({li(10, 1)}, {}, {}) == SAVA_ERR_EXECUTION);
  // Infinite loop hits the instruction budget.
  EXPECT(fx.run({enc(Opcode::JMP, 0, 0, 0, 0, 0)}, {}, {}) == SAVA_ERR_TIMEOUT);
}

}  // namespace

int main() {
  const std::vector<std::pair<const char*, std::function<void()>>> tests = {
      {"scalar_loop", testScalarLoop},
      {"dma_2d", testDma2d},
      {"mxu_f32", [] { testMxu(false); }},
      {"mxu_bf16", [] { testMxu(true); }},
      {"hazard_detection", testHazardDetection},
      {"vpu", testVpu},
      {"timing", testTiming},
      {"errors", testErrors},
  };
  for (const auto& [name, fn] : tests) {
    int before = g_failures;
    fn();
    std::printf("[%s] %s\n", g_failures == before ? "PASS" : "FAIL", name);
  }
  return g_failures == 0 ? 0 : 1;
}
