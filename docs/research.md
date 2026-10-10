# Background research: LLM accelerators and IREE integration

This note records why SAVA looks the way it does. It surveys a few real
inference architectures, notes what SAVA borrows from each, and compares the
options for connecting a new accelerator to IREE.

## 1. What LLM inference asks of hardware

* **Prefill** (processing the prompt) is dominated by large GEMMs and is
  usually compute-bound.
* **Decode** (generating one token at a time) is dominated by GEMV with the
  weight matrices: each weight byte is used about once per token. It is
  memory-bandwidth-bound, so the time to stream the weights (and the KV
  cache) sets the token rate.
* The remaining work is element-wise or reductions: softmax, RMSNorm/LayerNorm,
  activations, RoPE. Its FLOP count is small, but it must not cost extra
  round-trips to memory, so fusing it with neighbouring ops matters.

That is why recent designs push on-chip SRAM capacity and bandwidth, use low
precision (bf16, fp8, block floating point), and let the compiler schedule
data movement explicitly.

## 2. Architectures surveyed

### d-Matrix Corsair
* PCIe inference card built on *Digital In-Memory Compute* (DIMC): SRAM
  arrays that also perform the matrix-vector multiplications. It uses
  chiplets connected by a die-to-die link (DMX Link).
* About 2 GB of on-chip SRAM ("performance memory", ~150 TB/s claimed) plus
  up to 256 GB of LPDDR5 "capacity memory" (~400 GB/s per card).
* Block floating-point / MX-style formats instead of IEEE FP16/BF16 compute.
  9.6 PFLOPS FP4 is quoted per card.
* **Taken for SAVA:** two memory tiers (large DRAM, SRAM where the work
  happens) with explicit, compiler-managed movement between them. Block
  floating point is a planned numeric extension.

### Google TPU
* Large weight-stationary systolic matrix units (MXU), a vector unit, and
  software-managed on-chip memory, compiled through XLA.
* **Taken for SAVA:** the weight-stationary MXU and its timing model (load a
  block of B, stream rows of A), bf16 inputs with f32 accumulation, and
  coarse tile-level instructions.

### Groq LPU (TSP)
* Deterministic, compiler-scheduled design. Hardware interlocks are removed
  and the compiler places every instruction in time. Hundreds of MB of SRAM
  serve as primary weight storage.
* **Taken for SAVA:** "the compiler owns synchronization". SAVA keeps simple
  `wait` instructions instead of full static scheduling, but makes missing
  synchronization a hard error in the simulator.

### Tenstorrent Tensix
* Each Tensix core pairs five small RISC-V cores (two for data movement over
  the NoC, three driving unpack/math/pack) with a matrix engine built around
  32x32 tiles, a SIMD vector unit and about 1-1.5 MB of local SRAM. Cores
  communicate over two NoCs.
* **Taken for SAVA:** a scalar control core issuing to separate data-movement
  and compute engines, 32x32 granularity, about 1 MiB of SRAM per core, and a
  multi-core chip (planned).

### Berkeley Gemmini
* Open-source systolic-array generator driven by RISC-V custom instructions:
  `config` instructions set strides and shapes, and `mvin`/`mvout`/`compute`
  move and use data.
* **Taken for SAVA:** configuration CSRs that the async commands sample when
  they are issued.

### AMD AI Engine (via iree-amd-aie)
* VLIW/SIMD tiles with local memory, programmed through an out-of-tree IREE
  compiler plugin plus a runtime driver.
* **Relevant as a template:** the best public example of a full IREE
  integration for a non-GPU accelerator (see §3).

## 3. Connecting a new accelerator to IREE

IREE separates a **compiler** (MLIR, which partitions models into *dispatches*
and lowers each one for a target) from a **runtime** whose HAL (Hardware
Abstraction Layer) hides devices behind one API (buffers, command buffers,
executables, semaphores). Built-in targets include `llvm-cpu`, `vmvx`,
`vulkan-spirv`, `rocm`, `cuda` and `metal-spirv`. New targets are added as
compiler **plugins** (`compiler/plugins/target/*` in-tree, or out-of-tree via
`-DIREE_CMAKE_PLUGIN_PATHS`, as iree-amd-aie does) together with a HAL
**driver**.

| Option | Effort | What you learn | Used in |
|---|---|---|---|
| **A. Stand-alone MLIR pipeline using IREE's tools** (`iree-opt` + Python bindings, custom backend outside IREE) | low, no LLVM build | linalg tiling, transform dialect, bufferization, lowering, codegen | **v0** |
| **B. IREE as the partitioner**: `iree-compile --compile-to=executable-sources`, compile each dispatch with `savac`, drive dispatches from a host program | medium | IREE's flow/stream model, dispatch ABI, host scheduling | M4a |
| **C. Full IREE target**: compiler plugin (HAL target backend + codegen pipeline + `sava` dialect in C++) and HAL driver over `sava_sim.h` | high, builds IREE from source | the real production architecture; `iree-run-module --device=sava` | M4b |
| **D. Custom dispatch / microkernels**: hand-written kernels linked into an existing target | low | IREE extension points | not planned |

The IREE CUDA backend post-mortem recommends doing the runtime (HAL) side
first, because the HAL can be tested with its CTS tests before any compiler
exists. Our C API ([`sava_sim.h`](../../sim/include/sava/sava_sim.h)) was
designed with that in mind: device memory, executable load and launch map
directly onto HAL buffers, executables and dispatch.

### Why start with option A
* The IREE pip packages (`iree-base-compiler`, `iree-base-runtime`) provide
  `iree-opt`, `iree-compile` and the MLIR Python bindings, with no
  multi-hour LLVM/IREE build.
* The interesting compiler work (schedules, memory spaces, bufferization,
  synchronization, instruction selection) is the same in all options. The
  SAVA-ready IR contract ([pipeline.md](compiler/pipeline.md)) is designed so
  that moving to options B and C changes the packaging, not the
  transformations.
* IREE's own `llvm-cpu` backend serves as the numerical reference
  (`examples/matmul/run.py`).

## 4. Key SAVA design decisions

| Decision | Alternatives considered | Reason |
|---|---|---|
| Coarse tile instructions (DMA 2D, MXU tile, VPU vector) | fine-grained VLIW, as in AIE or Groq | matches what MLIR lowering naturally produces; keeps kernels short and the simulator fast |
| Explicit `wait` + strict hazard checker | hardware scoreboarding / semaphores | makes synchronization a compiler problem that is visible in the stats, and catches bugs deterministically |
| Configuration CSRs sampled at issue | wide instructions with every operand | 64-bit fixed encoding; teaches state tracking (CSR caching) in codegen |
| Separate SRAM address space, memory space `1` in MLIR | unified address space | maps directly to `memref` memory spaces and `promote_tensor to 1` |
| bf16 native, f32 at 1/4 rate | f32 only | realistic for LLMs, and leaves room for a mixed-precision pass |
| JSON ISA spec → generated C++ header | hand-written in both C++ and Python | single source of truth, like TableGen |
| C API (`sava_sim.h`) as the device interface | Python-only simulator | the same API can back a C/C++ IREE HAL driver later |

## Sources

* d-Matrix: [Corsair product brief](https://d-matrix.ai/pdf/d-Matrix-Corsair-Product-Brief-FINAL.pdf),
  [Chips and Cheese: Corsair, 256GB of LPDDR for AI models](https://chipsandcheese.com/p/d-matrix-corsair-256gb-of-lpddr-for),
  [SiliconANGLE coverage (Nov 2025)](https://siliconangle.com/2025/11/12/chip-startup-d-matrix-raises-275m-speed-inference-memory-compute/)
* Groq: [LPU architecture](https://groq.com/lpu-architecture)
* Tenstorrent: [TT-Metalium guide](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md),
  [Tensix RISC-V programming guide](https://docs.tenstorrent.com/tt-vscode-toolkit/riscv-guide/)
* IREE: [deployment configurations](https://iree.dev/guides/deployment-configurations/),
  [C API (compiler plugins)](https://iree.dev/reference/bindings/c-api/),
  [CUDA backend blog post](https://iree.dev/community/blog/2021-10-15-cuda-backend/),
  [iree-amd-aie (out-of-tree plugin)](https://github.com/nod-ai/iree-amd-aie)
* Gemmini: Genc et al., "Gemmini: Enabling Systematic Deep-Learning
  Architecture Evaluation via Full-Stack Integration", DAC 2021.
