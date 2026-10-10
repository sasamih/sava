# SAVA v0: Architecture Specification

SAVA (*Systolic Accelerator for Vector & Attention*, named after the river) is
a fictional inference accelerator for transformer/LLM workloads. It exists so
we can practise building a real compiler stack (MLIR/IREE → binary → device)
against a well-defined target. This document is the hardware contract. The
instruction-level details are in [sava-v0-isa.md](sava-v0-isa.md), and the
machine-readable form is in [`arch/`](../../arch).

| Item | Value |
|---|---|
| Version | v0 (ISA 0.1) |
| Simulated | 1 core (the chip has 4) |
| Machine-readable spec | [`arch/isa.json`](../../arch/isa.json), [`arch/sava-v0.json`](../../arch/sava-v0.json) |
| Reference model | [`sim/`](../../sim) (C++ functional + approximately-timed simulator) |

---

## 1. Design goals

1. **LLM inference first.** The workloads are GEMM-heavy prefill and
   GEMV-heavy, memory-bound decode, plus element-wise and reduction ops
   (softmax, norms, activations). So the hardware pairs a large matrix engine
   with a smaller vector engine, and uses bf16 with f32 accumulation.
2. **The compiler is in charge.** As on Groq, Tenstorrent and d-Matrix
   Corsair, the hardware has no caches and no automatic dependency tracking
   between units. Data movement (DMA), on-chip memory allocation and
   synchronization are explicit, so the compiler has real work to do and that
   work can be measured.
3. **Coarse-grained, tile-level instructions.** One instruction moves a 2D tile
   or multiplies two tiles. Kernels are short loops of large operations. This
   is the level MLIR's `linalg`/`scf`/`memref` lowering naturally reaches.
4. **Simple enough to simulate exactly.** Every architectural effect is
   deterministic, and the timing model is a handful of formulas
   (section 6).

## 2. System overview

```
 Host (x86, runtime)                      SAVA card
+--------------------+   PCIe   +-----------------------------------------------+
| application        |<-------->|  Device DRAM (LPDDR; 256 MiB in simulator)    |
| SAVA runtime       |          |        ^            ^            ^            |
| (Python today,     |          |        | NoC (v1)   |            |            |
|  IREE HAL later)   |          |  +-----+----+ +-----+----+ +-----+----+ ...   |
+--------------------+          |  |  core 0  | |  core 1  | |  core 2  |  x4   |
                                |  +----------+ +----------+ +----------+       |
                                +-----------------------------------------------+
```

* The **host** allocates device DRAM, uploads weights and activations, loads
  an executable (`.sbin`) and launches it with buffer addresses as arguments.
* A **chip** has 4 identical **cores** that share DRAM. The v0 simulator models
  a single core. `CORE_ID` and `NUM_CORES` already exist so that multi-core
  code can be written against the ISA today.

## 3. Core microarchitecture

```
                 +--------------------------------------------------------+
                 |                     SAVA core                          |
   instruction   |  +-----------------+                                   |
   memory ------>|  |  Scalar control |  issue (1 instr / cycle)          |
                 |  |  unit (SCU)     |-----+-----------+-----------+     |
                 |  |  32 x 64-bit GPR|     |           |           |     |
                 |  |  CSRs           |     v           v           v     |
                 |  +--------+--------+  [DMA q:8]  [MXU q:4]   [VPU q:4]  |
                 |           | lw/sw      |           |           |       |
                 |           v            v           v           v       |
                 |  +---------------------------------------------------+ |
                 |  |          SRAM scratchpad  (1 MiB, software managed)| |
                 |  +---------------------------------------------------+ |
                 |                        ^                               |
                 +------------------------|-------------------------------+
                                          v
                                   device DRAM (64 B/cycle, 200 cycles latency)
```

### 3.1 Scalar control unit (SCU)

* In-order, single-issue, one instruction per cycle. All instructions,
  including the commands for the other units, are fetched and issued here.
* 32 general-purpose 64-bit registers. `x0` always reads as zero. `x1..x8`
  (`a0..a7`) carry kernel arguments.
* Integer ALU, compare-and-branch, 32-bit loads and stores to SRAM (`lw`/`sw`),
  and CSR access.
* No data cache and no direct DRAM access. All DRAM traffic goes through the
  DMA engine.

### 3.2 Asynchronous units and command queues

The DMA engine, the MXU and the VPU each have an in-order command queue. When
the SCU issues a command it

1. **samples the configuration CSRs** (shape, strides, data type) at issue
   time, so the CSRs can be reprogrammed right after the command is issued;
2. pushes the command into the unit's queue, or stalls if the queue is full;
3. moves on to the next instruction without waiting for the command to
   complete.

Commands within one unit execute and complete in order. **There is no
ordering between different units.** The program must use `wait` to order
them (section 5).

### 3.3 DMA engine

* Moves 2D strided blocks between DRAM and SRAM: `DMA_ROWS` rows of
  `DMA_ROW_BYTES` bytes, with independent row strides on both sides.
* Does not interpret data types. Any byte alignment is allowed.
* Each transfer carries a **tag** (0-15) used for selective waiting.
* Bandwidth: 64 B/cycle. Latency: 200 cycles. Consecutive transfers overlap
  their latency but share the bandwidth.

### 3.4 Matrix unit (MXU)

* A 32x32 **weight-stationary systolic array** with f32 accumulators.
* `mxu.mm C, A, B` computes `C = A·B` or `C += A·B` on SRAM operands. `A` is
  MxK, `B` is KxN, and `C` is MxN in f32. Each operand has its own row stride.
* The input type is set by `MXU_DTYPE`: **bf16** is native. **f32** is
  supported at 1/4 throughput (multi-pass), which mainly helps bring-up and
  testing.
* Per-instruction limits: `M, N, K ≤ 128`. The MXU sequencer splits the
  operation into 32x32 blocks of `B` internally. Larger problems must be tiled
  by the compiler.
* Peak: 32·32 = 1024 MAC/cycle = **2.05 TFLOP/s bf16 per core** at 1 GHz
  (8.2 TFLOP/s per chip).

### 3.5 Vector unit (VPU)

* 16 f32 lanes, operating on contiguous vectors in SRAM. The length comes
  from `VPU_LEN`.
* Element-wise arithmetic, scalar-broadcast forms, `exp`, `rsqrt`, `relu`,
  full reductions (sum and max), and f32↔bf16 conversion.
* Used for accumulator initialisation, activation functions, normalisation and
  softmax building blocks, and data-type conversion.

### 3.6 SRAM scratchpad

* 1 MiB per core, byte-addressed, in a separate address space from DRAM.
  Address 0 is the first SRAM byte.
* Fully software managed: no caching and no coherence. The compiler decides
  where every tile lives.
* Bank conflicts and port limits are **not modeled** in v0.
* Alignment: 4 bytes for f32 operands and 2 bytes for bf16. The compiler uses
  64-byte alignment.

## 4. Address spaces

| Space | Size (v0 sim) | Accessed by | Addressing |
|---|---|---|---|
| DRAM | 256 MiB | DMA only (and the host) | byte address, 64-bit |
| SRAM | 1 MiB per core | DMA (one side), MXU, VPU, `lw`/`sw` | byte address |
| Instruction memory | per kernel | SCU fetch | instruction index (pc) |
| CSRs | 32 entries | `csrw`/`csrwi`/`csrr` | CSR id |

Any access outside a space traps (`SAVA_ERR_EXECUTION`). The error message
includes the pc and the disassembled instruction.

## 5. Synchronization model

* `wait MASK` stalls the SCU until every outstanding command selected by
  `MASK` has completed. Bits 0-15 select DMA tags, bit 16 selects the MXU and
  bit 17 selects the VPU. `fence` is shorthand for `wait all`.
* `halt` implicitly waits for all units before the kernel finishes.
* **Contract:** if two commands on *different* units touch overlapping SRAM
  bytes and at least one of them writes, the later command may only be issued
  after a `wait` that covers the earlier one. This covers RAW, WAR and WAW.
  Scalar `lw`/`sw` count as a separate unit.
* Commands on the *same* unit need no synchronization, because they run in
  order.

**Hazard checking (simulator).** The simulator applies each command's
functional effect when the command is issued, in program order. That matches
the hardware only for programs that obey the contract above. In strict mode
(the default), the simulator tracks the SRAM footprint of every un-waited
command and traps with `SAVA_ERR_HAZARD` on the first violation, naming both
instructions. This makes missing synchronization a deterministic compile-time
bug instead of a flaky numerical error. Strict mode is deliberately
conservative: a command that has finished in simulated time but was never
waited for still counts as un-synchronized.

## 6. Timing model

Time is counted in core cycles at 1 GHz. The SCU issues one instruction per
cycle. Each async unit `u` keeps a `free_at(u)` timestamp:

```
start(cmd)  = max(issue_cycle, free_at(u))
free_at(u)  = start + duration(cmd)
done(cmd)   = free_at(u) + extra_latency(cmd)
```

| Command | duration | extra_latency |
|---|---|---|
| `dma.ld` / `dma.st` | `max(1, ceil(rows·row_bytes / 64))` | 200 |
| `mxu.mm` | `(ceil(K/32)·ceil(N/32)·(32 + M) + 32) · (4 if f32 else 1)` | 0 |
| `v.*` | `ceil(len/16) · (4 if exp/rsqrt else 1) + 8` | 0 |

* A full queue (DMA 8, MXU 4, VPU 4) stalls the SCU until the oldest entry
  completes.
* `wait` advances the SCU to the latest `done` among the selected commands.
* The MXU formula models weight-stationary operation: load a 32x32 block of B
  (32 cycles), stream M rows of A through it, and pay one fill/drain per
  instruction.

The simulator reports total cycles, per-unit busy cycles and utilisation,
stall cycles, DMA bytes, MACs and VPU elements (`sava-sim --stats`,
`Stats.report()`).

### Roofline (per core, bf16)

* Compute peak: 2048 FLOP/cycle. DRAM bandwidth: 64 B/cycle.
* **Ridge point: 32 FLOP/byte.** Decode-time GEMV (arithmetic intensity
  ≈ 1 FLOP/byte with bf16 weights) is therefore heavily memory-bound. Large
  prefill GEMMs are compute-bound once tiles are reused enough. This is the
  same trade-off that SRAM-heavy designs such as Groq and Corsair are built
  around.

## 7. Numerics

| Type | Where | Notes |
|---|---|---|
| f32 | MXU inputs (1/4 rate), MXU accumulators, VPU | IEEE-754 binary32 |
| bf16 | MXU inputs (native), storage | f32→bf16 rounds to nearest-even (`v.cvt.bf16`). NaN maps to a quiet NaN. |
| i32, i8 | storage / DMA only | Compute support is reserved for future versions |

The MXU accumulates in f32, summing over `k` in increasing order. Results
can therefore differ from a reference that reduces in a different order by
normal f32 rounding (about 1e-5 relative for K ≈ 512). Block floating-point /
MX formats, as used by Corsair, are a planned extension.

## 8. Software interface

* **Kernel ABI**: `a0..a7` hold argument values. In v0 these are DRAM base
  addresses of buffers, in the order of the executable's argument table. All
  other registers and CSRs start at zero. See
  [ISA: kernel ABI](sava-v0-isa.md#kernel-abi).
* **Executable format**: `.sbin`, see
  [ISA: executable format](sava-v0-isa.md#executable-format).
* **Driver API**: [`sim/include/sava/sava_sim.h`](../../sim/include/sava/sava_sim.h)
  has device create/destroy, DRAM read/write, load, run and stats. A real
  PCIe driver would expose the same operations. An IREE HAL driver will be
  built on it.

## 9. Known simplifications in v0

These are deliberate, and each one is a candidate for a later iteration:

* Single core. No NoC, no inter-core synchronization.
* The functional effect of a command is applied when it is issued. This is
  only valid with the hazard checker enabled.
* No SRAM banking, no DRAM page or bank effects, and DRAM bandwidth is not
  shared with other cores.
* VPU operations are 1D and contiguous only (no strided or row-wise ops yet).
* No interrupts, no exceptions visible to software, and no instruction cache.
  The whole program sits in instruction memory.
* Integer and low-precision (i8, MX) compute is not implemented.

## 10. Version history

| Version | Changes |
|---|---|
| v0 / ISA 0.1 | Initial definition: SCU, DMA, MXU (f32/bf16), VPU (1D f32), strict hazard model, single-core simulator. |
