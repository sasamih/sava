# Roadmap

The project grows in milestones. Each one has a hardware/simulator side, a
compiler side, a measurable result (correctness tests and simulator stats),
and the MLIR/IREE concepts it teaches.

## M0: Skeleton (done)

* SAVA v0 spec, ISA JSON → generated C++ header, C++ simulator with C API, CLI
  and unit tests.
* Python assembler, `.sbin` format, ctypes runtime.
* `savac`: DPS frontend → transform-dialect tiling and SRAM promotion →
  bufferization → SAVA-ready IR → codegen → `.sbin`.
* End-to-end matmul checked against NumPy and IREE `llvm-cpu`.

## M1: Make the matmul fast (compiler only)

Baseline: 256x512x256 f32 runs in 211k cycles at 78% MXU utilisation, with
the DMA serialized behind it. The goal is to measure every change with
`run.py` stats.

1. **bf16 compute.** Write a pass (Python bindings or a transform script) that
   inserts f32→bf16 conversion of the matmul inputs. Choose where it happens:
   on the host for weights, or on-chip with `v.cvt.bf16`. Expect about 4x on
   the MXU.
2. **Remove the `v.fill`.** The first K iteration can run `mxu.mm` with
   accumulate=0. This is a pattern rewrite on `linalg.fill` + `linalg.matmul`,
   or a peeled first iteration (`transform.loop.peel`).
3. **Double buffering.** Ping-pong A/B buffers so that the DMA for iteration
   k+1 overlaps the MXU work of iteration k. Two parts: the schedule
   (`transform.structured.multibuffer`/`memref.multi_buffer`, or software
   pipelining) and smarter synchronization in codegen (wait on a specific
   tag instead of fencing the loop).
4. **Better tile selection** for bf16 and for GEMV shapes (M=1, decode).

## M2: Element-wise fusion and the VPU

* ISA 0.2: VPU row-wise ops (`VPU_ROWS`/strides), broadcast of a row vector
  (bias), row reductions. These are prerequisites for softmax and norms.
* Lower element-wise `linalg.generic` (bias add, ReLU/GELU, residual add) to
  VPU code, fused into the matmul tile loop
  (`transform.structured.fuse_into_containing_op`).
* Example: `examples/mlp/` with Linear → GELU → Linear, imported from PyTorch
  through `iree-turbine` (or written in MLIR).
* Intermediate tensors in DRAM: a DRAM scratch arena described in the `.sbin`
  and allocated by the runtime.

## M3: A real MLIR dialect for SAVA (C++)

* Out-of-tree MLIR project (CMake, TableGen ODS) with a `sava` dialect:
  `sava.dma_load`, `sava.dma_store`, `sava.mxu_mm`, `sava.vpu.*`, `sava.wait`,
  and memory-space attributes.
* Move instruction selection into conversion patterns
  (`linalg/memref → sava`). Write synchronization insertion as a proper
  dataflow analysis over the `sava` dialect. Use `mlir-translate`-style
  emission for asm/sbin.
* Build against a pinned LLVM (or IREE's `third_party/llvm-project`) in the dev
  container, with ccache.

## M4: Full IREE integration

* **M4a: IREE as the partitioner.** Compile a multi-op model with
  `iree-compile --compile-to=executable-sources`, compile each dispatch with
  `savac`, and run the dispatch sequence from a host program.
* **M4b: IREE plugin + HAL driver.**
  * Compiler: a `sava` HAL target backend plugin (`-DIREE_CMAKE_PLUGIN_PATHS`),
    a codegen pipeline that reuses the transform script and the `sava`
    dialect, and serialization of the executable variant into `.sbin`.
  * Runtime: a `sava` HAL driver (device, allocator, executable,
    command buffer) on top of `sava_sim.h`. Run models with
    `iree-run-module --device=sava`, and pass the HAL CTS.

## M5: Toward an LLM block

* Multi-core (4 cores): work partitioning across cores (`CORE_ID`), a shared
  DRAM bandwidth model, a NoC, and cross-core synchronization.
* Attention: QK^T, softmax (row max/sum/exp on the VPU), ·V. Optionally a
  flash-attention-style tiled schedule.
* Decode: a KV cache in DRAM and memory-bound GEMV analysis against the
  roofline (ridge point 32 FLOP/B).
* Numerics: int8 and MX/block floating point (Corsair-style) on the MXU.

## Small exercises (any time)

* Make `examples/asm/relu_scale.s` double-buffered by hand and compare cycles.
* Add a `--trace` view that shows the timeline per unit (e.g. Chrome trace
  JSON from the simulator).
* Make tiling handle sizes that do not divide evenly (padding with
  `transform.structured.pad`, or peeling).
* Add an `sava-objdump`-style listing that interleaves MLIR locations with
  instructions (propagate `loc` through codegen).
