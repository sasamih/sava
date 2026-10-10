# SAVA compiler (`savac`): pipeline v0

`savac` compiles a kernel written in MLIR (`linalg` on tensors, which is what
torch-mlir, IREE's ONNX importer or StableHLO lowering produce) into a SAVA
executable. v0 uses **IREE's MLIR toolchain** (`iree-opt` and the
`iree.compiler` Python bindings, both from the `iree-base-compiler` pip
package) as the transformation engine. The SAVA-specific parts are written in
Python, so the whole stack can be read and changed without building LLVM.

```
 matmul.mlir ──► frontend ──► tiling ──► iree-opt ──► iree-opt ──► iree-opt ──► codegen ──► asm ──► .sbin
 (tensors)       (DPS ABI)   (schedule)  (tile+promote) (bufferize) (lower)     (Python)    (Python)
```

```sh
python -m sava.compiler examples/matmul/matmul.mlir -o build/matmul.sbin --dump-dir build/matmul -S
```

`--dump-dir` writes every stage, plus `reproduce.sh` with the exact
`iree-opt` command lines, so each step can be re-run and changed by hand.

| File | Produced by | Contents |
|---|---|---|
| `00-input.mlir` | user | linalg on tensors |
| `01-dps.mlir` | `frontend.py` | output turned into a destination memref |
| `02-transform.mlir` | `tiling.py` | transform-dialect schedule (editable) |
| `03-tiled.mlir` | `iree-opt` | loops + SRAM promotion, still tensors |
| `04-bufferized.mlir` | `iree-opt` | memrefs: DRAM arguments, SRAM `alloc`s |
| `05-lowered.mlir` | `iree-opt` | **SAVA-ready IR**: index math, `reinterpret_cast` |
| `06-kernel.s` | `codegen.py` | SAVA assembly |
| `<name>.sbin` | `sava.asm` | executable |

---

## Stage 1: frontend and ABI ([frontend.py](../../python/sava/compiler/frontend.py))

Frontends return results by value. The SAVA ABI (like IREE's dispatch ABI)
passes every buffer, outputs included, as an argument. The frontend rewrites
the function into **destination-passing style** using the Python MLIR
bindings:

```mlir
func.func @matmul(%arg0: tensor<256x512xf32>, %arg1: tensor<512x256xf32>,
                  %arg2: memref<256x256xf32>) {
  ...
  %2 = linalg.matmul ins(%arg0, %arg1 : ...) outs(%1 : tensor<256x256xf32>) -> tensor<256x256xf32>
  bufferization.materialize_in_destination %2 in restrict writable %arg2
      : (tensor<256x256xf32>, memref<256x256xf32>) -> ()
  return
}
```

Together with `--eliminate-empty-tensors` this lets bufferization compute the
result **directly in the output buffer**, with no temporary DRAM allocation
and no final copy. (We tried `--buffer-results-to-out-params` first. It leaves
a temporary plus a `memref.copy` behind, because a top-level function has no
call site to hoist the allocation into.)

## Stage 2: schedule ([tiling.py](../../python/sava/compiler/tiling.py))

For every `linalg.matmul` the compiler picks tile sizes that respect the
hardware:

* `tm ≤ MXU max_m`, `tn ≤ max_n`, `tk ≤ max_k` (128), so each tile is exactly
  one `mxu.mm`;
* the tiles divide the problem (v0 has no remainder handling);
* `A + B + C` tiles fit in half of the SRAM (the other half is reserved for
  double buffering later).

It then generates a **transform dialect** script, which is MLIR's way of
writing a schedule as IR:

```mlir
%mm0_mn, %lm, %ln = transform.structured.fuse %mm0 tile_sizes [128, 128, 0]   // M, N loops + fuse fill
%mm0_fill = transform.get_producer_of_operand %mm0_mn[2]
%mm0_acc  = transform.get_operand %mm0_fill[1]
transform.structured.promote_tensor to 1 %mm0_acc                             // C tile -> SRAM
%mm0_mnk, %lk = transform.structured.tile_using_for %mm0_mn tile_sizes [0, 0, 128]  // K loop
transform.structured.promote_tensor to 1 %mm0_a                               // A tile -> SRAM
transform.structured.promote_tensor to 1 %mm0_b                               // B tile -> SRAM
```

Memory space `1` means SRAM throughout the stack. `--transform-script FILE`
replaces the generated script, which is the easiest way to experiment with
other schedules. The test `test_custom_transform_script` uses this to switch
to 32x64x32 tiles.

## Stage 3: tiling and promotion (`iree-opt`)

```
--eliminate-empty-tensors --canonicalize
--transform-preload-library=transform-library-paths=02-transform.mlir --transform-interpreter
--canonicalize --cse
```

The result is still on tensors. `bufferization.alloc_tensor` with
`memory_space = 1` marks the future SRAM buffers, and `materialize_in_destination`
marks the copies into them:

```mlir
%3 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<128x128xf32>
%4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<128x128xf32>) -> tensor<128x128xf32>
%5 = scf.for %arg7 = %c0 to %c512 step %c128 iter_args(%arg8 = %4) -> (tensor<128x128xf32>) {
  %6 = bufferization.alloc_tensor() <{memory_space = 1 : i64}> : tensor<128x128xf32>
  %7 = bufferization.materialize_in_destination %extracted_slice_1 in %6 : ...
  ...
  %10 = linalg.matmul ins(%7, %9 : ...) outs(%arg8 : tensor<128x128xf32>) -> tensor<128x128xf32>
```

## Stage 4: bufferization (`iree-opt`)

```
--one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map"
--canonicalize --cse
```

Tensors become memrefs. Function arguments are DRAM buffers (default memory
space). `alloc_tensor` becomes `memref.alloc` in space 1, and
`materialize_in_destination` becomes `memref.copy`.

## Stage 5: lowering to SAVA-ready IR (`iree-opt`)

```
--expand-strided-metadata --lower-affine --loop-invariant-code-motion --canonicalize --cse
```

`memref.subview` chains become `memref.reinterpret_cast` of the base buffer
with an explicitly computed element offset, so all address arithmetic is
ordinary `arith` on `index`. LICM hoists the invariant parts out of the inner
loop. The result maps almost one-to-one onto the hardware:

```mlir
scf.for %arg3 = %c0 to %c256 step %c128 {                 // M tiles
  %0 = arith.muli %arg3, %c512 overflow<nsw> : index
  %1 = arith.muli %arg3, %c256 overflow<nsw> : index
  scf.for %arg4 = %c0 to %c256 step %c128 {               // N tiles
    %alloc = memref.alloc() alignment = 64 : memref<128x128xf32, 1>            // C in SRAM
    linalg.fill ins(%cst : f32) outs(%alloc : memref<128x128xf32, 1>)          // -> v.fill
    scf.for %arg5 = %c0 to %c512 step %c128 {             // K tiles
      %3 = arith.addi %0, %arg5 : index
      %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%3], sizes: [128, 128], strides: [512, 1] ...
      %alloc_1 = memref.alloc() alignment = 64 : memref<128x128xf32, 1>
      memref.copy %reinterpret_cast_0, %alloc_1 ...                            // -> dma.ld
      ...
      linalg.matmul ins(%alloc_1, %alloc_3 : ...) outs(%alloc : ...)           // -> mxu.mm
    }
    ...
    memref.copy %alloc, %reinterpret_cast ...                                  // -> dma.st
  }
}
```

### The SAVA-ready IR contract

This is the interface between the "MLIR part" and the "backend part". Any
schedule (generated or hand-written) must lower to these ops:

| Op | Constraint | Lowered to |
|---|---|---|
| `func.func` | ≤ 8 identity-layout DRAM memref args, static shapes | `.arg` table, `a0..a7` |
| `arith.constant/addi/subi/muli/divsi/remsi/index_cast` | index/integer | constant folding, `li/addi/add/slli/muli/...` |
| `scf.for` | constant positive step, no `iter_args` | counted loop, `blt` |
| `memref.alloc` | memory space 1, static shape | static SRAM address |
| `memref.reinterpret_cast`, `memref.extract_strided_metadata` | static strides | address arithmetic |
| `memref.copy` | DRAM↔SRAM, 2D-expressible, unit inner stride | `dma.ld` / `dma.st` |
| `memref.copy` | SRAM↔SRAM, contiguous 32-bit | `v.copy` |
| `linalg.fill` | contiguous f32 SRAM, constant value | `v.fill` |
| `linalg.matmul` | SRAM operands, row-major, within MXU limits | `mxu.mm` (accumulate) |
| `memref.dealloc` | | ignored |
| `func.return` | | `halt` |

## Stage 6: code generation ([codegen.py](../../python/sava/compiler/codegen.py))

The codegen walks the SAVA-ready IR once and does what a production backend
would split into separate passes:

* **Memory planning.** Each `memref.alloc` site gets a fixed SRAM address
  from a bump allocator. This is correct because a given alloc site's
  instances never live at the same time in this IR. Overflowing SRAM is a
  compile error.
* **Register allocation.** Scoped allocation from `x9..x31`. Values defined in
  a loop body are freed at the end of that body. There is no spilling, so
  running out of registers is a compile error.
* **Instruction selection** follows the table above. Index constants are
  folded and only materialized when needed. Multiplication by a power of two
  becomes a shift.
* **Synchronization insertion.** The codegen tracks un-waited commands (unit,
  tag, SRAM buffers read and written). Before issuing a command it emits the
  smallest `wait` that resolves every conflict with another unit (RAW, WAR or
  WAW). **Loops are fenced** at entry and at the end of every iteration. That
  is always correct but serializes the pipeline (see the stats below and
  [roadmap](../roadmap.md) M1).
* **CSR caching.** `csrwi` is skipped when the value is already known. The
  cache is cleared at labels, where several control-flow paths meet.

The assembler ([`sava/asm.py`](../../python/sava/asm.py)) turns the text into
an `.sbin`. The simulator's hazard checker then acts as an independent check
that the inserted synchronization is right.

## Results and what to improve

`python examples/matmul/run.py` for 256x512 · 512x256 in f32:

```
cycles              211307  (211.31 us @ 1.0 GHz)
dma  ops     36  util  17.4%
mxu  ops     16  util  78.5%  macs 33554432  (317.6 GFLOP/s)
```

Peak f32 is 512 GFLOP/s and peak bf16 is 2048 GFLOP/s. The gap comes from
fully serialized DMA→MXU→DMA iterations and from f32 inputs. Both are
roadmap items and good exercises.

## Relation to IREE, and where this is going

| v0 (this repo) | Full IREE integration (roadmap M4) |
|---|---|
| One function = one kernel | IREE partitions the model into dispatches (`flow`/`stream`) |
| `frontend.py` DPS rewrite | `iree_tensor_ext.dispatch.tensor.load/store` on HAL bindings |
| `tiling.py` + transform script | the same transform script, run from a `sava` codegen pipeline |
| `codegen.py` (Python) | a C++ `sava` dialect + conversion passes in an IREE compiler plugin |
| `.sbin` | `hal.executable` variant serialized by a `sava` HAL target backend |
| `sava.runtime` (ctypes) | an IREE HAL driver on top of `sava_sim.h`; `iree-run-module --device=sava` |

The MLIR-side work (schedule, bufferization, the SAVA-ready contract) carries
over unchanged. What changes is who calls it and where the result is
packaged.

## Known limitations (v0)

* Only `linalg.fill` + `linalg.matmul` kernels (f32 or bf16 inputs, f32
  output), with no element-wise epilogues yet. A function may contain several
  *independent* matmuls (each with its own output). *Chained* matmuls, where
  one result feeds another, need a DRAM scratch area for the intermediate
  tensor and are rejected by codegen ("DRAM allocations inside kernels")
  until M2.
* Static shapes, and tile sizes must divide the problem (no padding or peeling).
* No `iter_args` loops and no `scf.if`.
* Synchronization is conservative (fenced loops). There is no double buffering.
