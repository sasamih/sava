# SAVA compiler (`savac`): pipeline v0

`savac` compiles a kernel written in MLIR (`linalg` on tensors, which is what
torch-mlir, IREE's ONNX importer or StableHLO lowering produce) into a SAVA
executable. v0 uses **IREE's MLIR toolchain** (`iree-opt` and the
`iree.compiler` Python bindings, both from the `iree-base-compiler` pip
package) as the transformation engine. The SAVA-specific parts are written in
Python, so the whole stack can be read and changed without building LLVM.

```
 matmul.mlir ─► frontend ─► tiling ─► iree-opt ─► iree-opt ─► hoist ─► iree-opt ─► double ─► iree-opt ─► codegen ─► asm ─► .sbin
 (tensors)     (DPS ABI)  (schedules) (tile+     (bufferize) (SRAM   (memref    buffer    (lower)    (Python)  (Python)
                                       promote)               allocs) schedule)  (Python)
```

```sh
python -m sava.compiler examples/matmul/matmul.mlir -o build/matmul.sbin --dump-dir build/matmul -S
```

`--dump-dir` writes every stage as soon as it is produced, plus
`reproduce.sh` with the exact `iree-opt` command lines, so each step can be
re-run and changed by hand. If a later stage fails, for example codegen
rejecting an op, the earlier stages are already on disk.

| File | Produced by | Contents |
|---|---|---|
| `00-input.mlir` | user | linalg on tensors |
| `01-dps.mlir` | `frontend.py` | output turned into a destination memref |
| `02-transform.mlir` | `tiling.py` | tensor-level schedule: tiling, SRAM promotion (editable: `--transform-script`) |
| `03-tiled.mlir` | `iree-opt` | loops + SRAM promotion, still tensors |
| `04-bufferized.mlir` | `iree-opt` | memrefs: DRAM arguments, SRAM `alloc`s inside the loops |
| `04a-hoisted.mlir` | `sram.py` | SRAM `alloc`s moved to the function entry |
| `04b-memref-transform.mlir` | `tiling.py` | memref-level schedule, e.g. double buffering (editable: `--post-bufferize-script`) |
| `04c-scheduled.mlir` | `iree-opt` | memref-level schedule applied |
| `04d-double-buffered.mlir` | `double_buffer.py` | planned matmuls: two SRAM slots, pipelined K loop |
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
  double buffering).

It also decides whether a matmul should be **double buffered**
(`MatmulPlan.double_buffer`): two SRAM slots for the A and B tiles, so the
DMA for K step k+1 overlaps the MXU work of step k. The conditions are:

* the K loop has at least two steps (`k / tk ≥ 2`), otherwise there is
  nothing to prefetch inside it;
* `2·(A + B) + C` fits in SRAM. With the half-SRAM tile budget this always
  holds, but the check is explicit so it survives budget changes.

For example, `matmul2k` (bf16, K = 2048, 16 K steps) is double buffered, while
96×40×24 (K fits in one tile) is not. The decision is recorded and tested
(`test_double_buffer_decision`), but **the transform script does not act on
it yet**. See [roadmap M1](../roadmap.md#m1-make-the-matmul-fast-compiler-only).

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

## Stage 4a: SRAM allocation hoisting ([sram.py](../../python/sava/compiler/sram.py))

After bufferization each promoted tile is a `memref.alloc` *inside* the loop
that uses it. A small Python pass moves every static SRAM allocation to the
function entry, keeping their order:

```mlir
func.func @matmul(...) {
  %alloc   = memref.alloc() alignment = 64 : memref<128x128xf32, 1>    // C
  %alloc_1 = memref.alloc() alignment = 64 : memref<128x128xbf16, 1>   // A
  %alloc_3 = memref.alloc() alignment = 64 : memref<128x128xbf16, 1>   // B
  scf.for ...
```

The generated code does not change, because codegen already gives every alloc
site a fixed SRAM address in program order. What changes is that the
memref-level schedule now sees loop-invariant buffers.
`transform.memref.multibuffer` requires that: the allocation must be defined
outside the loop it multi-buffers.

IREE's own hoisting (`iree-codegen-hoist-statically-bound-allocations`,
`transform.iree.hoist_static_alloc`) is not used because it inserts
`memref.dealloc` at the end of the function. Multibuffering rejects
allocations that have users outside the loop, and SAVA never frees SRAM
anyway.

## Stage 4b/4c: memref-level schedule (`iree-opt`)

```
--transform-preload-library=transform-library-paths=04b-memref-transform.mlir
--transform-interpreter --canonicalize --cse
```

The second transform script runs on buffers instead of tensors. This is where
transformations that reason about *storage* belong, such as double buffering
(M1): two SRAM slots per A/B tile, unrolling the K loop, prefetching. Matmuls
still carry their `sava.op_id`, so a script can find a specific matmul and
reach its buffers through its operands.

The generated script (`tiling.make_post_bufferize_script`) is an empty hook.
It only records, as comments, which matmuls are planned for double
buffering. Pass your own script with `--post-bufferize-script FILE` (`savac`
and `examples/matmul/run.py`).

Syntax note for IREE 3.12: inherent attributes of some transform ops must be
written as properties, e.g. `transform.memref.multibuffer %a <{factor = 2 : i64}>`
and `transform.get_parent_op %mm <{op_name = "scf.for"}>`, while
`transform.loop.unroll %loop factor = 2` uses a keyword. The plain
`{factor = 2}` form is rejected.

## Stage 4d: double buffering ([double_buffer.py](../../python/sava/compiler/double_buffer.py))

For every matmul with `MatmulPlan.double_buffer`, a Python pass rewrites the
K loop into a software-pipelined form with **two SRAM slots** for the A and B
tiles:

```mlir
%alloc_0 = memref.alloc() : memref<2x128x128xbf16, 1>                 // A, two slots
%subview   = memref.subview %alloc_0[0, 0, 0] ... offset: 0             // slot 0 (static)
%subview_1 = memref.subview %alloc_0[1, 0, 0] ... offset: 16384         // slot 1 (static)
...
memref.copy A[k=0] -> slot 0                                           // prologue
scf.for %j = 0 to 1792 step 256 {                                      // unrolled by 2
  memref.copy A[j+128] -> slot 1                                       // prefetch ...
  linalg.matmul ins(slot 0) ...                                        // ... overlaps this
  memref.copy A[j+256] -> slot 0
  linalg.matmul ins(slot 1) ...
}
memref.copy A[1920] -> slot 1                                          // epilogue
linalg.matmul ins(slot 0) ...
linalg.matmul ins(slot 1) ...
```

* **Two slots**: the DMA filling the next tile must not overwrite the tile the
  MXU is reading.
* **Unrolled by 2**: every access names a constant slot. `wait` takes an
  immediate mask, so codegen must know statically which DMA filled which
  slot. This is also why the plan requires an even number of K steps.
* **Prefetch before the matmul**: `wait mxu` waits for *all* MXU commands. If
  the DMA for step k were issued after the matmul of step k-1, it would have
  to wait for the MXU and the overlap would be lost.

The pass does not insert any waits. Overlap follows from codegen's generic
synchronization (stage 6) applied to this structure. MLIR's generic tools
were tried first: `transform.memref.multibuffer` picks the slot at run time
(`(k floordiv 128) mod 2`), and `transform.loop.pipeline` moves the address
computation instead of the copies and introduces memref `iter_args`. IREE's
GPU pipeline also uses dedicated passes for this
(`iree-codegen-gpu-multi-buffering`, `iree-codegen-gpu-pipelining`).

If a matmul does not match the expected pattern, for example because a custom
schedule already unrolled its loop, it is skipped and the reason is written
as a comment at the top of `04d-double-buffered.mlir`. `--no-double-buffer`
(`savac`, `run.py`) or `compile_mlir(..., double_buffering=False)` skips the
stage, which is useful for measuring its effect.

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
* **Synchronization insertion.** The codegen tracks un-waited commands: their
  unit, DMA tag, and the **byte ranges** they read and write (SRAM views with
  a constant offset are tracked exactly; dynamic offsets fall back to the
  whole allocation). Before issuing a command it emits the smallest `wait`
  that resolves every conflict with another unit (RAW, WAR or WAW). Because
  ranges are exact, the two slots of a double-buffered tile do not conflict
  with each other.
* **Loop-carried dependencies.** Loops are not fenced. While generating a
  body the codegen records every access. At the end of the body it waits only
  for in-flight commands that conflict with one of those accesses, because the
  next iteration would repeat them. Waits needed inside one iteration are
  already in place. Loops that may run zero times are still fenced at entry,
  because their in-body waits might not execute.
* **CSR caching.** `csrwi` is skipped when the value is already known. The
  cache is cleared at labels, where several control-flow paths meet.

The assembler ([`sava/asm.py`](../../python/sava/asm.py)) turns the text into
an `.sbin`. The simulator's hazard checker then acts as an independent check
that the inserted synchronization is right.

## Results and what to improve

`python examples/matmul/run.py [--mlir FILE] [--no-double-buffer]`:

| Kernel | fenced loops (M0) | precise sync, no double buffering | double buffered |
|---|---|---|---|
| `matmul.mlir`, 256x512x256 f32 | 211 307 cycles, MXU 78.5% | 207 183 | **180 033, MXU 92.1%** |
| `matmul2k.mlir`, 1024x2048x1024 bf16 | 4 075 507, MXU 65.1% | 4 009 523 | **2 834 171, MXU 93.7%** |

The compiled `matmul2k` kernel is within 0.3% of the hand-written reference
`examples/matmul/matmul2k_db_reference.s` (2 825 466 cycles). Peak f32 is
512 GFLOP/s and peak bf16 is 2048 GFLOP/s. The remaining gap comes from
per-C-tile overhead (`v.fill`, the prologue DMA and the `dma.st` of C, none of
which overlap the MXU yet) and from f32 inputs. See [roadmap](../roadmap.md)
M1.

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
  matmuls. *Chained* matmuls, where one result feeds another, work while the
  intermediate result fits in a single tile: it stays in SRAM and is handed
  over with `v.copy`. A larger intermediate is tiled and needs a DRAM scratch
  buffer, which codegen rejects ("DRAM allocations inside kernels") until M2.
* Static shapes, and tile sizes must divide the problem (no padding or peeling).
* No `iter_args` loops and no `scf.if`.
* Double buffering covers the K loop only (an even number of K steps). The
  next C tile's prologue and the `dma.st` of the previous C tile are not
  overlapped yet.
