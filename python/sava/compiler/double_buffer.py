"""Double buffering of matmul K loops (runs on memref IR, stage 04d).

For every matmul planned with `MatmulPlan.double_buffer`, the K loop

    scf.for %k = lb to ub step s {
      <address computation>            // subviews of A and B for step %k
      memref.copy A[%k] -> %a_tile     // dma.ld
      memref.copy B[%k] -> %b_tile     // dma.ld
      linalg.matmul ins(%a_tile, %b_tile) outs(%c_tile)
    }

is rewritten into a software-pipelined form with two SRAM slots per input:

    %a2 = memref.alloc() : memref<2x...>            // slot 0 / slot 1 are static
    load(k = lb)       -> slot 0                     // prologue
    scf.for %j = lb to ub - 2s step 2s {             // unrolled by 2
      load(k = j + s)  -> slot 1                     // prefetch ...
      matmul(slot 0)                                 // ... overlaps this matmul
      load(k = j + 2s) -> slot 0
      matmul(slot 1)
    }
    load(k = ub - s)   -> slot 1                     // epilogue
    matmul(slot 0)
    matmul(slot 1)

Why each piece:
  * two slots: the DMA writing the next tile must not overwrite the tile the
    MXU is reading;
  * unrolling by 2: every access names a *constant* slot. `wait` takes an
    immediate mask, so the code generator must know statically which DMA
    filled which slot;
  * prefetch before the matmul: `wait mxu` waits for all MXU commands, so the
    DMA for step k+1 has to be issued before the matmul of step k, otherwise
    waiting for the MXU before the DMA would serialize them again.

Overlap is not encoded here: the code generator only inserts the waits the
data dependencies require (codegen.py), and with this structure they allow
the DMA and the MXU to run in parallel.

MLIR's generic tools were evaluated first: `transform.memref.multibuffer`
computes the slot at run time (`(k floordiv s) mod 2`), and
`transform.loop.pipeline` moves the address computation rather than the
copies and introduces memref iter_args. Hence a dedicated pass, as in IREE's
GPU pipeline (iree-codegen-gpu-multi-buffering / gpu-pipelining).
"""

from __future__ import annotations

import math

from iree.compiler import ir
from iree.compiler.dialects import arith, memref, scf

from .tiling import SRAM_SPACE, MatmulPlan


class DoubleBufferError(Exception):
    pass


def _op(x):
    """Operation for an Operation or OpView."""
    return getattr(x, "operation", x)


def _const_index(value: ir.Value) -> int | None:
    owner = value.owner
    op = getattr(owner, "operation", None)
    if op is None or op.name != "arith.constant":
        return None
    return ir.IntegerAttr(op.attributes["value"]).value


def _find_matmul(module: ir.Module, op_id: int):
    found = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    walk(child.operation)
        if op.name == "linalg.matmul" and "sava.op_id" in op.attributes \
                and ir.IntegerAttr(op.attributes["sava.op_id"]).value == op_id:
            found.append(op)

    walk(module.operation)
    return found


def _sram_alloc(value: ir.Value):
    op = getattr(value.owner, "operation", None)
    if op is None or op.name != "memref.alloc":
        return None
    mt = ir.MemRefType(value.type)
    if mt.memory_space is None or ir.IntegerAttr(mt.memory_space).value != SRAM_SPACE:
        return None
    return op


def _check(cond: bool, op_id: int, msg: str):
    if not cond:
        raise DoubleBufferError(f"matmul #{op_id}: cannot double buffer: {msg}")


def _two_slots(alloc_op) -> tuple[ir.Value, ir.Value]:
    """Replaces nothing yet: creates a 2-slot allocation next to `alloc_op` and
    returns static views of slot 0 and slot 1."""
    old = ir.MemRefType(alloc_op.results[0].type)
    shape = list(old.shape)
    space = old.memory_space
    with ir.InsertionPoint(alloc_op), alloc_op.location:
        new_type = ir.MemRefType.get([2] + shape, old.element_type, memory_space=space)
        alloc2 = memref.AllocOp(new_type, [], [], alignment=ir.IntegerAttr.get(
            ir.IntegerType.get_signless(64), 64)).result
        strides = [math.prod(shape[i + 1:]) for i in range(len(shape))]
        views = []
        for slot in range(2):
            view_type = ir.MemRefType.get(
                shape, old.element_type,
                layout=ir.StridedLayoutAttr.get(slot * math.prod(shape), strides),
                memory_space=space)
            views.append(memref.SubViewOp(
                view_type, alloc2, [], [], [],
                [slot] + [0] * len(shape), [1] + shape, [1] * (len(shape) + 1)).result)
    return views[0], views[1]


def _clone(ops, mapping: dict, ip: ir.InsertionPoint):
    """Clones `ops` at `ip`, remapping operands through `mapping` (updated with results)."""
    for op in ops:
        new = op.operation.clone(ip=ip)
        for i, v in enumerate(new.operands):
            if v in mapping:
                new.operands[i] = mapping[v]
        for old_r, new_r in zip(op.results, new.results):
            mapping[old_r] = new_r


def double_buffer_matmul(module: ir.Module, op_id: int):
    found = _find_matmul(module, op_id)
    # Several copies exist e.g. when tiling fused this matmul into a consumer's loop.
    _check(len(found) == 1, op_id, f"expected one matmul with this id, found {len(found)}")
    mm = found[0]
    loop = _op(mm.parent)
    _check(loop.name == "scf.for", op_id, "the matmul is not directly inside its K loop")
    _check(len(loop.operands) == 3, op_id, "K loop has iter_args")
    lb, ub, step = (_const_index(v) for v in loop.operands)
    _check(None not in (lb, ub, step), op_id, "K loop bounds are not constant")
    steps = (ub - lb) // step
    _check(steps >= 2 and steps % 2 == 0, op_id, f"needs an even number >= 2 of K steps, has {steps}")

    body = loop.regions[0].blocks[0]
    iv = body.arguments[0]
    a_val, b_val = mm.operands[0], mm.operands[1]
    a_alloc, b_alloc = _sram_alloc(a_val), _sram_alloc(b_val)
    _check(a_alloc is not None and b_alloc is not None, op_id, "A/B are not SRAM allocations")

    body_ops = [op for op in body.operations if op.operation.name != "scf.yield"]
    loads = [op for op in body_ops if _op(op) != mm]
    for op in loads:
        name = op.operation.name
        _check(name in ("memref.subview", "memref.copy") or name.startswith("arith."), op_id,
               f"unexpected op in the K loop: {name}")
        if name == "memref.copy":
            _check(op.operands[1] in (a_val, b_val), op_id, "copy into an unexpected buffer")
    for val, alloc in ((a_val, a_alloc), (b_val, b_alloc)):
        users = [u.owner for u in val.uses]
        _check(all(_op(u.parent) == loop for u in users), op_id,
               "the A/B tile is used outside the K loop")

    a_slots, b_slots = _two_slots(a_alloc), _two_slots(b_alloc)
    index = ir.IndexType.get()

    def load(k: ir.Value, slot: int, ip):
        _clone(loads, {iv: k, a_val: a_slots[slot], b_val: b_slots[slot]}, ip)

    def matmul(slot: int, ip):
        _clone([mm], {a_val: a_slots[slot], b_val: b_slots[slot]}, ip)

    before = ir.InsertionPoint(loop)
    with loop.location:
        with before:
            c = {v: arith.ConstantOp(index, v).result for v in (lb, ub - 2 * step, 2 * step, ub - step)}
        load(c[lb], 0, before)                                   # prologue
        if steps > 2:
            with before:
                new_loop = scf.ForOp(c[lb], c[ub - 2 * step], c[2 * step])
            inner = ir.InsertionPoint(new_loop.body)
            j = new_loop.induction_variable
            with inner:
                c_s = arith.ConstantOp(index, step).result
                c_2s = arith.ConstantOp(index, 2 * step).result
                k1 = arith.AddIOp(j, c_s).result
            load(k1, 1, inner)
            matmul(0, inner)
            with inner:
                k2 = arith.AddIOp(j, c_2s).result
            load(k2, 0, inner)
            matmul(1, inner)
            with inner:
                scf.YieldOp([])
        load(c[ub - step], 1, before)                            # epilogue
        matmul(0, before)
        matmul(1, before)

    loop.erase()
    for alloc in (a_alloc, b_alloc):
        alloc.erase()


def apply_double_buffering(module: ir.Module, plans: list[MatmulPlan]) -> list[str]:
    """Double buffers every planned matmul.

    The plan is made before tiling; a custom schedule (--transform-script) may
    produce a K loop that does not fit the pattern. Such matmuls are left
    alone and `plan.double_buffer` is cleared. Returns one remark per skip.
    """
    remarks = []
    for plan in plans:
        if plan.double_buffer:
            try:
                double_buffer_matmul(module, plan.op_id)
            except DoubleBufferError as e:
                plan.double_buffer = False
                remarks.append(f"skipped: {e}")
    return remarks
