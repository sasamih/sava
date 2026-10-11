"""SRAM allocation hoisting (runs on bufferized IR, stage 04a).

After bufferization every promoted tile is a `memref.alloc` in memory space 1
*inside* the loop that uses it, i.e. conceptually a fresh buffer per
iteration. SAVA's SRAM is statically planned (codegen gives every alloc site a
fixed address), so moving these allocations to the function entry does not
change the generated code -- but it gives the memref-level schedule (stage
04b) loop-invariant buffers to work with: `transform.memref.multibuffer`
requires the allocation to be defined outside the loop it multi-buffers.

Hoisting is valid because a fresh allocation has undefined contents: reusing
one buffer across iterations can only be observed by a read-before-write,
which would already read undefined data.

Why not IREE's `iree-codegen-hoist-statically-bound-allocations` /
`transform.iree.hoist_static_alloc`? They insert `memref.dealloc` ops at the
end of the function, and multibuffering rejects allocations with users outside
the loop. SAVA never frees SRAM, so no deallocations are needed.
"""

from __future__ import annotations

from iree.compiler import ir

from .frontend import find_kernel
from .tiling import SRAM_SPACE


def _is_hoistable(op) -> bool:
    if op.operation.name != "memref.alloc" or len(op.operands) != 0:
        return False
    mt = ir.MemRefType(op.results[0].type)
    space = mt.memory_space
    return space is not None and ir.IntegerAttr(space).value == SRAM_SPACE and mt.has_static_shape


def hoist_sram_allocations(module: ir.Module) -> int:
    """Moves static SRAM allocations out of all loops to the kernel's entry.

    Relative order is preserved, so SRAM addresses assigned by codegen do not
    change. Returns the number of allocations moved.
    """
    fn = find_kernel(module)
    entry = fn.regions[0].blocks[0]
    anchor = entry.operations[0]
    moved = 0

    def walk(block, nested: bool):
        nonlocal moved
        for op in list(block.operations):
            for region in op.regions:
                for inner in region.blocks:
                    walk(inner, True)
            if nested and _is_hoistable(op):
                op.operation.move_before(anchor)
                moved += 1

    walk(entry, False)
    return moved
