"""MLIR -> SAVA compiler.

Pipeline (see docs/compiler/pipeline.md):

    input (linalg on tensors)
      -> frontend.py   ABI: results become destination memrefs (DPS)
      -> tiling.py     tile sizes + transform-dialect script (tiling, SRAM promotion)
      -> pipeline.py   iree-opt: transform interpreter, bufferization, lowering
      -> codegen.py    "SAVA-ready" IR -> SAVA assembly
      -> sava.asm      assembly -> .sbin
"""

from .pipeline import CompileError, CompileResult, compile_mlir

__all__ = ["CompileError", "CompileResult", "compile_mlir"]
