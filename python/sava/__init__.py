"""SAVA software stack: ISA tools, compiler and host runtime.

    sava.isa       ISA description loaded from arch/isa.json
    sava.asm       assembler / disassembler
    sava.sbin      executable container (.sbin) reader and writer
    sava.runtime   host runtime on top of the simulator C API
    sava.compiler  MLIR -> SAVA compiler (uses IREE's MLIR tooling)
"""

__version__ = "0.1.0"
