"""Code generation: "SAVA-ready" MLIR -> SAVA assembly.

Input contract (what pipeline.py produces, see docs/compiler/pipeline.md):

    func.func with memref arguments in DRAM (memory space 0)
    arith.constant / addi / subi / muli / divsi / remsi / index_cast   (index math)
    scf.for without iter_args                                          (loops)
    memref.alloc in memory space 1                                     (SRAM buffers)
    memref.reinterpret_cast / extract_strided_metadata                 (views)
    memref.copy DRAM <-> SRAM                                          -> dma.ld / dma.st
    linalg.fill on SRAM (f32)                                          -> v.fill
    linalg.matmul on SRAM                                              -> mxu.mm
    memref.dealloc (ignored), func.return                              -> halt

Responsibilities that a real backend would split into separate passes:

  * memory planning   - every SRAM alloc gets a static address (bump allocator),
  * register allocation - scoped allocation of x9..x31 (a0..a7 hold arguments),
  * synchronization   - a `wait` is inserted before any command that conflicts
                        with an un-waited command of another unit; loops are
                        fenced at entry and at the end of each iteration,
  * CSR caching       - redundant csrwi writes are skipped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from iree.compiler import ir

from ..isa import WAIT_MXU, WAIT_VPU, f32_bits
from .tiling import DEFAULT_MATMUL_MAPS, ELEM_BYTES, SRAM_SPACE

DYNAMIC = ir.ShapedType.get_dynamic_stride_or_offset()
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1


class CodegenError(Exception):
    pass


@dataclass(frozen=True)
class Const:
    value: int


@dataclass(frozen=True)
class FConst:
    value: float


@dataclass(frozen=True)
class Reg:
    name: str


@dataclass
class Buffer:
    space: str                # "dram" | "sram"
    base: int | str           # sram: byte address; dram: argument register
    offset: Const | Reg       # in elements
    shape: tuple[int, ...]
    strides: tuple[int, ...]  # in elements
    elem: str
    ident: str                # identity used for dependency tracking

    @property
    def esize(self) -> int:
        return ELEM_BYTES[self.elem]

    def as_2d(self) -> tuple[int, int, int] | None:
        """(rows, cols, row_stride) if the view is a 2D strided block."""
        if not self.shape:
            return 1, 1, 1
        if self.strides[-1] != 1:
            return None
        for i in range(len(self.shape) - 2):
            if self.strides[i] != self.strides[i + 1] * self.shape[i + 1]:
                return None
        cols = self.shape[-1]
        rows = math.prod(self.shape[:-1])
        row_stride = self.strides[-2] if len(self.shape) >= 2 else cols
        return rows, cols, row_stride

    def is_contiguous(self) -> bool:
        b = self.as_2d()
        return b is not None and (b[0] == 1 or b[2] == b[1])


@dataclass
class _Pending:
    unit: str  # dma | mxu | vpu
    tag: int
    reads: set[str]
    writes: set[str]


@dataclass
class KernelArg:
    name: str
    dtype: str
    shape: tuple[int, ...]
    reads: bool = False
    writes: bool = False

    @property
    def kind(self) -> str:
        if self.reads and self.writes:
            return "inout"
        return "out" if self.writes else "in"


class _RegAlloc:
    def __init__(self):
        self.free = [f"x{i}" for i in range(31, 8, -1)]  # x9..x31
        self.scopes: list[list[str]] = [[]]

    def alloc(self) -> str:
        if not self.free:
            raise CodegenError("out of scalar registers (v0 has no spilling)")
        r = self.free.pop()
        self.scopes[-1].append(r)
        return r

    def release(self, r: str):
        for scope in self.scopes:
            if r in scope:
                scope.remove(r)
                self.free.append(r)
                return

    def push(self):
        self.scopes.append([])

    def pop(self):
        for r in self.scopes.pop():
            self.free.append(r)


def _memref_layout(t: ir.Type) -> tuple[ir.MemRefType, tuple[int, ...], int]:
    mt = ir.MemRefType(t)
    strides, offset = mt.get_strides_and_offset()
    if any(s == DYNAMIC for s in strides):
        raise CodegenError(f"dynamic strides are not supported: {t}")
    if not mt.has_static_shape:
        raise CodegenError(f"dynamic shapes are not supported: {t}")
    return mt, tuple(strides), offset


def _memory_space(mt: ir.MemRefType) -> int:
    ms = mt.memory_space
    return 0 if ms is None else ir.IntegerAttr(ms).value


def _op_text(op) -> str:
    text = str(op).splitlines()[0].strip()
    return text if len(text) <= 110 else text[:107] + "..."


class KernelCodegen:
    def __init__(self, fn, cfg: dict, arg_names: list[str] | None = None):
        self.fn = fn
        self.cfg = cfg
        self.arg_names = arg_names
        self.lines: list[str] = []
        self.values: dict[ir.Value, object] = {}
        self.regs = _RegAlloc()
        self.args: list[KernelArg] = []
        self.sram_next = 0
        self.sram_buffers = 0
        self.csr_cache: dict[str, int] = {}
        self.pending: list[_Pending] = []
        self.next_tag = 0
        self.next_label = 0

    # ---- emission helpers ---------------------------------------------------

    def emit(self, text: str):
        self.lines.append(f"    {text}")

    def comment(self, text: str):
        self.lines.append(f"    # {text}")

    def label(self, name: str):
        self.lines.append(f"{name}:")
        self.csr_cache.clear()  # state may differ along incoming edges

    def new_label(self, hint: str) -> str:
        self.next_label += 1
        return f".L{self.next_label}_{hint}"

    @staticmethod
    def check_imm(v: int, what: str) -> int:
        if not INT32_MIN <= v <= INT32_MAX:
            raise CodegenError(f"{what} {v} does not fit in a 32-bit immediate")
        return v

    def set_csr(self, name: str, value: int):
        if self.csr_cache.get(name) == value:
            return
        self.emit(f"csrwi {name}, {self.check_imm(value, name)}")
        self.csr_cache[name] = value

    # ---- values ---------------------------------------------------------------

    def get(self, v: ir.Value):
        try:
            return self.values[v]
        except KeyError:
            raise CodegenError(f"use of unsupported/unknown value {v}") from None

    def in_reg(self, x: Const | Reg) -> tuple[str, bool]:
        """Returns (register, is_temporary)."""
        if isinstance(x, Reg):
            return x.name, False
        if x.value == 0:
            return "zero", False
        t = self.regs.alloc()
        self.emit(f"li {t}, {self.check_imm(x.value, 'constant')}")
        return t, True

    def address(self, buf: Buffer) -> tuple[str, bool]:
        """Materializes the byte address of a buffer view: (register, is_temp)."""
        eb = buf.esize
        if isinstance(buf.offset, Const):
            byte_off = buf.offset.value * eb
            if buf.space == "sram":
                t = self.regs.alloc()
                self.emit(f"li {t}, {self.check_imm(buf.base + byte_off, 'address')}")
                return t, True
            if byte_off == 0:
                return buf.base, False
            t = self.regs.alloc()
            self.emit(f"addi {t}, {buf.base}, {self.check_imm(byte_off, 'offset')}")
            return t, True
        t = self.regs.alloc()
        if eb & (eb - 1) == 0:
            self.emit(f"slli {t}, {buf.offset.name}, {eb.bit_length() - 1}")
        else:
            self.emit(f"muli {t}, {buf.offset.name}, {eb}")
        if buf.space == "sram":
            self.emit(f"addi {t}, {t}, {buf.base}")
        else:
            self.emit(f"add {t}, {t}, {buf.base}")
        return t, True

    def release(self, *regs_and_flags):
        for r, temp in regs_and_flags:
            if temp:
                self.regs.release(r)

    # ---- synchronization ------------------------------------------------------

    @staticmethod
    def _mask(p: _Pending) -> int:
        return {"dma": 1 << p.tag, "mxu": WAIT_MXU, "vpu": WAIT_VPU}[p.unit]

    @staticmethod
    def _mask_text(mask: int) -> str:
        parts = [f"dma{t}" for t in range(16) if mask & (1 << t)]
        if mask & WAIT_MXU:
            parts.append("mxu")
        if mask & WAIT_VPU:
            parts.append("vpu")
        return "|".join(parts)

    def wait(self, mask: int):
        self.emit(f"wait {self._mask_text(mask)}")
        self.pending = [p for p in self.pending if not self._mask(p) & mask]

    def sync(self, unit: str, reads: set[str], writes: set[str]):
        mask = 0
        for p in self.pending:
            if p.unit != unit and (writes & (p.reads | p.writes) or reads & p.writes):
                mask |= self._mask(p)
        if mask:
            self.wait(mask)

    def fence(self):
        mask = 0
        for p in self.pending:
            mask |= self._mask(p)
        if mask:
            self.wait(mask)

    def record(self, unit: str, reads: set[str], writes: set[str], tag: int = 0):
        self.pending.append(_Pending(unit, tag, reads, writes))

    # ---- driver ---------------------------------------------------------------

    def run(self) -> str:
        block = self.fn.regions[0].blocks[0]
        if len(block.arguments) > 8:
            raise CodegenError("at most 8 kernel arguments are supported (a0..a7)")
        for i, arg in enumerate(block.arguments):
            if not isinstance(arg.type, ir.MemRefType):
                raise CodegenError(f"argument {i} must be a memref, got {arg.type}")
            mt, strides, offset = _memref_layout(arg.type)
            if _memory_space(mt) != 0 or offset != 0:
                raise CodegenError(f"argument {i} must be an identity-layout DRAM memref")
            elem = str(mt.element_type)
            name = self.arg_names[i] if self.arg_names else f"arg{i}"
            self.args.append(KernelArg(name, elem, tuple(mt.shape)))
            self.values[arg] = Buffer("dram", f"a{i}", Const(0), tuple(mt.shape), strides,
                                      elem, f"arg{i}")
        self.gen_block(block)

        fn_name = ir.StringAttr(self.fn.attributes["sym_name"]).value
        header = [
            f"# SAVA kernel generated by sava.compiler from @{fn_name}",
            f"# SRAM: {self.sram_buffers} buffers, {self.sram_next} bytes",
            f".kernel {fn_name}",
        ]
        for a in self.args:
            dims = "x".join(map(str, a.shape)) or "scalar"
            header.append(f".arg {a.name} {a.kind} {a.dtype} {dims}")
        return "\n".join(header + [""] + self.lines) + "\n"

    def gen_block(self, block):
        for op in block.operations:
            name = op.operation.name
            handler = getattr(self, "op_" + name.replace(".", "_"), None)
            if handler is None:
                raise CodegenError(f"unsupported operation in SAVA-ready IR:\n  {_op_text(op)}\n"
                                   f"  at {op.location}")
            handler(op)

    # ---- arith ----------------------------------------------------------------

    def op_arith_constant(self, op):
        attr = op.attributes["value"]
        res = op.results[0]
        if isinstance(attr, ir.IntegerAttr):
            self.values[res] = Const(ir.IntegerAttr(attr).value)
        elif isinstance(attr, ir.FloatAttr):
            self.values[res] = FConst(ir.FloatAttr(attr).value)
        else:
            raise CodegenError(f"unsupported constant {_op_text(op)}")

    def _binary(self, op, fold, reg_reg: str, reg_imm=None):
        a, b = (self.get(v) for v in op.operands)
        res = op.results[0]
        if isinstance(a, Const) and isinstance(b, Const):
            self.values[res] = Const(fold(a.value, b.value))
            return
        rd = self.regs.alloc()
        if reg_imm and isinstance(b, Const) and reg_imm(rd, a, b.value):
            pass
        elif reg_imm and isinstance(a, Const) and reg_reg in ("add", "mul") and reg_imm(rd, b, a.value):
            pass
        else:
            ra, rb = self.in_reg(a), self.in_reg(b)
            self.emit(f"{reg_reg} {rd}, {ra[0]}, {rb[0]}")
            self.release(ra, rb)
        self.values[res] = Reg(rd)

    def op_arith_addi(self, op):
        def imm(rd, x, c):
            self.emit(f"addi {rd}, {x.name}, {self.check_imm(c, 'immediate')}")
            return True
        self._binary(op, lambda x, y: x + y, "add", imm)

    def op_arith_subi(self, op):
        def imm(rd, x, c):
            self.emit(f"addi {rd}, {x.name}, {self.check_imm(-c, 'immediate')}")
            return True
        self._binary(op, lambda x, y: x - y, "sub", imm)

    def op_arith_muli(self, op):
        def imm(rd, x, c):
            if c > 0 and c & (c - 1) == 0:
                self.emit(f"slli {rd}, {x.name}, {c.bit_length() - 1}")
            else:
                self.emit(f"muli {rd}, {x.name}, {self.check_imm(c, 'immediate')}")
            return True
        self._binary(op, lambda x, y: x * y, "mul", imm)

    def op_arith_divsi(self, op):
        self._binary(op, lambda x, y: int(x / y), "div")

    def op_arith_remsi(self, op):
        self._binary(op, lambda x, y: x - y * int(x / y), "rem")

    op_arith_divui = op_arith_divsi
    op_arith_remui = op_arith_remsi

    def op_arith_index_cast(self, op):
        self.values[op.results[0]] = self.get(op.operands[0])

    op_arith_index_castui = op_arith_index_cast

    # ---- control flow -----------------------------------------------------------

    def op_scf_for(self, op):
        if len(op.operands) != 3:
            raise CodegenError("scf.for with iter_args is not supported")
        lb, ub, step = (self.get(v) for v in op.operands)
        if not isinstance(step, Const) or step.value <= 0:
            raise CodegenError("scf.for needs a positive constant step")
        body = op.regions[0].blocks[0]
        iv = self.regs.alloc()
        self.values[body.arguments[0]] = Reg(iv)
        head, end = self.new_label("loop"), self.new_label("end")

        def fmt(x):
            return str(x.value) if isinstance(x, Const) else x.name
        self.comment(f"scf.for {iv} = {fmt(lb)} to {fmt(ub)} step {step.value}")
        ub_reg = self.in_reg(ub)
        self.fence()
        if isinstance(lb, Const):
            self.emit(f"li {iv}, {self.check_imm(lb.value, 'loop bound')}")
        else:
            self.emit(f"mv {iv}, {lb.name}")
        if not (isinstance(lb, Const) and isinstance(ub, Const) and lb.value < ub.value):
            self.emit(f"bge {iv}, {ub_reg[0]}, {end}")
        self.label(head)
        self.regs.push()
        self.gen_block(body)
        self.regs.pop()
        self.fence()
        self.emit(f"addi {iv}, {iv}, {self.check_imm(step.value, 'loop step')}")
        self.emit(f"blt {iv}, {ub_reg[0]}, {head}")
        self.label(end)
        self.release(ub_reg)
        self.regs.release(iv)

    def op_scf_yield(self, op):
        if len(op.operands):
            raise CodegenError("scf.yield with values is not supported")

    def op_func_return(self, op):
        self.emit("halt")

    # ---- memory -----------------------------------------------------------------

    def op_memref_alloc(self, op):
        mt, strides, offset = _memref_layout(op.results[0].type)
        if _memory_space(mt) != SRAM_SPACE:
            raise CodegenError(f"DRAM allocations inside kernels are not supported: {_op_text(op)}")
        elem = str(mt.element_type)
        nbytes = math.prod(mt.shape) * ELEM_BYTES[elem]
        align = self.cfg["sram"]["alignment"]
        addr = (self.sram_next + align - 1) // align * align
        self.sram_next = addr + nbytes
        if self.sram_next > self.cfg["sram"]["bytes"]:
            raise CodegenError(f"SRAM overflow: {self.sram_next} > {self.cfg['sram']['bytes']} bytes; "
                               "use smaller tiles")
        self.sram_buffers += 1
        self.comment(f"sram[0x{addr:x}..0x{addr + nbytes:x}) = {mt}")
        self.values[op.results[0]] = Buffer("sram", addr, Const(0), tuple(mt.shape), strides, elem,
                                            f"sram{self.sram_buffers}")

    def op_memref_dealloc(self, op):
        pass

    def op_memref_reinterpret_cast(self, op):
        src = self.get(op.operands[0])
        mt, strides, offset = _memref_layout(op.results[0].type)
        if offset == DYNAMIC:
            off = self.get(op.operands[1])
        else:
            off = Const(offset)
        self.values[op.results[0]] = Buffer(src.space, src.base, off, tuple(mt.shape), strides,
                                            src.elem, src.ident)

    def op_memref_extract_strided_metadata(self, op):
        src = self.get(op.operands[0])
        mt, strides, offset = _memref_layout(op.operands[0].type)
        results = list(op.results)
        self.values[results[0]] = Buffer(src.space, src.base, Const(0), (), (), src.elem, src.ident)
        self.values[results[1]] = src.offset if offset == DYNAMIC else Const(offset)
        rank = len(mt.shape)
        for i in range(rank):
            self.values[results[2 + i]] = Const(mt.shape[i])
            self.values[results[2 + rank + i]] = Const(strides[i])

    def _track_arg(self, buf: Buffer, *, read: bool = False, write: bool = False):
        if buf.ident.startswith("arg"):
            a = self.args[int(buf.ident[3:])]
            a.reads |= read
            a.writes |= write

    def op_memref_copy(self, op):
        src, dst = (self.get(v) for v in op.operands)
        self.comment(_op_text(op))
        if src.space == "sram" and dst.space == "sram":
            if not (src.is_contiguous() and dst.is_contiguous()) or src.esize != 4:
                raise CodegenError("SRAM->SRAM copies must be contiguous 32-bit data")
            self.sync("vpu", {src.ident}, {dst.ident})
            self.set_csr("VPU_LEN", math.prod(src.shape))
            d, s = self.address(dst), self.address(src)
            self.emit(f"v.copy {d[0]}, {s[0]}")
            self.release(d, s)
            self.record("vpu", {src.ident}, {dst.ident})
            return
        if src.space == dst.space:
            raise CodegenError("DRAM->DRAM copies are not supported")
        load = src.space == "dram"
        dram, sram = (src, dst) if load else (dst, src)
        db, sb = dram.as_2d(), sram.as_2d()
        if db is None or sb is None or db[:2] != sb[:2]:
            raise CodegenError(f"copy is not expressible as a 2D DMA: {_op_text(op)}")
        rows, cols, _ = db
        eb = dram.esize
        self.sync("dma", {src.ident}, {dst.ident})
        self.set_csr("DMA_ROWS", rows)
        self.set_csr("DMA_ROW_BYTES", cols * eb)
        self.set_csr("DMA_DRAM_STRIDE", db[2] * eb)
        self.set_csr("DMA_SRAM_STRIDE", sb[2] * eb)
        s_addr, d_addr = self.address(sram), self.address(dram)
        tag = self.next_tag
        self.next_tag = (self.next_tag + 1) % self.cfg["dma"]["tags"]
        self.emit(f"{'dma.ld' if load else 'dma.st'} {s_addr[0]}, {d_addr[0]}, {tag}")
        self.release(s_addr, d_addr)
        self.record("dma", {src.ident}, {dst.ident}, tag)
        self._track_arg(dram, read=load, write=not load)

    # ---- compute ------------------------------------------------------------------

    def op_linalg_fill(self, op):
        value, out = self.get(op.operands[0]), self.get(op.operands[1])
        self.comment(_op_text(op))
        if out.space != "sram" or out.elem != "f32" or not out.is_contiguous():
            raise CodegenError("linalg.fill is only supported on contiguous f32 SRAM buffers")
        if not isinstance(value, (Const, FConst)):
            raise CodegenError("linalg.fill value must be a constant")
        self.sync("vpu", set(), {out.ident})
        self.set_csr("VPU_LEN", math.prod(out.shape))
        d = self.address(out)
        self.emit(f"v.fill {d[0]}, {float(value.value)!r}")
        self.release(d)
        self.record("vpu", set(), {out.ident})

    def op_linalg_matmul(self, op):
        if "indexing_maps" in op.attributes and str(op.attributes["indexing_maps"]) != DEFAULT_MATMUL_MAPS:
            raise CodegenError("only plain (non-transposed) linalg.matmul is supported")
        a, b, c = (self.get(v) for v in op.operands)
        self.comment(_op_text(op))
        for name, buf in (("A", a), ("B", b), ("C", c)):
            if buf.space != "sram" or len(buf.shape) != 2 or buf.strides[1] != 1:
                raise CodegenError(f"matmul operand {name} must be a row-major 2D SRAM view")
        if a.elem != b.elem or a.elem not in ("f32", "bf16") or c.elem != "f32":
            raise CodegenError("matmul must be f32 x f32 -> f32 or bf16 x bf16 -> f32")
        m, k = a.shape
        n = b.shape[1]
        mxu = self.cfg["mxu"]
        if m > mxu["max_m"] or n > mxu["max_n"] or k > mxu["max_k"]:
            raise CodegenError(f"matmul tile {m}x{n}x{k} exceeds MXU limits "
                               f"{mxu['max_m']}x{mxu['max_n']}x{mxu['max_k']}; tile it first")
        self.sync("mxu", {a.ident, b.ident, c.ident}, {c.ident})
        self.set_csr("MXU_M", m)
        self.set_csr("MXU_N", n)
        self.set_csr("MXU_K", k)
        self.set_csr("MXU_LDA", a.strides[0] * a.esize)
        self.set_csr("MXU_LDB", b.strides[0] * b.esize)
        self.set_csr("MXU_LDC", c.strides[0] * 4)
        self.set_csr("MXU_DTYPE", 1 if a.elem == "bf16" else 0)
        rc, ra, rb = self.address(c), self.address(a), self.address(b)
        self.emit(f"mxu.mm {rc[0]}, {ra[0]}, {rb[0]}, 1")  # 1 = accumulate into C
        self.release(rc, ra, rb)
        self.record("mxu", {a.ident, b.ident, c.ident}, {c.ident})


def generate(module: ir.Module, cfg: dict, arg_names: list[str] | None = None) -> str:
    from .frontend import find_kernel
    return KernelCodegen(find_kernel(module), cfg, arg_names).run()
