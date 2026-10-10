"""Reader/writer for SAVA executables (.sbin).

The layout is documented in docs/hardware/sava-v0-isa.md and mirrored by
sim/src/executable.h.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

MAGIC = b"SAVA"
VERSION = (0, 1)
HEADER = struct.Struct("<4sHHIIIIII")  # 32 bytes
ARG = struct.Struct("<24sBBBB4II")  # 48 bytes
MAX_ARGS = 8
MAX_RANK = 4

ARG_KINDS = ("in", "out", "inout")
DTYPES = ("f32", "bf16", "i32", "i8")
DTYPE_BYTES = {"f32": 4, "bf16": 2, "i32": 4, "i8": 1}


@dataclass
class ArgSpec:
    name: str
    kind: str  # in | out | inout
    dtype: str  # f32 | bf16 | i32 | i8
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        n = DTYPE_BYTES[self.dtype]
        for d in self.shape:
            n *= d
        return n


@dataclass
class Executable:
    name: str
    code: list[int]
    args: list[ArgSpec] = field(default_factory=list)


def write(exe: Executable) -> bytes:
    if len(exe.args) > MAX_ARGS:
        raise ValueError(f"at most {MAX_ARGS} kernel arguments are supported")
    args_off = HEADER.size
    code_off = args_off + ARG.size * len(exe.args)
    code_off = (code_off + 7) & ~7
    name_off = code_off + 8 * len(exe.code)
    out = bytearray(HEADER.pack(MAGIC, *VERSION, code_off, len(exe.code), args_off,
                                len(exe.args), name_off, 0))
    for a in exe.args:
        if len(a.shape) > MAX_RANK:
            raise ValueError(f"argument '{a.name}' has rank > {MAX_RANK}")
        dims = list(a.shape) + [0] * (MAX_RANK - len(a.shape))
        out += ARG.pack(a.name.encode()[:23], ARG_KINDS.index(a.kind), DTYPES.index(a.dtype),
                        len(a.shape), 0, *dims, 0)
    out += b"\0" * (code_off - len(out))
    for w in exe.code:
        out += struct.pack("<Q", w)
    out += exe.name.encode() + b"\0"
    return bytes(out)


def read(data: bytes) -> Executable:
    magic, major, _minor, code_off, code_count, args_off, args_count, name_off, _flags = \
        HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError("not a SAVA executable")
    if major != VERSION[0]:
        raise ValueError(f"unsupported sbin version {major}")
    args = []
    for i in range(args_count):
        name, kind, dtype, rank, _, *rest = ARG.unpack_from(data, args_off + ARG.size * i)
        dims = rest[:MAX_RANK]
        args.append(ArgSpec(name.rstrip(b"\0").decode(), ARG_KINDS[kind], DTYPES[dtype],
                            tuple(dims[:rank])))
    code = list(struct.unpack_from(f"<{code_count}Q", data, code_off))
    name = data[name_off:data.index(b"\0", name_off)].decode()
    return Executable(name, code, args)
