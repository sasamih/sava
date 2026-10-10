"""SAVA ISA and hardware configuration, loaded from arch/*.json."""

from __future__ import annotations

import functools
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

ARCH_DIR = Path(os.environ.get("SAVA_ARCH_DIR", Path(__file__).resolve().parents[2] / "arch"))

WAIT_MXU = 1 << 16
WAIT_VPU = 1 << 17
WAIT_ALL = 0x3FFFF


@dataclass(frozen=True)
class Operand:
    field: str  # rd | rs1 | rs2 | rs3 | imm
    kind: str  # reg | csr | int | f32 | pcrel | mask


@dataclass(frozen=True)
class Instr:
    mnemonic: str
    opcode: int
    unit: str
    operands: tuple[Operand, ...]
    desc: str


class Isa:
    def __init__(self, data: dict):
        self.version: str = data["version"]
        self.fields: dict[str, tuple[int, int]] = {
            name: (f["lsb"], f["bits"]) for name, f in data["encoding"]["fields"].items()
        }
        self.num_regs: int = data["registers"]["count"]
        self.reg_aliases: dict[str, int] = dict(data["registers"]["aliases"])
        self.csrs: dict[str, int] = {c["name"]: c["id"] for c in data["csrs"]}
        self.csr_names: dict[int, str] = {v: k for k, v in self.csrs.items()}
        self.instrs: dict[str, Instr] = {}
        self.by_opcode: dict[int, Instr] = {}
        for i in data["instructions"]:
            instr = Instr(
                mnemonic=i["mnemonic"],
                opcode=int(i["opcode"], 16),
                unit=i["unit"],
                operands=tuple(Operand(f, k) for f, k in i["operands"]),
                desc=i["desc"],
            )
            self.instrs[instr.mnemonic] = instr
            self.by_opcode[instr.opcode] = instr

    def reg(self, name: str) -> int:
        name = name.strip()
        if name in self.reg_aliases:
            return self.reg_aliases[name]
        if name.startswith("x") and name[1:].isdigit():
            r = int(name[1:])
            if 0 <= r < self.num_regs:
                return r
        raise ValueError(f"unknown register '{name}'")

    def reg_name(self, r: int) -> str:
        for alias, idx in self.reg_aliases.items():
            if idx == r:
                return alias
        return f"x{r}"

    def encode(self, mnemonic: str, **fields: int) -> int:
        instr = self.instrs[mnemonic]
        word = instr.opcode << self.fields["opcode"][0]
        for name, value in fields.items():
            lsb, bits = self.fields[name]
            mask = (1 << bits) - 1
            if name == "imm":
                if not -(1 << 31) <= value < (1 << 32):
                    raise ValueError(f"immediate {value} does not fit in 32 bits")
            elif not 0 <= value <= mask:
                raise ValueError(f"field {name}={value} out of range")
            word |= (value & mask) << lsb
        return word

    def decode(self, word: int) -> tuple[Instr, dict[str, int]]:
        def get(name: str) -> int:
            lsb, bits = self.fields[name]
            return (word >> lsb) & ((1 << bits) - 1)

        opcode = get("opcode")
        if opcode not in self.by_opcode:
            raise ValueError(f"illegal instruction 0x{word:016x}")
        fields = {name: get(name) for name in self.fields if name != "opcode"}
        imm = fields["imm"]
        fields["imm"] = imm - (1 << 32) if imm & (1 << 31) else imm
        return self.by_opcode[opcode], fields


@functools.cache
def load_isa() -> Isa:
    with open(ARCH_DIR / "isa.json") as f:
        return Isa(json.load(f))


@functools.cache
def load_config(name: str = "sava-v0") -> dict:
    with open(ARCH_DIR / f"{name}.json") as f:
        return json.load(f)


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def bits_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]
