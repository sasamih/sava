"""SAVA assembler and disassembler.

Syntax (see docs/hardware/sava-v0-isa.md#assembly-syntax):

    # comment (also ';')
    .kernel matmul                  # kernel name
    .arg A in f32 128x256           # kernel argument -> register a0, a1, ...
    loop:                           # label
        dma.ld x10, a0, 0           # mnemonic operands...
        wait dma0|dma1|mxu          # symbolic wait mask
        v.fill x11, 0.0             # float immediate

Pseudo-instructions:
    mv rd, rs     -> addi rd, rs, 0
    fence         -> wait all

Usage: python -m sava.asm kernel.s -o kernel.sbin
"""

from __future__ import annotations

import argparse
import re
import sys

from . import sbin
from .isa import WAIT_ALL, WAIT_MXU, WAIT_VPU, Isa, bits_f32, f32_bits, load_isa

_LABEL_RE = re.compile(r"^([A-Za-z_.$][\w.$]*):")


class AsmError(Exception):
    pass


def _parse_int(text: str) -> int:
    return int(text.replace("_", ""), 0)


def _parse_mask(text: str) -> int:
    mask = 0
    for part in text.split("|"):
        part = part.strip()
        if part == "all":
            mask |= WAIT_ALL
        elif part == "mxu":
            mask |= WAIT_MXU
        elif part == "vpu":
            mask |= WAIT_VPU
        elif re.fullmatch(r"dma\d+", part):
            tag = int(part[3:])
            if tag > 15:
                raise ValueError(f"DMA tag {tag} out of range")
            mask |= 1 << tag
        else:
            mask |= _parse_int(part)
    return mask


def _split_operands(text: str) -> list[str]:
    return [t.strip() for t in text.split(",")] if text.strip() else []


def assemble(source: str, isa: Isa | None = None) -> sbin.Executable:
    isa = isa or load_isa()
    name = "kernel"
    args: list[sbin.ArgSpec] = []
    lines: list[tuple[int, str, list[str]]] = []  # (line_no, mnemonic, operands)
    labels: dict[str, int] = {}

    # Pass 1: directives, labels, pseudo-instruction expansion.
    for line_no, raw in enumerate(source.splitlines(), 1):
        line = re.split(r"[#;]", raw, maxsplit=1)[0].strip()
        while (m := _LABEL_RE.match(line)):
            label = m.group(1)
            if label in labels:
                raise AsmError(f"line {line_no}: duplicate label '{label}'")
            labels[label] = len(lines)
            line = line[m.end():].strip()
        if not line:
            continue
        if line.startswith("."):
            parts = line.split()
            if parts[0] == ".kernel" and len(parts) == 2:
                name = parts[1]
            elif parts[0] == ".arg" and len(parts) == 5:
                _, arg_name, kind, dtype, dims = parts
                if kind not in sbin.ARG_KINDS or dtype not in sbin.DTYPES:
                    raise AsmError(f"line {line_no}: bad .arg '{line}'")
                shape = tuple(int(d) for d in dims.split("x")) if dims != "scalar" else ()
                args.append(sbin.ArgSpec(arg_name, kind, dtype, shape))
            else:
                raise AsmError(f"line {line_no}: unknown directive '{line}'")
            continue
        mnemonic, _, rest = line.partition(" ")
        ops = _split_operands(rest)
        if mnemonic == "mv":
            mnemonic, ops = "addi", ops + ["0"]
        elif mnemonic == "fence":
            mnemonic, ops = "wait", ["all"]
        lines.append((line_no, mnemonic, ops))

    # Pass 2: encode.
    code = []
    for pc, (line_no, mnemonic, ops) in enumerate(lines):
        instr = isa.instrs.get(mnemonic)
        if instr is None:
            raise AsmError(f"line {line_no}: unknown instruction '{mnemonic}'")
        if len(ops) != len(instr.operands):
            raise AsmError(f"line {line_no}: '{mnemonic}' expects {len(instr.operands)} "
                           f"operands, got {len(ops)}")
        fields = {}
        try:
            for op, text in zip(instr.operands, ops):
                if op.kind == "reg":
                    value = isa.reg(text)
                elif op.kind == "csr":
                    if text in isa.csrs:
                        value = isa.csrs[text]
                    elif text[:1].isdigit():
                        value = _parse_int(text)
                    else:
                        raise AsmError(f"line {line_no}: unknown CSR '{text}'")
                elif op.kind == "int":
                    value = _parse_int(text)
                elif op.kind == "f32":
                    value = f32_bits(float(text))
                elif op.kind == "pcrel":
                    if text in labels:
                        value = labels[text] - pc
                    elif text[:1] in "-+0123456789":
                        value = _parse_int(text)
                    else:
                        raise AsmError(f"line {line_no}: unknown label '{text}'")
                elif op.kind == "mask":
                    value = _parse_mask(text)
                else:
                    raise AssertionError(op.kind)
                fields[op.field] = value
            code.append(isa.encode(mnemonic, **fields))
        except ValueError as e:
            raise AsmError(f"line {line_no}: {e}") from None
    if len(args) > sbin.MAX_ARGS:
        raise AsmError(f"at most {sbin.MAX_ARGS} .arg directives are supported")
    return sbin.Executable(name, code, args)


def disassemble_word(word: int, pc: int = 0, isa: Isa | None = None) -> str:
    isa = isa or load_isa()
    instr, fields = isa.decode(word)
    parts = []
    for op in instr.operands:
        v = fields[op.field]
        if op.kind == "reg":
            parts.append(isa.reg_name(v))
        elif op.kind == "csr":
            parts.append(isa.csr_names.get(v, str(v)))
        elif op.kind == "f32":
            parts.append(repr(bits_f32(v)))
        elif op.kind == "pcrel":
            parts.append(f"@{pc + v}")
        elif op.kind == "mask":
            parts.append(hex(v & 0xFFFFFFFF))
        else:
            parts.append(str(v))
    return f"{instr.mnemonic} {', '.join(parts)}".strip()


def disassemble(exe: sbin.Executable) -> str:
    isa = load_isa()
    out = [f".kernel {exe.name}"]
    for a in exe.args:
        out.append(f".arg {a.name} {a.kind} {a.dtype} {'x'.join(map(str, a.shape)) or 'scalar'}")
    for pc, w in enumerate(exe.code):
        out.append(f"    {disassemble_word(w, pc, isa):<40} # {pc}")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m sava.asm", description="SAVA assembler")
    p.add_argument("input", help="assembly file (.s), or .sbin with --disasm")
    p.add_argument("-o", "--output", help="output .sbin")
    p.add_argument("--disasm", action="store_true", help="disassemble an .sbin")
    args = p.parse_args(argv)
    if args.disasm:
        with open(args.input, "rb") as f:
            sys.stdout.write(disassemble(sbin.read(f.read())))
        return 0
    with open(args.input) as f:
        try:
            exe = assemble(f.read())
        except AsmError as e:
            print(f"{args.input}: error: {e}", file=sys.stderr)
            return 1
    output = args.output or args.input.rsplit(".", 1)[0] + ".sbin"
    with open(output, "wb") as f:
        f.write(sbin.write(exe))
    return 0


if __name__ == "__main__":
    sys.exit(main())
