import pytest

from sava import sbin
from sava.asm import AsmError, assemble, disassemble, disassemble_word
from sava.isa import WAIT_MXU, f32_bits, load_isa


def test_encoding_fields():
    isa = load_isa()
    exe = assemble("addi x10, a0, -4\nhalt\n")
    instr, fields = isa.decode(exe.code[0])
    assert instr.mnemonic == "addi"
    assert fields["rd"] == 10 and fields["rs1"] == 1 and fields["imm"] == -4


def test_labels_are_pc_relative():
    exe = assemble("top:\n nop\n nop\n blt x9, x10, top\n jmp end\n nop\nend:\n halt\n")
    isa = load_isa()
    assert isa.decode(exe.code[2])[1]["imm"] == -2
    assert isa.decode(exe.code[3])[1]["imm"] == 2


def test_symbolic_operands():
    isa = load_isa()
    exe = assemble("csrwi MXU_K, 64\nwait dma0|dma3|mxu\nv.fill x9, 1.5\nfence\n")
    assert isa.decode(exe.code[0])[1]["rd"] == isa.csrs["MXU_K"]
    assert isa.decode(exe.code[1])[1]["imm"] == 0b1001 | WAIT_MXU
    assert isa.decode(exe.code[2])[1]["imm"] & 0xFFFFFFFF == f32_bits(1.5)
    assert isa.decode(exe.code[3])[0].mnemonic == "wait"


def test_directives_and_sbin_roundtrip():
    src = ".kernel k\n.arg A in f32 4x8\n.arg C out bf16 4\n li x9, 1\n halt\n"
    exe = assemble(src)
    back = sbin.read(sbin.write(exe))
    assert back.name == "k"
    assert back.code == exe.code
    assert [(a.name, a.kind, a.dtype, a.shape) for a in back.args] == [
        ("A", "in", "f32", (4, 8)), ("C", "out", "bf16", (4,))]
    assert back.args[0].nbytes == 128


def test_disassemble_roundtrip():
    src = "li x9, 7\nmxu.mm x9, x10, x11, 1\ncsrr x12, CYCLE\nhalt\n"
    exe = assemble(src)
    text = disassemble(exe)
    assert "mxu.mm x9, x10, x11, 1" in text
    assert "csrr x12, CYCLE" in text
    assert assemble(text).code == exe.code
    assert disassemble_word(exe.code[0]) == "li x9, 7"


@pytest.mark.parametrize("src, msg", [
    ("frobnicate x1", "unknown instruction"),
    ("add x1, x2", "expects 3 operands"),
    ("add x1, x2, x99", "unknown register"),
    ("jmp nowhere", "unknown label"),
    ("csrwi NOPE, 1", "unknown CSR"),
    ("a:\na:\n halt", "duplicate label"),
    (".arg A in f64 4", "bad .arg"),
])
def test_errors(src, msg):
    with pytest.raises(AsmError, match=msg):
        assemble(src)
