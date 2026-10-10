# SAVA v0: Instruction Set Reference (ISA 0.1)

The authoritative definition is [`arch/isa.json`](../../arch/isa.json). The
simulator's decoder is generated from it, and the assembler reads it at run
time. A test ([`python/tests/test_docs.py`](../../python/tests/test_docs.py))
checks that every instruction and CSR there is documented on this page.

## Encoding

Every instruction is one 64-bit little-endian word with a single format:

```
 63      56 55   51 50   46 45   41 40   36 35  32 31                      0
+----------+-------+-------+-------+-------+------+-------------------------+
|  opcode  |  rd   |  rs1  |  rs2  |  rs3  | rsvd |          imm            |
+----------+-------+-------+-------+-------+------+-------------------------+
     8         5       5       5       5      4               32
```

* `imm` is sign-extended to 64 bits unless an instruction says otherwise.
* Branch targets are **pc-relative in instructions**: `pc_next = pc + imm`.
* Fields an instruction does not use must be zero.

## Registers

| Register | Alias | Role |
|---|---|---|
| `x0` | `zero` | Always reads 0. Writes are ignored. |
| `x1`-`x8` | `a0`-`a7` | Kernel arguments (see [kernel ABI](#kernel-abi)). |
| `x9`-`x31` | | General purpose. The compiler allocates from these. |

All registers are 64 bits wide. Arithmetic wraps modulo 2^64.

## CSRs

Configuration CSRs are **sampled when a command is issued**. Changing them
afterwards does not affect commands already in flight.

| Id | Name | Access | Meaning |
|---|---|---|---|
| 0 | `DMA_ROWS` | rw | Rows per DMA transfer |
| 1 | `DMA_ROW_BYTES` | rw | Bytes per row |
| 2 | `DMA_SRAM_STRIDE` | rw | Row pitch in SRAM (bytes) |
| 3 | `DMA_DRAM_STRIDE` | rw | Row pitch in DRAM (bytes) |
| 8 | `MXU_M` | rw | Rows of A and C (1..128) |
| 9 | `MXU_N` | rw | Columns of B and C (1..128) |
| 10 | `MXU_K` | rw | Reduction size (1..128) |
| 11 | `MXU_LDA` | rw | Row pitch of A in bytes (≥ K·esize) |
| 12 | `MXU_LDB` | rw | Row pitch of B in bytes (≥ N·esize) |
| 13 | `MXU_LDC` | rw | Row pitch of C in bytes (≥ N·4) |
| 14 | `MXU_DTYPE` | rw | A/B element type: 0 = f32, 1 = bf16 (C is always f32) |
| 16 | `VPU_LEN` | rw | Elements per `v.*` instruction |
| 24 | `CYCLE` | ro | Current cycle count |
| 25 | `CORE_ID` | ro | This core's index (0 in v0) |
| 26 | `NUM_CORES` | ro | Cores taking part in the kernel (1 in v0) |

Accessing an undefined CSR, or writing a read-only one, traps.

## Instructions

Notation: `R[x]` is a register, `SRAM32[a]` is a 32-bit SRAM word, `sext` is
sign extension. "Unit" is where the instruction executes: `scalar` runs
synchronously in the SCU, while `dma`/`mxu`/`vpu` are asynchronous commands
(see [architecture §3.2](sava-v0-architecture.md#32-asynchronous-units-and-command-queues)).

### Control and constants (scalar)

| Opcode | Syntax | Semantics |
|---|---|---|
| 0x00 | `nop` | No operation. |
| 0x01 | `halt` | Wait for all units, then end the kernel. |
| 0x02 | `li rd, imm` | `R[rd] = sext(imm)` |

### Integer ALU (scalar)

| Opcode | Syntax | Semantics |
|---|---|---|
| 0x10 | `add rd, rs1, rs2` | `R[rd] = R[rs1] + R[rs2]` |
| 0x11 | `sub rd, rs1, rs2` | `R[rd] = R[rs1] - R[rs2]` |
| 0x12 | `mul rd, rs1, rs2` | `R[rd] = R[rs1] * R[rs2]` (low 64 bits) |
| 0x13 | `and rd, rs1, rs2` | bitwise and |
| 0x14 | `or rd, rs1, rs2` | bitwise or |
| 0x15 | `xor rd, rs1, rs2` | bitwise xor |
| 0x16 | `sll rd, rs1, rs2` | `R[rd] = R[rs1] << (R[rs2] & 63)` |
| 0x17 | `srl rd, rs1, rs2` | logical shift right |
| 0x18 | `slt rd, rs1, rs2` | `R[rd] = (R[rs1] < R[rs2]) ? 1 : 0`, signed |
| 0x19 | `div rd, rs1, rs2` | signed division, truncating. Traps if the divisor is 0. |
| 0x1a | `rem rd, rs1, rs2` | signed remainder. Traps if the divisor is 0. |
| 0x20 | `addi rd, rs1, imm` | `R[rd] = R[rs1] + sext(imm)` |
| 0x21 | `muli rd, rs1, imm` | `R[rd] = R[rs1] * sext(imm)` |
| 0x22 | `slli rd, rs1, imm` | `R[rd] = R[rs1] << (imm & 63)` |
| 0x23 | `srli rd, rs1, imm` | logical shift right by immediate |
| 0x24 | `andi rd, rs1, imm` | `R[rd] = R[rs1] & sext(imm)` |

### Branches (scalar)

| Opcode | Syntax | Semantics |
|---|---|---|
| 0x30 | `beq rs1, rs2, label` | branch if equal |
| 0x31 | `bne rs1, rs2, label` | branch if not equal |
| 0x32 | `blt rs1, rs2, label` | branch if less than (signed) |
| 0x33 | `bge rs1, rs2, label` | branch if greater or equal (signed) |
| 0x34 | `jmp label` | unconditional jump |

### SRAM access and CSRs (scalar)

| Opcode | Syntax | Semantics |
|---|---|---|
| 0x38 | `lw rd, rs1, imm` | `R[rd] = sext(SRAM32[R[rs1] + imm])`. The address must be 4-byte aligned. |
| 0x39 | `sw rs2, rs1, imm` | `SRAM32[R[rs1] + imm] = R[rs2][31:0]` |
| 0x3c | `csrw CSR, rs1` | `CSR = R[rs1]` (the CSR id is encoded in `rd`) |
| 0x3d | `csrwi CSR, imm` | `CSR = sext(imm)` (the CSR id is encoded in `rd`) |
| 0x3e | `csrr rd, CSR` | `R[rd] = CSR` (the CSR id is encoded in `rs1`) |

### DMA (unit `dma`, asynchronous)

| Opcode | Syntax | Semantics |
|---|---|---|
| 0x40 | `dma.ld rs1, rs2, tag` | For `r` in `[0, DMA_ROWS)`: copy `DMA_ROW_BYTES` bytes from DRAM `R[rs2] + r·DMA_DRAM_STRIDE` to SRAM `R[rs1] + r·DMA_SRAM_STRIDE`. |
| 0x41 | `dma.st rs1, rs2, tag` | The same transfer in the other direction, SRAM `R[rs1]` → DRAM `R[rs2]`. |

`tag` is an immediate in 0..15 that can later be waited on with `wait dmaN`.

### Matrix unit (unit `mxu`, asynchronous)

| Opcode | Syntax | Semantics |
|---|---|---|
| 0x48 | `mxu.mm rs1, rs2, rs3, flags` | `C = (flags & 1 ? C : 0) + A·B`, where `C` = SRAM `R[rs1]` (MxN f32, pitch `MXU_LDC`), `A` = SRAM `R[rs2]` (MxK, pitch `MXU_LDA`), `B` = SRAM `R[rs3]` (KxN, pitch `MXU_LDB`). The element type of A and B comes from `MXU_DTYPE`. Flag bit 0 = accumulate. |

### Vector unit (unit `vpu`, asynchronous)

All operands are SRAM addresses of contiguous f32 vectors of `VPU_LEN`
elements, except where a bf16 type is stated. `imm` operands are f32 values,
written as float literals in assembly.

| Opcode | Syntax | Semantics |
|---|---|---|
| 0x50 | `v.fill dst, imm` | `dst[i] = imm` |
| 0x51 | `v.add dst, a, b` | `dst[i] = a[i] + b[i]` |
| 0x52 | `v.sub dst, a, b` | `dst[i] = a[i] - b[i]` |
| 0x53 | `v.mul dst, a, b` | `dst[i] = a[i] * b[i]` |
| 0x54 | `v.max dst, a, b` | `dst[i] = max(a[i], b[i])` |
| 0x55 | `v.adds dst, a, imm` | `dst[i] = a[i] + imm` |
| 0x56 | `v.muls dst, a, imm` | `dst[i] = a[i] * imm` |
| 0x57 | `v.exp dst, a` | `dst[i] = exp(a[i])` (4x cost) |
| 0x58 | `v.relu dst, a` | `dst[i] = max(a[i], 0)` |
| 0x59 | `v.rsqrt dst, a` | `dst[i] = 1/sqrt(a[i])` (4x cost) |
| 0x5a | `v.rsum dst, a` | `dst[0] = Σ a[i]` |
| 0x5b | `v.rmax dst, a` | `dst[0] = max a[i]` |
| 0x5c | `v.cvt.bf16 dst, a` | bf16 `dst[i]` = round-to-nearest-even(f32 `a[i]`) |
| 0x5d | `v.cvt.f32 dst, a` | f32 `dst[i]` = bf16 `a[i]` |
| 0x5e | `v.copy dst, a` | `dst[i] = a[i]` (any 32-bit data) |

All sources are read before the destination is written, so in-place
operations (`dst == a`) are well defined.

### Synchronization (scalar)

| Opcode | Syntax | Semantics |
|---|---|---|
| 0x60 | `wait mask` | Stall until every outstanding command selected by `mask` has completed: bits 0-15 = DMA tags, bit 16 = MXU, bit 17 = VPU. |

Assembly accepts symbolic masks such as `wait dma0|dma3|mxu`, `wait vpu` and
`wait all`.

## Assembly syntax

```asm
# comments start with '#' or ';'
.kernel relu_scale            # kernel name stored in the .sbin
.arg x in f32 8192            # argument 0 -> a0: name kind dtype dims (AxBxC or 'scalar')
.arg y out f32 8192           # argument 1 -> a1; kind is in | out | inout
loop:                         # labels: [A-Za-z_.$][\w.$]*
    dma.ld x12, a0, 0
    wait dma0
    v.muls x13, x12, 2.0      # float immediate
    blt x10, x11, loop
    halt
```

Pseudo-instructions: `mv rd, rs` → `addi rd, rs, 0`. `fence` → `wait all`.
Tools: `python -m sava.asm k.s -o k.sbin` assembles a file, and
`python -m sava.asm --disasm k.sbin` or `sava-sim k.sbin --disasm`
disassembles one.

## Kernel ABI

* On entry, `a0..a7` hold the kernel's arguments in the order of the
  executable's argument table. In v0 every argument is a buffer, and the
  register holds its **DRAM byte address**. The runtime aligns buffers to
  4 KiB.
* All other registers, every CSR and the contents of SRAM are **undefined**
  on entry. The simulator zeroes the registers and CSRs, but code must not
  depend on SRAM contents.
* A kernel ends with `halt`. Running past the end of the program is an error.
* Argument kinds: `in` (read only), `out` (written; initial contents
  undefined) and `inout`. The kind is metadata used by the runtime. The
  hardware does not enforce it.

This mirrors IREE's HAL executable ABI, where a dispatch receives a list of
buffer *bindings* plus push constants. Push constants (scalar arguments) are
reserved for a later version.

## Executable format

`.sbin` is a small little-endian container. It is produced by
[`python/sava/sbin.py`](../../python/sava/sbin.py) and parsed by
[`sim/src/executable.cpp`](../../sim/src/executable.cpp).

```
offset  size  field
------  ----  ---------------------------------------------------------
Header (32 bytes)
0       4     magic "SAVA"
4       2     version_major (0)
6       2     version_minor (1)
8       4     code_offset   (8-byte aligned)
12      4     code_count    (number of 64-bit instructions)
16      4     args_offset
20      4     args_count    (<= 8)
24      4     name_offset   (NUL-terminated kernel name)
28      4     flags         (0)
Argument record (48 bytes each)
0       24    name (NUL padded)
24      1     kind  (0 in, 1 out, 2 inout)
25      1     dtype (0 f32, 1 bf16, 2 i32, 3 i8)
26      1     rank  (0..4)
27      1     reserved
28      16    dims[4] (u32)
44      4     reserved
```

## Example

The core of the generated matmul kernel (the full output is
`build/examples/matmul/06-kernel.s`):

```asm
.L5_loop:                                   # K loop
    csrwi DMA_ROWS, 128
    csrwi DMA_ROW_BYTES, 512                # 128 f32 per row
    csrwi DMA_DRAM_STRIDE, 2048             # A is 256x512 f32 in DRAM
    csrwi DMA_SRAM_STRIDE, 512
    li x18, 65536                           # A tile in SRAM
    slli x19, x17, 2                        # element offset -> bytes
    add x19, x19, a0
    dma.ld x18, x19, 0
    ...                                     # B tile, tag 1
    wait dma0|dma1                          # MXU reads what the DMAs wrote
    csrwi MXU_M, 128
    ...
    mxu.mm x21, x20, x22, 1                 # C += A x B
    wait mxu
    addi x15, x15, 128
    blt x15, x16, .L5_loop
```
