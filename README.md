# SAVA: an accelerator and its compiler stack, built for learning

SAVA is a **fictional LLM inference accelerator** plus everything a real
hardware team would build around it: a hardware spec, a simulator, an ISA
toolchain, a runtime and an MLIR/IREE-based compiler. The goal is to learn ML
compilers by working on a target that behaves like a real one: explicit DMA,
SRAM scratchpads, an asynchronous matrix engine, and synchronization the
compiler is responsible for.

```
 linalg (MLIR) ──savac──► SAVA assembly ──asm──► .sbin ──runtime──► simulator ──► results + perf counters
   │  IREE iree-opt: transform dialect tiling, SRAM promotion, bufferization
   └─ golden reference: IREE llvm-cpu
```

## Repository layout

| Path | What |
|---|---|
| [`arch/`](arch) | **Single source of truth** for the hardware: ISA (`isa.json`) and microarchitecture parameters (`sava-v0.json`) |
| [`docs/hardware/`](docs/hardware) | [Architecture spec](docs/hardware/sava-v0-architecture.md), [ISA reference](docs/hardware/sava-v0-isa.md) |
| [`docs/compiler/`](docs/compiler) | [Compiler pipeline](docs/compiler/pipeline.md) with the IR of every stage |
| [`docs/research.md`](docs/research.md) | Survey (Corsair, TPU, Groq, Tenstorrent, Gemmini) and IREE integration options |
| [`docs/roadmap.md`](docs/roadmap.md) | Next milestones and exercises |
| [`sim/`](sim) | C++17 simulator: functional + approximately-timed model, hazard checker, C API (`sava_sim.h`), `sava-sim` CLI, unit tests |
| [`tools/gen_isa_header.py`](tools/gen_isa_header.py) | Generates the simulator's ISA header from `arch/*.json` (TableGen-style) |
| [`python/sava/`](python/sava) | `asm` (assembler), `sbin` (executable format), `runtime` (ctypes host runtime), `compiler` (`savac`) |
| [`examples/`](examples) | `matmul/` end-to-end MLIR example, `asm/` hand-written kernel |
| [`docker/`](docker), [`.devcontainer/`](.devcontainer) | Reproducible toolchain: Ubuntu 24.04, clang 18, CMake, Ninja, IREE 3.12 pip packages |

## Getting started

Everything runs in a Linux container, so the host only needs Docker (Docker
Desktop with WSL2 on Windows).

**Option 1: VS Code Dev Container.** Open the folder and run *Dev Containers:
Reopen in Container*.

**Option 2: shell scripts.**

```sh
scripts/dev.sh            # Git Bash / Linux / macOS: builds the image on first use, opens a shell
scripts\dev.ps1           # PowerShell
scripts/dev.sh python examples/matmul/run.py   # or run a single command
```

Inside the container (`/workspace` is this repo):

```sh
cmake --preset dev                 # configure (Ninja, clang, build/)
cmake --build --preset dev         # simulator, CLI, tests
ctest --preset dev                 # C++ unit tests + Python tests (pytest)

python examples/matmul/run.py      # compile -> run on SAVA -> compare with numpy and IREE CPU
```

Expected output (abridged):

```
compiled @matmul: 58 instructions, stages in build/examples/matmul/
  matmul #0: 256x512 * 512x256, tiles M=128 N=128 K=128
SAVA simulator vs numpy:   max abs error 8.392e-05
SAVA simulator vs IREE CPU: max abs error 2.670e-05
cycles              211307  (211.31 us @ 1.0 GHz)
mxu  ops     16  util  78.5%  macs 33554432  (317.6 GFLOP/s)
PASS
```

## Tools

```sh
# Compiler: MLIR -> .sbin (+ every intermediate stage and a reproduce.sh for iree-opt)
python -m sava.compiler examples/matmul/matmul.mlir -o build/matmul.sbin --dump-dir build/matmul -S
python -m sava.compiler model.mlir --transform-script my_schedule.mlir    # try your own schedule

# Assembler / disassembler
python -m sava.asm examples/asm/relu_scale.s -o build/relu_scale.sbin
python -m sava.asm --disasm build/relu_scale.sbin

# Simulator CLI (inputs/outputs as .npy, bound to the kernel's .arg table in order)
build/sim/sava-sim build/matmul.sbin -i a.npy -i b.npy -o c.npy --stats
build/sim/sava-sim build/matmul.sbin --disasm
build/sim/sava-sim build/relu_scale.sbin -i x.npy -o y.npy --trace=trace.txt
```

## Where to start reading

1. [Architecture spec](docs/hardware/sava-v0-architecture.md): what the
   machine is and how its timing and synchronization work.
2. [`examples/asm/relu_scale.s`](examples/asm/relu_scale.s): the ISA used by
   hand. Delete a `wait` and watch the simulator report the hazard.
3. [Compiler pipeline](docs/compiler/pipeline.md), then run
   `examples/matmul/run.py` and read `build/examples/matmul/0*-*.mlir` in
   order.
4. [Roadmap](docs/roadmap.md): pick the next milestone (M1: bf16 + double
   buffering is a good first one).
