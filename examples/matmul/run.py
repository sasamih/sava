#!/usr/bin/env python3
"""End-to-end example: compile matmul.mlir for SAVA, run it on the simulator
and check the result against NumPy and against IREE's own CPU backend.

    python examples/matmul/run.py [--mlir FILE] [--trace] [--no-iree]

Intermediate IR of every stage is written to build/examples/<name>/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from sava.compiler import compile_mlir  # noqa: E402
from sava.isa import load_config  # noqa: E402
from sava.runtime import run_executable  # noqa: E402


def iree_reference(source: str, func: str, inputs):
    """Compiles the same MLIR with IREE for the host CPU and runs it."""
    import iree.compiler as ireec
    import iree.runtime as ireert

    vmfb = ireec.compile_str(source, target_backends=["llvm-cpu"],
                             extra_args=["--iree-llvmcpu-target-cpu=host"])
    module = ireert.load_vm_flatbuffer(vmfb, driver="local-task")
    return np.asarray(module[func](*inputs))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mlir", default=str(Path(__file__).with_name("matmul.mlir")))
    p.add_argument("--trace", help="write an instruction trace to this file")
    p.add_argument("--no-iree", action="store_true", help="skip the IREE CPU reference")
    args = p.parse_args()

    source = Path(args.mlir).read_text()
    dump_dir = REPO / "build" / "examples" / Path(args.mlir).stem
    result = compile_mlir(source, dump_dir=dump_dir)
    print(f"compiled @{result.name}: {len(result.executable.code)} instructions, "
          f"stages in {dump_dir.relative_to(REPO)}/")
    for plan in result.plans:
        print(f"  matmul #{plan.op_id}: {plan.m}x{plan.k} * {plan.k}x{plan.n}, "
              f"tiles M={plan.tiles[0]} N={plan.tiles[1]} K={plan.tiles[2]}")

    rng = np.random.default_rng(0)
    inputs = [rng.standard_normal(a.shape, dtype=np.float32)
              for a in result.executable.args if a.kind != "out"]

    (out,), stats = run_executable(result.image, inputs, trace=args.trace)
    expected = inputs[0] @ inputs[1]
    err = float(np.max(np.abs(out - expected)))
    print(f"\nSAVA simulator vs numpy:   max abs error {err:.3e}")
    ok = np.allclose(out, expected, rtol=1e-4, atol=1e-4)

    if not args.no_iree:
        ref = iree_reference(source, result.name, inputs)
        err_iree = float(np.max(np.abs(out - ref)))
        print(f"SAVA simulator vs IREE CPU: max abs error {err_iree:.3e}")
        ok &= np.allclose(out, ref, rtol=1e-4, atol=1e-4)

    print("\n" + stats.report(load_config()["clock_ghz"]))
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
