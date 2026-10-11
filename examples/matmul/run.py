#!/usr/bin/env python3
"""End-to-end example: compile matmul.mlir for SAVA, run it on the simulator
and check the result against NumPy and against IREE's own CPU backend.

    python examples/matmul/run.py [--mlir FILE] [--transform-script FILE]
                                  [--post-bufferize-script FILE] [--asm FILE]
                                  [--no-double-buffer] [--trace FILE] [--no-iree]

Intermediate IR of every stage is written to build/examples/<name>/.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from sava import sbin  # noqa: E402
from sava.asm import assemble  # noqa: E402
from sava.compiler import CompileError, compile_mlir  # noqa: E402
from sava.isa import load_config  # noqa: E402
from sava.runtime import from_bf16, run_executable, to_bf16  # noqa: E402


def make_input(rng, arg) -> np.ndarray:
    """Random input for a kernel argument. bf16 travels as uint16 bit patterns."""
    x = rng.standard_normal(arg.shape, dtype=np.float32)
    if arg.dtype == "f32":
        return x
    if arg.dtype == "bf16":
        return to_bf16(x)
    raise NotImplementedError(f"no input generator for {arg.dtype}")


def as_f64(x: np.ndarray, dtype: str) -> np.ndarray:
    return (from_bf16(x) if dtype == "bf16" else x).astype(np.float64)


def iree_reference(source: str, func: str, inputs, dtypes):
    """Compiles the same MLIR with IREE for the host CPU and runs it."""
    import iree.compiler as ireec
    import iree.runtime as ireert

    vmfb = ireec.compile_str(source, target_backends=["llvm-cpu"],
                             extra_args=["--iree-llvmcpu-target-cpu=host"])
    config = ireert.Config("local-task")
    ctx = ireert.SystemContext(config=config)
    ctx.add_vm_module(ireert.VmModule.copy_buffer(ctx.instance, vmfb))

    def to_device(x, dtype):
        # NumPy has no bf16: the uint16 bits are tagged as bf16 explicitly,
        # otherwise IREE rejects the argument as ui16.
        if dtype != "bf16":
            return x
        return config.device.allocator.allocate_buffer_copy(
            memory_type=ireert.MemoryType.DEVICE_LOCAL, allowed_usage=ireert.BufferUsage.DEFAULT,
            device=config.device, buffer=x, element_type=ireert.HalElementType.BFLOAT_16)

    fn = ctx.modules.module[func]
    return np.asarray(fn(*[to_device(x, d) for x, d in zip(inputs, dtypes)]))


REL_TOL = 1e-5  # max |error| / max |reference|; f32 accumulation is ~1e-6 for K in the thousands


def check(name: str, out: np.ndarray, ref: np.ndarray) -> bool:
    err = float(np.max(np.abs(out - ref)))
    rel = err / max(float(np.max(np.abs(ref))), np.finfo(np.float32).tiny)
    ok = rel <= REL_TOL
    print(f"SAVA simulator vs {name + ':':<12} max abs error {err:.3e}, "
          f"relative {rel:.2e}  {'ok' if ok else f'> {REL_TOL:.0e}'}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mlir", default=str(Path(__file__).with_name("matmul.mlir")))
    p.add_argument("--trace", help="write an instruction trace to this file")
    p.add_argument("--no-iree", action="store_true", help="skip the IREE CPU reference")
    p.add_argument("--transform-script",
                   help="use this transform-dialect schedule instead of the generated one")
    p.add_argument("--post-bufferize-script",
                   help="use this memref-level schedule (stage 04b) instead of the generated one")
    p.add_argument("--asm", help="run this (hand-edited) assembly instead of the compiled "
                                 "kernel; --mlir still provides the reference")
    p.add_argument("--no-double-buffer", action="store_true",
                   help="skip double buffering (stage 04d), to measure its effect")
    args = p.parse_args()

    source = Path(args.mlir).read_text()
    script = Path(args.transform_script).read_text() if args.transform_script else None
    post = Path(args.post_bufferize_script).read_text() if args.post_bufferize_script else None
    dump_dir = REPO / "build" / "examples" / Path(args.mlir).stem
    try:
        result = compile_mlir(source, dump_dir=dump_dir, transform_script=script,
                              post_bufferize_script=post,
                              double_buffering=not args.no_double_buffer)
    except CompileError as e:
        print(f"compile error: {e}")
        return 1
    print(f"compiled @{result.name}: {len(result.executable.code)} instructions, "
          f"stages in {dump_dir.relative_to(REPO)}/")
    image = result.image
    if args.asm:
        hand = assemble(Path(args.asm).read_text())
        if [(a.kind, a.dtype, a.shape) for a in hand.args] != \
                [(a.kind, a.dtype, a.shape) for a in result.executable.args]:
            print(f"error: .arg directives in {args.asm} do not match {args.mlir}")
            return 1
        image = sbin.write(hand)
        print(f"  running hand-written {args.asm} ({len(hand.code)} instructions)")
    for name, path in (("schedule", args.transform_script),
                       ("memref schedule", args.post_bufferize_script)):
        if path:
            print(f"  {name}: {path}")
    if not script:
        for plan in result.plans:
            print(f"  matmul #{plan.op_id}: {plan.m}x{plan.k} * {plan.k}x{plan.n}, "
                  f"tiles M={plan.tiles[0]} N={plan.tiles[1]} K={plan.tiles[2]}, "
                  f"double_buffer={plan.double_buffer}")

    rng = np.random.default_rng(0)
    in_args = [a for a in result.executable.args if a.kind != "out"]
    dtypes = [a.dtype for a in in_args]
    inputs = [make_input(rng, a) for a in in_args]

    (out,), stats = run_executable(image, inputs, trace=args.trace)
    # Exact (float64) reference. Every f32 implementation, SAVA included, differs
    # from it by rounding that grows with K, so the error is measured relative to
    # the magnitude of the result instead of with a fixed absolute tolerance.
    # bf16 inputs are converted exactly, so only the f32 accumulation differs.
    exact = as_f64(inputs[0], dtypes[0]) @ as_f64(inputs[1], dtypes[1])
    print()
    ok = check("exact (f64)", out, exact)

    if not args.no_iree:
        ok &= check("IREE CPU", out, iree_reference(source, result.name, inputs, dtypes))

    print("\n" + stats.report(load_config()["clock_ghz"]))
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
