"""savac: compile an MLIR kernel to a SAVA executable.

    python -m sava.compiler model.mlir -o model.sbin --dump-dir build/model
"""

import argparse
import sys
from pathlib import Path

from .pipeline import CompileError, compile_mlir


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="savac", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="input .mlir (linalg on tensors)")
    p.add_argument("-o", "--output", help="output .sbin (default: <input>.sbin)")
    p.add_argument("--dump-dir", help="write every intermediate stage to this directory")
    p.add_argument("--transform-script", help="use this transform-dialect script instead of "
                                              "the generated one")
    p.add_argument("-S", action="store_true", help="print the generated assembly")
    args = p.parse_args(argv)

    script = Path(args.transform_script).read_text() if args.transform_script else None
    try:
        result = compile_mlir(Path(args.input).read_text(), dump_dir=args.dump_dir,
                              transform_script=script)
    except CompileError as e:
        print(f"savac: error: {e}", file=sys.stderr)
        return 1
    output = Path(args.output) if args.output else Path(args.input).with_suffix(".sbin")
    output.write_bytes(result.image)
    if args.S:
        sys.stdout.write(result.asm)
    for plan in result.plans:
        tm, tn, tk = plan.tiles
        print(f"savac: matmul #{plan.op_id} {plan.m}x{plan.n}x{plan.k} ({plan.elem}): "
              f"tiles {tm}x{tn}x{tk}", file=sys.stderr)
    print(f"savac: wrote {output} ({len(result.executable.code)} instructions)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
