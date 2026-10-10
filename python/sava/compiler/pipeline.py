"""Compilation pipeline driver.

Each stage's IR is kept (and optionally dumped) so every step can be
inspected and re-run by hand with iree-opt; `reproduce.sh` in the dump
directory contains the exact commands.

    00-input.mlir       as given (linalg on tensors)
    01-dps.mlir         results turned into destination memrefs   [frontend.py]
    02-transform.mlir   transform-dialect schedule                [tiling.py]
    03-tiled.mlir       tiled + SRAM-promoted, still on tensors   [iree-opt]
    04-bufferized.mlir  memrefs (DRAM args, SRAM allocs)          [iree-opt]
    05-lowered.mlir     "SAVA-ready": explicit index math         [iree-opt]
    06-kernel.s         SAVA assembly                             [codegen.py]
    <name>.sbin         executable                                [sava.asm]
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from iree.compiler import ir

from .. import sbin
from ..asm import assemble
from ..isa import load_config
from . import codegen, frontend, tiling


class CompileError(Exception):
    pass


TILE_PASSES = [
    "--eliminate-empty-tensors",
    "--canonicalize",
    "--transform-interpreter",
    "--canonicalize",
    "--cse",
]
BUFFERIZE_PASSES = [
    "--one-shot-bufferize=bufferize-function-boundaries "
    "function-boundary-type-conversion=identity-layout-map",
    "--canonicalize",
    "--cse",
]
LOWER_PASSES = [
    "--expand-strided-metadata",
    "--lower-affine",
    "--loop-invariant-code-motion",
    "--canonicalize",
    "--cse",
]


@dataclass
class CompileResult:
    name: str
    stages: dict[str, str]
    asm: str
    executable: sbin.Executable
    image: bytes
    plans: list[tiling.MatmulPlan]
    commands: list[str] = field(default_factory=list)


def _iree_opt() -> str:
    path = shutil.which("iree-opt")
    if path:
        return path
    try:
        from iree.compiler.tools.binaries import find_tool
        return find_tool("iree-opt")
    except Exception as e:  # pragma: no cover
        raise CompileError("iree-opt not found; pip install iree-base-compiler") from e


def _run_iree_opt(text: str, flags: list[str], stage: str) -> str:
    proc = subprocess.run([_iree_opt(), *flags], input=text, capture_output=True, text=True)
    if proc.returncode != 0:
        raise CompileError(f"iree-opt failed in stage '{stage}':\n{proc.stderr}")
    return proc.stdout


def compile_mlir(source: str, *, dump_dir: str | Path | None = None,
                 transform_script: str | None = None, config: str = "sava-v0") -> CompileResult:
    cfg = load_config(config)
    stages: dict[str, str] = {"00-input.mlir": source}
    commands: list[str] = []

    with ir.Context():
        try:
            module = ir.Module.parse(source)
        except Exception as e:
            raise CompileError(f"cannot parse input: {e}") from None
        try:
            sig = frontend.to_destination_passing_style(module)
            plans = tiling.plan_module(module, cfg)
        except (frontend.FrontendError, tiling.TilingError) as e:
            raise CompileError(str(e)) from None
        stages["01-dps.mlir"] = str(module)

    stages["02-transform.mlir"] = transform_script or tiling.make_transform_script(plans)

    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(dump_dir) / "02-transform.mlir" if dump_dir else Path(tmp) / "t.mlir"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(stages["02-transform.mlir"])
        tile_flags = list(TILE_PASSES)
        tile_flags.insert(2, f"--transform-preload-library=transform-library-paths={script_path}")
        steps = [("03-tiled.mlir", "01-dps.mlir", tile_flags),
                 ("04-bufferized.mlir", "03-tiled.mlir", BUFFERIZE_PASSES),
                 ("05-lowered.mlir", "04-bufferized.mlir", LOWER_PASSES)]
        for out, inp, flags in steps:
            stages[out] = _run_iree_opt(stages[inp], flags, out)
            shown = [f.replace(str(script_path), "02-transform.mlir") for f in flags]
            commands.append(f"iree-opt {inp} {' '.join(shlex.quote(f) for f in shown)} -o {out}")

    arg_names = [f"in{i}" for i in range(sig.num_inputs)] + [f"out{i}" for i in range(sig.num_outputs)]
    with ir.Context():
        lowered = ir.Module.parse(stages["05-lowered.mlir"])
        try:
            asm = codegen.generate(lowered, cfg, arg_names)
        except codegen.CodegenError as e:
            raise CompileError(f"codegen: {e}") from None
    stages["06-kernel.s"] = asm
    exe = assemble(asm)
    image = sbin.write(exe)

    result = CompileResult(sig.name, stages, asm, exe, image, plans, commands)
    if dump_dir:
        dump(result, Path(dump_dir))
    return result


def dump(result: CompileResult, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for name, text in result.stages.items():
        (out / name).write_text(text)
    (out / f"{result.name}.sbin").write_bytes(result.image)
    (out / "reproduce.sh").write_text(
        "#!/bin/sh\n# Re-runs the iree-opt stages of this compilation by hand.\nset -e\n"
        f"cd \"$(dirname \"$0\")\"\n" + "\n".join(result.commands) + "\n")
