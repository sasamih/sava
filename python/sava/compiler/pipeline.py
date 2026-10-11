"""Compilation pipeline driver.

Each stage's IR is kept (and optionally dumped) so every step can be
inspected and re-run by hand with iree-opt; `reproduce.sh` in the dump
directory contains the exact commands. Stages are written to the dump
directory as soon as they are produced, so a failing later stage still
leaves the earlier ones on disk.

    00-input.mlir            as given (linalg on tensors)
    01-dps.mlir              results turned into destination memrefs   [frontend.py]
    02-transform.mlir        tensor-level schedule (tiling, promotion) [tiling.py]
    03-tiled.mlir            tiled + SRAM-promoted, still on tensors   [iree-opt]
    04-bufferized.mlir       memrefs (DRAM args, SRAM allocs)          [iree-opt]
    04a-hoisted.mlir         SRAM allocs moved to the function entry   [sram.py]
    04b-memref-transform.mlir memref-level schedule (multibuffer, ...) [tiling.py]
    04c-scheduled.mlir       memref-level schedule applied             [iree-opt]
    04d-double-buffered.mlir planned matmuls double buffered           [double_buffer.py]
    05-lowered.mlir          "SAVA-ready": explicit index math         [iree-opt]
    06-kernel.s              SAVA assembly                             [codegen.py]
    <name>.sbin              executable                                [sava.asm]
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
from . import codegen, double_buffer, frontend, sram, tiling


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
SCHEDULE_PASSES = [
    "--transform-interpreter",
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


class _Stages:
    """Collects stage outputs and writes them to the dump directory right away."""

    def __init__(self, dump_dir: Path | None):
        self.dump_dir = dump_dir
        self.texts: dict[str, str] = {}
        self.commands: list[str] = []
        if dump_dir:
            dump_dir.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, name: str) -> str:
        return self.texts[name]

    def __setitem__(self, name: str, text: str):
        self.texts[name] = text
        if self.dump_dir:
            (self.dump_dir / name).write_text(text)
            self._write_reproduce()

    def iree_opt(self, out: str, inp: str, flags: list[str], scripts: dict[str, Path]):
        """Runs iree-opt on stage `inp`; `scripts` maps stage names to the files passed in flags."""
        self[out] = _run_iree_opt(self[inp], flags, out)
        shown = []
        for f in flags:
            for name, path in scripts.items():
                f = f.replace(str(path), name)
            shown.append(f)
        self.commands.append(f"iree-opt {inp} {' '.join(shlex.quote(f) for f in shown)} -o {out}")
        self._write_reproduce()

    def note(self, text: str):
        self.commands.append(f"# {text}")

    def _write_reproduce(self):
        if self.dump_dir:
            (self.dump_dir / "reproduce.sh").write_text(
                "#!/bin/sh\n# Re-runs the iree-opt stages of this compilation by hand.\nset -e\n"
                "cd \"$(dirname \"$0\")\"\n" + "\n".join(self.commands) + "\n")


def compile_mlir(source: str, *, dump_dir: str | Path | None = None,
                 transform_script: str | None = None,
                 post_bufferize_script: str | None = None,
                 double_buffering: bool = True,
                 config: str = "sava-v0") -> CompileResult:
    """Compiles a linalg-on-tensors kernel to a SAVA executable.

    `transform_script` replaces the generated tensor-level schedule (stage 02),
    `post_bufferize_script` the generated memref-level schedule (stage 04b).
    `double_buffering=False` skips stage 04d (useful to measure its effect).
    """
    cfg = load_config(config)
    stages = _Stages(Path(dump_dir) if dump_dir else None)
    stages["00-input.mlir"] = source

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
    stages["04b-memref-transform.mlir"] = (post_bufferize_script
                                           or tiling.make_post_bufferize_script(plans))

    with tempfile.TemporaryDirectory() as tmp:
        script_dir = stages.dump_dir or Path(tmp)
        scripts = {}
        for name in ("02-transform.mlir", "04b-memref-transform.mlir"):
            scripts[name] = script_dir / name
            scripts[name].write_text(stages[name])

        def preload(name: str) -> str:
            return f"--transform-preload-library=transform-library-paths={scripts[name]}"

        tile_flags = list(TILE_PASSES)
        tile_flags.insert(2, preload("02-transform.mlir"))
        stages.iree_opt("03-tiled.mlir", "01-dps.mlir", tile_flags, scripts)
        stages.iree_opt("04-bufferized.mlir", "03-tiled.mlir", BUFFERIZE_PASSES, scripts)

        with ir.Context():
            bufferized = ir.Module.parse(stages["04-bufferized.mlir"])
            sram.hoist_sram_allocations(bufferized)
            stages["04a-hoisted.mlir"] = str(bufferized)
        stages.note("04a-hoisted.mlir is produced in Python: sava.compiler.sram.hoist_sram_allocations")

        stages.iree_opt("04c-scheduled.mlir", "04a-hoisted.mlir",
                        [preload("04b-memref-transform.mlir"), *SCHEDULE_PASSES], scripts)

        with ir.Context(), ir.Location.unknown():
            scheduled = ir.Module.parse(stages["04c-scheduled.mlir"])
            if double_buffering:
                remarks = double_buffer.apply_double_buffering(scheduled, plans)
            else:
                remarks = ["disabled (double_buffering=False)"]
                for plan in plans:
                    plan.double_buffer = False
            header = "".join(f"// double buffering {r}\n" for r in remarks)
            stages["04d-double-buffered.mlir"] = header + str(scheduled)
        stages.note("04d-double-buffered.mlir is produced in Python: "
                    "sava.compiler.double_buffer.apply_double_buffering")
        stages.iree_opt("05-lowered.mlir", "04d-double-buffered.mlir", LOWER_PASSES, scripts)

    arg_names = [f"in{i}" for i in range(sig.num_inputs)] + [f"out{i}" for i in range(sig.num_outputs)]
    with ir.Context():
        lowered = ir.Module.parse(stages["05-lowered.mlir"])
        try:
            asm = codegen.generate(lowered, cfg, arg_names)
        except codegen.CodegenError as e:
            where = f" (stages so far are in {stages.dump_dir})" if stages.dump_dir else ""
            raise CompileError(f"codegen{where}: {e}") from None
    stages["06-kernel.s"] = asm
    exe = assemble(asm)
    image = sbin.write(exe)
    if stages.dump_dir:
        (stages.dump_dir / f"{sig.name}.sbin").write_bytes(image)

    return CompileResult(sig.name, stages.texts, asm, exe, image, plans, stages.commands)
