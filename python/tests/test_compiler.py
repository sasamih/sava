import numpy as np
import pytest

from conftest import requires_sim
from iree.compiler import ir

from sava.compiler import CompileError, compile_mlir
from sava.compiler.tiling import choose_matmul_tiles, plan_module
from sava.isa import load_config
from sava.runtime import from_bf16, run_executable, to_bf16


def matmul_mlir(m, n, k, in_t="f32"):
    return f"""
func.func @mm(%a: tensor<{m}x{k}x{in_t}>, %b: tensor<{k}x{n}x{in_t}>) -> tensor<{m}x{n}xf32> {{
  %z = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<{m}x{n}xf32>
  %c = linalg.fill ins(%z : f32) outs(%e : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
  %r = linalg.matmul ins(%a, %b : tensor<{m}x{k}x{in_t}>, tensor<{k}x{n}x{in_t}>)
                     outs(%c : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>
  return %r : tensor<{m}x{n}xf32>
}}
"""


def test_tile_selection_respects_limits():
    cfg = load_config()
    assert choose_matmul_tiles(256, 256, 512, "f32", cfg) == (128, 128, 128)
    assert choose_matmul_tiles(96, 40, 24, "f32", cfg) == (96, 40, 24)
    tm, tn, tk = choose_matmul_tiles(1, 4096, 4096, "bf16", cfg)  # decode-style GEMV
    assert tm == 1 and tn <= 128 and tk <= 128


def plan_for(m, n, k, in_t="f32"):
    with ir.Context():
        (plan,) = plan_module(ir.Module.parse(matmul_mlir(m, n, k, in_t)), load_config())
    return plan


@pytest.mark.parametrize("m, n, k, in_t, expected", [
    (1024, 1024, 2048, "bf16", True),   # 16 K steps
    (128, 128, 256, "f32", True),       # 2 K steps: the smallest case that can overlap
    (128, 128, 128, "f32", False),      # 1 K step: nothing to prefetch
    (128, 128, 384, "f32", False),      # 3 K steps: odd, the loop is unrolled by 2
    (96, 40, 24, "f32", False),         # K fits in a single tile
    (96, 40, 24, "bf16", False),
    (1, 4096, 4096, "bf16", True),      # decode-style GEMV
])
def test_double_buffer_decision(m, n, k, in_t, expected):
    plan = plan_for(m, n, k, in_t)
    assert plan.double_buffer == expected
    if plan.double_buffer:
        # Invariant: both A/B slots and the accumulator fit in SRAM.
        assert plan.sram_bytes(2) <= load_config()["sram"]["bytes"]


def test_double_buffer_footprint():
    plan = plan_for(1024, 1024, 2048, "bf16")
    assert plan.tiles == (128, 128, 128)
    # A and B: 128x128 bf16 = 32 KiB each; C: 128x128 f32 = 64 KiB.
    assert plan.sram_bytes() == (32 + 32 + 64) * 1024
    assert plan.sram_bytes(2) == (2 * (32 + 32) + 64) * 1024


@pytest.mark.parametrize("k, loop_matmuls", [(256, 0), (512, 2), (2048, 2)])
def test_double_buffering_structure(k, loop_matmuls):
    """Two static slots per input; prologue + loop unrolled by 2 + epilogue."""
    r = compile_mlir(matmul_mlir(128, 128, k, "bf16"))
    ir_text = r.stages["04d-double-buffered.mlir"]
    assert ir_text.count("memref.alloc() alignment = 64 : memref<2x128x128xbf16, 1>") == 2
    assert "offset: 16384>" in ir_text            # slot 1 is a constant view
    assert ir_text.count("linalg.matmul") == loop_matmuls + 2   # + epilogue
    # 2 K steps need no loop; more steps keep one loop that advances by 2 tiles.
    assert ("step %c256" in ir_text) == (loop_matmuls > 0)


def test_double_buffering_can_be_disabled():
    r = compile_mlir(matmul_mlir(128, 128, 512, "bf16"), double_buffering=False)
    assert "2x128x128" not in r.stages["04d-double-buffered.mlir"]
    assert not r.plans[0].double_buffer


UNROLL_K_LOOP = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} attributes {sava.op_id = 0 : i64}
        in %root : (!transform.any_op) -> !transform.any_op
    %loop = transform.get_parent_op %mm <{op_name = "scf.for"}> : (!transform.any_op) -> !transform.op<"scf.for">
    transform.loop.unroll %loop factor = 2 : !transform.op<"scf.for">
    transform.yield
  }
}
"""


@requires_sim
def test_double_buffering_skips_unexpected_loops():
    """A user schedule that already unrolled the K loop no longer fits the pattern:
    the matmul is skipped with a remark instead of failing the compilation."""
    r = compile_mlir(matmul_mlir(128, 128, 512), post_bufferize_script=UNROLL_K_LOOP)
    assert not r.plans[0].double_buffer
    assert r.stages["04d-double-buffered.mlir"].startswith(
        "// double buffering skipped: matmul #0: cannot double buffer: expected one matmul")
    a = np.random.default_rng(2).standard_normal((128, 512), dtype=np.float32)
    b = np.random.default_rng(3).standard_normal((512, 128), dtype=np.float32)
    (c,), _ = run_executable(r.image, [a, b])
    np.testing.assert_allclose(c, a @ b, rtol=1e-4, atol=1e-3)


@requires_sim
@pytest.mark.parametrize("m, n, k, in_t", [
    (128, 128, 256, "f32"), (128, 64, 512, "f32"), (128, 256, 1024, "bf16"), (64, 128, 256, "bf16"),
])
def test_double_buffered_matmul_is_correct(m, n, k, in_t):
    r = compile_mlir(matmul_mlir(m, n, k, in_t))
    assert r.plans[0].double_buffer
    rng = np.random.default_rng(k)
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)
    if in_t == "bf16":
        a, b = to_bf16(a), to_bf16(b)
        ref = from_bf16(a).astype(np.float64) @ from_bf16(b)
    else:
        ref = a.astype(np.float64) @ b
    (c,), _ = run_executable(r.image, [a, b])   # strict hazard checking validates the waits
    assert np.max(np.abs(c - ref)) / np.max(np.abs(ref)) < 1e-5


@requires_sim
def test_double_buffering_speedup():
    src = matmul_mlir(128, 256, 2048, "bf16")
    rng = np.random.default_rng(1)
    inputs = [to_bf16(rng.standard_normal(s, dtype=np.float32)) for s in ((128, 2048), (2048, 256))]
    (c0,), plain = run_executable(compile_mlir(src, double_buffering=False).image, inputs)
    (c1,), db = run_executable(compile_mlir(src).image, inputs)
    np.testing.assert_array_equal(c0, c1)
    assert db.cycles < 0.8 * plain.cycles
    assert db.utilization("mxu") > 0.9


def test_stages_are_recorded(tmp_path):
    r = compile_mlir(matmul_mlir(64, 64, 64), dump_dir=tmp_path)
    for stage in ["00-input.mlir", "01-dps.mlir", "02-transform.mlir", "03-tiled.mlir",
                  "04-bufferized.mlir", "04a-hoisted.mlir", "04b-memref-transform.mlir",
                  "04c-scheduled.mlir", "04d-double-buffered.mlir", "05-lowered.mlir",
                  "06-kernel.s", "mm.sbin", "reproduce.sh"]:
        assert (tmp_path / stage).exists(), stage
    assert "bufferization.materialize_in_destination" in r.stages["01-dps.mlir"]
    assert "memref<64x64xf32, 1>" in r.stages["04-bufferized.mlir"]
    assert "mxu.mm" in r.asm and "dma.ld" in r.asm and "dma.st" in r.asm
    assert [(a.name, a.kind) for a in r.executable.args] == [("in0", "in"), ("in1", "in"), ("out0", "out")]


def test_sram_allocations_are_hoisted():
    r = compile_mlir(matmul_mlir(128, 128, 256), double_buffering=False)
    before, after = r.stages["04-bufferized.mlir"], r.stages["04a-hoisted.mlir"]
    first_loop = after.index("scf.for")
    assert after.count("memref.alloc") == before.count("memref.alloc") == 3
    assert all(i < first_loop for i in _find_all(after, "memref.alloc"))
    assert "memref.dealloc" not in after  # deallocs would block multibuffering
    # SRAM addresses are assigned in program order; hoisting keeps that order.
    assert "sram[0x0..0x10000)" in r.asm


def _find_all(text, needle):
    i = text.find(needle)
    while i != -1:
        yield i
        i = text.find(needle, i + 1)


def test_generated_post_bufferize_script_records_the_plan():
    r = compile_mlir(matmul_mlir(128, 128, 256))
    script = r.stages["04b-memref-transform.mlir"]
    assert "matmul #0 (sava.op_id = 0): double buffering planned" in script
    # It does not transform anything (yet).
    assert r.stages["04c-scheduled.mlir"].count("memref.alloc") == 3


def test_post_bufferize_script_is_applied():
    script = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} attributes {sava.op_id = 0 : i64}
        in %root : (!transform.any_op) -> !transform.any_op
    transform.annotate %mm "sava.visited" : !transform.any_op
    transform.yield
  }
}
"""
    r = compile_mlir(matmul_mlir(64, 64, 64), post_bufferize_script=script)
    assert "sava.visited" in r.stages["04c-scheduled.mlir"]
    assert "sava.visited" not in r.stages["04a-hoisted.mlir"]


CHAINED = """
func.func @f(%a: tensor<NxNxf32>, %b: tensor<NxNxf32>) -> tensor<NxNxf32> {
  %z = arith.constant 0.0 : f32
  %e = tensor.empty() : tensor<NxNxf32>
  %c = linalg.fill ins(%z : f32) outs(%e : tensor<NxNxf32>) -> tensor<NxNxf32>
  %ab = linalg.matmul ins(%a, %b : tensor<NxNxf32>, tensor<NxNxf32>) outs(%c : tensor<NxNxf32>) -> tensor<NxNxf32>
  %abb = linalg.matmul ins(%ab, %b : tensor<NxNxf32>, tensor<NxNxf32>) outs(%c : tensor<NxNxf32>) -> tensor<NxNxf32>
  return %abb : tensor<NxNxf32>
}
"""


@requires_sim
def test_chained_matmul_within_one_tile():
    """The intermediate result fits in one SRAM tile, so it never touches DRAM."""
    r = compile_mlir(CHAINED.replace("NxN", "64x64"))
    rng = np.random.default_rng(5)
    a, b = (rng.standard_normal((64, 64), dtype=np.float32) for _ in range(2))
    (out,), _ = run_executable(r.image, [a, b])
    np.testing.assert_allclose(out, (a @ b) @ b, rtol=1e-4, atol=1e-3)
    assert "v.copy" in r.asm  # SRAM -> SRAM hand-over of the intermediate


def test_stages_are_dumped_even_when_codegen_fails(tmp_path):
    # 256x256 intermediates are tiled, so they need a DRAM scratch buffer (not supported yet).
    chained = CHAINED.replace("NxN", "256x256")
    with pytest.raises(CompileError, match="DRAM allocations"):
        compile_mlir(chained, dump_dir=tmp_path)
    assert (tmp_path / "05-lowered.mlir").exists()
    assert not (tmp_path / "06-kernel.s").exists()
    assert "04a-hoisted.mlir is produced in Python" in (tmp_path / "reproduce.sh").read_text()


@requires_sim
@pytest.mark.parametrize("m, n, k", [(32, 32, 32), (128, 64, 256), (96, 40, 24), (8, 256, 384)])
def test_matmul_f32(m, n, k):
    r = compile_mlir(matmul_mlir(m, n, k))
    rng = np.random.default_rng(m * n + k)
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)
    (c,), stats = run_executable(r.image, [a, b])
    np.testing.assert_allclose(c, a @ b, rtol=1e-4, atol=1e-4)
    assert stats.mxu_macs == m * n * k


@requires_sim
def test_matmul_bf16():
    m, n, k = 64, 128, 256
    r = compile_mlir(matmul_mlir(m, n, k, in_t="bf16"))
    assert "csrwi MXU_DTYPE, 1" in r.asm
    rng = np.random.default_rng(7)
    a = to_bf16(rng.standard_normal((m, k), dtype=np.float32))
    b = to_bf16(rng.standard_normal((k, n), dtype=np.float32))
    (c,), stats = run_executable(r.image, [a, b])
    np.testing.assert_allclose(c, from_bf16(a) @ from_bf16(b), rtol=1e-4, atol=1e-3)


@requires_sim
def test_custom_transform_script(tmp_path):
    """Smaller tiles via a user-supplied schedule still produce correct code."""
    r0 = compile_mlir(matmul_mlir(128, 128, 128))
    script = r0.stages["02-transform.mlir"].replace("tile_sizes [128, 128, 0]", "tile_sizes [32, 64, 0]") \
                                           .replace("tile_sizes [0, 0, 128]", "tile_sizes [0, 0, 32]")
    r = compile_mlir(matmul_mlir(128, 128, 128), transform_script=script)
    a = np.random.default_rng(3).standard_normal((128, 128), dtype=np.float32)
    (c,), stats = run_executable(r.image, [a, a])
    np.testing.assert_allclose(c, a @ a, rtol=1e-4, atol=1e-4)
    assert stats.mxu_ops == 4 * 2 * 4


@pytest.mark.parametrize("src, msg", [
    ("func.func @f(%a: tensor<?x4xf32>) -> tensor<?x4xf32> { return %a : tensor<?x4xf32> }",
     "dynamic shapes"),
    ("func.func @f(%a: tensor<4xf32>) -> tensor<4xf32> { return %a : tensor<4xf32> }",
     "no linalg.matmul"),
    ("func.func @f(%a: tensor<4xf32>) { return }", "no results"),
])
def test_unsupported_inputs(src, msg):
    with pytest.raises(CompileError, match=msg):
        compile_mlir(src)
