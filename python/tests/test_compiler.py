import numpy as np
import pytest

from conftest import requires_sim
from sava.compiler import CompileError, compile_mlir
from sava.compiler.tiling import choose_matmul_tiles
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


def test_stages_are_recorded(tmp_path):
    r = compile_mlir(matmul_mlir(64, 64, 64), dump_dir=tmp_path)
    for stage in ["00-input.mlir", "01-dps.mlir", "02-transform.mlir", "03-tiled.mlir",
                  "04-bufferized.mlir", "05-lowered.mlir", "06-kernel.s", "mm.sbin", "reproduce.sh"]:
        assert (tmp_path / stage).exists(), stage
    assert "bufferization.materialize_in_destination" in r.stages["01-dps.mlir"]
    assert "memref<64x64xf32, 1>" in r.stages["04-bufferized.mlir"]
    assert "mxu.mm" in r.asm and "dma.ld" in r.asm and "dma.st" in r.asm
    assert [(a.name, a.kind) for a in r.executable.args] == [("in0", "in"), ("in1", "in"), ("out0", "out")]


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
