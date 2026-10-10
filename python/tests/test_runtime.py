import numpy as np
import pytest

from conftest import requires_sim
from sava import sbin
from sava.asm import assemble
from sava.runtime import SimError, from_bf16, run_executable, to_bf16

pytestmark = requires_sim


def relu_scale_image(repo, drop_line: str | None = None) -> bytes:
    src = (repo / "examples" / "asm" / "relu_scale.s").read_text()
    if drop_line:
        src = "\n".join(l for l in src.splitlines() if l.strip().split("#")[0].strip() != drop_line)
    return sbin.write(assemble(src))


def test_hand_written_kernel(repo):
    x = np.random.default_rng(1).standard_normal(8192).astype(np.float32)
    (y,), stats = run_executable(relu_scale_image(repo), [x])
    np.testing.assert_array_equal(y, np.maximum(x, 0) * 2)
    assert stats.dma_bytes_in == stats.dma_bytes_out == x.nbytes
    assert stats.vpu_ops == 16 and stats.dma_ops == 16


@pytest.mark.parametrize("missing", ["wait dma0", "wait vpu", "wait dma1"])
def test_missing_wait_is_a_hazard(repo, missing):
    x = np.ones(8192, np.float32)
    with pytest.raises(SimError, match="hazard") as e:
        run_executable(relu_scale_image(repo, drop_line=missing), [x])
    assert e.value.status == "HAZARD"


def test_non_strict_mode_runs_racy_program(repo):
    x = np.ones(8192, np.float32)
    run_executable(relu_scale_image(repo, drop_line="wait dma1"), [x], strict_hazards=False)


def test_argument_validation(repo):
    with pytest.raises(ValueError, match="expects f32"):
        run_executable(relu_scale_image(repo), [np.ones(8192, np.float64)])
    with pytest.raises(ValueError, match="missing input"):
        run_executable(relu_scale_image(repo), [])


def test_trap_reports_pc():
    image = sbin.write(assemble("li x9, 0x7fffffff\nlw x10, x9, 0\nhalt\n"))
    with pytest.raises(SimError, match=r"pc=1 \(lw"):
        run_executable(image, [])


def test_timeout():
    image = sbin.write(assemble("l: jmp l\n"))
    with pytest.raises(SimError, match="budget"):
        run_executable(image, [], max_instructions=1000)


def test_bf16_helpers():
    x = np.array([1.0, -2.5, 3.140625, 1e-3], np.float32)
    np.testing.assert_allclose(from_bf16(to_bf16(x)), x, rtol=1e-2)
    assert to_bf16(np.float32([1.0]))[0] == 0x3F80
