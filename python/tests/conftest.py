import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))


@pytest.fixture
def repo() -> Path:
    return REPO


def _sim_available() -> bool:
    from sava.runtime import find_library
    try:
        find_library()
        return True
    except FileNotFoundError:
        return False


requires_sim = pytest.mark.skipif(not _sim_available(), reason="simulator library not built")
