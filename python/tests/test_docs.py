"""Keeps the hardware documentation in sync with arch/isa.json."""

import re

from sava.isa import load_isa


def test_every_instruction_and_csr_is_documented(repo):
    doc = (repo / "docs" / "hardware" / "sava-v0-isa.md").read_text()
    isa = load_isa()
    # Instructions appear as `mnemonic operands...`, CSRs as `NAME`.
    missing = [m for m in isa.instrs if not re.search(rf"`{re.escape(m)}[ `]", doc)]
    missing += [c for c in isa.csrs if f"`{c}`" not in doc]
    assert not missing, f"undocumented in sava-v0-isa.md: {', '.join(missing)}"
