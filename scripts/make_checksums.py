"""Write SHA-256 checksums for the tracked release artifacts.

Usage:
    python scripts/make_checksums.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "paper/main.pdf",
    "models/kirk_glassbox.json",
    "models/kirk_glassbox.npz",
    "models/kirk_glassbox_certificate.json",
    "models/kirk_glassbox_compact.json",
    "models/kirk_glassbox_compact_certificate.json",
    "models/kirk_glassbox_full.json",
    "models/kirk_glassbox_full_certificate.json",
    "models/kirk_residual.json",
    "models/kirk_verify.npz",
    "models/termgraph_report.json",
    "assets/figures/fig_architecture.png",
    "assets/figures/fig_budget.png",
    "assets/figures/fig_edit.png",
    "assets/figures/fig_five_seed.png",
    "assets/figures/fig_parallel.png",
    "assets/figures/fig_scaling.png",
    "assets/figures/fig_synthetic.png",
    "assets/figures/fig_terms_main.png",
    "assets/figures/fig_terms_threeway.png",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    lines = []
    for rel in FILES:
        path = ROOT / rel
        if not path.exists():
            raise FileNotFoundError(rel)
        lines.append(f"{sha256(path)}  {rel}")
    (ROOT / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(f"wrote SHA256SUMS ({len(lines)} files)")


if __name__ == "__main__":
    main()
