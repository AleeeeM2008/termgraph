"""Release checks: voice, assets, paths, numbers, and paper build.

Usage:
    python scripts/check_repo.py
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PAPER = ROOT / "paper" / "main.tex"
PDF = ROOT / "paper" / "main.pdf"
REPORT = ROOT / "models" / "termgraph_report.json"
FIGS = ROOT / "assets" / "figures"

problems: list[str] = []


def check(condition: bool, message: str):
    if not condition:
        problems.append(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    readme = README.read_text()
    paper = PAPER.read_text()
    report = json.loads(REPORT.read_text())

    # First-person singular, not plural.
    for name, text in (("README.md", readme), ("paper/main.tex", paper)):
        for match in re.finditer(r"\b(we|our|us)\b", text, flags=re.IGNORECASE):
            problems.append(f"{name}: first-person plural '{match.group(0)}'")

    # Journal style checks.
    for name, text in (("README.md", readme), ("paper/main.tex", paper)):
        check("—" not in text, f"{name}: em dash present")
        check("–" not in text, f"{name}: en dash present")
    check("\\usepackage[preprint,nohyperref]{jmlr2e}" in paper,
          "paper: not using jmlr2e preprint style")
    check("\\usepackage{caption}" not in paper,
          "paper: caption package reintroduced")
    check("pmery641@gmail.com" not in paper, "paper: email should live in README")

    stock_phrases = (
        "not only", "it is important to note", "in conclusion", "delve",
        "crucial", "pivotal", "landscape", "furthermore", "moreover",
        "seamless", "leverage", "comprehensive", "novel", "state-of-the-art",
        "paradigm", "realm", "underscore", "showcase", "holistic", "synergy",
        "empower", "foster", "navigate", "nuanced", "multifaceted",
        "paramount",
    )
    for name, text in (("README.md", readme), ("paper/main.tex", paper)):
        lowered = text.lower()
        for phrase in stock_phrases:
            if phrase in lowered:
                problems.append(f"{name}: stock phrase '{phrase}'")

    # Author and disclaimer.
    for name, text in (("README.md", readme), ("paper/main.tex", paper)):
        check("Patricio Alejandro Mery Caro" in text, f"{name}: missing author")
    check("pmery641@gmail.com" in readme, "README: missing email")
    check("DeepSeek V4.1 Flash" in readme, "README: missing disclaimer")

    # Figures referenced by README and paper exist.
    for path in re.findall(r"assets/figures/([\w.-]+\.png)", readme):
        check((FIGS / path).exists(), f"README figure missing: {path}")
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", paper):
        candidate = FIGS / name
        if not candidate.suffix:
            candidate = FIGS / (name + ".png")
        check(candidate.exists(), f"paper figure missing: {name}")

    # Paper compiled.
    check(PDF.exists(), "paper/main.pdf missing (run make paper)")

    # No absolute local paths in tracked source.
    python_files = list((ROOT / "src").rglob("*.py")) + list((ROOT / "experiments").rglob("*.py"))
    python_files += list((ROOT / "scripts").glob("*.py")) + list((ROOT / "tests").rglob("*.py"))
    for path in python_files:
        text = path.read_text()
        if str(Path.home()) in text:
            problems.append(f"absolute path in {path.relative_to(ROOT)}")

    # Key numbers in README/paper match the report.
    def has(text: str, value: float, digits: int = 4):
        return f"{value:.{digits}f}" in text

    perf = report["performance"]
    for text, name in ((readme, "README"), (paper, "paper")):
        check(has(text, perf["five_seed_compiled_auc"]),
              f"{name}: max AUC not in text")
        check(has(text, perf["ebm_ga2m_auc"]),
              f"{name}: EBM AUC not in text")
        check(has(text, perf["compact_5term_auc"]),
              f"{name}: compact AUC not in text")
    check(has(paper, report["hardware"]["fast_thread8_speedup"], 2),
          "paper: speedup not in text")
    check(has(readme, report["hardware"]["fast_thread8_speedup"], 2),
          "README: speedup not in text")

    # Report test status.
    result = report["tests"]["result"]
    check("passed" in result, "report: tests did not pass")

    # Release checksums must match the shipped files.
    sums = ROOT / "SHA256SUMS"
    checksum_count = 0
    check(sums.exists(), "SHA256SUMS missing (run make checksums)")
    if sums.exists():
        for line in sums.read_text().splitlines():
            if not line.strip():
                continue
            digest, rel = line.split("  ", 1)
            path = ROOT / rel
            checksum_count += 1
            check(path.exists(), f"checksum target missing: {rel}")
            if path.exists():
                check(sha256(path) == digest, f"checksum mismatch: {rel}")

    readme_figures = len(re.findall(r"assets/figures/([\w.-]+\.png)", readme))
    paper_figures = len(re.findall(r"\\includegraphics", paper))

    print("release checks")
    print(f"  README figures: {readme_figures}")
    print(f"  paper figures:  {paper_figures}")
    print(f"  tests:          {result}")
    print(f"  checksums:      {checksum_count} files")
    print(f"  paper:          {PDF.stat().st_size/1024:.0f} KB")
    print("  style:          no em dashes, no stock phrases, JMLR preprint")
    if problems:
        print("\nFAILED")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print("\nOK: release checks passed")


if __name__ == "__main__":
    main()
