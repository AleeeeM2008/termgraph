"""Benchmark parallel deployment inference on the compiled KIRK artifact."""
from __future__ import annotations
import os

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KIRK = Path(os.environ.get("KIRK_ROOT", Path(__file__).resolve().parents[1].parent / "TESTAstronomy"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "legacy"))

from kirk_termgraph import load_data  # noqa: E402
from termgraph.ir import TermGraph  # noqa: E402
from termgraph.runtime import predict  # noqa: E402


def bench(graph, X, n_jobs, backend, repeats=3):
    predict(graph, X[:100], n_jobs=n_jobs, backend=backend)  # warm up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = predict(graph, X, n_jobs=n_jobs, backend=backend)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), out


def main():
    graph = TermGraph.from_json(ROOT / "models" / "kirk_glassbox.json")
    X, Y, labels, feats, tr, ho = load_data()
    rng = np.random.default_rng(0)
    n = 200_000
    idx = rng.choice(len(ho), n, replace=True)
    Xb = X[ho][idx]
    print(f"artifact terms={len(graph.terms)} rows={n}", flush=True)

    serial_t, ref = bench(graph, Xb, 1, "thread")
    rows = [{
        "backend": "serial", "n_jobs": 1, "seconds": serial_t,
        "rows_per_s": n / serial_t, "max_abs_diff": 0.0,
    }]
    print(f"serial        {serial_t:.3f}s  {n/serial_t:,.0f} rows/s", flush=True)
    for backend in ("thread", "process"):
        for nj in (2, 4, 8):
            t, out = bench(graph, Xb, nj, backend)
            diff = float(np.max(np.abs(out - ref)))
            rows.append({
                "backend": backend, "n_jobs": nj, "seconds": t,
                "rows_per_s": n / t, "max_abs_diff": diff,
                "speedup": serial_t / t,
            })
            print(
                f"{backend:8s} {nj:2d}  {t:.3f}s  {n/t:,.0f} rows/s  "
                f"speedup {serial_t/t:.2f}x  maxdiff {diff:.2e}",
                flush=True,
            )

    out = {
        "n_rows": n,
        "n_terms": len(graph.terms),
        "max_order": graph.max_order,
        "rows": rows,
    }
    path = ROOT / "experiments" / "results" / "kirk_parallel.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
