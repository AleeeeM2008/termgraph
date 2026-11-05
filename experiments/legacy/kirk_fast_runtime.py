"""Benchmark fused runtime and binary artifact against the baseline."""
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
from termgraph.fast_runtime import FastTermGraph  # noqa: E402
from termgraph.ir import TermGraph  # noqa: E402
from termgraph.runtime import predict as parallel_predict  # noqa: E402


def bench(fn, repeats=3):
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), out


def main():
    X, Y, labels, feats, tr, ho = load_data()
    rng = np.random.default_rng(0)
    n = 200_000
    Xb = X[ho][rng.choice(len(ho), n, replace=True)]

    t0 = time.perf_counter()
    g = TermGraph.from_json(ROOT / "models" / "kirk_glassbox.json")
    t_json = time.perf_counter() - t0
    fast = FastTermGraph(g)

    base_t, ref = bench(lambda: g.predict(Xb))
    fast_t, fast_out = bench(lambda: fast.predict(Xb))
    rows = [
        {"name": "baseline serial", "seconds": base_t, "rows_per_s": n / base_t,
         "speedup": 1.0, "max_abs_diff": 0.0},
        {"name": "fast serial", "seconds": fast_t, "rows_per_s": n / fast_t,
         "speedup": base_t / fast_t,
         "max_abs_diff": float(np.max(np.abs(fast_out - ref)))},
    ]
    for nj in (2, 4, 8):
        t, out = bench(lambda nj=nj: parallel_predict(
            g, Xb, n_jobs=nj, backend="thread"))
        rows.append({
            "name": f"baseline thread-{nj}", "n_jobs": nj,
            "seconds": t, "rows_per_s": n / t, "speedup": base_t / t,
            "max_abs_diff": float(np.max(np.abs(out - ref))),
        })
        t, out = bench(lambda nj=nj: fast.predict_parallel(Xb, n_jobs=nj))
        rows.append({
            "name": f"fast thread-{nj}", "n_jobs": nj,
            "seconds": t, "rows_per_s": n / t, "speedup": base_t / t,
            "max_abs_diff": float(np.max(np.abs(out - ref))),
        })
    for r in rows:
        print(
            f"{r['name']:20s} {r['seconds']:.4f}s  {r['rows_per_s']:,.0f} rows/s  "
            f"speedup {r['speedup']:.2f}x  maxdiff {r['max_abs_diff']:.2e}"
        )

    npz = ROOT / "models" / "kirk_glassbox.npz"
    g.to_npz(npz)
    t0 = time.perf_counter()
    g2 = TermGraph.from_npz(npz)
    t_npz = time.perf_counter() - t0

    out = {
        "n_rows": n,
        "n_terms": len(g.terms),
        "rows": rows,
        "artifact": {
            "json_bytes": (ROOT / "models" / "kirk_glassbox.json").stat().st_size,
            "npz_bytes": npz.stat().st_size,
            "json_load_s": t_json,
            "npz_load_s": t_npz,
            "npz_predict_max_abs_diff": float(np.max(np.abs(g2.predict(Xb[:5000]) - ref[:5000]))),
        },
    }
    path = ROOT / "experiments" / "results" / "kirk_fast_runtime.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["artifact"], indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
