"""Phase 6 — measured adaptation overhead (manuscript: 실측 시간/메모리).

Aggregates the timing runs produced by scripts/run_kci_overhead.sh. Those runs use
`--timing`, which synchronizes CUDA around every window and forces max_workers=1;
they were executed serially with the machine otherwise idle.

Two latency figures are reported and they mean different things:
  round_sec_*      — the 15-round warm-up-excluded sample the existing instrument
                     already used (efficiency.py:_sample_window). Kept for continuity.
  all_round_sec_*  — over every window of the run. This is the honest per-window
                     latency and is what the manuscript should quote.

Total runtime is reported as measured, plus an extrapolation to the full client
set for datasets whose timing run was capped with max_clients (per-window latency
does not depend on how many clients are in the run; total runtime does).
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ARMS = (
    ("backbone", "Backbone only (no TTA)"),
    ("no_gate", "DELTA- (always adapt)"),
    ("delta", "DELTA (drift gate 1.0)"),
)
FULL_CLIENTS = {"murata": 30, "solar": 137, "electricity": 321, "traffic": 862}

FIELDS = [
    "dataset", "arm", "description", "n_clients_measured", "n_clients_full",
    "windows_per_client", "total_windows_measured",
    "wall_clock_sec", "wall_clock_sec_extrapolated_full",
    "latency_per_window_mean_ms", "latency_per_window_median_ms",
    "latency_per_window_p95_ms",
    "adapt_sec_median_ms", "adapt_sec_p95_ms",
    "peak_vram_allocated_mb", "peak_vram_reserved_mb", "peak_rss_mb",
    "synchronized", "max_workers", "run_dir",
]


def latest(path: Path) -> Path | None:
    runs = sorted(p.parent for p in path.rglob("metrics.json"))
    return runs[-1] if runs else None


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root / "runs/kci_revision_20260815_ov")
    parser.add_argument("--output-dir", type=Path, default=root / "docs/kci_revision")
    args = parser.parse_args()

    rows: list[dict] = []
    for dataset in ("murata", "solar", "electricity", "traffic"):
        for arm, description in ARMS:
            run_dir = latest(args.root / f"{dataset}_{arm}")
            if run_dir is None:
                continue
            m = json.loads((run_dir / "metrics.json").read_text())
            n_measured = len(m["per_client"])
            n_full = FULL_CLIENTS[dataset]
            rounds = m.get("round_count", 0)
            per_client_windows = rounds // n_measured if n_measured else 0
            wall = m.get("wall_clock_sec", 0.0)
            rows.append({
                "dataset": dataset,
                "arm": arm,
                "description": description,
                "n_clients_measured": n_measured,
                "n_clients_full": n_full,
                "windows_per_client": per_client_windows,
                "total_windows_measured": rounds,
                "wall_clock_sec": f"{wall:.3f}",
                "wall_clock_sec_extrapolated_full": (
                    f"{wall * n_full / n_measured:.3f}" if n_measured else ""
                ),
                "latency_per_window_mean_ms": f"{1000 * wall / rounds:.4f}" if rounds else "",
                "latency_per_window_median_ms": f"{1000 * m.get('all_round_sec_median', 0):.4f}",
                "latency_per_window_p95_ms": f"{1000 * m.get('all_round_sec_p95', 0):.4f}",
                "adapt_sec_median_ms": f"{1000 * m.get('local_adapt_sec_median', 0):.4f}",
                "adapt_sec_p95_ms": f"{1000 * m.get('local_adapt_sec_p95', 0):.4f}",
                "peak_vram_allocated_mb": f"{m.get('peak_vram_mb', 0):.2f}",
                "peak_vram_reserved_mb": f"{m.get('peak_vram_reserved_mb', 0):.2f}",
                "peak_rss_mb": f"{m.get('peak_rss_mb', 0):.1f}",
                "synchronized": m.get("measurement_synchronized", False),
                "max_workers": m.get("max_workers", 1),
                "run_dir": str(run_dir),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "adaptation_overhead.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)")

    # environment provenance — required for any timing claim
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"
        torch_v, cuda_v = torch.__version__, torch.version.cuda
    except Exception:
        gpu = torch_v = cuda_v = "n/a"
    try:
        cpu = subprocess.run(["bash", "-lc", "lscpu | grep 'Model name' | head -1"],
                             capture_output=True, text=True, timeout=10).stdout.split(":", 1)[-1].strip()
    except Exception:
        cpu = "n/a"
    env = {
        "gpu": gpu, "cpu": cpu, "torch": torch_v, "cuda": cuda_v,
        "python": platform.python_version(), "platform": platform.platform(),
    }
    (args.output_dir / "adaptation_overhead_env.json").write_text(json.dumps(env, indent=2) + "\n")
    print(f"Wrote {args.output_dir / 'adaptation_overhead_env.json'}")
    print("\n환경: " + " | ".join(f"{k}={v}" for k, v in env.items() if k in ("gpu", "torch", "cuda")))

    hdr = (f"{'dataset':<10}{'arm':<10}{'cli':>5}{'wall(s)':>10}{'mean(ms)':>10}"
           f"{'median':>9}{'p95':>9}{'adapt med':>11}{'VRAM alloc':>11}{'reserved':>10}{'RSS(MB)':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    last = None
    for r in rows:
        if last is not None and r["dataset"] != last:
            print()
        last = r["dataset"]
        print(f"{r['dataset']:<10}{r['arm']:<10}{r['n_clients_measured']:>5}"
              f"{float(r['wall_clock_sec']):>10.1f}{float(r['latency_per_window_mean_ms']):>10.4f}"
              f"{float(r['latency_per_window_median_ms']):>9.4f}{float(r['latency_per_window_p95_ms']):>9.4f}"
              f"{float(r['adapt_sec_median_ms']):>11.4f}{float(r['peak_vram_allocated_mb']):>11.2f}"
              f"{float(r['peak_vram_reserved_mb']):>10.2f}{float(r['peak_rss_mb']):>9.1f}")

    print("\n적응 오버헤드 (backbone 대비 per-window median latency):")
    for dataset in ("murata", "traffic"):
        base = next((r for r in rows if r["dataset"] == dataset and r["arm"] == "backbone"), None)
        if base is None:
            continue
        b = float(base["latency_per_window_median_ms"])
        for arm, _ in ARMS[1:]:
            r = next((x for x in rows if x["dataset"] == dataset and x["arm"] == arm), None)
            if r is None:
                continue
            v = float(r["latency_per_window_median_ms"])
            print(f"  {dataset:<10} {arm:<10} {v:.4f} ms  =  backbone x{v / b:.2f}  (+{v - b:.4f} ms)")


if __name__ == "__main__":
    main()
