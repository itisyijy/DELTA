"""Phase 4 — DELTA component ablation table (manuscript Table 7).

Four variants at the DELTA operating point, one component disabled at a time:

  DELTA            drift gate ON   rollback ON   reset ON
  w/o drift gate   drift gate OFF  rollback ON   reset ON   (= DELTA-, reused)
  w/o rollback     drift gate ON   rollback OFF  reset ON
  w/o reset        drift gate ON   rollback ON   reset OFF

`w/o drift gate` reuses the existing control runs: with drift_gate_threshold = 0.0
`should_adapt` short-circuits before the score is computed (policy.py:54-55), so
that arm is identical under either drift_score_space and needs no re-run.

Only variants whose runs have finished are emitted; missing ones are listed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.report_delta_distributions import (  # noqa: E402
    DATASETS,
    DEFAULT_STAMP,
    load_run,
    load_sources,
)

NO_HARM = 0.5
FIELDS = [
    "dataset", "variant", "drift_gate", "rollback_guard", "reset",
    "mse", "mae", "smape", "relative_delta_mse_percent",
    "gate_skip_rate", "rollback_skip_rate", "adaptation_rate",
    "reset_count", "reset_per_adaptation",
    "n_clients", "clients_over_no_harm",
    "worst_client_rel_percent", "client_rel_p95", "client_rel_p50",
    "run_dir",
]

VARIANTS = (
    ("DELTA", "ON", "ON", "ON"),
    ("w/o drift gate", "OFF", "ON", "ON"),
    ("w/o rollback guard", "ON", "OFF", "ON"),
    ("w/o reset", "ON", "ON", "OFF"),
)


def latest(path: Path) -> Path | None:
    if (path / "metrics.json").exists():
        return path
    runs = sorted(p.parent for p in path.rglob("metrics.json"))
    return runs[-1] if runs else None


def row(dataset: str, variant: tuple, run_dir: Path, backbone: dict) -> dict:
    metrics, config = load_run(run_dir)
    s = metrics["diagnostic_summary"]
    n_clients = len(metrics["per_client"])

    ids = sorted(set(backbone["per_client"]) & set(metrics["per_client"]))
    base = np.array([backbone["per_client"][i]["mse"] for i in ids])
    arm = np.array([metrics["per_client"][i]["mse"] for i in ids])
    rel = 100.0 * (arm - base) / base

    return {
        "dataset": dataset,
        "variant": variant[0],
        "drift_gate": variant[1],
        "rollback_guard": variant[2],
        "reset": variant[3],
        "mse": f"{metrics['avg']['mse']:.10f}",
        "mae": f"{metrics['avg']['mae']:.10f}",
        "smape": f"{metrics['avg']['smape']:.6f}",
        "relative_delta_mse_percent": f"{100.0 * (metrics['avg']['mse'] - backbone['avg']['mse']) / backbone['avg']['mse']:.6f}",
        "gate_skip_rate": f"{s['drift_skip_rate']:.9f}",
        "rollback_skip_rate": f"{s['rollback_skip_rate']:.9f}",
        "adaptation_rate": f"{s['adapt_rate']:.9f}",
        "reset_count": round(s["reset_steps"] * n_clients),
        "reset_per_adaptation": f"{s['reset_rate_given_adapt']:.9f}",
        "n_clients": n_clients,
        "clients_over_no_harm": int((rel > NO_HARM).sum()),
        "worst_client_rel_percent": f"{rel.max():.6f}",
        "client_rel_p95": f"{np.quantile(rel, 0.95):.6f}",
        "client_rel_p50": f"{np.quantile(rel, 0.5):.6f}",
        "run_dir": str(run_dir),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", type=Path, default=DEFAULT_STAMP)
    parser.add_argument("--delta-root", type=Path, default=root / "runs/kci_revision_20260814_b1")
    parser.add_argument("--ablation-root", type=Path, default=root / "runs/kci_revision_20260814_ab")
    parser.add_argument("--space", default="zscore", choices=("raw", "zscore"))
    parser.add_argument("--output-dir", type=Path, default=root / "docs/kci_revision")
    args = parser.parse_args()

    sources = load_sources(args.stamp)
    rows: list[dict] = []
    missing: list[str] = []

    for dataset in DATASETS:
        backbone, _ = load_run(sources[dataset, "backbone"])
        candidates = {
            "DELTA": (
                args.delta_root / f"{dataset}_{args.space}_gate1p0"
                if args.space == "zscore"
                else sources[dataset, "gate_1p0"]
            ),
            "w/o drift gate": sources[dataset, "control"],
            "w/o rollback guard": args.ablation_root / f"{dataset}_no_rollback",
            "w/o reset": args.ablation_root / f"{dataset}_no_reset",
        }
        for variant in VARIANTS:
            run_dir = latest(Path(candidates[variant[0]]))
            if run_dir is None:
                missing.append(f"{dataset} / {variant[0]}")
                continue
            rows.append(row(dataset, variant, run_dir, backbone))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "ablation_table7.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows, drift_score_space={args.space})")
    if missing:
        print("\nNOT YET AVAILABLE:")
        for item in missing:
            print(f"  - {item}")

    hdr = (f"{'dataset':<12}{'variant':<20}{'MSE':>11}{'relΔMSE%':>10}{'adapt':>8}"
           f"{'gateskip':>10}{'rbskip':>8}{'reset':>8}{'reset/ad':>10}{'>0.5%':>7}{'worst%':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    last = None
    for r in rows:
        if last is not None and r["dataset"] != last:
            print()
        last = r["dataset"]
        print(
            f"{r['dataset']:<12}{r['variant']:<20}{float(r['mse']):>11.6f}"
            f"{float(r['relative_delta_mse_percent']):>10.4f}{float(r['adaptation_rate']):>8.3f}"
            f"{float(r['gate_skip_rate']):>10.3f}{float(r['rollback_skip_rate']):>8.3f}"
            f"{r['reset_count']:>8,}{float(r['reset_per_adaptation']):>10.6f}"
            f"{r['clients_over_no_harm']:>7}{float(r['worst_client_rel_percent']):>9.4f}"
        )


if __name__ == "__main__":
    main()
