"""Threshold sensitivity — NOT a selection experiment.

The published threshold (1.0) was chosen by comparing candidates on the **final
test split**: each candidate is a full TTA run over test (loop.py:424) and the
selection metric is that run's test ΔMSE (report_kcc_drift_gate_sweep.py). No
validation-based selection path exists anywhere in the repository. Selection data
and reporting data are therefore not separated, so this table is reported as a
*sensitivity analysis* only and no new optimum is picked from it.

Aggregates whichever threshold runs exist, for a given drift_score_space.
"""

from __future__ import annotations

import argparse
import csv
import json
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
    "dataset", "drift_score_space", "threshold",
    "mse", "mae", "smape", "relative_delta_mse_percent",
    "adaptation_rate", "drift_skip_rate", "rollback_skip_rate",
    "reset_count", "reset_per_adaptation",
    "n_clients", "clients_over_no_harm", "worst_client_rel_percent",
    "run_dir",
]


def find_runs(roots: list[Path], dataset: str, space: str) -> dict[float, Path]:
    """Map threshold -> run dir, from <root>/<dataset>_<space>_gate<tag>/."""
    out: dict[float, Path] = {}
    for root in roots:
        for d in sorted(root.glob(f"{dataset}_{space}_gate*")):
            tag = d.name.rsplit("_gate", 1)[1]
            leaves = sorted(p.parent for p in d.rglob("metrics.json"))
            if leaves:
                out[float(tag.replace("p", "."))] = leaves[-1]
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", type=Path, default=DEFAULT_STAMP)
    parser.add_argument(
        "--roots", nargs="+", type=Path,
        default=[root / "runs/kci_revision_20260814_b1",
                 root / "runs/kci_revision_20260814_t3g03",
                 root / "runs/kci_revision_20260814_t3g05",
                 *sorted(root.glob("runs/kci_revision_20260815_t07_*"))],
    )
    parser.add_argument("--space", default="zscore", choices=("raw", "zscore"))
    parser.add_argument("--output-dir", type=Path, default=root / "docs/kci_revision")
    args = parser.parse_args()

    sources = load_sources(args.stamp)
    rows: list[dict] = []
    by_threshold: dict[float, list[float]] = {}

    for dataset in DATASETS:
        backbone, _ = load_run(sources[dataset, "backbone"])
        # threshold 0.0 (= no gate) is the control arm and is space-independent
        runs = {0.0: sources[dataset, "control"]}
        runs.update(find_runs(args.roots, dataset, args.space))

        for threshold in sorted(runs):
            metrics, config = load_run(Path(runs[threshold]))
            s = metrics["diagnostic_summary"]
            n_clients = len(metrics["per_client"])
            ids = sorted(set(backbone["per_client"]) & set(metrics["per_client"]))
            base = np.array([backbone["per_client"][i]["mse"] for i in ids])
            arm = np.array([metrics["per_client"][i]["mse"] for i in ids])
            rel = 100.0 * (arm - base) / base
            rel_avg = 100.0 * (metrics["avg"]["mse"] - backbone["avg"]["mse"]) / backbone["avg"]["mse"]

            rows.append({
                "dataset": dataset,
                "drift_score_space": "n/a (gate off)" if threshold == 0.0 else args.space,
                "threshold": threshold,
                "mse": f"{metrics['avg']['mse']:.10f}",
                "mae": f"{metrics['avg']['mae']:.10f}",
                "smape": f"{metrics['avg']['smape']:.6f}",
                "relative_delta_mse_percent": f"{rel_avg:.6f}",
                "adaptation_rate": f"{s['adapt_rate']:.9f}",
                "drift_skip_rate": f"{s['drift_skip_rate']:.9f}",
                "rollback_skip_rate": f"{s['rollback_skip_rate']:.9f}",
                "reset_count": round(s["reset_steps"] * n_clients),
                "reset_per_adaptation": f"{s['reset_rate_given_adapt']:.9f}",
                "n_clients": n_clients,
                "clients_over_no_harm": int((rel > NO_HARM).sum()),
                "worst_client_rel_percent": f"{rel.max():.6f}",
                "run_dir": str(runs[threshold]),
            })
            by_threshold.setdefault(threshold, []).append(s["adapt_rate"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "threshold_sensitivity.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows, space={args.space})")

    thresholds = sorted(by_threshold)
    print(f"\n{'threshold':>10}" + "".join(f"{d[:11]:>13}" for d in DATASETS)
          + f"{'평균 적응률':>14}{'no-harm':>10}")
    print("-" * (10 + 13 * len(DATASETS) + 24))
    for t in thresholds:
        cells = []
        worst_ok = True
        for d in DATASETS:
            r = next((x for x in rows if x["dataset"] == d and x["threshold"] == t), None)
            if r is None:
                cells.append(f"{'-':>13}")
                worst_ok = False
                continue
            v = float(r["relative_delta_mse_percent"])
            cells.append(f"{v:>13.4f}")
            worst_ok &= v <= NO_HARM
        n = len(by_threshold[t])
        mean_adapt = sum(by_threshold[t]) / n
        flag = "OK" if (worst_ok and n == len(DATASETS)) else ("일부" if n < len(DATASETS) else "위반")
        print(f"{t:>10.1f}" + "".join(cells) + f"{mean_adapt:>14.4f}{flag:>10}")
    print("\nΔMSE(%) vs backbone. threshold 0.0 = gate off (DELTA-).")
    print("이 표는 selection이 아니라 sensitivity analysis다 — 후보 평가에 final test가 쓰였다.")


if __name__ == "__main__":
    main()
