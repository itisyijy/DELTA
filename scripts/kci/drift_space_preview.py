"""Offline preview of the drift score before/after the unit-space correction (B1).

The shipped ``compute_drift_score`` (scripts/tta/policy.py:36-45) compares a window
that is *already* z-scored against the *raw-unit* train statistics.  This script
recomputes, for every test window of every client, both

  raw     — the shipped formula, max(|(m - mu)/s|, |(sd - sigma)/s|)
  zscore  — the unit-consistent form, max(|m|, |sd - 1|)

and reports the skip rates each would induce.  Read-only: no run is executed and
no existing artifact is touched.

The ``raw`` branch is asserted against the logged ``drift_skip_rate`` so that a
mismatch fails loudly instead of producing a plausible-looking table.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.report_delta_distributions import (  # noqa: E402
    DATASETS,
    DEFAULT_STAMP,
    load_clients,
    load_run,
    load_sources,
)

THRESHOLDS = (0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
PERCENTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
EXACT_TOL = 1e-9


def window_moments(client, start: int, steps: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-window mean and population std, in the stored (z-scored) space."""
    windows = torch.from_numpy(client.values[:, 0]).unfold(0, k, 1)[start : start + steps]
    means = windows.mean(1).numpy().astype(np.float64)
    stds = windows.std(1, unbiased=False).numpy().astype(np.float64)
    return means, stds


def scores_raw(means: np.ndarray, stds: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Shipped formula — mirrors scripts/tta/policy.py:41-45."""
    scale = max(sigma, 1e-8)
    return np.maximum(np.abs((means - mu) / scale), np.abs((stds - sigma) / scale))


def scores_zscore(means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Unit-consistent form: in z-space the train stats are mu=0, sigma=1."""
    return np.maximum(np.abs(means), np.abs(stds - 1.0))


def summarize(scores: np.ndarray) -> dict:
    quant = np.quantile(scores, PERCENTILES)
    return {
        "n_windows": int(scores.size),
        **{f"skip@{t}": float((scores < t).mean()) for t in THRESHOLDS},
        "frac_exactly_1p0": float((np.abs(scores - 1.0) < EXACT_TOL).mean()),
        **{f"p{int(q * 100):02d}": float(v) for q, v in zip(PERCENTILES, quant)},
    }


def collect(config: dict) -> dict[str, np.ndarray]:
    clients = load_clients(config)
    model = config["model"]
    k = int(config["tta"]["k_ratio"] * model["pred_len"])
    test_start, test_end = clients[0].split_indices["test"]
    start = test_start + model["seq_len"]
    steps = test_end - (test_start + model["seq_len"] + k - 1)

    raw, zsc = [], []
    for client in clients:
        means, stds = window_moments(client, start, steps, k)
        raw.append(scores_raw(means, stds, client.global_mean, client.global_std))
        zsc.append(scores_zscore(means, stds))
    return {
        "raw": np.concatenate(raw),
        "zscore": np.concatenate(zsc),
        "n_clients": len(clients),
        "windows_per_client": steps,
        "k": k,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", type=Path, default=DEFAULT_STAMP)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs/kci_revision",
    )
    args = parser.parse_args()

    sources = load_sources(args.stamp)
    rows: list[dict] = []
    report: dict[str, dict] = {}

    for dataset in DATASETS:
        metrics, config = load_run(sources[dataset, "gate_1p0"])
        assert config["tta"]["drift_gate_threshold"] == 1.0
        collected = collect(config)

        logged = metrics["diagnostic_summary"]["drift_skip_rate"]
        recomputed = float((collected["raw"] < 1.0).mean())
        assert np.isclose(recomputed, logged, atol=1e-15), (
            f"{dataset}: raw skip@1.0 {recomputed!r} != logged {logged!r}"
        )

        report[dataset] = {"logged_drift_skip_rate": logged, "k": collected["k"]}
        for space in ("raw", "zscore"):
            stats = summarize(collected[space])
            report[dataset][space] = stats
            rows.append(
                {
                    "dataset": dataset,
                    "space": space,
                    "n_clients": collected["n_clients"],
                    "windows_per_client": collected["windows_per_client"],
                    **stats,
                }
            )
        print(
            f"[{dataset}] verified raw skip@1.0 == logged drift_skip_rate "
            f"({logged:.15f}); k={collected['k']} clients={collected['n_clients']}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "drift_space_preview.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.output_dir / "drift_space_preview.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {csv_path}\nWrote {json_path}")


if __name__ == "__main__":
    main()
