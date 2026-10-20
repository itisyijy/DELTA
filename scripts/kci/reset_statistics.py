"""Emit reset / gate / rollback counters per dataset x variant from existing runs.

Read-only aggregation over the canonical ``sources.tsv`` mapping.  Nothing is
re-executed and no existing artifact is modified.

Note on units: ``diagnostic_summary`` is the *unweighted mean across clients* of
each per-client counter (scripts/tta/loop.py:530-533), so ``total_steps`` is the
window count **per client**.  Global totals are recovered by multiplying by the
client count, which this script does explicitly.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.report_delta_distributions import (  # noqa: E402
    DATASETS,
    DEFAULT_STAMP,
    load_run,
    load_sources,
)

VARIANTS = (
    ("control", "w/o drift gate (DELTA-)"),
    ("gate_0p3", "drift gate 0.3"),
    ("gate_0p5", "drift gate 0.5"),
    ("gate_1p0", "DELTA (drift gate 1.0)"),
)
# 본문 기준이 교정판(zscore)이므로 gate_1p0 은 교정 run 으로 치환한다.
# control/0.3/0.5 는 raw sweep 그대로 두면 기준이 섞이므로 space 인자로 함께 전환한다.
CORRECTED_ROOTS = {
    "gate_0p3": "runs/kci_revision_20260814_t3g03",
    "gate_0p5": "runs/kci_revision_20260814_t3g05",
    "gate_1p0": "runs/kci_revision_20260814_b1",
}

FIELDS = [
    "dataset",
    "variant",
    "description",
    "drift_gate_threshold",
    "n_clients",
    "windows_per_client",
    "total_test_windows",
    "gate_passed_windows",
    "adaptation_windows",
    "rollback_skipped_windows",
    "drift_skipped_windows",
    "reset_count",
    "adapt_rate",
    "drift_skip_rate",
    "rollback_skip_rate",
    "reset_per_total_windows",
    "reset_per_adaptation",
    "rate_sum_check",
    "run_dir",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", type=Path, default=DEFAULT_STAMP)
    parser.add_argument("--space", default="zscore", choices=("raw", "zscore"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "docs/kci_revision/reset_statistics.csv",
    )
    args = parser.parse_args()

    sources = load_sources(args.stamp)
    rows = []
    for dataset in DATASETS:
        for variant, description in VARIANTS:
            path = sources[dataset, variant]
            if args.space == "zscore" and variant in CORRECTED_ROOTS:
                # variant "gate_0p3" -> run dir "<dataset>_zscore_gate0p3"
                tag = variant.split("_", 1)[1]
                cand = (Path(__file__).resolve().parents[2]
                        / CORRECTED_ROOTS[variant] / f"{dataset}_zscore_gate{tag}")
                leaves = sorted(p.parent for p in cand.rglob("metrics.json"))
                if not leaves:
                    raise FileNotFoundError(f"corrected run missing: {cand}")
                path = leaves[-1]
            metrics, config = load_run(path)
            summary = metrics["diagnostic_summary"]
            n_clients = len(metrics["per_client"])

            # per-client means -> global counts
            steps = summary["total_steps"]
            adapted = summary["adapted_steps"]
            drift = summary["drift_skips"]
            rollback = summary["rollback_skips"]
            resets = summary["reset_steps"]

            total_windows = steps * n_clients
            adapt_windows = adapted * n_clients
            drift_windows = drift * n_clients
            rollback_windows = rollback * n_clients
            reset_windows = resets * n_clients
            # gate-passed = everything the drift gate let through (before rollback)
            gate_passed = total_windows - drift_windows

            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "description": description,
                    "drift_gate_threshold": config["tta"]["drift_gate_threshold"],
                    "n_clients": n_clients,
                    "windows_per_client": steps,
                    "total_test_windows": round(total_windows),
                    "gate_passed_windows": round(gate_passed),
                    "adaptation_windows": round(adapt_windows),
                    "rollback_skipped_windows": round(rollback_windows),
                    "drift_skipped_windows": round(drift_windows),
                    "reset_count": round(reset_windows),
                    "adapt_rate": f"{summary['adapt_rate']:.9f}",
                    "drift_skip_rate": f"{summary['drift_skip_rate']:.9f}",
                    "rollback_skip_rate": f"{summary['rollback_skip_rate']:.9f}",
                    "reset_per_total_windows": f"{reset_windows / total_windows:.9f}",
                    "reset_per_adaptation": f"{summary['reset_rate_given_adapt']:.9f}",
                    "rate_sum_check": f"{summary['adapt_rate'] + summary['drift_skip_rate'] + summary['rollback_skip_rate']:.9f}",
                    "run_dir": str(path),
                }
            )

            # the three decision categories must partition every window
            # runs predating the acceptance/hard gate (2026-04-12) omit those keys
            total_rate = (
                summary["adapt_rate"]
                + summary["drift_skip_rate"]
                + summary["rollback_skip_rate"]
                + summary.get("accept_gate_skip_rate", 0.0)
                + summary.get("hard_gate_skip_rate", 0.0)
            )
            assert abs(total_rate - 1.0) < 1e-6, f"{dataset}/{variant}: rates sum to {total_rate}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output} ({len(rows)} rows)")

    hdr = f"{'dataset':<12}{'variant':<10}{'total':>12}{'gatepass':>11}{'adapt':>11}{'reset':>8}{'reset/tot':>11}{'reset/adapt':>12}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['dataset']:<12}{r['variant']:<10}{r['total_test_windows']:>12,}"
            f"{r['gate_passed_windows']:>11,}{r['adaptation_windows']:>11,}"
            f"{r['reset_count']:>8,}{float(r['reset_per_total_windows']):>11.6f}"
            f"{float(r['reset_per_adaptation']):>12.6f}"
        )


if __name__ == "__main__":
    main()
