"""Per-client relative delta-MSE of DELTA vs the frozen backbone (all 1,350 clients).

Pure post-processing of existing metrics.json artifacts — no run is executed.
Emits the full per-client vector (which report_delta_distributions.py computes
and discards) plus the summary statistics the manuscript needs.
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

# a backbone MSE this small makes the relative metric meaningless (division blowup)
DEGENERATE_BASE_MSE = 1e-6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", type=Path, default=DEFAULT_STAMP)
    parser.add_argument("--variant", default="gate_1p0")
    parser.add_argument(
        "--space", default="zscore", choices=("raw", "zscore"),
        help="본문 기준. zscore 면 gate_1p0 을 교정 DELTA run 으로 치환한다.",
    )
    parser.add_argument(
        "--delta-root", type=Path,
        default=Path(__file__).resolve().parents[2] / "runs/kci_revision_20260814_b1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs/kci_revision",
    )
    args = parser.parse_args()

    sources = load_sources(args.stamp)
    rows: list[dict] = []
    summary: list[dict] = []

    for dataset in DATASETS:
        backbone, _ = load_run(sources[dataset, "backbone"])
        variant_dir = sources[dataset, args.variant]
        if args.space == "zscore" and args.variant == "gate_1p0":
            cand = args.delta_root / f"{dataset}_zscore_gate1p0"
            leaves = sorted(p.parent for p in cand.rglob("metrics.json"))
            if not leaves:
                raise FileNotFoundError(f"corrected DELTA run missing: {cand}")
            variant_dir = leaves[-1]
        variant, _ = load_run(variant_dir)
        ids = sorted(set(backbone["per_client"]) & set(variant["per_client"]))
        base = np.array([backbone["per_client"][i]["mse"] for i in ids])
        gate = np.array([variant["per_client"][i]["mse"] for i in ids])
        rel = 100.0 * (gate - base) / base

        for cid, b, g, r in zip(ids, base, gate, rel):
            rows.append(
                {
                    "dataset": dataset,
                    "client_id": cid,
                    "mse_backbone": f"{b:.12g}",
                    "mse_delta": f"{g:.12g}",
                    "delta_mse": f"{g - b:.12g}",
                    "relative_delta_mse_percent": f"{r:.9f}",
                    "exceeds_no_harm_0p5": int(r > 0.5),
                    "degenerate_backbone_mse": int(b < DEGENERATE_BASE_MSE),
                }
            )

        q1, med, q3, p95 = np.quantile(rel, (0.25, 0.5, 0.75, 0.95))
        degenerate = base < DEGENERATE_BASE_MSE
        summary.append(
            {
                "dataset": dataset,
                "n": len(ids),
                "median": f"{med:.6f}",
                "q1": f"{q1:.6f}",
                "q3": f"{q3:.6f}",
                "p95": f"{p95:.6f}",
                "max": f"{rel.max():.6f}",
                "min": f"{rel.min():.6f}",
                "count_gt_0p5": int((rel > 0.5).sum()),
                "count_gt_0": int((rel > 0).sum()),
                "n_degenerate_backbone_mse": int(degenerate.sum()),
                "worst_client_id": ids[int(np.argmax(rel))],
                "worst_client_backbone_mse": f"{base[int(np.argmax(rel))]:.6g}",
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_client = args.output_dir / "client_relative_delta_mse.csv"
    with per_client.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.output_dir / "client_relative_delta_mse_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print(f"Wrote {per_client} ({len(rows)} clients)")
    print(f"Wrote {summary_path}\n")
    hdr = f"{'dataset':<12}{'n':>5}{'median':>10}{'Q1':>10}{'Q3':>10}{'p95':>10}{'max':>10}{'>0.5%':>7}{'degen':>7}"
    print(hdr)
    print("-" * len(hdr))
    for s in summary:
        print(
            f"{s['dataset']:<12}{s['n']:>5}{float(s['median']):>10.4f}{float(s['q1']):>10.4f}"
            f"{float(s['q3']):>10.4f}{float(s['p95']):>10.4f}{float(s['max']):>10.4f}"
            f"{s['count_gt_0p5']:>7}{s['n_degenerate_backbone_mse']:>7}"
        )


if __name__ == "__main__":
    main()
