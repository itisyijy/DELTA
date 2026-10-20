"""Phase 2 — rate-matched random gate: aggregate the seed sweep and place DELTA in it.

Reads every finished random-gate run under the given roots, computes the same
metrics reported for DELTA, and summarizes the seed distribution (mean, std,
95% CI) so DELTA's single result can be located inside it.

The random arm books its skips under skip_reason="drift_gate" (the gate override
drives the same code path), so `drift_skip_rate` is the *random* skip rate here.
Both the gate-pass rate and the final adaptation rate are reported, since the
rollback guard still runs afterwards and can move the two apart.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.report_delta_distributions import (  # noqa: E402
    DEFAULT_STAMP,
    load_run,
    load_sources,
)

NO_HARM = 0.5
PER_SEED_FIELDS = [
    "dataset", "arm", "gate_seed", "mse", "mae", "smape",
    "relative_delta_mse_percent", "gate_pass_rate", "adaptation_rate",
    "skip_rate_by_gate", "rollback_skip_rate", "reset_per_adaptation",
    "clients_over_no_harm", "worst_client_rel_percent", "run_dir",
]


def arm_metrics(run_dir: Path, backbone: dict) -> dict:
    metrics, config = load_run(run_dir)
    summary = metrics["diagnostic_summary"]
    ids = sorted(set(backbone["per_client"]) & set(metrics["per_client"]))
    base = np.array([backbone["per_client"][i]["mse"] for i in ids])
    arm = np.array([metrics["per_client"][i]["mse"] for i in ids])
    rel = 100.0 * (arm - base) / base
    return {
        "mse": metrics["avg"]["mse"],
        "mae": metrics["avg"]["mae"],
        "smape": metrics["avg"]["smape"],
        "relative_delta_mse_percent": 100.0
        * (metrics["avg"]["mse"] - backbone["avg"]["mse"]) / backbone["avg"]["mse"],
        # gate-pass = everything the gate let through, before the rollback guard
        "gate_pass_rate": summary["adapt_rate"] + summary["rollback_skip_rate"],
        "adaptation_rate": summary["adapt_rate"],
        "skip_rate_by_gate": summary["drift_skip_rate"],
        "rollback_skip_rate": summary["rollback_skip_rate"],
        "reset_per_adaptation": summary["reset_rate_given_adapt"],
        "clients_over_no_harm": int((rel > NO_HARM).sum()),
        "worst_client_rel_percent": float(rel.max()),
        "gate_seed": metrics.get("gate_seed"),
        "run_dir": str(run_dir),
    }


def ci95(values: np.ndarray) -> tuple[float, float]:
    if values.size < 2:
        return (float("nan"), float("nan"))
    sem = stats.sem(values)
    half = sem * stats.t.ppf(0.975, values.size - 1)
    return (float(values.mean() - half), float(values.mean() + half))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", type=Path, default=DEFAULT_STAMP)
    parser.add_argument(
        "--random-roots", nargs="+", type=Path, required=True,
        help="Directories holding <dataset>_random_s<seed>/ run dirs",
    )
    parser.add_argument(
        "--space", default="raw", choices=("raw", "zscore"),
        help=(
            "Which DELTA arm to compare against. MUST match the drift_score_space of "
            "the run the random arms were rate-matched to — the two arms are only "
            "comparable at the same adaptation budget."
        ),
    )
    parser.add_argument("--delta-root", type=Path, default=root / "runs/kci_revision_20260814_b1")
    parser.add_argument("--output-dir", type=Path, default=root / "docs/kci_revision")
    args = parser.parse_args()

    def delta_run(dataset: str) -> Path:
        if args.space == "raw":
            return sources[dataset, "gate_1p0"]
        path = args.delta_root / f"{dataset}_zscore_gate1p0"
        runs = sorted(p.parent for p in path.rglob("metrics.json"))
        if not runs:
            raise FileNotFoundError(f"no corrected DELTA run under {path}")
        return runs[-1]

    sources = load_sources(args.stamp)
    per_seed: list[dict] = []
    summary_rows: list[dict] = []
    markdown: list[str] = [
        "# Rate-matched random gate — summary",
        "",
        "Each random arm receives, per client, exactly the number of windows DELTA's",
        "drift gate passed at threshold 1.0, drawn uniformly without replacement from the",
        "same eligible window set. Backbone, adapter initialization, optimizer, learning",
        "rate, loss, rollback guard, reset rule, evaluation windows and preprocessing are",
        "identical — window selection is the only difference.",
        "",
    ]

    by_dataset: dict[str, list[dict]] = {}
    for random_root in args.random_roots:
        for run_parent in sorted(random_root.glob("*_random_s*")):
            dataset = run_parent.name.split("_random_s")[0]
            leaves = sorted(p.parent for p in run_parent.rglob("metrics.json"))
            if not leaves:
                continue
            backbone, _ = load_run(sources[dataset, "backbone"])
            row = arm_metrics(leaves[-1], backbone)
            by_dataset.setdefault(dataset, []).append(row)
            per_seed.append(
                {"dataset": dataset, "arm": "random gate", **{
                    k: (f"{v:.10f}" if isinstance(v, float) else v)
                    for k, v in row.items() if k in
                    {"gate_seed", "mse", "mae", "smape", "relative_delta_mse_percent",
                     "gate_pass_rate", "adaptation_rate", "skip_rate_by_gate",
                     "rollback_skip_rate", "reset_per_adaptation",
                     "clients_over_no_harm", "worst_client_rel_percent", "run_dir"}
                }}
            )

    if not by_dataset:
        print("No finished random-gate runs found under:", *args.random_roots, sep="\n  ")
        return

    for dataset, rows in sorted(by_dataset.items()):
        backbone, _ = load_run(sources[dataset, "backbone"])
        delta = arm_metrics(delta_run(dataset), backbone)
        # the two arms are only comparable at the same budget; fail loudly otherwise
        rand_pass = rows[0]["gate_pass_rate"]
        assert abs(delta["gate_pass_rate"] - rand_pass) < 1e-6, (
            f"{dataset}: DELTA gate-pass {delta['gate_pass_rate']:.6f} != random "
            f"{rand_pass:.6f} — the random arms were matched to a different run. "
            f"Pass --space {'zscore' if args.space == 'raw' else 'raw'}?"
        )
        per_seed.append(
            {"dataset": dataset, "arm": "DELTA (drift gate)", "gate_seed": "",
             **{k: (f"{v:.10f}" if isinstance(v, float) else v)
                for k, v in delta.items() if k in
                {"mse", "mae", "smape", "relative_delta_mse_percent", "gate_pass_rate",
                 "adaptation_rate", "skip_rate_by_gate", "rollback_skip_rate",
                 "reset_per_adaptation", "clients_over_no_harm",
                 "worst_client_rel_percent", "run_dir"}}}
        )

        markdown.append(f"## {dataset} — {len(rows)} random seeds")
        markdown.append("")
        markdown.append("| metric | random mean | random std | random 95% CI | DELTA | DELTA percentile in random |")
        markdown.append("|---|---:|---:|---|---:|---:|")

        for metric in ("mse", "mae", "smape", "relative_delta_mse_percent",
                       "gate_pass_rate", "adaptation_rate", "rollback_skip_rate",
                       "reset_per_adaptation", "clients_over_no_harm"):
            values = np.array([r[metric] for r in rows], dtype=float)
            lo, hi = ci95(values)
            d = float(delta[metric])
            pct = 100.0 * float((values < d).mean())
            summary_rows.append({
                "dataset": dataset, "metric": metric, "n_seeds": len(values),
                "random_mean": f"{values.mean():.10f}",
                "random_std": f"{values.std(ddof=1) if values.size > 1 else 0.0:.10f}",
                "random_ci95_low": f"{lo:.10f}", "random_ci95_high": f"{hi:.10f}",
                "random_min": f"{values.min():.10f}", "random_max": f"{values.max():.10f}",
                "delta_value": f"{d:.10f}",
                "delta_percentile_in_random": f"{pct:.2f}",
                "delta_better_than_random_mean": int(
                    d < values.mean() if metric != "gate_pass_rate" else abs(d - values.mean()) < 1e-9
                ),
            })
            markdown.append(
                f"| {metric} | {values.mean():.6g} | {values.std(ddof=1) if values.size > 1 else 0.0:.3g} | "
                f"[{lo:.6g}, {hi:.6g}] | {d:.6g} | {pct:.1f}% |"
            )
        markdown.append("")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = args.output_dir / "rate_matched_random_results.csv"
    with per_seed_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_SEED_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_seed)

    summary_path = args.output_dir / "rate_matched_random_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    md_path = args.output_dir / "rate_matched_random_summary.md"
    md_path.write_text("\n".join(markdown))

    print(f"Wrote {per_seed_path} ({len(per_seed)} rows)")
    print(f"Wrote {summary_path}")
    print(f"Wrote {md_path}")
    print("\n" + "\n".join(markdown))


if __name__ == "__main__":
    main()
