"""Figure: drift-score timeline for a representative client (Murata and Electricity).

The representative client is not hand-picked: it is the client whose gate
activation rate (1 - drift_skip_rate, read from per_client_diagnostics) is
closest to that dataset's median.  Ties break on the lower client index.

Window-level drift scores are not persisted by any run, so they are recomputed
offline with the same formula the engine uses; the recomputation is checked
against each run's logged drift_skip_rate before anything is drawn.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.report_delta_distributions import (  # noqa: E402
    DEFAULT_STAMP,
    load_clients,
    load_run,
    load_sources,
)
from vizstyle import (  # noqa: E402
    AQUA,
    BLUE,
    PAPER_WIDTH_IN,
    DATASET_LABEL,
    GRID,
    ORANGE,
    SURFACE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    apply_style,
    despine,
    fs,
    lw,
    save,
)

THRESHOLD = 1.0
PANEL_DATASETS = ("murata", "electricity")


def client_scores(client, start: int, steps: int, k: int, space: str = "raw") -> np.ndarray:
    """Vectorized drift score over windows.

    space="raw"    -> scripts/tta/policy.py:41-45 (shipped formula)
    space="zscore" -> compute_drift_score_zspace: in z-space the train stats are
                      mu=0, sigma=1, so the score is max(|z_mean|, |z_std - 1|).
    """
    import torch

    windows = torch.from_numpy(client.values[:, 0]).unfold(0, k, 1)[start : start + steps]
    means = windows.mean(1).numpy().astype(np.float64)
    stds = windows.std(1, unbiased=False).numpy().astype(np.float64)
    if space == "zscore":
        return np.maximum(np.abs(means), np.abs(stds - 1.0))
    scale = max(client.global_std, 1e-8)
    return np.maximum(
        np.abs((means - client.global_mean) / scale),
        np.abs((stds - client.global_std) / scale),
    )


def pick_representative(diagnostics: dict) -> tuple[str, float, float]:
    """Client whose gate activation rate is closest to the dataset median."""
    ids = sorted(diagnostics)
    activation = np.array([1.0 - diagnostics[c]["drift_skip_rate"] for c in ids])
    median = float(np.median(activation))
    idx = int(np.argmin(np.abs(activation - median)))
    return ids[idx], float(activation[idx]), median


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", type=Path, default=DEFAULT_STAMP)
    parser.add_argument("--output", type=Path, default=root / "docs/kci_revision/fig_drift_score_timeline")
    parser.add_argument("--notes", type=Path, default=root / "docs/kci_revision/representative_client_selection.md")
    parser.add_argument("--space", default="zscore", choices=("raw", "zscore"))
    parser.add_argument("--delta-root", type=Path, default=root / "runs/kci_revision_20260814_b1")
    args = parser.parse_args()

    sources = load_sources(args.stamp)

    import matplotlib.pyplot as plt

    apply_style()
    # 폭은 조판이 정한 17.5 cm 고정. 커진 글자만큼 세로만 늘린다.
    fig, axes = plt.subplots(2, 1, figsize=(PAPER_WIDTH_IN, 6.2))

    notes = [
        "# Representative client selection (drift-score timeline figure)",
        "",
        "Selection rule, applied identically to both datasets: the client whose **gate",
        "activation rate** (`1 - per_client_diagnostics[cid].drift_skip_rate`, from the",
        "`gate_1p0` run) is **closest to that dataset's median activation rate**. No manual",
        "choice is involved; ties would break on the sorted client id.",
        "",
        "Window-level drift scores are not stored by any run, so they are recomputed offline",
        "with the formula at `scripts/tta/policy.py:41-45`. The recomputation is verified",
        "against the run's logged `drift_skip_rate` before plotting.",
        "",
    ]

    for ax, dataset in zip(axes, PANEL_DATASETS):
        run_dir = sources[dataset, "gate_1p0"]
        if args.space == "zscore":
            cand = args.delta_root / f"{dataset}_zscore_gate1p0"
            leaves = sorted(p.parent for p in cand.rglob("metrics.json"))
            if not leaves:
                raise FileNotFoundError(f"corrected DELTA run missing: {cand}")
            run_dir = leaves[-1]
        metrics, config = load_run(run_dir)
        assert config["tta"]["drift_gate_threshold"] == THRESHOLD
        assert config["tta"].get("drift_score_space", "raw") == args.space
        diagnostics = metrics["per_client_diagnostics"]
        cid, activation, median = pick_representative(diagnostics)

        clients = load_clients(config)
        by_id = {c.client_id: c for c in clients}
        client = by_id[cid]

        model = config["model"]
        k = int(config["tta"]["k_ratio"] * model["pred_len"])
        test_start, test_end = client.split_indices["test"]
        start = test_start + model["seq_len"]
        steps = test_end - (test_start + model["seq_len"] + k - 1)
        scores = client_scores(client, start, steps, k, space=args.space)

        recomputed_skip = float((scores < THRESHOLD).mean())
        logged_skip = diagnostics[cid]["drift_skip_rate"]
        assert np.isclose(recomputed_skip, logged_skip, atol=1e-12), (
            f"{dataset}/{cid}: recomputed skip {recomputed_skip} != logged {logged_skip}"
        )

        x = np.arange(scores.size)
        activated = scores >= THRESHOLD

        # full-height bands mark the skipped windows; the y range never reaches 0,
        # so filling to the score would imply a baseline that is not on screen
        blended = ax.get_xaxis_transform()
        ax.fill_between(
            x, 0, 1, where=~activated, transform=blended, color=ORANGE, alpha=0.22,
            linewidth=0, zorder=1, step="mid",
        )
        ax.plot(x, scores, color=BLUE, linewidth=lw(0.5), zorder=3)
        ax.axhline(THRESHOLD, color=TEXT_PRIMARY, linewidth=lw(1.0),
                   linestyle=(0, (4, 3)), zorder=4)

        ax.set_xlim(0, scores.size)
        ax.set_ylabel("drift score")
        # 커진 글자로는 한 줄(약 6.9 in)이 축 폭(약 5.7 in)을 넘어 잘린다. 두 줄로 나눈다.
        ax.set_title(
            f"{DATASET_LABEL[dataset]} — client {cid}\n"
            f"gate activation {activation * 100:.1f}%, "
            f"dataset median {median * 100:.1f}%",
            loc="left", color=TEXT_PRIMARY, fontsize=fs(9), pad=21,
        )
        despine(ax)
        ax.grid(axis="x", visible=False)

        lo, hi = float(scores.min()), float(scores.max())
        span_lo = min(lo, THRESHOLD)
        span_hi = max(hi, THRESHOLD)
        pad = max((span_hi - span_lo) * 0.16, 0.02)
        ax.set_ylim(span_lo - pad, span_hi + pad)
        # place the threshold caption just above the line, wherever the line lands
        _, ytop = ax.get_ylim()
        ax.annotate(
            f"threshold {THRESHOLD}",
            xy=(0.995, THRESHOLD), xycoords=("axes fraction", "data"),
            xytext=(0, 4), textcoords="offset points",
            ha="right", va="bottom", fontsize=fs(7.4), color=TEXT_SECONDARY,
        )
        exact = float(np.isclose(scores, THRESHOLD, atol=1e-9).mean())
        ax.text(
            0.005, 1.005,
            f"{scores.size:,} test windows · {activated.mean() * 100:.1f}% activate the gate"
            + (f" · {exact * 100:.1f}% sit exactly at {THRESHOLD}" if exact > 0.01 else ""),
            transform=ax.transAxes, ha="left", va="bottom", fontsize=fs(7.2), color=TEXT_MUTED,
        )

        notes.extend(
            [
                f"## {DATASET_LABEL[dataset]}",
                "",
                f"- run: `{run_dir}`",
                f"- drift_score_space: **{args.space}**",
                f"- selected client: **{cid}**",
                f"- its gate activation rate: {activation:.9f}",
                f"- dataset median activation rate: {median:.9f}",
                f"- test windows: {scores.size:,}",
                f"- recomputed drift skip rate: {recomputed_skip:.15f}",
                f"- logged `drift_skip_rate`:   {logged_skip:.15f}  (match asserted, atol 1e-12)",
                f"- score min / median / max: {lo:.6f} / {np.median(scores):.6f} / {hi:.6f}",
                f"- fraction of windows with score exactly {THRESHOLD}: {exact:.6f}",
                "",
            ]
        )
        print(f"[{dataset}] client {cid}: activation {activation:.4f} (median {median:.4f}), "
              f"skip recomputed {recomputed_skip:.12f} == logged {logged_skip:.12f}")

    axes[-1].set_xlabel("test window index")
    # 제목·부제를 캔버스 안(y<1)으로 내린다. 축 밖에 글자가 있으면 bbox="tight" 가
    # 캔버스를 넓혀 조판 폭 지정이 무의미해지므로, 이 그림도 exact_size 로 저장한다.
    fig.text(
        0.012, 0.992,
        "Drift score over the test stream for a representative client",
        ha="left", va="top", fontsize=fs(10.5), color=TEXT_PRIMARY,
    )
    fig.text(
        0.012, 0.955,
        "Shaded = window skipped by the drift gate (score below threshold).",
        ha="left", va="top", fontsize=fs(7.8), color=TEXT_SECONDARY,
    )
    fig.tight_layout(rect=(0.004, 0.004, 0.996, 0.925))
    save(fig, args.output, exact_size=True)

    args.notes.write_text("\n".join(notes))
    print(f"Wrote {args.notes}")


if __name__ == "__main__":
    main()
