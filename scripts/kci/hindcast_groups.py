"""Phase 3 — drift-gate selectivity: does the gate actually pick the harder windows?

Consumes the per-window traces written by `run_affine_local.py --dump-windows`
and compares the pre-adaptation hindcast error between the windows the gate let
through and the windows it skipped.

Two comparisons, as specified:
  (1) actual_update == True   vs  skipped_by_drift == True
  (2) drift_gate_pass == True vs  skipped_by_drift == True

Effect size is Cliff's delta.  The naive O(n1*n2) pairwise count is impossible
here (Solar alone is 1.4M windows, so ~5e11 pairs), so it is computed from the
Mann-Whitney U statistic via the exact identity

    delta = 2*U1/(n1*n2) - 1

which holds when ties are handled by midranks, since U1 = #(x>y) + 0.5*#(x==y)
and #(x>y) - #(x<y) = 2*U1 - n1*n2.  A brute-force check on a small subsample
guards the identity at runtime.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vizstyle import (  # noqa: E402
    PAPER_WIDTH_IN,
    AQUA,
    BLUE,
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

COMPARISONS = (
    ("adapted_vs_drift_skipped", "actual_update", "Updated", "Skipped by drift gate"),
    ("gatepass_vs_drift_skipped", "drift_gate_pass", "Passed drift gate", "Skipped by drift gate"),
)


def cliffs_delta(x: np.ndarray, y: np.ndarray, *, verify_n: int = 400) -> float:
    """Cliff's delta via the Mann-Whitney U identity, with a brute-force guard."""
    if x.size == 0 or y.size == 0:
        return float("nan")
    u1 = stats.mannwhitneyu(x, y, alternative="two-sided", method="asymptotic").statistic
    delta = 2.0 * u1 / (x.size * y.size) - 1.0

    rng = np.random.default_rng(0)
    xs = rng.choice(x, size=min(verify_n, x.size), replace=False)
    ys = rng.choice(y, size=min(verify_n, y.size), replace=False)
    diff = xs[:, None] - ys[None, :]
    brute = (np.sign(diff).sum()) / (xs.size * ys.size)
    sub_u1 = stats.mannwhitneyu(xs, ys, alternative="two-sided", method="asymptotic").statistic
    sub_delta = 2.0 * sub_u1 / (xs.size * ys.size) - 1.0
    assert np.isclose(brute, sub_delta, atol=1e-9), (
        f"Cliff's delta identity check failed: brute {brute} vs U-based {sub_delta}"
    )
    return float(delta)


def magnitude(delta: float) -> str:
    """Romano et al. (2006) thresholds."""
    a = abs(delta)
    if np.isnan(a):
        return "n/a"
    return "negligible" if a < 0.147 else "small" if a < 0.33 else "medium" if a < 0.474 else "large"


def describe(values: np.ndarray) -> dict:
    if values.size == 0:
        return {k: float("nan") for k in ("n", "mean", "median", "std", "q1", "q3")} | {"n": 0}
    q1, med, q3 = np.quantile(values, (0.25, 0.5, 0.75))
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(med),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "q1": float(q1),
        "q3": float(q3),
    }


def load_trace(path: Path) -> dict[str, np.ndarray]:
    import numpy.lib.recfunctions  # noqa: F401

    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    return {name: rows[name] for name in rows.dtype.names}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traces", nargs="+", type=Path, required=True,
        help="window_trace.csv files produced by --dump-windows",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "docs/kci_revision")
    parser.add_argument(
        "--consolidated",
        type=Path,
        default=root / "runs/kci_revision_20260814_w/drift_gate_window_diagnostics.csv",
        help=(
            "Where to write the concatenated per-window diagnostics. Kept under runs/ "
            "(gitignored) because it is ~100 MB; docs/ holds the summary only."
        ),
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    apply_style()

    stats_rows: list[dict] = []
    traces: dict[str, dict[str, np.ndarray]] = {}

    for path in args.traces:
        trace = load_trace(path)
        dataset = str(trace["dataset"][0])
        traces[dataset] = trace

        err = trace["l_hind_pre"].astype(np.float64)
        finite = np.isfinite(err)
        if not finite.all():
            print(f"[{dataset}] dropping {int((~finite).sum()):,} non-finite l_hind_pre rows")

        skipped = (trace["skipped_by_drift"] == 1) & finite
        for key, mask_col, label_a, label_b in COMPARISONS:
            group_a = err[(trace[mask_col] == 1) & finite]
            group_b = err[skipped]
            delta = cliffs_delta(group_a, group_b)
            for label, values in ((label_a, group_a), (label_b, group_b)):
                stats_rows.append(
                    {"dataset": dataset, "comparison": key, "group": label,
                     **describe(values),
                     "cliffs_delta": f"{delta:.6f}", "magnitude": magnitude(delta)}
                )
            print(f"[{dataset}] {key}: n={group_a.size:,} vs {group_b.size:,}  "
                  f"median {np.median(group_a):.6g} vs {np.median(group_b):.6g}  "
                  f"Cliff's delta = {delta:+.4f} ({magnitude(delta)})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.output_dir / "drift_gate_selectivity_stats.csv"
    with stats_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats_rows[0]))
        writer.writeheader()
        writer.writerows(stats_rows)
    print(f"\nWrote {stats_path}")

    # concatenate the per-run traces into the single named deliverable
    args.consolidated.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with args.consolidated.open("w") as out:
        for i, path in enumerate(args.traces):
            with path.open() as handle:
                header = handle.readline()
                if i == 0:
                    out.write(header)
                for line in handle:
                    out.write(line)
                    total += 1
    size_mb = args.consolidated.stat().st_size / 1024 ** 2
    print(f"Wrote {args.consolidated} ({total:,} rows, {size_mb:.1f} MB)")

    # ── figure: one panel per dataset, two comparisons side by side ──────────
    datasets = list(traces)
    # 2단 조판 양단 폭 17.5 cm. 제목/부제는 논문 캡션으로 옮겼다.
    # 폭은 고정이므로, 커진 글자가 들어갈 자리는 세로로만 만든다.
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(PAPER_WIDTH_IN, 4.9), squeeze=False
    )
    panel_notes = []
    for ax, dataset in zip(axes[0], datasets):
        trace = traces[dataset]
        err = trace["l_hind_pre"].astype(np.float64)
        finite = np.isfinite(err)
        skipped = err[(trace["skipped_by_drift"] == 1) & finite]
        updated = err[(trace["actual_update"] == 1) & finite]
        gatepass = err[(trace["drift_gate_pass"] == 1) & finite]

        groups = [updated, gatepass, skipped]
        labels = ["Updated", "Passed\ndrift gate", "Skipped by\ndrift gate"]
        colors = [BLUE, AQUA, ORANGE]

        bp = ax.boxplot(
            groups, positions=[0, 1, 2], widths=0.55, patch_artist=True, showfliers=False,
            medianprops=dict(color=SURFACE, linewidth=lw(1.1)),
            whiskerprops=dict(color=TEXT_SECONDARY, linewidth=lw(1.0)),
            capprops=dict(color=TEXT_SECONDARY, linewidth=lw(1.0)),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor(SURFACE)
            patch.set_linewidth(lw(1.2))

        # y 이름은 두 패널 모두에 단다. 한쪽에만 달면 그 절반만 이름 폭(86 px)을
        # 더 먹어 지면이 정확히 반반으로 나뉘지 않는다. 두 패널은 y 범위도
        # 다르므로(1e-12~1e0 vs 1e-8~1e0) 각자 이름을 갖는 편이 읽기에도 맞다.
        ax.set_ylabel("pre-update hindcast error", fontsize=fs(8))
        ax.set_yscale("log")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(labels)
        ax.set_title(dataset.capitalize(), loc="left", color=TEXT_PRIMARY, fontsize=fs(9), pad=4)
        despine(ax)
        ax.grid(axis="x", visible=False)

        d_update = cliffs_delta(updated, skipped)
        d_gate = cliffs_delta(gatepass, skipped)
        ratio = np.median(skipped) / np.median(updated)
        # 효과크기 주석은 축 안이 아니라 패널 아래에 둔다. 축 안에 두면 상자와
        # 겹치지 않게 위쪽으로 4 decade 넘는 빈 공간을 잡아야 해서, 정작 보여줄
        # 상자가 그만큼 납작해진다.
        panel_notes.append((
            ax,
            f"Cliff's d ({magnitude(d_update)} effect)\n"
            f"updated {d_update:+.3f} · gate-passed {d_gate:+.3f}\n"
            f"median error: skipped is {ratio:,.0f}× updated",
        ))
        ylo, yhi = ax.get_ylim()
        # 위쪽은 수염 끝이 눈금선에 붙지 않을 만큼만 남긴다.
        ax.set_ylim(ylo / 30.0, yhi * 3.0)
        for i, values in enumerate(groups):
            ax.text(i, ylo / 22.0, f"n={values.size:,}", ha="center", va="bottom",
                    fontsize=fs(7.0), color=TEXT_MUTED)

    # 오른쪽 여백: 마지막 눈금 라벨("Skipped by")이 축 밖으로 반쯤 튀어나오므로
    # 커진 글자 폭만큼 지면 끝에서 떼어 놓는다.
    # 지면을 정확히 반으로 가른다: 두 축의 왼쪽 끝이 딱 0.5 만큼 떨어지도록
    # left / right / wspace 를 맞췄다 (축 폭은 여전히 서로 같다).
    #   패널0 0.108-0.493 · 패널1 0.608-0.993, 각 절반의 앞 0.100 은 이름+눈금 몫
    fig.subplots_adjust(left=0.108, right=0.993, top=0.935, bottom=0.285,
                        wspace=(0.5 - 0.385) / 0.385)
    # 주석도 각 절반의 왼쪽 끝에 건다 -> 패널 덩어리와 시작선이 맞는다.
    for column, (_, note) in enumerate(panel_notes):
        fig.text(
            0.008 + column * 0.500, 0.015, note,
            ha="left", va="bottom", fontsize=fs(9.0), color=TEXT_SECONDARY,
            linespacing=1.45,
        )
    save(fig, args.output_dir / "fig_drift_gate_selectivity", exact_size=True)


if __name__ == "__main__":
    main()
