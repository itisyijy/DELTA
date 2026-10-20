"""Figure: how DELTA disposes of every test window (adaptation efficiency).

The three categories partition the window set exactly (verified: they sum to 1
in all four datasets).  Reset is NOT a skip category — it is a post-update event —
so it is annotated separately rather than stacked.

Direct value labels are mandatory here, not decorative: the aqua slot sits at
2.74:1 against the chart surface, below the 3:1 line, and the documented relief
for that is visible labels.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vizstyle import (  # noqa: E402
    PAPER_WIDTH_IN,
    CAT_COLORS,
    CAT_HATCH,
    CAT_LABEL,
    DATASET_LABEL,
    DATASET_ORDER,
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

CATEGORIES = ("adaptation", "drift_skipped", "rollback_skipped")
COLUMN = {
    "adaptation": "adapt_rate",
    "drift_skipped": "drift_skip_rate",
    "rollback_skipped": "rollback_skip_rate",
}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=root / "docs/kci_revision/reset_statistics.csv")
    parser.add_argument("--variant", default="gate_1p0")
    parser.add_argument("--output", type=Path, default=root / "docs/kci_revision/fig_adaptation_decisions")
    args = parser.parse_args()

    rows = {
        r["dataset"]: r
        for r in csv.DictReader(args.input.open())
        if r["variant"] == args.variant
    }

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    apply_style()
    # 제목/부제는 논문 캡션으로 옮겼다. 축 밖 텍스트를 두지 않아야 exact_size 가 성립한다.
    # 폭은 조판이 정한 17.5 cm 고정이므로, 커진 글자만큼 세로만 늘려 여백을 만든다.
    fig, ax = plt.subplots(figsize=(PAPER_WIDTH_IN, 5.1))

    ypos = list(range(len(DATASET_ORDER)))[::-1]
    gap = 0.004  # 2px-equivalent surface gap between stacked segments

    for y, dataset in zip(ypos, DATASET_ORDER):
        row = rows[dataset]
        fractions = [float(row[COLUMN[c]]) for c in CATEGORIES]
        total = sum(fractions)
        assert abs(total - 1.0) < 1e-6, f"{dataset}: categories sum to {total}"

        left = 0.0
        for category, frac in zip(CATEGORIES, fractions):
            width = max(frac - gap, 0.0)
            ax.barh(
                y, width, left=left, height=0.52,
                color=CAT_COLORS[category], edgecolor=SURFACE, linewidth=lw(0.8),
                hatch=CAT_HATCH[category], zorder=2,
            )
            if frac >= 0.055:
                ax.text(
                    left + frac / 2, y, f"{frac * 100:.1f}%",
                    ha="center", va="center", fontsize=fs(8),
                    color=SURFACE if category == "adaptation" else TEXT_PRIMARY,
                    zorder=3,
                )
            left += frac

        # 라벨을 넣기엔 얇은 구간은 막대 바로 위에 적는다 (축 안쪽 유지)
        thin = [(c, f) for c, f in zip(CATEGORIES, fractions) if f < 0.055]
        if thin:
            note = "  ".join(f"{CAT_LABEL[c].split('(')[0].strip()} {f * 100:.1f}%" for c, f in thin)
            ax.text(0.995, y + 0.36, note, ha="right", va="center",
                    fontsize=fs(7.0), color=TEXT_MUTED)

    # y 축에는 데이터셋 이름만 남긴다. 창 수·reset 은 아래 주석 블록으로 뺐다 —
    # 세 줄짜리 라벨이 지면 폭의 37 % 를 먹어 정작 막대가 좁아지던 문제.
    ax.set_yticks(ypos)
    ax.set_yticklabels([DATASET_LABEL[d] for d in DATASET_ORDER])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of test windows")
    despine(ax, keep=("bottom",))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", pad=6, labelsize=fs(9))

    # 범례는 축이 아니라 figure 에 붙인다. 커진 글자로는 3 개를 한 줄에 둘 폭이
    # 안 나와 2 열(2 행)로 접는다.
    fig.legend(
        handles=[
            Patch(facecolor=CAT_COLORS[c], edgecolor=SURFACE, hatch=CAT_HATCH[c], label=CAT_LABEL[c])
            for c in CATEGORIES
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.185), ncol=2, handlelength=1.6,
        columnspacing=2.0, handletextpad=0.6,
    )
    # y 축에서 뺀 창 수·reset 은 막대 순서 그대로 하단에 적는다.
    fig.text(
        0.022, 0.012,
        "\n".join(
            f"{DATASET_LABEL[d]} — {int(rows[d]['total_test_windows']):,} test windows"
            f" · reset {int(rows[d]['reset_count']):,}"
            f" ({float(rows[d]['reset_per_adaptation']) * 100:.2f}% of updates)"
            for d in DATASET_ORDER
        ),
        ha="left", va="bottom", fontsize=fs(7.2), color=TEXT_SECONDARY, linespacing=1.55,
    )
    # 왼쪽은 데이터셋 이름 한 줄만 받으면 되고, 오른쪽은 "100%" 눈금이 잘리지
    # 않을 만큼 남긴다. 아래는 눈금·축이름 / 범례 / 주석 블록 3 단을 받는다.
    fig.subplots_adjust(left=0.145, right=0.955, top=0.985, bottom=0.395)
    save(fig, args.output, exact_size=True)

    print()
    for dataset in DATASET_ORDER:
        row = rows[dataset]
        print(
            f"{dataset:<12} adapt={float(row['adapt_rate']):.4f} "
            f"drift_skip={float(row['drift_skip_rate']):.4f} "
            f"rollback_skip={float(row['rollback_skip_rate']):.4f} "
            f"sum={float(row['rate_sum_check']):.9f} "
            f"reset={int(row['reset_count']):,}"
        )


if __name__ == "__main__":
    main()
