"""Figure: per-client relative delta-MSE distribution (DELTA vs backbone), 1,350 clients.

Reads docs/kci_revision/client_relative_delta_mse.csv (produced by client_delta_mse.py).

Axis note: IQRs are ~0.03-0.14 percentage points wide while the tails reach
-3.06 and +4.31, so a linear axis collapses every box to a line.  The y axis is
therefore symlog with a linear window of +/-0.2 around zero — the whole linear
window covers every median and quartile, and the tails stay at their true values
rather than being clipped or dropped.
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vizstyle import (  # noqa: E402
    PAPER_WIDTH_IN,
    BLUE,
    DATASET_LABEL,
    DATASET_ORDER,
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

LINTHRESH = 0.2
NO_HARM = 0.5


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=root / "docs/kci_revision/client_relative_delta_mse.csv")
    parser.add_argument("--output", type=Path, default=root / "docs/kci_revision/fig_client_relative_delta_mse")
    args = parser.parse_args()

    values = collections.defaultdict(list)
    degenerate = collections.defaultdict(list)
    for row in csv.DictReader(args.input.open()):
        rel = float(row["relative_delta_mse_percent"])
        values[row["dataset"]].append(rel)
        if int(row["degenerate_backbone_mse"]):
            degenerate[row["dataset"]].append((row["client_id"], rel, float(row["mse_backbone"])))

    import matplotlib.pyplot as plt

    apply_style()
    # 2단 조판 양단 폭 17.5 cm = 6.89 in. 제목/부제는 논문 캡션으로 옮겼다.
    # 폭은 고정이므로 커진 글자만큼 세로만 늘린다.
    fig, ax = plt.subplots(figsize=(PAPER_WIDTH_IN, 4.4))

    data = [np.array(values[d]) for d in DATASET_ORDER]
    positions = np.arange(len(DATASET_ORDER))

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.62,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color=SURFACE, linewidth=lw(1.3)),
        whiskerprops=dict(color=TEXT_SECONDARY, linewidth=lw(1.0)),
        capprops=dict(color=TEXT_SECONDARY, linewidth=lw(1.0)),
        flierprops=dict(
            marker="o", markersize=lw(3.0), markerfacecolor="none",
            markeredgecolor=TEXT_MUTED, markeredgewidth=lw(0.7),
        ),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(BLUE)
        patch.set_edgecolor(SURFACE)
        patch.set_linewidth(lw(1.2))

    ax.axhline(0.0, color=TEXT_SECONDARY, linewidth=lw(1.0), zorder=1)
    ax.axhline(NO_HARM, color=ORANGE, linewidth=lw(1.2), linestyle=(0, (4, 3)), zorder=1)

    ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=1.8)
    ax.set_yticks([-3, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 3])
    ax.set_yticklabels(["-3", "-1", "-0.5", "-0.2", "0", "0.2", "0.5", "1", "3"])
    ax.axhspan(-LINTHRESH, LINTHRESH, color=GRID, alpha=0.35, zorder=0, linewidth=0)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{DATASET_LABEL[d]}\nn={len(values[d])}" for d in DATASET_ORDER]
    )
    # 한 줄이면 세로로 둬도 폭을 거의 안 먹는다. symlog 축 설명은 라벨에서 빼고
    # 논문 캡션으로 넘겼다 (모듈 docstring 에 근거 유지).
    ax.set_ylabel("Relative ΔMSE vs backbone (%)")
    despine(ax)
    ax.grid(axis="x", visible=False)

    # annotate every client above the no-harm margin — never silently dropped
    for i, dataset in enumerate(DATASET_ORDER):
        arr = np.array(values[dataset])
        for cid, rel, base in degenerate[dataset]:
            if rel <= NO_HARM:
                continue
            ax.annotate(
                f"client {cid}: +{rel:.2f}%,  backbone MSE {base:.1e}\n"
                f"degenerate baseline — kept, not removed",
                xy=(i - 0.08, rel), xytext=(i - 0.20, rel),
                fontsize=fs(8.0), color=ORANGE, ha="right", va="center",
                linespacing=1.45,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=SURFACE,
                          edgecolor=ORANGE, linewidth=lw(0.7), alpha=0.95),
                arrowprops=dict(arrowstyle="-", color=ORANGE, linewidth=lw(0.9),
                                shrinkA=4, shrinkB=3),
            )
        over = int((arr > NO_HARM).sum())
        # 글자를 키운 뒤 "... over +0.5%" 원문은 한 칸(1.29 in)을 넘어 옆 라벨과
        # 겹친다. 부등호로 줄여 칸 안에 들어오게 한다.
        # y 는 축 비율로 고정한다. symlog + linscale 을 키우면 데이터 좌표상의
        # -4.4 가 축 바닥에 붙어버려 라벨이 축선과 겹친다.
        ax.text(
            i, 0.02, f"{over} client{'' if over == 1 else 's'} >+0.5%",
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom",
            fontsize=fs(7.0), color=ORANGE if over else TEXT_MUTED,
        )

    # linscale 을 키운 만큼 로그 구간이 압축된다. 아래로는 -3.06 꼬리와 카운트
    # 라벨이 겹치지 않을 만큼, 위로는 주석 상자가 잘리지 않을 만큼 잡는다.
    ax.set_ylim(-12.0, 30.0)
    ax.set_xlim(-0.6, len(DATASET_ORDER) - 0.4)
    # 왼쪽은 눈금 숫자 + 한 줄 라벨, 아래는 2줄 눈금 라벨만 받으면 된다.
    fig.subplots_adjust(left=0.098, right=0.995, top=0.985, bottom=0.125)
    save(fig, args.output, exact_size=True)

    print(f"\n{'dataset':<12}{'n':>5}{'median':>9}{'Q1':>9}{'Q3':>9}{'p95':>9}{'max':>9}{'>0.5%':>7}")
    for dataset in DATASET_ORDER:
        arr = np.array(values[dataset])
        q1, med, q3, p95 = np.quantile(arr, (0.25, 0.5, 0.75, 0.95))
        print(f"{dataset:<12}{len(arr):>5}{med:>9.4f}{q1:>9.4f}{q3:>9.4f}{p95:>9.4f}"
              f"{arr.max():>9.4f}{int((arr > NO_HARM).sum()):>7}")


if __name__ == "__main__":
    main()
