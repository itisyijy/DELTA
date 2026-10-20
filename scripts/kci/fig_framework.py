"""그림 1 — DELTA 전체 프레임워크 (벡터 재작성).

원고의 그림 1 은 편집 가능한 원본 없이 docx 안에 래스터 PNG(8678x5754)로만 들어 있어,
동일 구조를 벡터로 재작성한다. 원본 대비 바뀐 점은 세 가지뿐이며 모두 근거가 있다:

1. `Global scaling + RevIN` -> `Per-client scaling + RevIN` (2곳)
   구현은 클라이언트별 train split 에 독립적으로 StandardScaler 를 적합한다
   (scripts/data/loader.py:77-93, parquet :134-172). pooled fit 코드는 없다.
2. ③ 단계에 **Reset rule** 블록 추가.
   원본 그림은 Drift gate -> Rollback guard -> Update φ only 3 단계만 그렸는데,
   §3.2 본문은 4 단계로 서술한다. 본문과 일치시킨다.
3. 최종 문구에서 "lightweight" 를 파라미터 기준임이 드러나도록 한정.
   실측상 연산 비용은 창당 약 11 배다 (adaptation_overhead.csv).

선택 규칙 순서는 구현(engine.py:264-420)과 일치하므로 그대로 둔다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vizstyle import (  # noqa: E402
    BLUE,
    GRID,
    ORANGE,
    PAPER_WIDTH_IN,
    SURFACE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    apply_style,
    fs,
    lw,
    save,
)

# 이 그림만 전역 배율(1.35)보다 낮게 간다. 상자 폭이 3 단 배치로 고정되어 있어
# 전역 배율을 그대로 받으면 "Per-client scaling + RevIN" 같은 긴 문구가 상자를
# 넘친다. 아래에서 3 단 폭도 함께 재배분해 (③ 이 남던 폭을 ①·② 로) 여유를 만들었다.
DIAGRAM_SCALE = 1.25

BOX_FACE = "#eef2fd"
BOX_EDGE = "#4a6fd4"
STAGE_EDGE = "#c9c8c2"
GREEN_FACE = "#eef9f2"
GREEN_EDGE = "#1baf7a"


def rbox(ax, x, y, w, h, text, *, face=BOX_FACE, edge=BOX_EDGE, size=7.4, color=TEXT_PRIMARY, weight="normal"):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
        linewidth=lw(0.9), facecolor=face, edgecolor=edge, zorder=2,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs(size, DIAGRAM_SCALE), color=color, zorder=3,
            linespacing=1.35, weight=weight)


def arrow(ax, x1, y1, x2, y2, *, color=BOX_EDGE, rad=0.0):
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
        linewidth=lw(0.9), color=color, zorder=2,
        connectionstyle=f"arc3,rad={rad}",
    ))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=root / "docs/kci_revision/fig_framework")
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    apply_style()
    # 폭은 조판이 정한 17.5 cm 고정. 커진 글자만큼 세로만 늘린다.
    fig, ax = plt.subplots(figsize=(PAPER_WIDTH_IN, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── 단계 컨테이너 ────────────────────────────────────────────────────────
    # ③ 의 상자는 문구가 짧아 폭이 남았고, ①·② 는 커진 글자에 폭이 모자란다.
    # 총 폭은 그대로 두고 ③ -> ①·② 로 재배분한다.
    stages = [
        (0.005, 0.290, "① Offline / Pretraining"),
        (0.310, 0.335, "② Online / Test Time"),
        (0.714, 0.281, "③ Selective Adaptation"),
    ]
    for sx, sw, title in stages:
        ax.add_patch(FancyBboxPatch(
            (sx, 0.235), sw, 0.735, boxstyle="round,pad=0,rounding_size=0.010",
            linewidth=lw(0.8), facecolor="none", edgecolor=STAGE_EDGE, zorder=1,
        ))
        ax.text(sx + 0.012, 0.935, title, ha="left", va="center",
                fontsize=fs(8.2, 1.15), color=TEXT_PRIMARY,
                weight="bold", zorder=3)

    # ── ① 사전학습 ───────────────────────────────────────────────────────────
    col1 = [
        "Historical train windows",
        "Per-client scaling + RevIN",     # 원본: Global scaling + RevIN
        "Train DLinear backbone  θ*",
        "Export pretrained  θ*",
    ]
    y, h, gap = 0.775, 0.115, 0.043
    for i, t in enumerate(col1):
        yy = y - i * (h + gap)
        rbox(ax, 0.022, yy, 0.256, h, t)
        if i < len(col1) - 1:
            arrow(ax, 0.150, yy, 0.150, yy - gap)

    ax.text(0.322, 0.898, "client c, window t", ha="left", va="center",
            fontsize=fs(6.8, DIAGRAM_SCALE), color=TEXT_MUTED, zorder=3)

    # ── ② 온라인 ─────────────────────────────────────────────────────────────
    col2 = [
        "Recent window  $X_{recent}$",
        "Per-client scaling + RevIN",     # 원본: Global scaling + RevIN
        "Frozen backbone  θ*",
        "Backbone forecast  $\\hat{Y}_{backbone}$",
        "Time-affine adapter  φ = (γ, δ)\n$\\hat{Y}_{final} = γ ⊙ \\hat{Y}_{backbone} + δ$",
    ]
    y2, h2, gap2 = 0.790, 0.088, 0.036
    ys2 = []
    for i, t in enumerate(col2):
        hh = h2 + (0.036 if i == len(col2) - 1 else 0.0)
        yy = y2 - i * (h2 + gap2) - (0.0 if i < len(col2) - 1 else 0.026)
        ys2.append(yy)
        rbox(ax, 0.327, yy, 0.310, hh, t, size=7.2)
        if i < len(col2) - 1:
            arrow(ax, 0.482, yy, 0.482, yy - gap2)

    # ── ③ 선택적 적응 ────────────────────────────────────────────────────────
    col3 = ["Drift gate", "Rollback guard", "Update φ only", "Reset rule"]  # Reset 신규
    y3, h3, gap3 = 0.775, 0.093, 0.038
    for i, t in enumerate(col3):
        yy = y3 - i * (h3 + gap3)
        is_new = t == "Reset rule"
        rbox(ax, 0.7495, yy, 0.210, h3, t,
             face="#fdf1ec" if is_new else BOX_FACE,
             edge=ORANGE if is_new else BOX_EDGE, size=7.6)
        if i < len(col3) - 1:
            arrow(ax, 0.8545, yy, 0.8545, yy - gap3)

    # adapter -> 적응 트리거 -> drift gate
        # 라벨이 들어갈 폭(약 0.4 in)을 ②·③ 상자 사이에 남겨 둔다.
    arrow(ax, 0.637, ys2[-1] + 0.055, 0.7495, 0.775 + h3 / 2, color=ORANGE, rad=-0.28)
    ax.text(0.697, 0.560, "adaptation\ntriggered", ha="center", va="center",
            fontsize=fs(6.8, DIAGRAM_SCALE), color=ORANGE, zorder=4, linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.18", facecolor=SURFACE, edgecolor="none"))

    # 사전학습 -> 온라인 (배포)
    arrow(ax, 0.150, 0.235, 0.150, 0.175, color=ORANGE)
    ax.annotate("", xy=(0.327, ys2[2] + h2 / 2), xytext=(0.150, 0.175),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=9, linewidth=lw(0.9),
                                color=ORANGE, connectionstyle="angle,angleA=0,angleB=90,rad=6"),
                zorder=2)
    ax.text(0.168, 0.150, "deploy pretrained θ*", ha="left", va="center",
            fontsize=fs(7.0, DIAGRAM_SCALE), color=ORANGE, zorder=3)

    # ── 최종 산출 ────────────────────────────────────────────────────────────
    rbox(ax, 0.512, 0.030, 0.474, 0.082,
         "Final forecast: no-harm accuracy · selective adaptation\n"
         "parameter-light adapter (0.30–0.52 % of the backbone)",
         face=GREEN_FACE, edge=GREEN_EDGE, size=7.0)
    arrow(ax, 0.482, ys2[-1], 0.482, 0.150, color=GREEN_EDGE)
    ax.plot([0.482, 0.749], [0.150, 0.150], color=GREEN_EDGE, linewidth=lw(0.9), zorder=2)
    arrow(ax, 0.749, 0.150, 0.749, 0.112, color=GREEN_EDGE)

    fig.subplots_adjust(left=0.002, right=0.998, top=0.995, bottom=0.005)
    save(fig, args.output, exact_size=True)
    print("\n원본 대비 변경: (1) Global scaling ×2 -> Per-client scaling, "
          "(2) Reset rule 블록 추가, (3) 최종 문구에 파라미터 기준 명시")


if __name__ == "__main__":
    main()
