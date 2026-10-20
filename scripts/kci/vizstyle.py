"""Shared figure style so every KCI figure reads as one system.

Palette values are the dataviz reference instance's first three categorical
slots, validated all-pairs in light mode with scripts/kci/validate_palette.py:
worst CVD dE 9.2 (deutan), worst normal-vision dE 24.0, both clear of the gates.
AQUA sits at 2.74:1 against the surface (below the 3:1 line), so every figure
that uses it must carry visible direct labels — that is the documented relief.

These are print figures for a Korean journal submission: light mode only, no
hover layer, grayscale-legible via hatching on the one composition chart.
"""

from __future__ import annotations

import contextlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a887f"
GRID = "#e3e2dd"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

# adaptation decision categories (mutually exclusive, sum to 1)
CAT_COLORS = {"adaptation": BLUE, "drift_skipped": ORANGE, "rollback_skipped": AQUA}
CAT_HATCH = {"adaptation": "", "drift_skipped": "///", "rollback_skipped": "..."}
CAT_LABEL = {
    "adaptation": "Adaptation (update applied)",
    "drift_skipped": "Skipped by drift gate",
    "rollback_skipped": "Skipped by update guard",
}

DATASET_ORDER = ("murata", "solar", "electricity", "traffic")
DATASET_LABEL = {
    "murata": "Murata",
    "solar": "Solar",
    "electricity": "Electricity",
    "traffic": "Traffic",
}


# 조판 폭(17.5 cm)은 레이아웃이 정하는 상수라 늘릴 수 없다. 인쇄물에서 글자를
# 실제로 키우는 방법은 폭 대비 글자 크기를 올리는 것뿐이므로, 모든 글자 크기를
# 이 한 배율로 곱한다. 그림 높이는 각 스크립트에서 같이 키워 여백을 확보한다.
FONT_SCALE = 1.35

# 선 두께는 글자만큼 키우면 그림이 뭉개진다. 별도의 완만한 배율을 쓴다.
LINE_SCALE = 1.2


def fs(size: float, scale: float | None = None) -> float:
    """Scale a per-figure explicit fontsize by the shared FONT_SCALE.

    scale 를 넘기면 그 그림에만 다른 배율을 쓴다 (프레임워크 도식처럼 상자 폭이
    고정되어 전역 배율을 그대로 받으면 글자가 상자를 넘치는 경우).
    """
    return round(size * (FONT_SCALE if scale is None else scale), 2)


def lw(width: float) -> float:
    """Scale a per-figure explicit linewidth by the shared LINE_SCALE."""
    return round(width * LINE_SCALE, 2)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": fs(9),
            "axes.titlesize": fs(10),
            "axes.labelsize": fs(9),
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": lw(0.8),
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": lw(0.6),
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": fs(8.5),
            "ytick.labelsize": fs(8.5),
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            # minor 도 같이 꺼야 한다. major 만 0 으로 두면 로그축에서 decade 안
            # 2~9 자리 minor tick 이 기본값 2 pt 로 그려진다. 축 범위가 넓은 패널은
            # matplotlib 이 minor 를 자동 억제하므로, 같은 그림 안에서도 패널마다
            # 눈금이 있고 없고가 갈린다 (Solar 에만 보이던 잔선).
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
            "legend.frameon": False,
            "legend.fontsize": fs(8.5),
            "text.color": TEXT_PRIMARY,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def despine(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


# 2단 조판 양단 폭 17.5 cm
PAPER_WIDTH_IN = 17.5 / 2.54


def save(fig, path, *, exact_size: bool = False) -> None:
    """Write both a PNG (for review) and a PDF (vector, for the manuscript).

    exact_size=True 는 bbox_inches="tight" 를 끈다. tight 는 축 밖 텍스트를 감싸려고
    캔버스를 넓히기 때문에, 조판 폭을 지정해야 하는 논문 그림에서는 figsize 가
    무의미해진다 (실측: 6.89 in 지정이 9.01 in 로 확장됨). exact_size 를 쓸 때는
    호출 측에서 fig.subplots_adjust 로 여백을 직접 잡아야 한다.

    주의: savefig(bbox_inches=None) 로는 tight 가 꺼지지 않는다. matplotlib 은
    bbox_inches 가 None 이면 rcParams["savefig.bbox"] 로 되돌리기 때문에, 여기서
    tight 를 실제로 끄려면 rc_context 로 그 rcParam 자체를 None 으로 덮어야 한다.
    (이 누락 때문에 exact_size=True 그림도 6.89 in 이 아니라 7.5 in 로 나갔다.)
    """
    path = str(path)
    ctx = plt.rc_context({"savefig.bbox": None}) if exact_size else contextlib.nullcontext()
    with ctx:
        fig.savefig(path + ".png")
        fig.savefig(path + ".pdf")
    if exact_size:
        w, h = fig.get_size_inches()
        print(f"Wrote {path}.png / .pdf  ({w * 2.54:.1f} x {h * 2.54:.1f} cm, exact)")
    else:
        print(f"Wrote {path}.png / .pdf")
    plt.close(fig)
