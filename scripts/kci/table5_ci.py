"""표 5 재생성 — zscore 기준 클라이언트 단위 bootstrap CI.

본문 기준이 raw -> zscore 로 바뀌면서 DELTA arm 이 달라졌으므로, (DELTA - backbone)
클라이언트별 MSE 차이의 신뢰구간을 다시 계산한다. 기존 `ci_summary.md`(raw, 2026-07-24)는
덮어쓰지 않고 새 파일로 낸다.

입력은 `client_relative_delta_mse.csv` (zscore 기준, 1,350행). 재표본은 클라이언트 단위
**복원추출** 10,000회, percentile 95% CI, `numpy.random.default_rng(0)` 고정.
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import numpy as np

ORDER = ["murata", "solar", "electricity", "traffic"]
EXPECTED_N = {"murata": 30, "solar": 137, "electricity": 321, "traffic": 862}
EXPECTED_REDUCTION = {"murata": 30.1, "solar": 24.5, "electricity": 72.4, "traffic": 77.6}
N_BOOT, SEED = 10_000, 0


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=root / "docs/kci_revision/client_relative_delta_mse.csv")
    parser.add_argument("--ablation", type=Path, default=root / "docs/kci_revision/ablation_table7.csv")
    parser.add_argument("--output", type=Path, default=root / "docs/kci_revision/table5_ci_zscore.csv")
    args = parser.parse_args()

    diffs = collections.defaultdict(list)
    mse_b = collections.defaultdict(list)
    mse_d = collections.defaultdict(list)
    for r in csv.DictReader(args.src.open()):
        diffs[r["dataset"]].append(float(r["delta_mse"]))
        mse_b[r["dataset"]].append(float(r["mse_backbone"]))
        mse_d[r["dataset"]].append(float(r["mse_delta"]))

    adapt = {}
    for r in csv.DictReader(args.ablation.open()):
        adapt[(r["dataset"], r["variant"])] = float(r["adaptation_rate"])

    rng = np.random.default_rng(SEED)
    rows = []
    for ds in ORDER:
        v = np.asarray(diffs[ds])
        n = v.size
        idx = rng.integers(0, n, size=(N_BOOT, n))
        boot = v[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])

        # 무결성: 클라이언트별 차이의 평균 == 집계 MSE 차이 (macro 평균이므로 성립해야 한다)
        assert abs(v.mean() - (np.mean(mse_d[ds]) - np.mean(mse_b[ds]))) < 1e-12, ds
        assert n == EXPECTED_N[ds], f"{ds}: n={n} != {EXPECTED_N[ds]}"

        base = adapt[(ds, "w/o drift gate")]
        gated = adapt[(ds, "DELTA")]
        reduction = 100 * (base - gated) / base
        assert abs(reduction - EXPECTED_REDUCTION[ds]) < 0.05, f"{ds}: {reduction:.2f}"

        rows.append({
            "dataset": ds,
            "n": n,
            "mean_diff_x1e4": round(v.mean() * 1e4, 2),
            "ci_low_x1e4": round(lo * 1e4, 2),
            "ci_high_x1e4": round(hi * 1e4, 2),
            "adapt_reduction_percent": round(reduction, 1),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.output}\n")

    hdr = f"{'dataset':<12}{'n':>5}{'평균차(x1e-4)':>15}{'95% CI (x1e-4)':>24}{'적응 감소율':>13}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ci = f"[{r['ci_low_x1e4']:+.2f}, {r['ci_high_x1e4']:+.2f}]"
        print(f"{r['dataset']:<12}{r['n']:>5}{r['mean_diff_x1e4']:>15,.2f}{ci:>24}"
              f"{r['adapt_reduction_percent']:>12.1f}%")

    # 검증 조건 4 — 정성적 결론이 유지되는가
    print("\n검증 조건 4 (정성적 결론):")
    by = {r["dataset"]: r for r in rows}
    solar_includes_zero = by["solar"]["ci_low_x1e4"] <= 0 <= by["solar"]["ci_high_x1e4"]
    elec_all_negative = by["electricity"]["ci_high_x1e4"] < 0
    print(f"  Solar CI 가 0 을 포함           : {solar_includes_zero}  "
          f"[{by['solar']['ci_low_x1e4']:+.2f}, {by['solar']['ci_high_x1e4']:+.2f}]")
    print(f"  Electricity CI 가 전 구간 음수  : {elec_all_negative}  "
          f"[{by['electricity']['ci_low_x1e4']:+.2f}, {by['electricity']['ci_high_x1e4']:+.2f}]")
    if not (solar_includes_zero and elec_all_negative):
        print("\n  ** 정성적 결론이 뒤집혔다 — 원고 §4.3 서술을 다시 손봐야 한다 **")


if __name__ == "__main__":
    main()
