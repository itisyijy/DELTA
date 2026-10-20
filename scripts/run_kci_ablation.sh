#!/bin/sh
# KCI 개정 Phase 4 — DELTA 구성요소 ablation.
#
# DELTA 동작점에서 한 번에 한 component만 끈다. 나머지 하이퍼파라미터는
# run_kcc_drift_gate_sweep.sh:91-104 와 완전히 동일하다.
#
#   delta         : gate 1.0 / rollback 3.0 / reset 2.5   -> 기존 *_gate_1p0 재사용 (실행 불필요)
#   no_drift_gate : gate 0.0                              -> 기존 *_control 재사용 (실행 불필요)
#   no_rollback   : rollback_threshold = inf              -> 신규
#   no_reset      : reset_threshold = inf                 -> 신규
#
# rollback 비활성에 큰 유한값(1e30)이 아니라 .inf 를 쓰는 이유 — 실측으로 확인:
#   RollbackGuard.should_skip 은 `l_recon_pre > threshold * avg` 이다 (engine.py:66).
#   rolling mean 이 정확히 0 이 되는 창이 있으면
#     threshold=1e30 -> `l > 1e30*0 = 0` -> True  -> 모든 창을 skip (정반대 동작)
#     threshold=inf  -> `l > inf*0 = nan` -> False -> 절대 skip 안 함 (의도한 동작)
#   avg 가 양수이거나 warm-up(avg=inf)인 경우에도 inf 는 항상 False 를 준다.
#
#   STAMP=20260814_ab DEVICE=cuda:1 VARIANTS="no_rollback no_reset" \
#     DATASETS="murata solar" sh scripts/run_kci_ablation.sh
set -eu

ROOT=/home/jylee/DELTA-KCC2026
PYTHON=${PYTHON:-/home/jylee/miniconda3/envs/kcc2026/bin/python}
DEVICE=${DEVICE:-cuda:1}
MAX_WORKERS=${MAX_WORKERS:-3}
DATASETS=${DATASETS:-"murata solar electricity traffic"}
VARIANTS=${VARIANTS:-"no_rollback no_reset"}
GATE=${GATE:-1.0}
SPACE=${SPACE:-raw}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

STAMP_ROOT="$ROOT/runs/kci_revision_$STAMP"
LOG_DIR="$ROOT/logs"
QUEUE_SH="$STAMP_ROOT/queue_ablation_$(echo "$DEVICE" | tr ':' '_').sh"
mkdir -p "$STAMP_ROOT" "$LOG_DIR"
printf "#!/bin/sh\nset -eu\n" > "$QUEUE_SH"

echo "=== KCI ablation | stamp=$STAMP | space=$SPACE | gate=$GATE | device=$DEVICE ==="
echo "Datasets: $DATASETS"
echo "Variants: $VARIANTS"

for dataset in $DATASETS; do
  for variant in $VARIANTS; do
    outdir="$STAMP_ROOT/${dataset}_${variant}"
    cfg="$outdir/config.yaml"
    prereq="$outdir/fed.yaml"
    log="$LOG_DIR/kci_${STAMP}_${dataset}_${variant}.log"
    mkdir -p "$outdir"

    "$PYTHON" - "$ROOT/configs/$dataset/fed_tta.yaml" "$cfg" "$prereq" "$DEVICE" "$outdir" "$GATE" "$SPACE" "$variant" <<'PY'
import sys
import yaml

base_cfg, dst, prereq_dst, device, outdir, gate, space, variant = sys.argv[1:]
with open(base_cfg) as f:
    cfg = yaml.safe_load(f)
cfg["device"] = device
cfg["output_dir"] = outdir
cfg.setdefault("tta", {})
# ── DELTA 동작점 (run_kcc_drift_gate_sweep.sh:91-104 와 동일) ──
cfg["tta"]["adapter_mode"] = "time_affine"
cfg["tta"]["k_ratio"] = 0.0625
cfg["tta"]["lr"] = 1e-4
cfg["tta"]["alpha"] = 0.3
cfg["tta"]["beta"] = 1.0
cfg["tta"]["lambda_anchor"] = 0.1
cfg["tta"]["sensitivity"] = 1.0
cfg["tta"]["max_boost"] = 5.0
cfg["tta"]["reset_threshold"] = 2.5
cfg["tta"]["drift_gate_threshold"] = float(gate)
cfg["tta"]["hard_gate_scale"] = 0.0
cfg["tta"]["hard_gate_min_history"] = 0
cfg["tta"]["acceptance_margin"] = -1.0
cfg["tta"]["drift_score_space"] = space
# rollback_threshold / rollback_window 는 fed_tta.yaml 에서 상속 (3.0 / 20)

# ── variant 별로 정확히 한 키만 바꾼다 ──
if variant == "delta":
    pass
elif variant == "no_drift_gate":
    cfg["tta"]["drift_gate_threshold"] = 0.0
elif variant == "no_rollback":
    cfg["tta"]["rollback_threshold"] = float("inf")
elif variant == "no_reset":
    cfg["tta"]["reset_threshold"] = float("inf")
else:
    raise SystemExit(f"unknown variant: {variant!r}")

with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
with open(prereq_dst, "w") as f:
    yaml.safe_dump({"baseline": "fed"}, f, sort_keys=False)

# 의도한 키 하나만 달라졌는지 즉시 재확인 (YAML 왕복 포함)
with open(dst) as f:
    roundtrip = yaml.safe_load(f)["tta"]
expected = {
    "delta": {},
    "no_drift_gate": {"drift_gate_threshold": 0.0},
    "no_rollback": {"rollback_threshold": float("inf")},
    "no_reset": {"reset_threshold": float("inf")},
}[variant]
for key, want in expected.items():
    got = roundtrip[key]
    assert got == want, f"{variant}: {key} round-tripped as {got!r}, expected {want!r}"
print(f"  [{variant}] verified: " + (", ".join(f"{k}={v}" for k, v in expected.items()) or "baseline"))
PY

    printf "%s\n" "cd $ROOT && $PYTHON -u -m scripts.run_affine_local --config $cfg --max-workers $MAX_WORKERS > $log 2>&1" >> "$QUEUE_SH"
    echo "[${dataset}_${variant}] -> $log"
  done
done

session="kci_${STAMP}_ablation_$(echo "$DEVICE" | tr ':' '_')"
tmux new-session -d -s "$session" "sh $QUEUE_SH"
echo "Queue session: $session -> $QUEUE_SH"
echo "Run root: $STAMP_ROOT"
