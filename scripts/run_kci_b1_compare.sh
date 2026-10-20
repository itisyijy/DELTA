#!/bin/sh
# KCI 개정 B1 — drift score 계산 공간 교정 전/후 비교 러너.
#
# run_kcc_drift_gate_sweep.sh 의 write_tta_cfg 와 동일한 하이퍼파라미터를 그대로 주입하고,
# tta.drift_score_space 한 키만 바꾼다. 기존 run 디렉터리는 읽지도 쓰지도 않으며 결과는
# runs/kci_revision_<STAMP>/ 아래에만 생성된다.
#
#   SPACE=raw     → 기존 구현 재현 (무해성 회귀 검증용)
#   SPACE=zscore  → 단위 정합 교정판
#
# 사용:
#   STAMP=20260814_0400 DEVICE=cuda:1 SPACE=zscore DATASETS="murata solar" \
#     sh scripts/run_kci_b1_compare.sh
set -eu

ROOT=/home/jylee/DELTA-KCC2026
PYTHON=${PYTHON:-/home/jylee/miniconda3/envs/kcc2026/bin/python}
DEVICE=${DEVICE:-cuda:1}
MAX_WORKERS=${MAX_WORKERS:-3}
DATASETS=${DATASETS:-"murata solar electricity traffic"}
SPACE=${SPACE:-zscore}
GATE=${GATE:-1.0}
LAUNCH_MODE=${LAUNCH_MODE:-sequential}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

STAMP_ROOT="$ROOT/runs/kci_revision_$STAMP"
LOG_DIR="$ROOT/logs"
GATE_TAG=$(echo "$GATE" | tr '.' 'p')
QUEUE_SH="$STAMP_ROOT/queue_${SPACE}_gate${GATE_TAG}_$(echo "$DEVICE" | tr ':' '_').sh"
mkdir -p "$STAMP_ROOT" "$LOG_DIR"
printf "#!/bin/sh\nset -eu\n" > "$QUEUE_SH"

write_tta_cfg() {
  "$PYTHON" - "$1" "$2" "$3" "$DEVICE" "$4" "$5" "$6" <<'PY'
import sys
import yaml

base_cfg, dst, prereq_dst, device, outdir, gate, space = sys.argv[1:]
with open(base_cfg) as f:
    cfg = yaml.safe_load(f)
cfg["device"] = device
cfg["output_dir"] = outdir
cfg.setdefault("tta", {})
# ── run_kcc_drift_gate_sweep.sh:91-104 와 동일 (한 글자도 바꾸지 않는다) ──
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
# ── B1 에서 바뀌는 단 하나의 키 ──
cfg["tta"]["drift_score_space"] = space
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
with open(prereq_dst, "w") as f:
    yaml.safe_dump({"baseline": "fed"}, f, sort_keys=False)
PY
}

echo "=== KCI B1 compare | stamp=$STAMP | space=$SPACE | gate=$GATE | device=$DEVICE ==="
echo "Datasets: $DATASETS"
for dataset in $DATASETS; do
  label="${dataset}_${SPACE}_gate$(echo "$GATE" | tr '.' 'p')"
  outdir="$STAMP_ROOT/$label"
  cfg="$outdir/config.yaml"
  prereq="$outdir/fed.yaml"
  log="$LOG_DIR/kci_${STAMP}_${label}.log"
  mkdir -p "$outdir"
  write_tta_cfg "$ROOT/configs/$dataset/fed_tta.yaml" "$cfg" "$prereq" "$outdir" "$GATE" "$SPACE"
  cmd="cd $ROOT && $PYTHON -u -m scripts.run_affine_local --config $cfg --max-workers $MAX_WORKERS > $log 2>&1"
  if [ "$LAUNCH_MODE" = "sequential" ]; then
    printf "%s\n" "$cmd" >> "$QUEUE_SH"
  else
    tmux new-session -d -s "kci_${STAMP}_${label}" "$cmd"
  fi
  echo "[$label] -> $log"
done

if [ "$LAUNCH_MODE" = "sequential" ]; then
  session="kci_${STAMP}_$(echo "${SPACE}_gate${GATE_TAG}_${DEVICE}" | tr ':' '_')"
  tmux new-session -d -s "$session" "sh $QUEUE_SH"
  echo "Sequential queue session: $session -> $QUEUE_SH"
fi

echo ""
echo "Run root: $STAMP_ROOT"
echo "Monitor:  grep -h 'Avg:\|Diagnostics:' $LOG_DIR/kci_${STAMP}_*.log"
