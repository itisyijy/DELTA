#!/bin/sh
# KCI 개정 Phase 3 — window-level drift-gate 진단 수집.
#
# DELTA(gate_1p0) 설정을 한 글자도 바꾸지 않고 --dump-windows 플래그만 추가해 재실행한다.
# 계측이 결과를 바꾸지 않았는지는 metrics.json을 기존 gate_1p0 run과 대조해 확인한다.
#
#   STAMP=20260814_w DEVICE=cuda:1 DATASETS="murata solar" sh scripts/run_kci_window_trace.sh
set -eu

ROOT=/home/jylee/DELTA-KCC2026
PYTHON=${PYTHON:-/home/jylee/miniconda3/envs/kcc2026/bin/python}
DEVICE=${DEVICE:-cuda:1}
MAX_WORKERS=${MAX_WORKERS:-3}
DATASETS=${DATASETS:-"murata solar"}
GATE=${GATE:-1.0}
SPACE=${SPACE:-raw}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

STAMP_ROOT="$ROOT/runs/kci_revision_$STAMP"
LOG_DIR="$ROOT/logs"
QUEUE_SH="$STAMP_ROOT/queue_trace_$(echo "$DEVICE" | tr ':' '_').sh"
mkdir -p "$STAMP_ROOT" "$LOG_DIR"
printf "#!/bin/sh\nset -eu\n" > "$QUEUE_SH"

for dataset in $DATASETS; do
  label="${dataset}_trace_gate$(echo "$GATE" | tr '.' 'p')"
  outdir="$STAMP_ROOT/$label"
  cfg="$outdir/config.yaml"
  prereq="$outdir/fed.yaml"
  log="$LOG_DIR/kci_${STAMP}_${label}.log"
  dump="$outdir/window_trace.csv"
  mkdir -p "$outdir"

  "$PYTHON" - "$ROOT/configs/$dataset/fed_tta.yaml" "$cfg" "$prereq" "$DEVICE" "$outdir" "$GATE" "$SPACE" <<'PY'
import sys
import yaml

base_cfg, dst, prereq_dst, device, outdir, gate, space = sys.argv[1:]
with open(base_cfg) as f:
    cfg = yaml.safe_load(f)
cfg["device"] = device
cfg["output_dir"] = outdir
cfg.setdefault("tta", {})
# run_kcc_drift_gate_sweep.sh:91-104 와 동일
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
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
with open(prereq_dst, "w") as f:
    yaml.safe_dump({"baseline": "fed"}, f, sort_keys=False)
PY

  printf "%s\n" "cd $ROOT && $PYTHON -u -m scripts.run_affine_local --config $cfg --max-workers $MAX_WORKERS --dump-windows $dump > $log 2>&1" >> "$QUEUE_SH"
  echo "[$label] -> $log  (trace: $dump)"
done

session="kci_${STAMP}_trace_$(echo "$DEVICE" | tr ':' '_')"
tmux new-session -d -s "$session" "sh $QUEUE_SH"
echo "Queue session: $session -> $QUEUE_SH"
echo "Run root: $STAMP_ROOT"
