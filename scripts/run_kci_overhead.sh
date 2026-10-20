#!/bin/sh
# KCI 개정 Phase 6 — 실제 adaptation overhead 측정.
#
# 세 방식을 동일 하드웨어에서 비교한다:
#   backbone   : TTA 없음 (run_backbone_eval)
#   no_gate    : DELTA- (drift_gate_threshold = 0.0, 항상 적응)
#   delta      : DELTA  (drift_gate_threshold = 1.0, 교정 score)
#
# 타이밍 유효성 조건 (지키지 않으면 숫자가 의미 없다):
#   * --timing 이 window 전후로 torch.cuda.synchronize() 를 호출한다
#   * max_workers = 1 — 클라이언트 스레드가 겹치면 per-window 지연시간이 무의미해진다
#   * 이 스크립트는 **순차** 실행하며, 실행 중 다른 작업이 없어야 한다
#
# Traffic 은 862 클라이언트를 1 워커로 돌리면 너무 오래 걸리므로 MAX_CLIENTS 로 자른다.
# per-window 지연시간은 클라이언트 수와 무관하므로 대표성이 있고, 전체 런타임은
# 리포트에서 창 수 x 클라이언트 수로 환산한다.
#
#   STAMP=20260815_ov DEVICE=cuda:1 DATASETS="murata traffic" sh scripts/run_kci_overhead.sh
set -eu

ROOT=/home/jylee/DELTA-KCC2026
PYTHON=${PYTHON:-/home/jylee/miniconda3/envs/kcc2026/bin/python}
DEVICE=${DEVICE:-cuda:1}
DATASETS=${DATASETS:-"murata traffic"}
SPACE=${SPACE:-zscore}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
# 데이터셋별 클라이언트 상한 (0 = 전부)
MAX_CLIENTS_murata=${MAX_CLIENTS_murata:-0}
MAX_CLIENTS_solar=${MAX_CLIENTS_solar:-40}
MAX_CLIENTS_electricity=${MAX_CLIENTS_electricity:-60}
MAX_CLIENTS_traffic=${MAX_CLIENTS_traffic:-100}

STAMP_ROOT="$ROOT/runs/kci_revision_$STAMP"
LOG_DIR="$ROOT/logs"
QUEUE_SH="$STAMP_ROOT/queue_overhead.sh"
mkdir -p "$STAMP_ROOT" "$LOG_DIR"
printf "#!/bin/sh\nset -eu\n" > "$QUEUE_SH"

echo "=== KCI overhead | stamp=$STAMP | device=$DEVICE | space=$SPACE ==="
echo "Datasets: $DATASETS   (순차 실행, max_workers=1, CUDA sync)"

for dataset in $DATASETS; do
  eval "maxc=\${MAX_CLIENTS_${dataset}}"
  for arm in backbone no_gate delta; do
    outdir="$STAMP_ROOT/${dataset}_${arm}"
    cfg="$outdir/config.yaml"
    prereq="$outdir/fed.yaml"
    log="$LOG_DIR/kci_${STAMP}_${dataset}_${arm}.log"
    mkdir -p "$outdir"

    "$PYTHON" - "$ROOT/configs/$dataset/fed_tta.yaml" "$cfg" "$prereq" "$DEVICE" "$outdir" "$SPACE" "$arm" "$maxc" <<'PY'
import sys
import yaml

base_cfg, dst, prereq_dst, device, outdir, space, arm, maxc = sys.argv[1:]
with open(base_cfg) as f:
    cfg = yaml.safe_load(f)
cfg["device"] = device
cfg["output_dir"] = outdir
if int(maxc) > 0:
    cfg["max_clients"] = int(maxc)
cfg.setdefault("tta", {})
# DELTA 동작점 (run_kcc_drift_gate_sweep.sh:91-104 와 동일)
cfg["tta"]["adapter_mode"] = "time_affine"
cfg["tta"]["k_ratio"] = 0.0625
cfg["tta"]["lr"] = 1e-4
cfg["tta"]["alpha"] = 0.3
cfg["tta"]["beta"] = 1.0
cfg["tta"]["lambda_anchor"] = 0.1
cfg["tta"]["sensitivity"] = 1.0
cfg["tta"]["max_boost"] = 5.0
cfg["tta"]["reset_threshold"] = 2.5
cfg["tta"]["hard_gate_scale"] = 0.0
cfg["tta"]["hard_gate_min_history"] = 0
cfg["tta"]["acceptance_margin"] = -1.0
cfg["tta"]["drift_score_space"] = space
cfg["tta"]["drift_gate_threshold"] = 0.0 if arm in ("backbone", "no_gate") else 1.0
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
with open(prereq_dst, "w") as f:
    yaml.safe_dump({"baseline": "fed"}, f, sort_keys=False)
PY

    if [ "$arm" = "backbone" ]; then
      cmd="cd $ROOT && $PYTHON -u -m scripts.run_backbone_eval --config $cfg --timing > $log 2>&1"
    else
      cmd="cd $ROOT && $PYTHON -u -m scripts.run_affine_local --config $cfg --max-workers 1 --timing > $log 2>&1"
    fi
    printf "%s\n" "$cmd" >> "$QUEUE_SH"
    echo "[${dataset}_${arm}] max_clients=${maxc:-all} -> $log"
  done
done

session="kci_${STAMP}_overhead"
tmux new-session -d -s "$session" "sh $QUEUE_SH"
echo "Queue session: $session -> $QUEUE_SH"
echo "Run root: $STAMP_ROOT"
