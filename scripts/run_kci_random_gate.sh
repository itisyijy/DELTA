#!/bin/sh
# KCI 개정 Phase 2 — rate-matched random gate baseline.
#
# 동일한 적응 예산에서 drift score 기반 창 선택이 무작위 선택보다 나은지 검증한다.
# DELTA(gate_1p0)가 클라이언트별로 통과시킨 창 수를 그대로 받아, 같은 eligible 창
# 집합에서 정확히 같은 개수를 무작위로 고른다. 그 외 backbone / adapter 초기화 /
# optimizer / lr / loss / rollback guard / reset rule / 평가 창 / 전처리는 전부 동일하다.
#
#   STAMP=20260814_r DEVICE=cuda:1 DATASET=murata SEEDS="0 1 2 ... 29" \
#     sh scripts/run_kci_random_gate.sh
set -eu

ROOT=/home/jylee/DELTA-KCC2026
PYTHON=${PYTHON:-/home/jylee/miniconda3/envs/kcc2026/bin/python}
DEVICE=${DEVICE:-cuda:1}
MAX_WORKERS=${MAX_WORKERS:-3}
DATASET=${DATASET:-murata}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
GATE=${GATE:-1.0}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

STAMP_ROOT="$ROOT/runs/kci_revision_$STAMP"
LOG_DIR="$ROOT/logs"
QUEUE_SH="$STAMP_ROOT/queue_random_${DATASET}.sh"
mkdir -p "$STAMP_ROOT" "$LOG_DIR"
printf "#!/bin/sh\nset -eu\n" > "$QUEUE_SH"

# DELTA reference run (gate-pass 개수의 출처).
# REFERENCE 를 넘기면 그 run 을 쓰고, 없으면 sources.tsv 의 canonical gate_1p0 매핑을 따른다.
# 교정판이 본문이면 교정 DELTA run 을 REFERENCE 로 넘겨야 예산이 맞는다.
if [ -n "${REFERENCE:-}" ]; then
  REFERENCE=$("$PYTHON" - "$REFERENCE" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
if not (p / "metrics.json").exists():
    runs = sorted(q.parent for q in p.rglob("metrics.json"))
    if not runs:
        raise SystemExit(f"no metrics.json under {p}")
    p = runs[-1]
print(p)
PY
)
else
  REFERENCE=$("$PYTHON" - "$ROOT" "$DATASET" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scripts.report_delta_distributions import DEFAULT_STAMP, load_sources
print(load_sources(DEFAULT_STAMP)[sys.argv[2], "gate_1p0"])
PY
)
fi
echo "=== KCI random gate | stamp=$STAMP | dataset=$DATASET | device=$DEVICE ==="
echo "Reference DELTA run: $REFERENCE"
echo "Seeds: $SEEDS"

cfgdir="$STAMP_ROOT/${DATASET}_random"
cfg="$cfgdir/config.yaml"
prereq="$cfgdir/fed.yaml"
mkdir -p "$cfgdir"

"$PYTHON" - "$ROOT/configs/$DATASET/fed_tta.yaml" "$cfg" "$prereq" "$DEVICE" "$cfgdir" "$GATE" <<'PY'
import sys
import yaml

base_cfg, dst, prereq_dst, device, outdir, gate = sys.argv[1:]
with open(base_cfg) as f:
    cfg = yaml.safe_load(f)
cfg["device"] = device
cfg["output_dir"] = outdir
cfg.setdefault("tta", {})
# run_kcc_drift_gate_sweep.sh:91-104 와 동일 — window selection 외에는 아무것도 바뀌지 않는다
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
cfg["tta"]["drift_score_space"] = "raw"
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
with open(prereq_dst, "w") as f:
    yaml.safe_dump({"baseline": "fed"}, f, sort_keys=False)
PY

for seed in $SEEDS; do
  outdir="$STAMP_ROOT/${DATASET}_random_s${seed}"
  seedcfg="$outdir/config.yaml"
  log="$LOG_DIR/kci_${STAMP}_${DATASET}_random_s${seed}.log"
  mkdir -p "$outdir"
  "$PYTHON" - "$cfg" "$seedcfg" "$outdir" <<'PY'
import sys
import yaml
src, dst, outdir = sys.argv[1:]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["output_dir"] = outdir
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
  printf "%s\n" "cd $ROOT && $PYTHON -u -m scripts.run_affine_local --config $seedcfg --max-workers $MAX_WORKERS --gate-mode random --gate-reference $REFERENCE --gate-seed $seed > $log 2>&1" >> "$QUEUE_SH"
done

session="kci_${STAMP}_random_${DATASET}"
tmux new-session -d -s "$session" "sh $QUEUE_SH"
echo "Queue session: $session -> $QUEUE_SH"
echo "Run root: $STAMP_ROOT"
echo "Monitor:  grep -h 'Avg:\|Random gate' $LOG_DIR/kci_${STAMP}_${DATASET}_random_s*.log"
