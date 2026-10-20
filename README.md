# DELTA: Deployment-safe Efficient Lightweight Test-time Adaptation

**연합학습 환경에서 장기 시계열 예측을 위한 배포 안전을 고려한 경량 테스트 시 적응 기법**
*Deployment-safe Efficient Lightweight Test-time Adaptation for Long-term Time-series Forecasting in Federated Learning*

KCC 2026 우수논문 · KCI 확장판의 공식 코드 저장소입니다.

📄 **논문**: [KCC 2026 (DBpia)](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12929376) · KCI 확장판 게재 예정 · 📦 **실험 결과 아카이브**: [Releases](https://github.com/itisyijy/DELTA/releases) (암호화)

---

## 개요

DELTA는 FedAvg로 학습한 DLinear 백본을 배포 후 **고정**하고, 클라이언트마다
**time-affine adapter(192개 파라미터)만** 테스트 시점에 갱신하는 선택적 TTA 방법입니다.

| 구성요소 | 역할 |
|---|---|
| **Time-affine adapter** | 백본 출력을 시간축 affine 변환으로 보정 (`a, b ∈ ℝ^{H×C}`, 백본 파라미터의 0.297–0.518%) |
| **Drift gate** | 학습 분할 통계로 표준화한 drift score가 임계값 τ 이상인 윈도우만 적응 |
| **Update guard** | 갱신 전 재구성 오차가 최근 20개 이력 평균의 3배를 넘으면 갱신 생략 |
| **Reset rule** | 갱신 전 hindcast MSE가 2.5를 넘으면 어댑터를 항등변환으로 초기화 |

<!-- 그림: docs/kci_revision/fig_framework.png (결과 아카이브 포함) -->

## 주요 결과

4개 데이터셋, 1,350개 클라이언트 (τ = 1.0):

- 데이터셋 평균 **적응 비율 0.846 → 0.406**
- Traffic 윈도우당 지연시간 **백본 대비 10.8× → 3.9×**
- 백본 대비 집계 ΔMSE **−0.097% ~ 0.030%** (no-harm 기준 0.5% 이내)

| Dataset | FedAvg backbone MSE | DELTA MSE | ΔMSE (%) | 적응 비율 |
|---|---:|---:|---:|---:|
| Murata | 0.3032 | 0.3032 | 0.026 | 0.544 |
| Solar | 0.2242 | 0.2242 | 0.005 | 0.639 |
| Electricity | 0.1533 | 0.1532 | −0.091 | 0.251 |
| Traffic | 0.4184 | 0.4185 | 0.030 | 0.190 |

Ablation 결과 정확도를 지키는 것은 reset rule이고, drift gate는 정확도가 아니라
**갱신 빈도를 줄이는 장치**입니다. 동일 적응 예산의 random gate와 비교한 MSE 차이는
0.005%p 이내입니다(논문 §4.5–4.6).

## 저장소 구조

```text
scripts/
├── run.py                   # 백본 학습 진입점 (centralized / FedAvg / legacy TTA)
├── run_affine_local.py      # DELTA 실행기 (time-affine adapter + gate/guard/reset)
├── run_backbone_eval.py     # 백본 단독 평가 (TTA 없음)
├── models/                  # DLinear, RevIN-DLinear
├── data/                    # CSV / parquet 로더
├── trainers/                # Centralized, FedAvg
├── tta/                     # adapter, loss, policy(drift gate), engine, loop
├── kci/                     # 논문 표·그림 생성 스크립트
├── run_dataset_pipeline.sh  # 백본 및 비교 baseline 학습
├── run_kcc_drift_gate_sweep.sh, run_kcc_multi_seed.sh, report_kcc_*.py
└── run_kci_*.sh             # KCI 확장 실험 (교정 drift score, ablation, random gate, overhead)
configs/{murata,solar,electricity,traffic}/
tests/
```

## 환경

- Python 3 + PyTorch (실험은 conda env `kcc2026`, NVIDIA RTX A4000에서 수행)
- `pip install torch numpy pandas pyarrow pyyaml scipy scikit-learn matplotlib pytest`

```bash
export PYTHON=python   # 또는 conda env의 python 경로
export DEVICE=cuda:0
$PYTHON -m pytest -q
```

## 데이터

| Dataset | 간격 | 입력 길이 | 예측 길이 | 커널 | 클라이언트 수 |
|---|---|---:|---:|---:|---:|
| Murata | 15 min | 192 | 96 | 49 | 30 |
| Solar | 10 min | 288 | 96 | 73 | 137 |
| Electricity | 1 hour | 336 | 96 | 25 | 321 |
| Traffic | 1 hour | 336 | 96 | 25 | 862 |

각 변량을 하나의 클라이언트로 두고 클라이언트별 학습 분할 통계로 표준화합니다.
Solar·Electricity·Traffic은 [LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)
배포본 CSV를 사용합니다. Murata는 비공개 산업 데이터로 저장소에 포함되지 않습니다.
`configs/<dataset>/*.yaml`의 `data_path`를 각자의 경로로 바꿔 사용하세요.

## 재현

### 1. 백본 학습

```bash
for DATASET in murata solar electricity traffic; do
  $PYTHON -m scripts.run --config configs/$DATASET/centralized.yaml --device $DEVICE
  $PYTHON -m scripts.run --config configs/$DATASET/fed.yaml --device $DEVICE
done
```

FedAvg는 로컬 3 epoch × 15 round, centralized는 15 epoch, batch 256, lr 1e-3, seed 0입니다.
Direct-weight TTA와 loop-based baseline은 `fed_tta.yaml`, `fed_tta_loop.yaml`
(`--auto-prereq`로 선행 체크포인트 자동 학습)로 실행합니다.

### 2. DELTA (논문 설정)

> ⚠ `configs/<dataset>/fed_tta.yaml`만으로는 DELTA가 재현되지 않습니다. 논문 설정은 아래
> 러너 스크립트가 config에 주입하며, **drift score는 표준화 공간(`SPACE=zscore`)**입니다.
> 코드 기본값 `drift_score_space: raw`는 KCC 원판과의 호환을 위해 남겨둔 것입니다.

```bash
# DELTA (τ=1.0) — 표 2, 표 3
STAMP=delta SPACE=zscore GATE=1.0 DEVICE=$DEVICE \
  DATASETS="murata solar electricity traffic" sh scripts/run_kci_b1_compare.sh
```

실제 적용되는 TTA 설정:

```yaml
tta:
  adapter_mode: time_affine
  k_ratio: 0.0625            # hindcast k = H/16 = 6
  lr: 1.0e-4
  alpha: 0.3                 # consistency
  beta: 1.0                  # hindcast
  lambda_anchor: 0.1
  sensitivity: 1.0
  max_boost: 5.0
  drift_score_space: zscore
  drift_gate_threshold: 1.0  # DELTA- 는 0.0
  rollback_threshold: 3.0    # update guard
  rollback_window: 20
  reset_threshold: 2.5
  grad_clip: 1.0
```

### 3. 논문 표·그림

| 논문 | 실행 | 집계 |
|---|---|---|
| 표 2 — 방법별 성능 | 1, 2단계 결과 | `scripts/report_kcc_*.py` |
| 표 3 — drift gate 임계값 민감도 | `run_kci_b1_compare.sh`에서 `GATE=0.3/0.5/0.7/1.0` (0.0 = DELTA-) | `scripts/kci/threshold_sensitivity.py` |
| 표 4 — 구성요소 ablation | `run_kci_ablation.sh` (`VARIANTS="no_rollback no_reset"`) | `scripts/kci/ablation_table7.py` |
| 그림 2 — 클라이언트별 ΔMSE | — | `scripts/kci/client_delta_mse.py`, `fig_client_delta_mse.py` |
| 그림 3 — 윈도우별 처리 결과, 지연시간 | `run_kci_overhead.sh` | `fig_adaptation_stacked_bar.py`, `adaptation_overhead.py` |
| 그림 4 — drift gate 선별성 | `run_kci_window_trace.sh` (`SPACE=zscore`) | `scripts/kci/hindcast_groups.py` |
| 그림 5 — drift score 흐름 | — | `scripts/kci/fig_drift_timeline.py` |
| §4.6 — rate-matched random gate | `run_kci_random_gate.sh` | `scripts/kci/random_gate_report.py --space zscore` |

분석 스크립트는 `runs/`의 결과를 읽어 `docs/kci_revision/`에 CSV와 그림을 씁니다.
셸 러너의 `PYTHON` 기본값은 작성자 환경 경로이므로 환경변수로 덮어쓰세요.

> 지연시간 측정은 `max_workers=1`로 순차 단독 실행해야 의미가 있습니다.
> 같은 `STAMP`로 러너를 동시에 두 번 띄우면 큐 파일이 덮어써지니 STAMP를 분리하세요.

## 결과 아카이브

`runs/`, `logs/`, `checkpoints/`(8개 백본), 원고, 실험 보고서와 manifest, 논문 그림 원본은
암호화한 아카이브로 [Releases](https://github.com/itisyijy/DELTA/releases)에 첨부되어 있습니다.
비밀번호는 저자에게 문의하세요.

```bash
gpg -d delta-kci-results.tar.zst.gpg | zstd -d | tar x
```

## 한계

임계값 τ=1.0과 최종 평가에 같은 테스트 분할을 사용했으므로, 표 3은 임계값 선택이 아니라
민감도 분석입니다. Reset 임계값 2.5와 update guard의 이력 길이 20·배수 3은 따로 탐색하지
않았습니다. Drift gate는 갱신 전 오차가 큰 윈도우를 주로 건너뛰는 경향이 있습니다(Cliff's δ
Murata −0.547, Solar −0.831).

## 인용

KCC 2026 논문: [DBpia NODE12929376](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12929376)
(서지 정보와 BibTeX는 DBpia 페이지의 인용하기를 이용하세요).
KCI 확장판은 게재 확정 후 추가합니다.

## 참고

- DLinear: Zeng et al., *Are Transformers Effective for Time Series Forecasting?*, AAAI 2023
- FedAvg: McMahan et al., AISTATS 2017
- RevIN: Kim et al., ICLR 2022
