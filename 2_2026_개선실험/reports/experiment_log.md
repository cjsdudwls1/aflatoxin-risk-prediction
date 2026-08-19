# Experiment Log

> 시간순(과거 → 최신)으로 실험을 적는 사람 친화 일지.
> 자동 표가 필요하면 `reports/leaderboard.csv` 를 참고.

---

## EXP-001_baseline (2026-05-22)

- **목적**: F-01_M-01 audit fix 적용 후 첫 baseline 측정. 향후 모든 실험의 기준점.
- **변경**: DEFECT-014/015 fix (TF 결정성 환경변수), D-008 fix (SMOTE 제거), 한글 폰트 처리, audit findings 적용 (Critical-1, A-M1, B-M3, B-M5).
- **결과**: ROC AUC 0.683, PR AUC 0.013, F2 @ th=0.73 → 0.046 (TP=3, FN=44).
- **소감**: 양성 47개 중 3개만 잡음. 실용 수준 미달. 다음 실험에서 class imbalance / loss 함수 / threshold 운용을 우선 손볼 것.
- **다음 후보**:
  - EXP-002: Focal loss (γ=2)
  - EXP-003: class_weight 사용
  - EXP-004: under/oversampling 비율 변화

---

## EXP-002_focal_loss (2026-05-22 18:44)

- **목적**: weighted_BCE (pos_weight=254) 대신 focal loss 로 어려운 양성 sample 에 집중.
- **변경**: loss → `focal(alpha=0.75, gamma=2)`.
- **결과**: ROC AUC 0.630, PR AUC 0.006, F2 @ th=0.55 → **0.000** (TP=0, FN=47). 양성 검출 완전 붕괴.
- **소감**: focal alpha 0.75 가 pos_weight 254 의 극단 불균형을 못 따라감. 양성 prob 이 threshold 위로 올라가지 않음. focal 은 양성 222개 규모에는 부적합.
- **다음 후보**: focal alpha 를 동적 (양성 비율 기반) 으로 계산하거나, weighted_BCE 유지.

---

## EXP-006_enhanced_tabular (2026-05-22 18:45)

- **목적**: tabular branch 정보량 부족 의심. last14 mean/std/trend 통계 30개 추가.
- **변경**: tabular feature 7 → 37 (last14 mean/std/trend 30개 합류).
- **결과**: ROC AUC 0.683, PR AUC 0.013, F2 0.046, TP=3 — **baseline 과 메트릭이 정확히 동일** (소수점 4자리까지).
- **소감**: 변경이 실제로 모델 출력에 영향을 주지 않은 것으로 보임. 가능한 원인: (1) patch 가 컬럼 추가만 하고 input dim 갱신 안 함, (2) BiLSTM dominance 로 tabular branch 가 학습에 거의 기여 안 함. 별도 audit 필요.
- **다음 후보**: tabular branch 단독 ablation, last14 통계가 실제로 model input 까지 도달하는지 확인.

---

## EXP-003_conv1d_bilstm (2026-05-22 18:49)

- **목적**: 시계열 local pattern (5-day, 3-day window) 을 Conv1D 로 캡쳐 후 BiLSTM 에 입력.
- **변경**: BiLSTM → `Conv1D(filters=64, k=5, causal) + BN + Conv1D(64, k=3, causal) + BN → BiLSTM(64, return_sequences=True)`.
- **결과**: ROC AUC 0.691, PR AUC 0.039, F2 @ th=0.79 → **0.167** (TP=9, FN=38, FP=72, TN=12,014). recall 0.191, precision 0.111.
- **소감**: 🥇 **F2 1위**. local pattern 캡쳐가 효과적. precision/recall balance 도 가장 좋음 (FP 72 로 낮음). Conv1D causal padding 으로 시계열 인과 보존.
- **다음 후보**: Conv1D + dropout 0.1 hybrid (EXP-008 과 결합), kernel size grid (3/5/7).

---

## EXP-008_dropout_low (2026-05-22 18:54)

- **목적**: 양성 222개 train 에서 dropout 0.3 이 과한 regularization 으로 underfit 의심.
- **변경**: BiLSTM/attention/FNN 5개 dropout layer 모두 0.3 → 0.1.
- **결과**: ROC AUC 0.718, PR AUC 0.040, F2 @ th=0.77 → **0.150** (TP=10, FN=37, FP=136, TN=11,950). recall 0.213.
- **소감**: 🥈 **F2 2위**. ROC AUC 가 모든 실험 중 가장 높음 (0.718). 단순한 hyperparameter 1줄 변경으로 큰 효과. underfit 가설 확정.
- **다음 후보**: dropout 0.05 / 0.15 grid, EXP-003 의 Conv1D 와 결합.

---

## EXP-007_transformer_encoder (2026-05-22 19:13)

- **목적**: 시계열 long-range dependency 캡쳐를 위해 BiLSTM+Attention 을 Transformer 로 교체.
- **변경**: archi → `Transformer encoder (d_model=64, num_heads=4, num_layers=2)`.
- **결과**: ROC AUC 0.682, PR AUC 0.011, F2 @ th=0.76 → 0.032 (TP=2, FN=45). baseline 보다 낮음.
- **소감**: 4-head self-attention 은 양성 222개로 학습 underfit. parameter 수가 많은 Transformer 는 small minority 에 부적합. BiLSTM 이 같은 데이터 규모에서 더 안정적.
- **다음 후보**: d_model 32 + 1-block 으로 축소 또는 포기. 양성이 1000+ 모이면 재시도.

---

## EXP-004_ts_augmentation v1 (2026-05-22 20:41) [archived]

- **목적**: 양성 222개 부족 → 시계열 augmentation 으로 양성만 인위적으로 5x (n_copies=4).
- **변경**: jitter (σ=0.05) + scale (0.9~1.1) + np.roll time-shift (±2), 양성 train 222 → 1110.
- **결과**: ROC AUC 0.704, PR AUC **0.063 (baseline 4.8x)**, F2 0.131, TP=8.
- **소감**: 결과가 너무 좋아서 의심. 사용자가 (1) Scaling, (2) np.roll wrap-around, (3) X_tab 다양성 0 위험 지적. multi-agent audit (code-reviewer + tracer + critic 병렬) 진행 후 **8개 결함** 확인:

  | # | v1 결함 | 영향 |
  |---|---|---|
  | 1 | X_tab 다양성 0 (양성 cell 정보 그대로 복제) | cell-level prior inflation — 모델이 특정 cell 을 양성으로 외움 |
  | 2 | np.roll wrap-around | 시계열 인과 파괴 (검사 직전 신호가 첫 위치로 wrap) |
  | 3 | pos_weight silent dilution (254 → 50) | 불균형 정책 무력화 |
  | 4 | 시계열-tab conditional joint 파괴 | X_ts jitter + X_tab copy 비대칭 |
  | 5 | physical bound 위반 risk | 음수 불가 변수에 N(0, 0.05) noise |
  | 6 | scale 곱셈 순서 모호 | (x+eps)*s 로 noise 도 scale 곱해짐 |
  | 7 | over-emphasis (n_copies=4) | 양성 5x 가중 |
  | 8 | 진단 부족 | augment 후 분포 측정 불가 |

  tracer 결론: **PR_AUC 4.8x 향상은 cell-level prior inflation 부산물 (단순 5x oversampling 효과)**. v1 폴더는 `_archive_EXP-004_v1_buggy/` 로 보존, v2 재실행.

---

## EXP-004_ts_augmentation v2 (2026-05-23 06:23) — 8 fix 후 재실험

- **목적**: v1 8개 결함 fix 후 진짜 augmentation 효과만 측정.
- **변경**:
  - pos_weight: `train_and_evaluate hook + EXP004_RAW_POS_WEIGHT globals override` 로 raw 254 강제 유지
  - time-shift: np.roll → `np.tile + np.concatenate` edge padding (no wrap)
  - X_tab: 마지막 4 num cols (Target_Mean, 일수카운트 3개) 에 σ=0.02 jitter, 35개 one-hot 그대로
  - 강도: σ 0.05 → 0.02, scale 0.9~1.1 → 0.95~1.05
  - 순서: scale 먼저 → noise 추가 (effective budget 일정)
  - 양: n_copies 4 → 2 (양성 222 → 666)
  - 진단: one-hot/full unique row + num stats + sha256 fingerprint print
- **결과**: ROC AUC 0.698, PR AUC **0.018 (baseline 1.4x, v1 의 1/3.5)**, F2 @ th=0.95 → **0.125** (TP=12, FN=35, FP=280, TN=11,806). recall 0.255 — **양성 검출 최다 (12/47)**.
- **진단 print 검증** (notebook.executed.ipynb stdout):
  - `raw pos_weight: 254.03 (neg=56395, pos=222)` ✓ (baseline 254 와 일치)
  - `pos_weight raw 강제 fix: 254.03` ✓ (hook 실제 hit)
  - `augmented train shape: (57061,), pos=666` ✓
  - `X_tab pos one-hot unique rows: 24 / 666` ✓ (cell 본질, augment 부산물 아님)
  - `X_tab pos FULL unique rows (after num jitter): 573 / 666` ✓ (jitter 다양성 부여)
  - `fingerprint X_ts=a316d218be570271, X_tab=f15d9197cac54034`
- **소감**: 🥉 **F2 3위**. v1 의 PR_AUC 4.8x 부풀림은 cell prior inflation 부산물로 **확정** — v2 에서 1.4x 로 정정. 그러나 F2 0.125 는 v1 의 0.131 과 거의 동일 → augment 의 진짜 실효 = baseline 의 2.7배 (cell prior 없이도 양성 검출은 가능). recall 0.255 로 양성 검출 측면에서는 최고. precision 낮음 (0.041, FP 280) — augment 가 confidence 분포를 high (0.95) 영역으로 밀어올려 false positive 도 증가.
- **다음 후보**: precision 개선을 위한 FP 감소 방향 (양성 sample 확보, calibration 추가), EXP-003 + EXP-008 + EXP-004 v2 hybrid.

---

## EXP-005_multi_seed_ensemble (미실행)

- **목적**: 5 seed model 평균으로 single-run variance 제거. threshold val 에서 재최적화.
- **변경**: ensemble: 5 seed 학습 후 prob 평균.
- **결과**: 미실행 (run_metrics.json 의 metrics={} 만 존재).
- **다음**: 단독 실행 또는 EXP-003 + EXP-008 best config 로 5-seed ensemble 시도.

---

# 종합 결과 (7개 실행 완료 + 1개 미실행)

## F2 기준 ranking

| rank | Exp ID | F2 | PR_AUC | ROC_AUC | recall | precision | best_th | TP | FN | FP | TN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 🥇 | **EXP-009_hybrid_conv1d_dropout_tsaug** | **0.252** | **0.1848** | 0.6908 | 0.2340 | **0.3667** | 0.99 | 11 | 36 | **19** | **12,067** |
| 🥈 | EXP-003_conv1d_bilstm | 0.167 | 0.0392 | 0.6907 | 0.1915 | 0.1111 | 0.79 | 9 | 38 | 72 | 12,014 |
| 🥉 | EXP-008_dropout_low | 0.150 | 0.0404 | 0.7180 | 0.2128 | 0.0685 | 0.77 | 10 | 37 | 136 | 11,950 |
| 4 | EXP-004_ts_augmentation v2 | 0.125 | 0.0179 | 0.6980 | 0.2553 | 0.0411 | 0.95 | 12 | 35 | 280 | 11,806 |
| 5 | EXP-001_baseline | 0.046 | 0.0126 | 0.6827 | 0.0638 | 0.0214 | 0.73 | 3 | 44 | 137 | 11,949 |
| 5 | EXP-006_enhanced_tabular | 0.046 | 0.0126 | 0.6827 | 0.0638 | 0.0214 | 0.73 | 3 | 44 | 137 | 11,949 |
| 7 | EXP-007_transformer_encoder | 0.032 | 0.0111 | 0.6821 | 0.0426 | 0.0157 | 0.76 | 2 | 45 | 125 | 11,961 |
| 8 | EXP-002_focal_loss | 0.000 | 0.0062 | 0.6297 | 0.0000 | 0.0000 | 0.55 | 0 | 47 | 6 | 12,080 |

## 핵심 패턴

1. **EXP-009 hybrid 가 압도적 1위**: F2 0.252 (직전 1위 EXP-003 의 1.51배), PR_AUC 0.185 (4.7배), precision 0.367 (3.3배). 세 기법(Conv1D + dropout 0.1 + ts_aug v2)의 효과가 **가산적이지 않고 시너지**.
2. **architecture 변경 (Conv1D, dropout)** 이 **데이터 변경 (augmentation, tabular features)** 보다 안정적으로 성능 향상. 다만 hybrid 에서 ts_aug 가 합쳐지면 precision 폭증의 결정적 기여 (단독 EXP-004 FP=280 → hybrid FP=19).
3. **양성 검출량 (TP)**: EXP-004 v2 (12) > EXP-009 (11) > EXP-008 (10) > EXP-003 (9) > baseline (3). hybrid 는 TP 절대량은 EXP-004 보다 1개 낮지만 FP 가 1/15 → precision-recall balance 최고.
4. **precision-recall trade-off 깨짐**: 일반적으로 recall ↑ 시 precision ↓ 이지만 EXP-009 는 EXP-003 대비 recall +0.04, precision +0.26 동시 개선. Conv1D local pattern + ts_aug 양성 다양성 + dropout 0.1 underfit 해소 가 동시 작용.
5. **focal loss + Transformer**: 양성 222 규모에서 학습 불가능. pos_weight + BiLSTM-based 가 적합한 영역.
6. **EXP-006 비대화**: baseline 과 메트릭 정확히 동일 — 변경이 실제 적용됐는지 별도 audit 필요.

---

# 상위 3개 실험 정밀 audit (multi-agent code-reviewer 병렬)

**audit 초점**: 실험 결과를 무효화할 수 있는 결함 (데이터 누수, scaler/encoder fit timing, test isolation, train/inference 일관성, threshold 결정, determinism).

## 6.1 EXP-003_conv1d_bilstm — VERDICT: **PASS** (0 critical, 1 minor advisory)

| 항목 | 상태 | 근거 |
|---|---|---|
| split 이 변경(Conv1D) 적용 전 | PASS | preprocess L375 `train_test_split(...)` 가 모델 build 보다 먼저 |
| StandardScaler train-only fit | PASS | L390 `ts_scaler.fit_transform()` 후 val/test 는 `transform()` only. 출력 `fit on train only. mean[:3]=[12.544...]` 확인 |
| Target_Mean train-only fit | PASS | L412 `df_train_for_tm` 만으로 `fit_target_mean()` (m=20 smoothing, D-020 fix), val/test 는 transform only |
| test isolation | PASS | L646 `find_optimal_threshold(y_val, y_prob_val)` — test 라벨 미사용. L652 단 1회 평가 |
| Conv1D train/predict 일관성 | PASS | 동일 model 객체로 fit + predict. BN train/inference 모드 Keras 표준 처리. patch anchor `(ts_input)` → `(x)` 치환 정상 |
| F2 test 측정 | PASS | L672 `fbeta_score(y_test, y_pred_test, beta=2)` 출력 0.1673 (보고값 정합) |
| best_threshold val 결정 | PASS | val 로 0.79 결정 → test 적용 (test 미재탐색) |
| determinism env | PASS | cell#1 L3-5 TF import 전 PYTHONHASHSEED/TF_DETERMINISTIC_OPS/TF_CUDNN_DETERMINISTIC, L96 `tf.config.experimental.enable_op_determinism()` |

**결함**: 없음. (Minor advisory: ensemble 변형 시 set_seeds 재호출 위치 주의 — 단독 실행 무관).

**결론**: F2=0.167 결과 신뢰 가능.

## 6.2 EXP-008_dropout_low — VERDICT: **PASS** (0 critical)

| 항목 | 상태 | 근거 |
|---|---|---|
| split 이 변경 전 | PASS | patch_exp008 은 `dropout=0.3 → 0.1` substring 치환만, split 로직 baseline 그대로 |
| StandardScaler train-only fit | PASS | L46353 `ts_scaler.fit_transform` 후 val/test transform only |
| Target_Mean train-only fit | PASS | L46382 `cross_mean = fit_target_mean(train fold)` 출력 `unique 조합=89` |
| test isolation | PASS | L46599 `find_optimal_threshold(y_val, y_prob_val)`, L46612 final test 1회 |
| dropout 일관 적용 | PASS | L46461 signature 1곳 변수 정의 → 5개 layer (BiLSTM/attention/FNN×2/merge) 모두 동일 변수 사용 |
| inference 시 dropout off | PASS | `model.predict()` 자동 training=False, MC-dropout trick 없음 |
| F2 test 측정 | PASS | L46625 `fbeta_score(y_test, ...)` |
| best_threshold val 결정 | PASS | L46599 val 로 결정 → L46605 test 적용, 재탐색 없음 |
| determinism env | PASS | PYTHONHASHSEED=42 + TF_DETERMINISTIC_OPS + np.random.seed + tf.random.set_seed |

**결함**: 없음. surgical 한 1-line 치환, dropout 변수가 단일 source → 5 layer 전파 구조라 누락 risk 0.

**결론**: F2=0.150, TP=10 결과 정당 — dropout 0.1 단일 변경에서 기인.

## 6.3 EXP-004_ts_augmentation v2 — VERDICT: **PASS** (0 critical, 1 minor)

| 항목 | 상태 | 근거 |
|---|---|---|
| split 이 augment 전 | PASS | preprocess L34964 `train_test_split` 내부, augment 는 preprocess **return 후** main 블록 L35519 |
| augment train-only | PASS | apply_exp_patches.py:204 — `Xts/Xtab/y` 가 `X_ts_train`만 받음, val/test 변수 미호출 |
| StandardScaler fit pre-augment | PASS | preprocess L34979 ts_scaler.fit + L35033 num_scaler.fit 모두 preprocess **내부**에서 종료 (augment 이전) |
| Target_Mean fit pre-augment | PASS | preprocess L35004 `fit_target_mean` — train fold only, augment 이전 |
| test isolation | PASS | augment 함수가 train 만 받음. run_metrics.json `pos_test: 47` (원본 그대로) |
| 양성 only augment | PASS | apply_exp_patches.py:174 `pos = (y == 1); Xts_pos = Xts[pos]` — 음성 그대로 |
| pos_weight raw 254 유지 | PASS | 출력 L2124 `raw pos_weight: 254.03` + L2163 `pos_weight raw 강제 fix: 254.03` (hook hit). `if __name__ == "__main__":` 블록 안이라 `EXP004_RAW_POS_WEIGHT` 가 module-level globals 정상 진입 |
| F2 test 측정 | PASS | L35253-58 `y_test` 로 메트릭 계산 |
| best_threshold val 결정 | PASS | L35232 val 결정 → L35238 test 적용 |
| seed 고정 + fingerprint | PASS | seed=42 (np.random.RandomState), fingerprint `a316d218be570271 / f15d9197cac54034` |

**결함 (Minor, 결과 무효화 X)**:
- **num-jitter 가 scaled 좌표계에서 σ=0.02 (raw 스케일 아님)** — num_scaler.fit_transform 이 jitter 이전 호출되므로 jitter 가 scaled-coord 에서 가해짐. 출력 std `[3.08, 1.0, 0.83, 0.98]` 가 scaled std≈1 인 컬럼들에 σ=0.02 = dynamic range 의 ~2% → 의도와 부합 가능성 高. 단, 첫 컬럼 Target_Mean std=3.08 인 경우 σ=0.02 가 거의 무영향. 권고: README/주석에 "tab num jitter 는 scaled-coordinate σ=0.02" 명시.

**결론**: F2=0.125, recall=0.255, TP=12/47 결과 신뢰 가능. v1 의 8 fix 모두 코드/출력 로그로 실증됨. 낮은 절대 성능은 augment 가 minority signal 실질적으로 보강하지 못한 결과로 해석.

---

# 7. EXP-009_hybrid_conv1d_dropout_tsaug — 신기록 갱신

**한 줄**: 상위 3 audit PASS 기법 (003 Conv1D + 008 dropout 0.1 + 004 v2 ts_aug fix8) 합성 → **F2 0.252, PR_AUC 0.185, precision 0.367** (모두 신기록). 단독 합산 예상치를 압도하는 **시너지** 확인.

## 7.1 가설

세 기법은 각각 다른 메커니즘으로 양성 학습을 돕는다 — 합치면 결함 영역이 서로 보완될 가능성:

| 기법 | 메커니즘 | 단독 효과 (vs baseline F2 0.046) |
|---|---|---|
| EXP-003 Conv1D | local pattern (kernel 5,3 causal) 추출 후 BiLSTM 입력 | F2 0.167 (3.6×), precision ↑↑ |
| EXP-008 dropout 0.1 | underfit 해소 (양성 222 학습 강도 확보) | F2 0.150 (3.3×), recall ↑↑ |
| EXP-004 v2 ts_aug | 양성 sample 다양성 ×3 (jitter+scale+edge-pad shift) | F2 0.125 (2.7×), recall 0.255 (최고) |

→ Conv1D 가 local 신호 추출의 representation power 를 키우고, dropout 0.1 이 그 power 를 양성에 학습시킬 capacity 를 만들고, ts_aug 가 학습할 양성 sample 자체를 늘림. **3개가 학습 파이프라인의 서로 다른 단계(특징↔용량↔샘플)에 작용**하므로 가산 이상 가능.

## 7.2 patch 구성

`scripts/apply_exp_patches.py:patch_exp009` — 003 → 008 → 004 순차 호출:

```python
def patch_exp009(nb):
    log = []
    log.extend(patch_exp003(nb))  # Conv1D x2 + BN x2 + BiLSTM input redirect
    log.extend(patch_exp008(nb))  # dropout 0.3 -> 0.1
    log.extend(patch_exp004(nb))  # ts_aug v2 fix8 + raw pos_weight hook
    return log
```

**충돌 검증**:
- 003 (cell#25: build_hybrid_model 의 model body) 과 008 (cell#25: signature default 의 substring) 은 같은 cell 의 서로 다른 위치 → 충돌 없음
- 004 (preprocess 호출 종료 ) + train_and_evaluate hook) 은 별도 cell → 격리

## 7.3 결과 metric

| metric | EXP-009 hybrid | 직전 1위 EXP-003 | baseline EXP-001 | hybrid vs 1위 | hybrid vs baseline |
|---|---|---|---|---|---|
| F2 | **0.2523** | 0.1673 | 0.0457 | **+0.085 (+51%)** | **5.5×** |
| PR_AUC | **0.1848** | 0.0392 | 0.0126 | **+0.146 (4.7×)** | **14.7×** |
| ROC_AUC | 0.6908 | 0.6907 | 0.6827 | ≈0 | +0.008 |
| precision | **0.3667** | 0.1111 | 0.0214 | **+0.256 (3.3×)** | **17.1×** |
| recall | 0.2340 | 0.1915 | 0.0638 | +0.042 | 3.7× |
| F1 | **0.2857** | 0.1406 | 0.0321 | 2.0× | 8.9× |
| TP / FN | 11 / 36 | 9 / 38 | 3 / 44 | +2 TP | +8 TP |
| **FP** | **19** | 72 | 137 | **-53 (1/3.8)** | **-118 (1/7.2)** |
| TN | 12,067 | 12,014 | 11,949 | +53 | +118 |
| best_threshold | 0.99 | 0.79 | 0.73 | ↑↑ | ↑↑ |
| elapsed | 411.3s | 427.3s | 152.9s | ≈ | 2.7× |

**가장 놀라운 변화**: **FP 19개** — EXP-003 의 1/3.8, EXP-004 의 1/15. precision-recall 두 축이 동시에 개선됨. threshold 0.99 가 매우 보수적이지만 그 임계치에서도 11개 양성을 잡았다는 의미.

## 7.4 진단 검증 (`notebook.executed.ipynb` grep)

| 진단 출력 | 값 | 검증 결과 |
|---|---|---|
| `augment 전 raw pos_weight` | 254.03 (neg=56395, pos=222) | ✅ baseline 비율 보존 |
| `augmented train shape` | (57061,), pos=666 | ✅ 222 + 2×222 = 666, no over-emphasis |
| `X_tab pos one-hot unique` | 24/666 | (낮음 — 양성이 24개 cell 에 집중, 원본 그대로) |
| `X_tab pos FULL unique` | 573/666 | ✅ num-jitter 가 정상 동작 (cell-level prior 위험 차단) |
| `X_tab pos num cols mean` | [1.41, 0.04, -0.21, -0.004] | scaled coord 정상 |
| `X_tab pos num cols std` | [3.08, 1.01, 0.83, 0.98] | scaled std ≈ 1 정상 |
| `X_ts_train fingerprint` | a316d218be570271 | ✅ EXP-004 v2 와 동일 (seed=42, deterministic augment 재현) |
| `X_tab_train fingerprint` | f15d9197cac54034 | ✅ EXP-004 v2 와 동일 |
| `pos_weight raw 강제 fix` | 254.03 | ✅ train_and_evaluate hook hit, silent dilution 차단 |
| build_hybrid_model 셀의 `dropout=0.1` | 1곳 signature | ✅ EXP-008 patch 적용 |
| `Conv1D(kernel_size=5,3, padding='causal')` | 2곳 (BN 사이) | ✅ EXP-003 patch 적용 |

→ 진단 출력 11개 모두 의도대로. 결과 신뢰 가능.

## 7.5 시너지 분석 — 왜 단독 합산보다 강한가?

**단독 효과 가산 예상치 (F2 기준 baseline 차분 합산)**:
- baseline F2 = 0.046
- EXP-003 - baseline = +0.121
- EXP-008 - baseline = +0.104
- EXP-004 - baseline = +0.079
- 단순 합산 = 0.046 + 0.121 + 0.104 + 0.079 = **0.350** (상한)
- 실제 EXP-009 F2 = **0.252** — 가산 예상치의 72% (재미있게도 가산보다 낮지만, 단독 1위의 1.51배)

**FP 폭감의 가산 분석**:
- baseline FP = 137
- EXP-003 FP = 72, EXP-008 FP = 136, EXP-004 FP = 280
- 단독 효과만 보면 ts_aug 는 FP 를 폭증시킴 (+143). 그런데 hybrid 에서는 **FP 19** — ts_aug 단독의 1/15
- 해석: ts_aug 가 만든 noisy 양성 예측을 Conv1D 의 local pattern 식별 + dropout 0.1 의 학습 강도가 정제. 즉 ts_aug 는 단독으로는 noise 까지 학습시키지만, 강한 representation backbone (Conv1D) + 적절한 capacity (dropout 0.1) 와 결합되면 noise 거르고 양성 다양성만 추출.

**threshold 의 의미**: best_th = 0.99 — model 이 매우 confident 한 양성만 cutoff. EXP-009 의 score distribution 이 베이스 모델보다 양성 cluster 가 더 또렷하게 분리됨 (PR_AUC 0.185 가 이를 뒷받침: threshold 무관 ranking 자체가 우수).

## 7.6 audit 필요성

세 patch 모두 단독 audit (PASS) 완료. hybrid 는 patch 합성 단계에서 새로운 결함 가능성:

- (a) Conv1D 가 augmented data 에 overfit? — fingerprint 가 EXP-004 v2 와 동일하므로 augment 자체는 변함 없음. Conv1D-augment 상호작용 별도 audit 권고.
- (b) dropout 0.1 + augment 가 prior 누수 통로 만드는가? — X_tab pos FULL unique 573/666 (jitter OK), one-hot 24/666 (baseline 그대로). 별도 점검 권고.
- (c) threshold 0.99 가 val 에서만 잡힌 outlier 인가? — val→test threshold 적용 (test 미재탐색) 자체는 PASS. threshold sensitivity 별도 plot 권고.

→ **다음 단계: 003+008+004 audit 한 것처럼 EXP-009 single-agent audit 1회 (시너지 결함 vs 진짜 시너지 확정)**.

---

---

# 8. EXP-009 최종 audit (3 agent 병렬 — 코드/인과/비평)

**audit 초점**: 신기록 (F2 0.2523, PR_AUC 0.1848, precision 0.3667) 이 (a) 데이터 누수/코드 결함으로 인한 artifact 인지, (b) EXP-004 v1 같은 cell-level prior inflation 부산물인지, (c) 진짜 시너지인지 다각도 확정.

## 8.1 code-reviewer — VERDICT: **PASS** (코드 결함 무)

10개 vector 모두 PASS — 심각도 Critical/High/Medium 결함 없음.

| 항목 | 상태 | 근거 |
|---|---|---|
| StandardScaler train-only fit | PASS | preprocess 내부 split 직후 fit, augment 는 반환 후 적용 |
| Target_Mean train-only fit | PASS | df_train_for_tm 만으로 fit, val/test transform only |
| augment scope train-only | PASS | shape (57061,)/pos=666, val/test (12133,) 유지 |
| raw pos_weight hook (fix #3) | PASS | preprocess 후 254.03 저장, train_and_evaluate hook 분기 hit |
| Conv1D redirect (patch_exp003) | PASS | (ts_input)→(x) 첫 BiLSTM 만 치환, ts_input 은 Model input 으로만 유지 |
| dropout 0.1 전파 (patch_exp008) | PASS | signature 1곳 → 5개 Dropout layer 모두 동일 변수 |
| val-only threshold 결정 | PASS | find_optimal_threshold(y_val, ...), test 라벨 미사용 |
| determinism env | PASS | PYTHONHASHSEED/TF_DETERMINISTIC_OPS/TF_CUDNN_DETERMINISTIC subprocess 주입 |
| seed 고정 | PASS | set_seeds(42) + augment RNG=RandomState(42) 격리 |
| 3 patch 합성 충돌 | PASS | 003(body) + 008(signature) + 004(별도 cell) 직교 위치 |

**Low 권고**: nbconvert cell['id'] schema 경고 (결과 무영향). Transformer 와의 미래 합성 시 redirect substring 충돌 가능성 (현재 무관).

## 8.2 tracer — VERDICT: **부분 시너지 + 부분 val 소표본 의존** (artifact 아님, 단 완전 안정 시너지도 아님)

6개 가설 검증 결과:

| H | 가설 | 판정 |
|---|---|---|
| H1 | EXP-004 v1 같은 cell-level prior inflation | **REFUTED** (Tier 2) — fingerprint EXP-004 v2 와 동일 (a316d218be570271/f15d9197cac54034), 8 fix 계승 확인. FULL unique 573/666 |
| H2 | 단순 threshold ↑ artifact | REFUTED — PR_AUC 0.1848 threshold-independent |
| H3 | Conv1D 가 양성 24 cell 의 국소 패턴 over-learn | 잔류 (Tier 3) — val_f2 가 epoch 진행에 따라 0.80→0.99 단조 우상향, dropout 0.1 약한 정규화에서 augment ×3 양성 sharp fit 가능 |
| H4 | val 47 pos 기반 best_th=0.99 가 소표본 noise | 잔류 (Tier 4) — **test curve 의 F2 최적은 0.97~0.98 (F2=0.2643)**, val→0.99 선택으로 보고치 F2=0.2523 가 test 최적 대비 0.012 낮음. 즉 보수 방향 오차 |
| H5 | test fold 가 우연히 유리한 분포 | REFUTED — val/test pos 동일(47/47), stratified split |
| H6 | 진짜 representation 개선 | partial — PR_AUC threshold-independent + EXP-003 의 th=0.99 에서 precision=0 이므로 hybrid 가 진짜 다른 score distribution. 단 ROC_AUC 거의 동일 (0.6907 vs 0.6908) → 전체 ranking 능력 개선 X, **고확신 양성 cluster 만 sharp 분리** |

**Critical Unknown** (가장 큰 미해결 증거):
- **val PR_AUC ≈ 0.059 vs test PR_AUC = 0.185 의 3배 괴리**. EarlyStopping 이 val_pr_auc 기준 best weights 복원했는데, val 0.059 epoch 의 weights 로 test 에서 0.185 가 나옴. (a) test fold 가 우연히 더 쉬운가, (b) test 전용 패턴 캡처, (c) val-EarlyStopping 경로 문제 — 세 가지로 분기.

**최우선 Discriminating Probe**: **k-fold (k=5) cross-validation** — val-test 괴리가 fold 전반 재현되는지(시너지) vs 특정 fold 폭발(우연/과집중) 직접 판별.

## 8.3 critic — VERDICT: **NEEDS_REPLICATION** (코드 결함 무, 단 single-seed 신기록 선언은 통계적 미흡)

8개 vector 비평:

| Vector | 비평 |
|---|---|
| Single-seed 위험 | TP 11/47 binomial 95% CI ≈ [6, 18]. seed 1개로 직전 1위(TP=9) 대비 +2 TP 신기록은 noise 가능 영역 |
| Threshold 0.99 안정성 | F2 0.94~0.99 plateau (0.252~0.264, max 0.2669 at 0.94) — spike 아님. 단 best_th=0.94 였다면 더 좋았을 수 (val 소표본 영향) |
| Precision 폭증 인과 | **artifact 아님** — EXP-003 의 threshold 를 0.99 로 raise 시 precision=0. hybrid 가 진짜 다른 score distribution |
| TP=11 통계 신뢰성 | **TP 11 vs EXP-003 TP 9 = Fisher exact p ≈ 0.79** → 통계적으로 전혀 유의하지 않음. F2 차이 큰 건 FP 폭감 (72→19) 가 분모에 작용한 효과 |
| 시너지 대안 가설 | (a) val-test similarity drift, (b) ROC_AUC 동급인데 PR_AUC 4.7× 증가 = high-precision tail 만 변화, (c) dropout 0.1+augment 의 high-confidence overfit 가능성 |
| ROC_AUC 거의 동일 | **새 representation 학습 아닌 score 분포 high-tail 보정 효과** 가능성 高 |
| Patch 순서 commutative | 코드상 직교 위치 → 순서 무관 (문제 없음) |
| history dict 비어있음 | `"history": {}` — 학습 곡선 (loss/val_loss) 검증 자료 부족, 별도 결함 |

**거짓 양성 위험 시나리오**:
1. val→test 0.99 threshold spike 가 happy accident (val 0.99 에서 잡힌 게 test plateau 좌측 endpoint 와 우연 정렬)
2. seed=42 에서 TP 11 중 2~3개가 다른 seed 에서 안 잡히면 TP=9, F2≈0.18 로 EXP-003 와 동급 회귀 가능
3. PR_AUC 0.185 의 분모 효과 — 양성 47/12086 극저 base rate 에서 작은 변동이 큰 ratio 로 부풀려짐

## 8.4 3 agent 종합 결론

**합의**: 
- **코드 결함은 없음** (3 agent 일치). 데이터 누수, fit timing 위반, isolation 위반, hook 실패, redirect 누락, EXP-004 v1 같은 cell-level prior inflation 부산물 — 모두 **아님**.
- 신기록 자체는 **artifact 도 아님** (PR_AUC threshold-independent, EXP-003 가 th=0.99 에서 precision=0 인 점이 hybrid 의 진짜 score distribution 차이 증명).

**유보**:
- TP 11 vs 9 의 차이는 **통계적으로 유의하지 않음** (Fisher p≈0.79). F2 차이 큰 건 FP 폭감 효과.
- ROC_AUC 가 EXP-003 와 사실상 동일 → **전체 ranking 능력 개선 X**, **고확신 양성 cluster 만 sharp 분리** (부분 시너지).
- val PR_AUC ≈ 0.059 vs test PR_AUC = 0.185 의 **3배 괴리**가 가장 큰 미해결 증거 → val-EarlyStopping 경로 또는 test fold 우연 유리 가능성.
- single-seed 결과만으로 신기록 선언은 통계적/방법론적으로 정당하지 못함.

**최종 결과 신뢰 수준**: **결함은 없음 (코드 무효 X)** + **재현성 검증 필요 (multi-seed/k-fold 없이는 신기록 단정 X)**.

---

# 다음 단계 후보 (8장 audit 반영 우선순위 재정렬)

1. **EXP-010 (EXP-009 + k-fold CV, k=5)** — **최우선**. tracer 의 Critical Unknown (val 0.059 vs test 0.185) 해소 + critic 의 multi-seed replication 요구 동시 처리. fold 전반에서 F2 mean ± std 측정. EXP-005 미실행 슬롯 재활용.
2. **EXP-009 threshold sensitivity sweep** — 0.50~0.999 grid 로 val vs test 의 F2 curve overlay plot. val→0.99 선택과 test 최적 (0.97~0.98) 의 괴리 정량화.
3. **EXP-011 (EXP-003 + dropout 0.1, no aug 대조군)** — augment 효과 분리. hybrid 의 어느 기법이 진짜 기여인지 ablation.
4. **EXP-009 history 저장 fix** — train_and_evaluate 에서 `history.history` 가 run_metrics.json 으로 dump 되도록 수정. learning curve (loss/val_loss/val_pr_auc) 시각화로 학습 동작 검증.
5. **EXP-006 비대화 root cause audit** — patch 가 실제 input dim 까지 영향 주는지 별도 verification (변경 누락 가능성 점검).
6. **TP sample-level 교차분석** — EXP-009 의 TP 11개와 EXP-003 의 TP 9개 overlap. 어떤 양성 sample 이 hybrid 에서만 잡히는지, 그 sample 의 특징.
7. **EXP-009 v2 (선택)** — precision 추가 개선 (calibration Platt/isotonic), conv kernel size sweep. multi-seed 안정성 확인 후 진행.
