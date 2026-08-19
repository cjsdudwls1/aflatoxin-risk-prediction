# EXP-004_ts_augmentation (v2 — audit 후 재실험)

## 한 줄 요약
양성 sample 시계열 augmentation 으로 BiLSTM robustness 강화. v1 결함 8개 fix 후 재실험.

## Base experiment
EXP-001_baseline

## v1 (archived as `_archive_EXP-004_v1_buggy`) 결함과 v2 fix
| # | v1 결함 | v2 fix |
|---|---|---|
| 1 | X_tab 다양성 0 — cell-level prior inflation (tabular branch 가 양성 외움) | num cols 4개에 jitter sigma=0.02 |
| 2 | np.roll wrap-around — 검사 직전 신호가 첫 위치로 wrap, 시계열 인과 파괴 | edge padding (np.tile + concat, 인과 보존) |
| 3 | pos_weight silent dilution — augment 후 254 → 50, baseline 불균형 정책 깨짐 | train_and_evaluate hook + EXP004_RAW_POS_WEIGHT globals() override |
| 4 | 시계열-tabular conditional joint 파괴 — X_ts jitter + X_tab copy 비대칭 | X_tab num cols 도 jitter (시계열-tab noise 균형) |
| 5 | physical bound 위반 risk — 일조/습도/강수 등 음수 불가 변수에 N(0,0.05) | noise sigma 0.05→0.02, scale 0.9~1.1→0.95~1.05 |
| 6 | scale 곱셈 순서 모호 — (x+eps)*s 로 noise 도 scale 곱해짐 | scale 먼저 → noise 추가 (effective budget 일정) |
| 7 | over-emphasis — n_copies=4 로 양성 5x 가중 | n_copies=4→2 |
| 8 | 진단 부족 — augment 후 분포 측정 불가 | unique row + num stats + sha256 fingerprint print |

## 가설
- v1 의 PR-AUC 4.8x 향상 (0.013 → 0.063) 은 H3(over-confidence by duplicate) + H5(augment 효과 미미) 의 부산물이었다 — 즉 단순 5x oversampling 효과.
- v2 fix 후에는 X_tab cell-level prior inflation 차단 + pos_weight raw 유지 + 인과 보존 augment 로, 진짜 generalization 향상만 측정.
- 기대 결과: F2 가 baseline 0.046 보다는 향상되되 v1 의 0.131 보다 낮을 가능성 (cell prior inflation 제거 효과). best_threshold 도 baseline 의 0.73 근처로 복귀해야 정상.

## 무엇을 바꿨나
- preprocess() 호출 직후 raw pos_weight 저장 (`EXP004_RAW_POS_WEIGHT`)
- `_ts_augment_positives_v2(n_copies=2, seed=42, n_tab_num=4)` 적용
  - 시계열: scale (0.95~1.05) 먼저 → noise (sigma=0.02) 추가 → edge padding shift (±2, no wrap)
  - X_tab: 마지막 4 num cols (`Target_Mean`, 독소/저습도/연속 무강수 일수) 에 noise sigma=0.02
- train_and_evaluate 내부 `pos_weight = calculate_pos_weight(y_train)` 다음에 globals override hook

## 실행

```powershell
$env:EXP_ID="EXP-004_ts_augmentation"
$env:CHANGED="ts_augmentation v2 (8 fix): pos_weight raw, no-wrap edge padding, X_tab num jitter, scale-first noise, n_copies=2, sigma=0.02"
$env:NOTES="v1 결함 8개 audit (code-reviewer + tracer + critic) 후 v2 재작성. v1 결과 PR_AUC 4.8x는 cell-level prior inflation 부산물로 판정."
$env:BASE_EXP="EXP-001_baseline"
python -m modal run src/modal_run.py::run_notebook
python src/update_leaderboard.py
```

## 결과
실행 후 `manifest.json` / `run_metrics.json` 에서 핵심 수치 정리.

## 진단 print 검증 항목 (stdout 확인)
- `[EXP-004 v2] augment 전 raw pos_weight: {X.XX}` — baseline 254 와 비교
- `[EXP-004 v2] pos_weight raw 강제 fix: 254.XX` — train_and_evaluate hook 작동 확인
- `[EXP-004 v2] X_tab pos one-hot unique rows: ~89 / 666` — cell-level cardinality
- `[EXP-004 v2] X_tab pos FULL unique rows: 666 / 666` — jitter 가 unique 보장 확인
- fingerprint sha256[:16] — reproducibility 추적용
