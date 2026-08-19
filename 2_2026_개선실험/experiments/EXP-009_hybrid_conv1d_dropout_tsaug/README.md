# EXP-009_hybrid_conv1d_dropout_tsaug

## 한 줄 요약
단독 audit 를 통과한 상위 3개 기법(EXP-003 Conv1D+BiLSTM, EXP-008 dropout 0.1, EXP-004 v2 시계열 augmentation)을 모두 합친 hybrid 실험. **F2 = 0.252 로 전체 실험 중 신기록.**

## Base experiment
EXP-001_baseline

## 가설
세 기법은 각각 **서로 다른 축**을 건드린다 — 축이 겹치지 않으면 합쳤을 때 효과가 더해질(상보적) 가능성이 높다.

| 출처 | 건드린 축 | 단독 효과 |
|---|---|---|
| EXP-003 | 모델 구조 (local pattern 추출) | recall↑ (TP 3→9) |
| EXP-008 | 정규화 강도 (underfit 완화) | recall↑ (TP 3→10) |
| EXP-004 v2 | 데이터 양 (양성 augmentation) | recall↑ (TP 3→12) |

- 단독 기법들은 모두 **recall 위주**였고 precision 은 낮았다(오탐 FP 가 72~280개).
- 결합하면 서로 다른 정보가 score 분포를 정교하게 만들어, **높은 threshold 에서 오탐이 줄고 precision 이 크게 오를 것**으로 기대.

## 무엇을 바꿨나 (3개 기법 union)
| 출처 | 변경 내용 |
|---|---|
| EXP-003 | BiLSTM 앞에 `Conv1D(kernel=5)` + `Conv1D(kernel=3)` + `BatchNorm` ×2 (causal padding) 추가, 첫 BiLSTM 입력을 Conv 출력으로 redirect |
| EXP-008 | `build_hybrid_model` 의 dropout 0.3 → 0.1 (양성 222개 underfit 완화) |
| EXP-004 v2 | 양성 시계열 augmentation fix8 — raw pos_weight 유지, no-wrap edge padding, X_tab num jitter, scale-first noise, n_copies=2, σ=0.02, scale 0.95~1.05 |

patch 합성은 `scripts/apply_exp_patches.py` 의 `patch_exp009` 가 담당 (patch_exp003 → patch_exp008 → patch_exp004 **순차 적용**).

## 실행

PowerShell:
```powershell
# 1) master notebook 에 patch_exp009 적용 (scripts/apply_exp_patches.py)
# 2) Modal GPU L4 에서 실행
$env:EXP_ID="EXP-009_hybrid_conv1d_dropout_tsaug"
$env:CHANGED="archi: Conv1D(k=5)+Conv1D(k=3)+BN x2 -> BiLSTM; hyper: dropout 0.3 -> 0.1; data: ts_aug v2 fix8"
$env:NOTES="EXP-003 + EXP-008 + EXP-004 v2 hybrid. Top-3 audit PASS technique union."
$env:BASE_EXP="EXP-001_baseline"
python -m modal run src/modal_run.py::run_notebook
python src/update_leaderboard.py
```
- 실행 환경: Modal GPU L4, Python 3.11, elapsed 411.3 s
- 결정성 env(`PYTHONHASHSEED=42`, `TF_DETERMINISTIC_OPS=1`, `TF_CUDNN_DETERMINISTIC=1`) 를 nbconvert subprocess 에 주입

## 결과
| 지표 | **EXP-009** | baseline(001) | EXP-003 | EXP-008 | EXP-004 v2 |
|---|---|---|---|---|---|
| F2 | **0.252** | 0.046 | 0.167 | 0.150 | 0.125 |
| PR-AUC | **0.185** | 0.013 | 0.039 | 0.040 | 0.018 |
| ROC-AUC | 0.691 | 0.683 | 0.691 | 0.718 | 0.698 |
| Precision | **0.367** | 0.021 | 0.111 | 0.068 | 0.041 |
| Recall | 0.234 | 0.064 | 0.191 | 0.213 | 0.255 |
| best threshold | 0.99 | 0.73 | 0.79 | 0.77 | 0.95 |
| TP / FN / FP / TN | **11 / 36 / 19 / 12067** | 3/44/137/11949 | 9/38/72/12014 | 10/37/136/11950 | 12/35/280/11806 |

- **PR-AUC 가 baseline 대비 약 14배, EXP-003 대비 약 4.7배** 상승.
- **Precision 0.367 은 전 실험 최고** — 오탐 FP 가 72~280개에서 **19개로 급감**한 것이 핵심.
- recall(0.234)은 중간 수준이지만 정밀도가 받쳐줘서 F2 신기록 달성.

### 진단 print 검증 (notebook.executed.ipynb stdout, 11개 모두 PASS)
- `raw pos_weight: 254.03 (neg=56395, pos=222)` — augment 전 불균형 정책 유지
- `augmented train shape: (57061,), pos=666` — n_copies=2 적용 (222 → 666)
- `X_tab pos one-hot unique rows: 24 / 666` / `FULL unique rows (after num jitter): 573 / 666` — num jitter 가 cell-level 다양성 확보
- `X_ts fingerprint: a316d218be570271` / `X_tab fingerprint: f15d9197cac54034` — **EXP-004 v2 와 동일** (augmentation 재현성 확인)
- `pos_weight raw 강제 fix: 254.03` — train_and_evaluate hook 작동
- `dropout=0.1` (signature) / `Conv1D(kernel_size=5/3, padding='causal')` — EXP-008·003 patch 적용 확인

## Audit 결과 (3 agent 병렬: code-reviewer / tracer / critic)
- **code-reviewer → PASS**: 데이터 누수·기법 무효화 없음. fingerprint 가 EXP-004 v2 와 동일해 augmentation 이 의도대로 작동.
- **tracer → 부분 시너지**: v1 의 cell-level prior inflation(tabular branch 가 양성을 외우는 부작용) 가설은 **REFUTED**. 단 **val PR-AUC 0.059 vs test PR-AUC 0.185 의 3배 괴리**가 미해결 (소표본 분산 가능성).
- **critic → NEEDS_REPLICATION**: TP 11 vs EXP-003 의 9 는 Fisher exact p≈0.79 로 **통계적 유의성 없음**. ROC-AUC 가 EXP-003 과 사실상 동일(0.691) → 새 ranking 능력이 아니라 score 분포의 **high-tail 보정** 효과.

**종합**: 코드 결함은 없고 신기록 수치 자체는 artifact 아님(EXP-003 은 th=0.99 에서 precision=0 이므로 hybrid 는 진짜 다른 score 분포를 만든다). 다만 **single-seed 결과**라 k-fold(k=5) replication 전까지 "확정 신기록" 선언은 보류.

## 소감 / 다음 단계
- 상보성 가설은 **precision 축에서 확인**됨 (FP 136~280 → 19). 구조 + 정규화 + 데이터가 각자 다른 약점을 메운 결과로 해석.
- audit 가 지적한 val-test 괴리와 단일 seed 한계 때문에, 보고서/논문에 핵심 결과로 쓰려면 **k-fold CV(EXP-010)로 안정성을 먼저 검증**하는 것이 안전.
