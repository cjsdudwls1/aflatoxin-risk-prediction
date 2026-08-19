# EXP-010_cv5_hybrid

## 한 줄 요약
EXP-009(F2=0.252, single-seed)의 안정성을 **5-fold Stratified CV**로 검증.
**F2 = 0.250 ± 0.036 으로 단일 실행값과 거의 일치 → 그 수치는 운이 아니라 재현 가능한 값**임을 확인.

## Base experiment
EXP-009_hybrid_conv1d_dropout_tsaug (모델·데이터 config 그대로, **평가 프로토콜만 CV로 교체**)

## 가설 / 목적
EXP-009 audit(critic)이 *"single-seed 결과라 k-fold replication 전까지 확정 신기록 보류"* 로 지적했다.
같은 config 를 5-fold CV 로 돌려 F2 의 **평균과 분산**을 측정한다.
→ 단일 측정값이 **재현 가능한 실력(skill)** 인지 아니면 **운(luck)** 인지를 구분하는 것이 목적.

## 무엇을 바꿨나 (EXP-009 대비 모델·데이터는 그대로)
모델 구조 / 하이퍼파라미터 / augmentation 은 EXP-009 와 **완전히 동일**. 평가 방식만 단일 70/15/15 split → 5-fold CV.

| 항목 | 내용 |
|---|---|
| CV | `StratifiedKFold(k=5, shuffle=True, seed=42)` — 양성 비율을 유지한 채 5분할 |
| per-fold split | 각 fold: test = 20% (held-out), 나머지 trainval 80% 를 train / val 로 분리(`test_size=0.1875` → val ≈ 전체의 15%) |
| 누수 방지 | 시계열 StandardScaler · Target_Mean(m-estimate) · OneHotEncoder · num scaler **전부 그 fold 의 train 에만 fit** → val/test 는 `transform` 만 |
| augmentation | ts_aug v2(fix8) 를 **각 fold 의 train 에만** 적용 (val/test 는 절대 augment 안 함) |
| threshold | fold 마다 **val 에서 F2 최적 임계치 결정 → test 1회 평가** (val→test isolation 유지) |
| pos_weight | augment 전 fold train 의 raw 불균형 비율 사용 (EXP-004 v2 정책 유지) |
| seed | fold i 마다 `set_seeds(42+i)` |

patch: `scripts/make_exp010_cv.py` — patch_exp003 + patch_exp008 로 모델 구조를 EXP-009 와 동일하게 만든 뒤, cell#25 의 single-split main 을 5-fold CV 루프로 치환.

## 실행
PowerShell:
```powershell
python scripts/make_exp010_cv.py          # EXP-010 notebook 생성 (patch + cell#25 CV 치환, compile 자체검증)
$env:EXP_ID="EXP-010_cv5_hybrid"
python -m modal run src/modal_run.py::run_notebook
```
- 실행 환경: Modal GPU L4, Python 3.11, **elapsed 1987.3 s** (약 33분, 5 fold × 학습)
- 결정성 env(`PYTHONHASHSEED=42`, `TF_DETERMINISTIC_OPS=1`, `TF_CUDNN_DETERMINISTIC=1`) 주입
- ⚠️ `run_metrics.json` 은 CV main 이라 비어있음(정상) — CV 결과는 `cv_*.{csv,json,png}` 에 별도 저장.
  modal_run.py 는 `/work/outputs/cv_*` 를 자동 회수하지 않으므로 `modal volume get aflatoxin-runs outputs/cv_results.csv ...` 로 회수함.

## 결과

### fold 별
| fold | F2 | ROC-AUC | PR-AUC | Precision | Recall | th | TP / FN / FP / TN | test 양성 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.2333 | 0.7039 | 0.1442 | 0.2917 | 0.2222 | 0.99 | 14 / 49 / 34 / 16080 | 63 |
| 2 | 0.2332 | 0.6018 | 0.0986 | 0.1758 | 0.2540 | 0.98 | 16 / 47 / 75 / 16039 | 63 |
| 3 | 0.2326 | 0.6976 | 0.1194 | 0.1818 | 0.2500 | 0.97 | 16 / 48 / 72 / 16041 | 64 |
| 4 | 0.3135 | 0.6954 | 0.1829 | 0.3725 | 0.3016 | 0.99 | 19 / 44 / 32 / 16081 | 63 |
| 5 | 0.2358 | 0.6133 | 0.1001 | 0.2273 | 0.2381 | 0.99 | 15 / 48 / 51 / 16062 | 63 |

### 요약 (mean ± std) vs EXP-009 단일값
| 지표 | **CV mean ± std** | median | (min ~ max) | EXP-009 단일 | 해석 |
|---|---|---|---|---|---|
| **F2** | **0.250 ± 0.036** | 0.233 | 0.233 ~ 0.314 | 0.252 | **거의 일치 → 재현됨** ✅ |
| ROC-AUC | 0.662 ± 0.050 | 0.695 | 0.602 ~ 0.704 | 0.691 | 단일값은 CV 상위 fold 수준 |
| PR-AUC | 0.129 ± 0.035 | 0.119 | 0.099 ~ 0.183 | 0.185 | 단일값(0.185) > CV max(0.183) → 분포 상단 |
| Precision | 0.250 ± 0.083 | 0.227 | 0.176 ~ 0.373 | 0.367 | 단일값 ≈ CV max(0.373) → 분포 상단, 분산 큼 |
| Recall | 0.253 ± 0.030 | 0.250 | 0.222 ~ 0.302 | 0.234 | 단일값은 CV 하위 |
| best threshold | 0.984 ± 0.009 | 0.99 | 0.97 ~ 0.99 | 0.99 | 매우 높은 임계치(공통 특성) |

> test 규모가 다름에 주의: EXP-009 single 은 15% test(양성 47), EXP-010 fold 는 20% test(양성 63~64). raw count 가 아니라 **비율 지표(F2/precision/recall)** 로 비교해야 한다.

## 핵심 판정
1. **F2 재현 확인** — CV 평균 0.250 ≈ 단일 실행 0.252. std 0.036(5개 중 4개 fold 가 0.233~0.236 에 밀집). EXP-009 의 F2 신기록은 **운이 아니라 안정적으로 재현되는 값**.
2. **단, EXP-009 단일 실행은 precision·PR-AUC·ROC-AUC 에서 운 좋게 분포 상단을 뽑았다** — EXP-009 의 precision 0.367 / PR-AUC 0.185 는 5-fold 중 가장 좋은 fold(fold 4: 0.373 / 0.183) 와 거의 같은 수준. CV 평균으로 보면 precision **0.250**, PR-AUC **0.129** 가 더 정직한 추정치. *F2 가 일치한 건 (precision 상위)×(recall 하위)가 상쇄됐기 때문.*
3. **보고용 권장 수치** — 단일 값 대신 **F2 = 0.250 ± 0.036 (5-fold CV)** 로 보고. precision / PR-AUC 도 CV 평균±std 로 쓰면 single-seed 낙관 편향을 피한다.

## "단일 측정값은 무조건 reject 되나?" 질문에 대한 답
- **무조건은 아니다.** 다만 0.5% 극불균형 + 소표본(fold test 양성 63개) 환경에서는 분할 운에 따라 지표가 크게 흔들리므로(이번에 precision 0.176~0.373, ROC-AUC 0.602~0.704 로 확인됨), 단일 값만 제시하면 *"운인지 실력인지"* 구분이 안 돼 신뢰도가 약하다.
- 이번 CV 가 정확히 그 약점을 메운다: F2 가 5-fold 에서 **0.250 ± 0.036 으로 안정** → 단일 실행값이 재현 가능함을 입증. 평균±std 로 보고하는 것이 표준이며 가장 방어적이다.

## 산출물
- `cv_results.csv` / `cv_summary.csv` / `cv_results.json` — fold 별 + 요약 통계
- `figures/predictions_fold1~5.npz` — fold 별 val/test 예측확률 + threshold sweep 배열 (곡선 figure 재현용)
- **저널 규격 그림**(`figures/` 폴더) — TIFF, **컬러(RGB)**, **300 dpi**, **84 mm**(1-column), 제목 없음. 색맹 친화(Okabe-Ito) 팔레트 + 선스타일/마커 이중 구분(흑백 복사 대비). 영문 캡션은 `figures/figure_captions.md`.
  - 요약: `Fig_CV_metrics_dotplot.tiff`(권장) / `Fig_CV_metrics_lines.tiff`(대안) / `Fig_confusion_matrix.tiff`
  - 곡선: `Fig_ROC.tiff` / `Fig_PR.tiff` / `Fig_threshold_sweep.tiff` / `Fig_prob_dist.tiff` / `Fig_lift_gain.tiff` / `Fig_calibration.tiff`
  - 생성(`figures/` 안에서 실행): `python make_journal_figs.py`(요약 3종, 상위 `cv_results.json` 사용) + `python make_curve_figs.py`(곡선 6종, `predictions_fold*.npz` 사용). `_preview_*.png`(200 dpi)는 검수용 — 제출 X.
- `cv_metrics.png` — 노트북 내장(제목·컬러·dpi120) → **저널 규격 아님**, 분석 기록용. 제출엔 위 TIFF 사용.
- `notebook.ipynb`(실행 출력 포함) / `execute.log` / `manifest.json` / `run_metrics.json`(CV main 이라 비어있음 — 정상)

> **[fix] `predictions.npz` 덮어쓰기 버그** — train_and_evaluate 가 fold마다 고정명으로 저장해 fold-5 만 생존했음.
> cell#25 를 fold별 파일명(`globals()['CV_FOLD_IDX']` 기반)으로 패치(code-reviewer APPROVE, 학습·집계 로직 불변 = 결정성 유지),
> 재실행으로 `predictions_fold1~5.npz` 5개 확보. 5개 모두 재계산 metric 이 cv_results.json 과 일치 확인.
> 생성 스크립트 `scripts/make_exp010_cv.py` 에도 동일 패치 반영(단계 e).

## 알려진 한계 / 다음 단계
- ROC-AUC / PR-AUC 의 fold 변동(std 0.050 / 0.035)은 test 양성이 fold 당 ~63개로 작아 생기는 **소표본 분산**.
- threshold 가 0.97~0.99 로 매우 높음 — score 분포가 양성에 낮은 확률을 주고 극단 high-tail 에서만 양성을 잡는 구조(EXP-009 와 동일 특성).
- **다음**: 이 CV 수치(F2 0.250±0.036)로 논문 Results 의 main number 를 확정.
