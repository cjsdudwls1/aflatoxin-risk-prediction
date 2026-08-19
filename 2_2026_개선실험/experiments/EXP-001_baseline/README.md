# EXP-001_baseline

## 한 줄 요약
F-01_M-01 audit fix 적용 후의 baseline 실행 — 향후 모든 실험의 비교 기준점.

## 무엇을 바꿨나
- DEFECT-014/015 fix: TF 결정성 환경변수를 nbconvert subprocess 환경에 명시 주입
- D-008 fix: SMOTE 제거
- 한글 폰트 처리 적용
- audit findings (Critical-1, A-M1, B-M3, B-M5) 코드 반영
- 사용자 결정 외 findings 는 보고서 한계점으로 서술 (코드 변경 없음)

## 결과 요약
| 지표 | 값 |
|---|---|
| ROC AUC | 0.6827 |
| PR AUC | 0.0126 |
| Best threshold (F2 max on val) | 0.73 |
| Precision @ best | 0.0214 |
| Recall @ best | 0.0638 |
| F2 @ best | 0.0457 |
| Confusion matrix (test) | TN=11949, FP=137, FN=44, TP=3 |

## 한계
- 양성 47개 중 3개만 검출 (놓침 44개) — 실용 수준 미달
- PR AUC 0.013 — 베이스라인(양성비율 0.0039)의 약 3.2배에 그침
- 다음 실험에서 개선 필요한 방향: class imbalance 처리, loss 함수, threshold 운용

## 재현 방법
```bash
cd "2026식약처/6. F-01_M-01 추가 수정 코드"
EXP_ID=EXP-001_baseline python -m modal run src/modal_run.py::run_notebook
```
git commit `b4f7af0` 시점의 `notebooks/3.모델링.executed.ipynb` 와 동일한 코드로 실행.

## 파일
- `notebook.executed.ipynb` — 실행 완료된 노트북 (셀 출력 포함)
- `run_metrics.json` — 메트릭 + threshold 곡선 원본
- `execute.log` — nbconvert 실행 로그
- `manifest.json` — 실험 메타데이터
