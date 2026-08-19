# EXP-003_conv1d_bilstm

## 한 줄 요약
<여기에 이번 실험의 목적을 한 줄로>

## Base experiment
EXP-001_baseline

## 가설
<왜 이 변경이 baseline 보다 좋을 것이라 생각하는가>

## 무엇을 바꿨나
- (예: loss: BCE -> Focal(gamma=2))
- (예: epochs: 30 -> 50)

## 실행

PowerShell:
```powershell
$env:EXP_ID="EXP-003_conv1d_bilstm"
$env:CHANGED="<변경점들; 으로 구분>"
$env:NOTES="<자유 메모>"
$env:BASE_EXP="EXP-001_baseline"
python -m modal run src/modal_run.py::run_notebook
python src/update_leaderboard.py
```

## 결과
(실행 후 `manifest.json` / `run_metrics.json` 참고. 핵심 수치 여기에도 손으로 정리.)

## 소감
(왜 좋아졌나/안 됐나, 다음에 시도할 가치)
