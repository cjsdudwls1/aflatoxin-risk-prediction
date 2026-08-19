# F-01_M-01 추가 수정 코드 - 실험 폴더

아플라톡신 검출 모델링 실험 폴더. 매 실험을 **한 폴더 한 묶음**으로 관리하고,
메트릭은 자동으로 leaderboard 에 누적되도록 구성되어 있다.

## 마스터 노트북과 실험 폴더의 관계 (핵심 개념)

- `notebooks/3.모델링.executed.ipynb` 는 **마스터 노트북**. "현재 baseline = 가장 최근에
  검증된 안정 코드" 이며, **직접 수정하지 않는다**.
- 새 실험은 `src/new_experiment.py` 가 마스터를 실험 폴더로 **복사**한 뒤,
  복사본을 수정하는 방식으로 진행한다. 마스터는 깨끗하게 보존되고, 실험에 쓰인
  코드는 결과와 같은 폴더에 그대로 박혀 있어 6개월 뒤에도 재현 가능하다.
- 좋은 실험이 나오면 그 실험의 `notebook.ipynb` 를 마스터에 promote (덮어쓰기 +
  git commit) 한다. 이때부터 새 실험들은 promote 된 마스터를 baseline 으로 삼는다.

## 마스터 promote 시점과 방법

### 언제 promote 하나
- 어떤 실험이 baseline 보다 명확히 좋고, 다음 실험들의 새 출발점으로 쓰는 게 합리적일 때.
  - 예: PR AUC / F2 가 의미 있게 상승하고 confusion matrix 도 더 나아진 경우.
- 단일 실험의 우연(랜덤 시드 변동, 데이터 누락 보정 등)이 아니라 **재현 가능한 개선** 일 때.
- 같은 변경 위에 새 실험을 쌓을 계획이 있을 때. 예: focal loss 가 효과적이면 그 다음
  실험은 focal loss 위에서 다른 변수를 변경하는 게 자연스러움.

### 어떻게 promote 하나 (예: `EXP-007_focal_v2` 를 새 마스터로)

PowerShell:
```powershell
Copy-Item "experiments/EXP-007_focal_v2/notebook.ipynb" `
          "notebooks/3.모델링.executed.ipynb" -Force
git add "notebooks/3.모델링.executed.ipynb"
git commit -m "promote EXP-007_focal_v2 to master baseline"
```

Bash:
```bash
cp "experiments/EXP-007_focal_v2/notebook.ipynb" \
   "notebooks/3.모델링.executed.ipynb"
git add "notebooks/3.모델링.executed.ipynb"
git commit -m "promote EXP-007_focal_v2 to master baseline"
```

### 주의사항
- **마스터에는 입력 노트북 (`notebook.ipynb`) 만 promote.** 실행본
  (`notebook.executed.ipynb`)은 출력 셀에 GPU/시드별 변동이 박혀 있어 마스터로 부적합.
  `new_experiment.py` 가 마스터를 복사할 때 실행본을 받으면 모든 새 실험이 옛 출력에
  오염된 상태에서 시작된다.
- promote 커밋은 **단독 커밋** 으로. 다른 변경과 섞으면 "어느 시점부터 어떤 baseline
  이었나" 가 git log 에서 흐려진다. 커밋 메시지에 promote 한 EXP_ID 를 반드시 포함.
- promote 직후 시작하는 새 실험은 `BASE_EXP` 환경변수를 promote 된 실험으로 지정
  (예: `BASE_EXP=EXP-007_focal_v2`). manifest.json 의 `base_exp` 가 올바르게 박힌다.
- promote 했어도 옛 실험 폴더는 그대로 둔다. 그 폴더의 `notebook.ipynb` 가 그 시점
  baseline 을 동결 보존한다.

### 안 좋은 promote 를 되돌리려면
- promote 커밋을 `git revert <sha>` 로 되돌리면 마스터가 promote 이전 상태로 복원.
- 또는 명시적으로 baseline 의 `notebook.ipynb` 를 다시 복사:
  ```bash
  cp experiments/EXP-001_baseline/notebook.ipynb notebooks/3.모델링.executed.ipynb
  ```

## 폴더 구조

```
6. F-01_M-01 추가 수정 코드/
├── README.md                    # ← 본 문서
├── notebooks/                   # 마스터 노트북 (직접 수정 금지 — 정본 보관)
│   ├── 1.5 대상기간 선정.ipynb
│   ├── 1.데이터 전처리+2.EDA.executed.ipynb
│   ├── 2. 특성공학.executed.ipynb
│   └── 3.모델링.executed.ipynb
├── data/                        # 1·2번 노트북 산출 데이터
│   ├── interim/df_fixed.pkl.gz
│   └── processed/df_enhanced.{parquet,pkl.gz}
├── src/                         # 재사용 코드/인프라
│   ├── new_experiment.py        # 실험 폴더 생성 + 마스터 복사 + README 스텁
│   ├── modal_run.py             # Modal GPU 실행 + 산출물 자동 저장
│   ├── update_leaderboard.py    # manifest.json → leaderboard.csv 갱신
│   ├── font_helper.py
│   ├── collect_env.py
│   └── requirements.txt
├── experiments/                 # 실험 결과 (한 실험 = 한 폴더)
│   └── EXP-001_baseline/
│       ├── notebook.ipynb            # 그 실험에 쓰인 입력 코드
│       ├── notebook.executed.ipynb   # 실행 후 셀 출력 포함된 결과 노트북
│       ├── run_metrics.json
│       ├── execute.log
│       ├── manifest.json             # 실험 메타 + 메트릭 요약
│       └── README.md                 # 변경점/가설/소감
├── reports/                     # 실험 비교용 자료
│   ├── leaderboard.csv          # 모든 실험을 한 행씩 자동 누적
│   └── experiment_log.md        # 사람 친화 실험 일지
└── _archive/                    # 옛 산출물 백업 (보관용, 안 건드림)
```

## 새 실험을 돌리는 방법 (표준 워크플로우)

### 1. (선택) 노트북 1·2번을 다시 돌려 데이터 갱신
새 특성공학을 시도할 때만. `data/processed/df_enhanced.parquet` 가 갱신되어야 한다.

### 2. 새 실험 폴더 만들기
```bash
python src/new_experiment.py EXP-002_focal_loss
```
생성되는 것:
- `experiments/EXP-002_focal_loss/notebook.ipynb` (마스터 복사본)
- `experiments/EXP-002_focal_loss/README.md` (변경점/가설 적을 스텁)

### 3. 실험 폴더 안의 notebook.ipynb 를 수정
이번 실험에서 바꿀 부분을 `experiments/EXP-002_focal_loss/notebook.ipynb` 에 적용한다.
같은 폴더의 `README.md` 에 변경점과 가설도 함께 적어두면 나중에 알아본다.
**마스터(`notebooks/3.모델링.executed.ipynb`)는 건드리지 않는다.**

### 4. Modal 에서 실행

PowerShell:
```powershell
$env:EXP_ID="EXP-002_focal_loss"
$env:CHANGED="loss: BCE -> Focal(gamma=2); epochs: 30 -> 50"
$env:NOTES="Focal loss로 양성 가중 강화"
$env:BASE_EXP="EXP-001_baseline"
python -m modal run src/modal_run.py::run_notebook
```

Bash:
```bash
EXP_ID=EXP-002_focal_loss \
CHANGED="loss: BCE -> Focal(gamma=2); epochs: 30 -> 50" \
NOTES="Focal loss로 양성 가중 강화" \
BASE_EXP=EXP-001_baseline \
python -m modal run src/modal_run.py::run_notebook
```

→ 같은 폴더에 `notebook.executed.ipynb`, `execute.log`, `run_metrics.json`,
`manifest.json` 이 자동 추가됨.

### 5. leaderboard 갱신
```bash
python src/update_leaderboard.py
```
→ `reports/leaderboard.csv` 가 모든 실험을 모아 다시 생성됨.
엑셀에서 정렬하면 어떤 실험이 좋았는지 한눈에 비교 가능.

### 6. (권장) experiment_log.md 에 한두 줄 일지 추가
"이번 실험이 왜 좋아졌는가/왜 안 됐는가" - 메트릭이 답하지 않는 부분.

## EXP_ID 명명 규칙

- 형식: `EXP-NNN_short_name`
  - `NNN` : 3자리 일련번호 (001, 002, ...)
  - `short_name` : 변경의 핵심을 1~3 단어 영문 snake_case 로
- 예시: `EXP-002_focal_loss`, `EXP-003_class_weight`, `EXP-004_undersample_3x`
- 영문/숫자/`-`/`_` 만. 띄어쓰기/한글/특수문자 금지 (모달 경로 안전).
- `new_experiment.py` 가 이 규칙을 강제로 검사하고, 폴더가 이미 있으면 에러로 막는다.
- `modal_run.py` 는 이미 `notebook.executed.ipynb` 가 있는 폴더에 대해 실행을 거부한다
  (실수로 baseline 덮어쓰기 방지).

## manifest.json 스키마

각 실험 폴더에 자동 생성. 사람도 읽기 좋고 leaderboard 도 이걸 기반으로 만듦.

```json
{
  "exp_id": "EXP-002_focal_loss",
  "started_at": "2026-05-23T01:23:45+00:00",
  "elapsed_sec": 152.9,
  "git_commit": "<sha>",
  "base_exp": "EXP-001_baseline",
  "changed": ["loss: BCE -> Focal(gamma=2)", "epochs: 30 -> 50"],
  "runtime": {"platform": "Modal GPU L4", "python": "3.11"},
  "summary_metrics": {
    "best_threshold": 0.73,
    "roc_auc": 0.71,
    "pr_auc": 0.018,
    "f2": 0.062,
    "precision": 0.025,
    "recall": 0.085,
    "confusion_matrix": {"tn": 11900, "fp": 186, "fn": 43, "tp": 4}
  },
  "notes": "Focal loss로 양성 가중 강화 - recall 약간 상승"
}
```

## 자주 묻는 질문

**Q. 마스터 노트북은 언제 업데이트하나?**
좋은 실험이 나와 "현재 baseline" 자리를 차지할 만하다고 판단되면, 그 실험의
`notebook.ipynb` 를 `notebooks/3.모델링.executed.ipynb` 에 복사 (덮어쓰기) 하고
git commit. 이 시점부터 새 실험은 새 마스터를 baseline 으로 삼는다.

**Q. 같은 EXP_ID 로 두 번 돌리면?**
- `new_experiment.py` 는 폴더가 이미 있으면 에러.
- `modal_run.py` 는 폴더에 `notebook.executed.ipynb` 가 이미 있으면 에러.
정말 다시 돌리고 싶으면 그 폴더의 실행 산출물(`notebook.executed.ipynb`,
`execute.log`, `run_metrics.json`, `manifest.json`)만 백업/삭제하고 다시 실행.

**Q. EXP_ID 안 정하고 `modal_run.py` 돌리면?**
임시 ID `EXP-YYYYMMDD-HHMMSS_adhoc` 가 자동 생성되지만, 실험 폴더는 미리
만들어져 있어야 한다. 임시 실험을 돌리려면 우선
`python src/new_experiment.py EXP-YYYYMMDD-HHMMSS_adhoc` 처럼 동일한 ID 로
폴더부터 만든다.

**Q. 옛 실험을 다시 돌리려면?**
그 폴더의 `notebook.ipynb` 가 그대로 보존되어 있다.
1) 그 폴더의 실행 산출물만 백업/삭제 후 같은 EXP_ID 로 재실행, 또는
2) `EXP-NNN_xxx_rerun` 같은 새 폴더에 그 notebook.ipynb 를 복사한 뒤 실행.

**Q. 한 실험이 망했는데 leaderboard 에서 빼고 싶다.**
`experiments/EXP-NNN_xxx/` 폴더를 통째로 `_archive/` 로 옮기고
`python src/update_leaderboard.py` 다시 실행하면 됩니다.

**Q. baseline 결과는 어디?**
`experiments/EXP-001_baseline/` - ROC AUC 0.683, PR AUC 0.013.
다음 실험들은 이걸 이겨야 합니다.

## _archive/ 안의 파일들

이전(정리 전) 산출물 백업.
- `modal_outputs`, `modal_outputs_dir` : 옛 modal volume get 산출물
- `_add_dump_cell.py` : 일회용 노트북 셀 삽입 스크립트
- `_run_logs/` : 노트북 직렬 실행 시 로컬 로그

다음 PR 에서 한 번에 정말 안 쓰는지 확인 후 정리할 수 있다.
