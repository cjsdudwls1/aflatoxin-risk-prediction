"""
새 실험 폴더 + 마스터 노트북 복사본 + README 스텁을 한 번에 만든다.

사용:
    python src/new_experiment.py EXP-002_focal_loss
    python src/new_experiment.py EXP-002_focal_loss --base EXP-001_baseline

동작:
1. experiments/<EXP_ID>/ 폴더 생성 (이미 있으면 에러)
2. 마스터 notebooks/3.모델링.executed.ipynb 를 그 폴더에 notebook.ipynb 로 복사
3. README.md 스텁 작성 (변경점/가설 적을 자리)
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

# Windows cp949 콘솔에서 UnicodeEncodeError 회피
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
MASTER_NOTEBOOK = PROJECT_ROOT / "notebooks" / "3.모델링.executed.ipynb"
EXP_DIR = PROJECT_ROOT / "experiments"

EXP_ID_RE = re.compile(r"^EXP-[A-Za-z0-9_\-]+$")


README_TEMPLATE = """# {exp_id}

## 한 줄 요약
<여기에 이번 실험의 목적을 한 줄로>

## Base experiment
{base}

## 가설
<왜 이 변경이 baseline 보다 좋을 것이라 생각하는가>

## 무엇을 바꿨나
- (예: loss: BCE -> Focal(gamma=2))
- (예: epochs: 30 -> 50)

## 실행

PowerShell:
```powershell
$env:EXP_ID="{exp_id}"
$env:CHANGED="<변경점들; 으로 구분>"
$env:NOTES="<자유 메모>"
$env:BASE_EXP="{base}"
python -m modal run src/modal_run.py::run_notebook
python src/update_leaderboard.py
```

## 결과
(실행 후 `manifest.json` / `run_metrics.json` 참고. 핵심 수치 여기에도 손으로 정리.)

## 소감
(왜 좋아졌나/안 됐나, 다음에 시도할 가치)
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("exp_id", help="실험 식별자 (예: EXP-002_focal_loss)")
    parser.add_argument(
        "--base",
        default="EXP-001_baseline",
        help="비교 기준 실험 (기본: EXP-001_baseline)",
    )
    args = parser.parse_args()

    exp_id = args.exp_id.strip()
    if not EXP_ID_RE.match(exp_id):
        print(f"[error] exp_id 형식 위반: {exp_id}", file=sys.stderr)
        print(
            "        형식: EXP-NNN_short_name (영문/숫자/-/_) - "
            "한글/공백 금지",
            file=sys.stderr,
        )
        return 2

    out_dir = EXP_DIR / exp_id
    if out_dir.exists():
        print(f"[error] 이미 존재: {out_dir}", file=sys.stderr)
        return 2

    if not MASTER_NOTEBOOK.exists():
        print(f"[error] 마스터 노트북 없음: {MASTER_NOTEBOOK}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True)
    notebook_copy = out_dir / "notebook.ipynb"
    shutil.copy2(MASTER_NOTEBOOK, notebook_copy)

    readme = out_dir / "README.md"
    readme.write_text(
        README_TEMPLATE.format(exp_id=exp_id, base=args.base),
        encoding="utf-8",
    )

    print(f"[created] {out_dir}")
    print(f"  +- notebook.ipynb  (마스터 복사본 - 자유롭게 수정)")
    print(f"  +- README.md       (변경점/가설 적을 자리)")
    print()
    print("다음 단계:")
    print(f"  1. notebook.ipynb 를 열어 수정")
    print(f"  2. README.md 에 변경점/가설 적기")
    print(f"  3. 실행:")
    print(f'     $env:EXP_ID="{exp_id}"')
    print(f"     python -m modal run src/modal_run.py::run_notebook")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
