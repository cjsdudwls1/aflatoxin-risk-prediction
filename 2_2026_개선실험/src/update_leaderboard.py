"""
experiments/EXP-*/ 폴더의 manifest.json 들을 모두 모아
reports/leaderboard.csv 를 (재)생성한다.

사용:
    python src/update_leaderboard.py

특징:
- experiments/EXP-*/manifest.json 가 있는 모든 폴더를 자동 수집.
- manifest.json 없거나 깨진 폴더는 경고만 찍고 건너뜀.
- 컬럼: exp_id, started_at, git_commit, roc_auc, pr_auc, f2, f1, precision,
  recall, accuracy, best_threshold, tp, fn, fp, tn, elapsed_sec, base_exp,
  changed, notes
- 시작시각 오름차순 정렬.
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys

# Windows cp949 콘솔에서 한글 출력 깨짐/에러 회피
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SRC_DIR = pathlib.Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent
EXP_DIR = PROJECT_ROOT / "experiments"
OUT_CSV = PROJECT_ROOT / "reports" / "leaderboard.csv"

COLUMNS = [
    "exp_id", "started_at", "git_commit",
    "roc_auc", "pr_auc", "f2", "f1", "precision", "recall", "accuracy",
    "best_threshold",
    "tp", "fn", "fp", "tn",
    "elapsed_sec", "base_exp", "changed", "notes",
]


def _row_from_manifest(m: dict) -> dict:
    sm = m.get("summary_metrics") or {}
    cm = sm.get("confusion_matrix") or {}
    return {
        "exp_id": m.get("exp_id"),
        "started_at": m.get("started_at"),
        "git_commit": (m.get("git_commit") or "")[:7],
        "roc_auc": sm.get("roc_auc"),
        "pr_auc": sm.get("pr_auc"),
        "f2": sm.get("f2"),
        "f1": sm.get("f1"),
        "precision": sm.get("precision"),
        "recall": sm.get("recall"),
        "accuracy": sm.get("accuracy"),
        "best_threshold": sm.get("best_threshold"),
        "tp": cm.get("tp"),
        "fn": cm.get("fn"),
        "fp": cm.get("fp"),
        "tn": cm.get("tn"),
        "elapsed_sec": m.get("elapsed_sec"),
        "base_exp": m.get("base_exp"),
        "changed": "; ".join(m.get("changed") or []),
        "notes": m.get("notes", ""),
    }


def main() -> int:
    if not EXP_DIR.exists():
        print(f"[error] experiments 폴더 없음: {EXP_DIR}", file=sys.stderr)
        return 1

    rows = []
    skipped = []
    for d in sorted(EXP_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith("EXP-"):
            continue
        manifest = d / "manifest.json"
        if not manifest.exists():
            skipped.append((d.name, "no manifest.json"))
            continue
        try:
            m = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception as e:
            skipped.append((d.name, f"parse error: {e}"))
            continue
        rows.append(_row_from_manifest(m))

    rows.sort(key=lambda r: (r.get("started_at") or ""))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[saved] {OUT_CSV} ({len(rows)} rows)")
    for name, reason in skipped:
        print(f"[skip ] {name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
