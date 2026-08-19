# -*- coding: utf-8 -*-
"""
실행된 노트북에서 5-fold CV 결과를 추출 (커널 실험 #2 결과표용)
================================================================
배경: 커널 변형 실험(EXP-012~015)은 cv_results.json/predictions npz 가
      로컬로 동기화되지 않고 notebook.executed.ipynb 만 회수됨. 또한 모든 변형이
      볼륨의 동일 경로(/work/outputs/predictions_fold*.npz)에 써서 OOF npz 는 신뢰 불가.
      => 실행 노트북에 인쇄된 per-fold 요약표를 ground truth 로 파싱한다.

파싱 대상(cell 25 stream 출력 안의 표):
   fold n_train n_val n_test test_pos best_threshold roc_auc pr_auc f2 precision recall tp fn fp tn

산출(메모리 규칙: pooled 주력):
   - pooled F2/precision/recall : fold별 tp/fn/fp/tn 합산
   - ROC-AUC/PR-AUC            : fold 평균±표준편차 (threshold-free; OOF npz 미회수라 fold평균 사용)
                                  *EXP-011 검증: pooled-OOF ROC 0.657 ≈ fold평균 0.660 → 충실한 근사

실행: python scripts/extract_nb_cv.py
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"

ORDER = [
    ("EXP-011_cv5_same_padding", "k5+k3", "현행 최종(기준)"),
    ("EXP-012_kernel7",          "k7+k3", "급성(종관 1회)"),
    ("EXP-013_kernel9",          "k9+k3", "중간 점검"),
    ("EXP-014_kernel15",         "k15+k3", "만성/계절"),
    ("EXP-015_multibranch_7_15", "k7∥k15", "multi-filter"),
]

# per-fold 데이터 행: 정수/실수 15개 토큰
ROW_RE = re.compile(
    r"^\s*([1-5])\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$"
)


def get_cell25_text(nbp: Path) -> str:
    nb = json.loads(nbp.read_text(encoding="utf-8"))
    # build_hybrid_model + CV 가 도는 셀: stream 출력 가장 큰 코드셀
    best, best_len = "", -1
    for c in nb["cells"]:
        if c.get("cell_type") != "code":
            continue
        t = "".join("".join(o.get("text", [])) for o in c.get("outputs", [])
                    if o.get("output_type") == "stream")
        if len(t) > best_len:
            best, best_len = t, len(t)
    return best


def parse_folds(text: str):
    folds = []
    for ln in text.splitlines():
        m = ROW_RE.match(ln)
        if m:
            g = m.groups()
            folds.append({
                "fold": int(g[0]), "roc_auc": float(g[6]), "pr_auc": float(g[7]),
                "f2": float(g[8]), "precision": float(g[9]), "recall": float(g[10]),
                "tp": int(g[11]), "fn": int(g[12]), "fp": int(g[13]), "tn": int(g[14]),
            })
    # 같은 fold 가 중복되면 마지막만(혹시 재인쇄)
    seen = {}
    for f in folds:
        seen[f["fold"]] = f
    return [seen[k] for k in sorted(seen)]


def summarize(folds):
    tp = sum(f["tp"] for f in folds); fn = sum(f["fn"] for f in folds)
    fp = sum(f["fp"] for f in folds); tn = sum(f["tn"] for f in folds)
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F2 = 5 * P * R / (4 * P + R) if (4 * P + R) else 0.0
    roc = np.array([f["roc_auc"] for f in folds])
    pr = np.array([f["pr_auc"] for f in folds])
    return {
        "n_folds": len(folds), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "pooled_precision": P, "pooled_recall": R, "pooled_f2": F2,
        "roc_mean": float(roc.mean()), "roc_std": float(roc.std(ddof=1) if len(roc) > 1 else 0),
        "pr_mean": float(pr.mean()), "pr_std": float(pr.std(ddof=1) if len(pr) > 1 else 0),
    }


def main():
    rows = []
    for name, kdesc, note in ORDER:
        nbp = EXP / name / "notebook.executed.ipynb"
        if not nbp.exists():
            print(f"[skip] {name:28s} {kdesc:8s} — executed notebook 없음(실행중)")
            continue
        text = get_cell25_text(nbp)
        folds = parse_folds(text)
        if len(folds) < 5:
            print(f"[warn] {name:28s} — fold {len(folds)}/5 만 파싱됨")
            if not folds:
                continue
        s = summarize(folds)
        rows.append((name, kdesc, note, s))
        print(f"[ok] {name:28s} pooled F2={s['pooled_f2']:.3f} "
              f"P={s['pooled_precision']:.3f} R={s['pooled_recall']:.3f} "
              f"ROC={s['roc_mean']:.3f}±{s['roc_std']:.3f} PR={s['pr_mean']:.3f}±{s['pr_std']:.3f} "
              f"| TP={s['tp']} FN={s['fn']} FP={s['fp']}")

    print("\n붙여넣기용 마크다운(주력=pooled; ROC/PR=fold평균):\n")
    print("| 실험 | 커널 | F2(pooled) | ROC-AUC(평균) | PR-AUC(평균) | precision(pooled) | recall(pooled) | 비고 |")
    print("|---|---|---|---|---|---|---|---|")
    for name, kdesc, note, s in rows:
        eid = name.split("_")[0]
        print(f"| {eid} | {kdesc} | {s['pooled_f2']:.3f} | "
              f"{s['roc_mean']:.3f}±{s['roc_std']:.3f} | {s['pr_mean']:.3f}±{s['pr_std']:.3f} | "
              f"{s['pooled_precision']:.3f} | {s['pooled_recall']:.3f} | {note} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
