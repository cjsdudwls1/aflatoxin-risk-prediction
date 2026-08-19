# -*- coding: utf-8 -*-
"""
커널 실험 결과 수집기 (교수 과제 #2 결과표 자동 생성)
======================================================
EXP-011(기준)과 EXP-012~015(커널 변형)의 성능을 **동일한 pooled(OOF) 방식**으로 산출해
reports/커널크기_설계근거.md 7절 표에 붙일 마크다운 행을 출력한다.

지표 규약(메모리 규칙: pooled 주력):
  - ROC-AUC, PR-AUC : 5개 fold 의 test 예측(figures/predictions_fold*.npz)을 전부 이어붙여
                      한 번에 계산(threshold-free, OOF pooled).
  - F2/precision/recall : cv_results.json 의 fold별 혼동행렬(tp/fn/fp/tn)을 합산해 계산
                          (각 fold 는 자기 val 에서 정한 임계값을 자기 test 에 적용 — 누수 없음).
  - 폴드평균±표준편차(summary)도 함께 출력(변동성 보조 지표).

미완료 실험은 자동 skip. 실행: python scripts/collect_kernel_results.py
"""
from __future__ import annotations
import json, glob, os, re, sys
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"

ORDER = [
    ("EXP-011_cv5_same_padding", "k5+k3", "현행 최종"),
    ("EXP-012_kernel7",          "k7+k3", "급성(종관 1회)"),
    ("EXP-013_kernel9",          "k9+k3", "중간 점검"),
    ("EXP-014_kernel15",         "k15+k3", "만성/계절"),
    ("EXP-015_multibranch_7_15", "k7∥k15", "multi-filter"),
]


def pooled_threshold_free(exp_dir: Path):
    """fold별 test 예측을 이어붙여 pooled ROC-AUC, PR-AUC 계산."""
    files = glob.glob(str(exp_dir / "figures" / "predictions_fold*.npz"))
    if not files:
        return None
    files = sorted(files, key=lambda p: int(re.search(r"fold(\d)", os.path.basename(p)).group(1)))
    ys, ps = [], []
    for f in files:
        z = np.load(f)
        ys.append(z["y_test"]); ps.append(z["y_prob_test"])
    y = np.concatenate(ys); p = np.concatenate(ps)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        roc = float(roc_auc_score(y, p))
        pr = float(average_precision_score(y, p))
    except Exception:
        roc = pr = float("nan")
    return {"n": int(len(y)), "pos": int(y.sum()), "roc": roc, "pr": pr}


def pooled_confusion(cvjson: dict):
    """fold별 혼동행렬 합산 -> pooled precision/recall/F2."""
    fr = cvjson["fold_results"]
    tp = sum(f["tp"] for f in fr); fn = sum(f["fn"] for f in fr)
    fp = sum(f["fp"] for f in fr); tn = sum(f["tn"] for f in fr)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) else 0.0
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": prec, "recall": rec, "f2": f2}


def fold_mean(cvjson: dict):
    s = cvjson.get("summary", {})
    def g(k):
        v = s.get(k, {})
        return (v.get("mean"), v.get("std")) if isinstance(v, dict) else (None, None)
    return {k: g(k) for k in ["f2", "roc_auc", "pr_auc", "precision", "recall"]}


def main():
    rows = []
    print("=" * 110)
    for name, kdesc, note in ORDER:
        d = EXP / name
        cvp = d / "cv_results.json"
        if not cvp.exists():
            print(f"[skip] {name:28s} {kdesc:8s} — cv_results.json 없음(실행중/미완료)")
            continue
        cv = json.loads(cvp.read_text(encoding="utf-8"))
        conf = pooled_confusion(cv)
        tf = pooled_threshold_free(d)
        fm = fold_mean(cv)
        roc = tf["roc"] if tf else fm["roc_auc"][0]
        pr = tf["pr"] if tf else fm["pr_auc"][0]
        # 마크다운 표 행 (pooled 주력)
        row = (f"| {name.split('_')[0]} | {kdesc} | {conf['f2']:.3f} | {roc:.3f} | "
               f"{conf['precision']:.3f} | {conf['recall']:.3f} | {pr:.3f} | {note} |")
        rows.append(row)
        fm_f2 = fm["f2"]; fm_roc = fm["roc_auc"]
        print(f"[ok] {name:28s} pooled F2={conf['f2']:.3f} ROC={roc:.3f} "
              f"P={conf['precision']:.3f} R={conf['recall']:.3f} PR={pr:.3f} "
              f"| TP={conf['tp']} FN={conf['fn']} FP={conf['fp']} "
              f"| fold평균 F2={fm_f2[0]:.3f}±{fm_f2[1]:.3f} ROC={fm_roc[0]:.3f}±{fm_roc[1]:.3f}")

    print("=" * 110)
    print("\n붙여넣기용 마크다운 표(주력=pooled):\n")
    print("| 실험 | 커널 | F2(pooled) | ROC-AUC | precision | recall | PR-AUC | 비고 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
