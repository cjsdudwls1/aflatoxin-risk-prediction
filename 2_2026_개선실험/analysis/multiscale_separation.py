# -*- coding: utf-8 -*-
"""
다중 스케일 판별 분석 (커널 크기 #2 의 핵심 근거)
================================================
질문: "Conv1D 커널을 며칠(W)로 잡아야 부적합/적합이 가장 잘 갈리는가?"

아이디어:
  - 커널 폭 W = "연속 W일 패턴을 보는 필터". 따라서 '연속 W일 동안 지속된 조건'이
    부적합을 가장 잘 가르는 W 를 데이터로 찾으면, 그 W 가 곧 권장 커널 길이다.
  - 각 후보 W 마다 '연속 W일 이동평균'을 구하고, 창 전체에서 그 최댓값/최솟값을 취해
    '가장 극단적인 지속 W일 조건' 스칼라를 만든다.
      * 기온 max  = 가장 더운 지속 W일 (Aspergillus 생장 유리)
      * 습도 min  = 가장 건조한 지속 W일 (기주식물 스트레스)
      * 강수 min  = 가장 가문 지속 W일 (가뭄 스트레스 -> 독소 생성)
      * 토양수분 min = 가장 마른 지속 W일
  - 이 스칼라가 부적합 vs 적합을 얼마나 가르는지 Cohen's d 와 단일변수 AUC 로 측정.
  - |d|(또는 AUC) 가 최대인 W = 경험적 최적 커널 길이.

출력: analysis/multiscale_results.json
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "df_enhanced.parquet"
OUT = ROOT / "analysis"
LABEL = "JDGMNT_WORD_NAME_encoded"
N_DAYS = 60

# (feature, direction) : direction='max' 면 지속 고온/고습 추적, 'min' 이면 지속 저습/가뭄
PROBES = [
    ("tmprt_150",    "max", "Sustained high mean-temp"),
    ("tmprt_150Top", "max", "Sustained high max-temp"),
    ("hd_150",       "min", "Sustained low humidity (dry spell)"),
    ("arvlty_300",   "min", "Sustained low precipitation (drought)"),
    ("solrad_Qy",    "max", "Sustained high solar radiation"),
    ("soil_Mitr_10", "min", "Sustained low soil moisture"),
]
WINDOWS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 21, 25, 30]


def build_matrix(df, feat):
    cols = [f"{feat}_{d:02d}" for d in range(N_DAYS)]
    return df[cols].to_numpy(dtype=np.float64)


def cohens_d(a, b):
    """집단 a(부적합), b(적합) 평균차 / 합동표준편차. 부호는 a-b."""
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if sp == 0:
        return 0.0
    return float((a.mean() - b.mean()) / sp)


def auc_mannwhitney(pos, neg):
    """단일변수 ROC-AUC = P(점수_pos > 점수_neg). rank 기반(빠름)."""
    x = np.concatenate([pos, neg])
    order = x.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    # 동점 평균 rank 보정
    # (간단화: 동점 영향 작으므로 평균 rank 근사)
    npos, nneg = len(pos), len(neg)
    sum_pos = ranks[:npos].sum()
    auc = (sum_pos - npos * (npos + 1) / 2.0) / (npos * nneg)
    return float(auc)


def windowed_extreme(X, W, direction):
    """연속 W일 이동평균을 구하고 창 전체에서 max 또는 min. (n,) 반환."""
    if W == 1:
        roll = X
    else:
        roll = sliding_window_view(X, W, axis=1).mean(axis=2)  # (n, 60-W+1)
    return roll.max(axis=1) if direction == "max" else roll.min(axis=1)


def main():
    print("[load]", DATA)
    df = pd.read_parquet(DATA)
    y = df[LABEL].to_numpy()
    pos_mask = y == 1
    neg_mask = y == 0
    res = {"n_pos": int(pos_mask.sum()), "n_neg": int(neg_mask.sum()),
           "windows": WINDOWS, "probes": {}}

    for feat, direction, desc in PROBES:
        X = build_matrix(df, feat)
        finite = np.isfinite(X).all(axis=1)
        curve_d, curve_auc = [], []
        for W in WINDOWS:
            s = windowed_extreme(X, W, direction)
            sp = s[finite & pos_mask]
            sn = s[finite & neg_mask]
            d = cohens_d(sp, sn)
            auc = auc_mannwhitney(sp, sn)
            # AUC 는 방향에 따라 <0.5 일 수 있으니 |AUC-0.5| 로 분리력 표시
            curve_d.append(round(d, 4))
            curve_auc.append(round(auc, 4))
        d_abs = [abs(v) for v in curve_d]
        best_i = int(np.argmax(d_abs))
        auc_sep = [abs(a - 0.5) for a in curve_auc]
        best_auc_i = int(np.argmax(auc_sep))
        res["probes"][feat] = {
            "desc": desc, "direction": direction,
            "cohens_d": curve_d, "auc": curve_auc,
            "best_W_by_d": WINDOWS[best_i],
            "best_d": curve_d[best_i],
            "best_W_by_auc": WINDOWS[best_auc_i],
            "best_auc": curve_auc[best_auc_i],
        }
        print(f"[probe] {feat:14s} {direction:3s} bestW(|d|)={WINDOWS[best_i]:2d} "
              f"d={curve_d[best_i]:+.3f}  bestW(AUC)={WINDOWS[best_auc_i]:2d} "
              f"AUC={curve_auc[best_auc_i]:.3f}")

    outp = OUT / "multiscale_results.json"
    with io.open(outp, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("[saved]", outp)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
