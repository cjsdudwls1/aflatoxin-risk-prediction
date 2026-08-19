# -*- coding: utf-8 -*-
"""
패턴/주기 탐지 분석 (교수 과제 #3 + #2 커널 크기 근거)
=====================================================
목적: 각 검사 샘플의 '직전 60일 날씨 창' 안에서 실제로 어떤 시간 규모(주기)가
      지배적인지 데이터로 직접 측정하여, Conv1D 커널 크기(5/7/9/15일) 선택의
      "왜 그 길이인지"에 대한 정량적 근거를 만든다.

방법:
  1) 자기상관함수(ACF): 한 변수가 며칠이 지나야 '자기 자신을 잊는지'(decorrelation
     length). 첫 ACF가 1/e(0.368) 아래로 떨어지는 lag = 날씨의 기억 길이.
  2) periodogram(주파수 분석/FFT): 60일 창을 주파수 성분으로 분해해 가장 강한
     주기(peak period)를 찾는다. 추세(계절 드리프트)는 선형 detrend로 제거.
  3) 판별(discriminative) 분석: 부적합 vs 적합 샘플의 평균 스펙트럼/ACF 차이.
     두 집단이 갈리는 주기 = 아플라톡신 신호가 실린 시간 규모 = 커널이 잡아야 할 길이.

출력: analysis/pattern_results.json, analysis/*.png
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "df_enhanced.parquet"
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)

# 분석 대상 날씨 변수 (사람이 읽을 라벨)
FEATURES = {
    "tmprt_150":     "Mean temperature (150cm)",
    "tmprt_150Top":  "Max temperature",
    "tmprt_150Lwet": "Min temperature",
    "hd_150":        "Humidity",
    "arvlty_300":    "Precipitation",
    "solrad_Qy":     "Solar radiation",
    "soil_Mitr_10":  "Soil moisture (10cm)",
    "일조":          "Sunshine hours",
}
N_DAYS = 60
LABEL = "JDGMNT_WORD_NAME_encoded"
MAXLAG = 30  # ACF 최대 lag


def build_matrix(df: pd.DataFrame, feat: str) -> np.ndarray:
    """(n_samples, 60) 행렬. 열 순서 day 0 -> 59 (오름차순)."""
    cols = [f"{feat}_{d:02d}" for d in range(N_DAYS)]
    return df[cols].to_numpy(dtype=np.float64)


def detrend_rows(X: np.ndarray) -> np.ndarray:
    """행마다 선형 추세(절편+기울기) 제거. 60일 안의 계절성 드리프트가
       저주파를 독점하는 것을 막아, 진짜 단주기 변동만 남긴다."""
    n, T = X.shape
    t = np.arange(T, dtype=np.float64)
    tc = t - t.mean()
    denom = (tc * tc).sum()
    slope = (X * tc).sum(axis=1) / denom          # (n,)
    intercept = X.mean(axis=1)                     # tc 평균0이므로 절편=행평균
    fit = intercept[:, None] + slope[:, None] * tc[None, :]
    return X - fit


def mean_acf(X: np.ndarray, maxlag: int = MAXLAG) -> np.ndarray:
    """행별 정규화 ACF를 구해 샘플 평균. (분산 큰 샘플이 지배하지 않도록 행별 정규화)"""
    Xd = detrend_rows(X)
    Xd = Xd - Xd.mean(axis=1, keepdims=True)
    var = (Xd * Xd).mean(axis=1)                   # lag0 autocov (=분산)
    good = var > 1e-9
    Xd = Xd[good]; var = var[good]
    n, T = Xd.shape
    acf = np.zeros(maxlag + 1)
    for k in range(maxlag + 1):
        cov = (Xd[:, : T - k] * Xd[:, k:]).mean(axis=1)  # 행별 lag-k autocov
        acf[k] = (cov / var).mean()
    return acf


def mean_periodogram(X: np.ndarray):
    """행별 detrend + Hann창 -> |rFFT|^2 평균. (freqs[cyc/day], power, periods[day])"""
    Xd = detrend_rows(X)
    n, T = Xd.shape
    w = np.hanning(T)
    Xw = Xd * w[None, :]
    F = np.fft.rfft(Xw, axis=1)
    power = (np.abs(F) ** 2).mean(axis=0)          # 평균 파워 스펙트럼
    freqs = np.fft.rfftfreq(T, d=1.0)              # cycles/day
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    return freqs, power, periods


def decorr_length(acf: np.ndarray) -> float:
    """ACF가 1/e 아래로 처음 떨어지는 lag(선형보간). '날씨의 기억 길이'(일)."""
    thr = 1.0 / np.e
    for k in range(1, len(acf)):
        if acf[k] < thr:
            a0, a1 = acf[k - 1], acf[k]
            if a0 == a1:
                return float(k)
            return float((k - 1) + (a0 - thr) / (a0 - a1))
    return float(len(acf) - 1)


def dominant_period(freqs, power, pmin=2.0, pmax=30.0) -> dict:
    """2~30일 범위에서 최대 파워 주기. (DC/추세 성분 제외)"""
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    mask = (periods >= pmin) & (periods <= pmax)
    if not mask.any():
        return {"period": None, "power": None}
    idx = np.argmax(power[mask])
    sel_p = periods[mask][idx]
    sel_pw = power[mask][idx]
    # 전체 파워 대비 비중(추세 제외 후)
    tot = power[freqs > 0].sum()
    return {"period_days": float(sel_p), "power": float(sel_pw),
            "rel_power": float(sel_pw / tot) if tot > 0 else None}


def main() -> int:
    print("[load]", DATA)
    df = pd.read_parquet(DATA)
    y = df[LABEL].to_numpy()
    pos = y == 1
    neg = y == 0
    print(f"[label] pos={pos.sum()} neg={neg.sum()} total={len(y)}")

    results = {"n_total": int(len(y)), "n_pos": int(pos.sum()),
               "n_neg": int(neg.sum()), "n_days": N_DAYS, "features": {}}

    for feat, label in FEATURES.items():
        X = build_matrix(df, feat)
        # NaN 행 제거 (해당 변수 기준)
        finite = np.isfinite(X).all(axis=1)
        Xall = X[finite]; ya = y[finite]
        if Xall.shape[0] < 50:
            print(f"[skip] {feat}: too few finite rows ({Xall.shape[0]})")
            continue
        acf_all = mean_acf(Xall)
        f, p_all, per = mean_periodogram(Xall)
        dom_all = dominant_period(f, p_all)
        dl_all = decorr_length(acf_all)

        # 라벨별
        Xp = Xall[ya[: len(Xall)] == 1] if False else X[finite & pos]
        Xn = X[finite & neg]
        Xp = Xp[np.isfinite(Xp).all(axis=1)]
        Xn = Xn[np.isfinite(Xn).all(axis=1)]
        acf_p = mean_acf(Xp) if len(Xp) > 20 else None
        acf_n = mean_acf(Xn) if len(Xn) > 20 else None
        fp, pp, _ = mean_periodogram(Xp) if len(Xp) > 20 else (f, None, per)
        fn, pn, _ = mean_periodogram(Xn) if len(Xn) > 20 else (f, None, per)

        results["features"][feat] = {
            "label": label,
            "n_used": int(Xall.shape[0]),
            "n_pos_used": int(len(Xp)),
            "n_neg_used": int(len(Xn)),
            "decorr_length_days": round(dl_all, 2),
            "dominant_period": dom_all,
            "acf": [round(float(a), 4) for a in acf_all],
            "acf_pos": [round(float(a), 4) for a in acf_p] if acf_p is not None else None,
            "acf_neg": [round(float(a), 4) for a in acf_n] if acf_n is not None else None,
            "periods_days": [round(float(x), 3) for x in per[1:]],   # skip DC
            "power_all": [round(float(x), 6) for x in p_all[1:]],
            "power_pos": [round(float(x), 6) for x in pp[1:]] if pp is not None else None,
            "power_neg": [round(float(x), 6) for x in pn[1:]] if pn is not None else None,
        }
        dp = dom_all.get("period_days")
        print(f"[feat] {feat:14s} n={Xall.shape[0]:6d} decorr={dl_all:5.2f}d "
              f"dom_period={dp if dp is None else round(dp,2)}d")

    # 도메인 일수 변수 분포(이미 7일 캡) 요약
    for c in ["독소 생성 최적 온도 일수", "저습도 일수", "연속 무강수 일수"]:
        if c in df.columns:
            vc = df[c].value_counts(dropna=False).sort_index()
            results.setdefault("domain_daycount", {})[c] = {
                str(k): int(v) for k, v in vc.items()}

    outp = OUT / "pattern_results.json"
    with io.open(outp, "w", encoding="utf-8") as fjs:
        json.dump(results, fjs, ensure_ascii=False, indent=2)
    print("[saved]", outp)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
