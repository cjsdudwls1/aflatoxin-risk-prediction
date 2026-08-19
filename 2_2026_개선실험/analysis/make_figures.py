# -*- coding: utf-8 -*-
"""패턴/주기 분석 결과 시각화 (교수 제출용 스냅샷). 라벨은 폰트 안전 위해 영어."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "analysis"
FIG = A / "figures"
FIG.mkdir(exist_ok=True)

pat = json.load(open(A / "pattern_results.json", encoding="utf-8"))
ms = json.load(open(A / "multiscale_results.json", encoding="utf-8"))

KEY = ["tmprt_150", "hd_150", "arvlty_300", "solrad_Qy", "soil_Mitr_10"]
NAME = {"tmprt_150": "Temperature", "hd_150": "Humidity",
        "arvlty_300": "Precipitation", "solrad_Qy": "Solar radiation",
        "soil_Mitr_10": "Soil moisture", "tmprt_150Top": "Max temp"}
COL = {"tmprt_150": "tab:red", "hd_150": "tab:blue", "arvlty_300": "tab:cyan",
       "solrad_Qy": "tab:orange", "soil_Mitr_10": "tab:brown",
       "tmprt_150Top": "tab:pink"}

fig, ax = plt.subplots(2, 2, figsize=(13, 9))

# (a) Mean ACF vs lag
for k in KEY:
    acf = pat["features"][k]["acf"]
    ax[0, 0].plot(range(len(acf)), acf, label=NAME[k], color=COL[k], lw=1.8)
ax[0, 0].axhline(1 / np.e, ls="--", c="gray", lw=1)
ax[0, 0].text(20, 1 / np.e + 0.02, "1/e (decorrelation)", color="gray", fontsize=8)
ax[0, 0].axhline(0, c="k", lw=0.6)
ax[0, 0].set_xlabel("Lag (days)"); ax[0, 0].set_ylabel("Mean autocorrelation")
ax[0, 0].set_title("(a) Autocorrelation: how long weather 'remembers' itself")
ax[0, 0].legend(fontsize=8); ax[0, 0].set_xlim(0, 30)

# (b) Mean power spectrum vs period (normalized)
for k in KEY:
    per = np.array(pat["features"][k]["periods_days"])
    pw = np.array(pat["features"][k]["power_all"]); pw = pw / pw.sum()
    o = np.argsort(per)
    ax[0, 1].plot(per[o], pw[o], label=NAME[k], color=COL[k], lw=1.6)
ax[0, 1].axvspan(5, 8, color="gold", alpha=0.18)
ax[0, 1].text(5.3, ax[0, 1].get_ylim()[1] * 0.0, "", fontsize=8)
ax[0, 1].set_xscale("log")
ax[0, 1].set_xlabel("Period (days, log)"); ax[0, 1].set_ylabel("Normalized power")
ax[0, 1].set_title("(b) Spectrum: weak synoptic band ~5-8 d (gold), no sharp cycle")
ax[0, 1].legend(fontsize=8)
for xt in [5, 7, 15, 30]:
    ax[0, 1].axvline(xt, ls=":", c="gray", lw=0.7)

# (c) Cohen's d vs window W
W = ms["windows"]
for k, info in ms["probes"].items():
    ax[1, 0].plot(W, info["cohens_d"], marker="o", ms=3,
                  label=f'{NAME.get(k,k)} ({info["direction"]})', color=COL.get(k, "k"))
ax[1, 0].axvline(7, ls=":", c="green"); ax[1, 0].axvline(15, ls=":", c="purple")
ax[1, 0].set_xlabel("Window length W (days)"); ax[1, 0].set_ylabel("Cohen's d (unfit vs fit)")
ax[1, 0].set_title("(c) Class separation vs window: flat for temp/solar")
ax[1, 0].legend(fontsize=7)

# (d) AUC vs window W
for k, info in ms["probes"].items():
    ax[1, 1].plot(W, info["auc"], marker="s", ms=3,
                  label=NAME.get(k, k), color=COL.get(k, "k"))
ax[1, 1].axhline(0.5, c="gray", lw=0.7)
ax[1, 1].axvline(7, ls=":", c="green"); ax[1, 1].axvline(15, ls=":", c="purple")
ax[1, 1].set_xlabel("Window length W (days)"); ax[1, 1].set_ylabel("Univariate AUC")
ax[1, 1].set_title("(d) AUC vs window: humidity U-shape -> needs short AND long")
ax[1, 1].legend(fontsize=7)

plt.tight_layout()
out = FIG / "pattern_analysis_overview.png"
plt.savefig(out, dpi=140)
print("[saved]", out)
