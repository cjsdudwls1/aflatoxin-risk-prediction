# -*- coding: utf-8 -*-
"""EXP-010 CV 결과 -> FSB(Food Science and Biotechnology) 저널 규격 그림 생성.

저널 그림 규정(Instructions for Authors):
  - 포맷 TIFF, 1단 폭 84.0 mm / 2단 폭 173 mm
  - 해상도: 컬러(RGB) 300 dpi / 그레이스케일 600 dpi / 순수 흑백 line art 1200 dpi
  - 그림에 제목(title) 넣지 않음(캡션은 본문 Figure captions 로 별도)
  - 다중 패널이면 (A),(B) 라벨

이 스크립트는 **컬러(RGB) 300 dpi, 84mm 1단** TIFF 로 출력한다.
색은 색맹 친화(Okabe-Ito) 팔레트 + 선스타일/마커 이중 구분(흑백 복사 대비).
입력: 상위 폴더(EXP-010 직속)의 cv_results.json. 출력: 이 폴더(figures/).
외부 의존성: matplotlib, pillow, numpy.

실행(PowerShell): $env:PYTHONUTF8="1"; python "make_journal_figs.py"
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
# cv_results.json 은 상위(EXP-010 직속)에 있다 — 표준 산출물은 직속 유지.
with open(os.path.join(BASE, "..", "cv_results.json"), encoding="utf-8") as f:
    cv = json.load(f)
folds = cv["fold_results"]
summary = cv["summary"]
METRICS = [("f2", "F2"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"),
           ("precision", "Precision"), ("recall", "Recall")]

# Okabe-Ito 색맹 안전 팔레트(8색)
OK = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7",
}

# Arial 우선(없으면 matplotlib 기본 sans-serif 로 폴백)
_arial = r"C:\Windows\Fonts\arial.ttf"
if os.path.exists(_arial):
    font_manager.fontManager.addfont(_arial)
    plt.rcParams["font.family"] = "Arial"

MM = 1 / 25.4
plt.rcParams.update({
    "font.size": 8, "axes.linewidth": 0.6,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.unicode_minus": False,
})


def save_tiff(fig, path):
    """300 dpi RGB TIFF 저장(FSB 컬러 규격). 알파 제거 위해 RGB 변환 후 LZW 무손실 재저장."""
    fig.savefig(path, format="tiff", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    Image.open(path).convert("RGB").save(
        path, format="TIFF", compression="tiff_lzw", dpi=(300, 300))


def save_png(fig, path):
    fig.savefig(path, format="png", dpi=200)


# ---------- (1) DOT PLOT (권장): x=지표, fold 점 + 평균±SD ----------
fig1, ax = plt.subplots(figsize=(84 * MM, 72 * MM), layout="constrained")
xpos = np.arange(len(METRICS))
jit = np.linspace(-0.06, 0.06, len(folds))  # 동일값 가로 분산(결정적)
for i, (key, _) in enumerate(METRICS):
    vals = [fr[key] for fr in folds]
    ax.scatter(xpos[i] - 0.14 + jit, vals, s=16, facecolors="none",
               edgecolors=OK["blue"], linewidths=0.9, zorder=3,
               label="Individual fold" if i == 0 else None)
    m, sd = summary[key]["mean"], summary[key]["std"]
    ax.errorbar(xpos[i] + 0.14, m, yerr=sd, fmt="D", ms=4, color=OK["vermillion"],
                ecolor=OK["vermillion"], elinewidth=1.0, capsize=2.5, capthick=1.0,
                zorder=4, label="Mean ± SD" if i == 0 else None)
    ax.text(xpos[i] + 0.14, m + sd + 0.025, f"{m:.3f}", ha="center",
            va="bottom", fontsize=6.0)
ax.set_xticks(xpos)
ax.set_xticklabels([lab for _, lab in METRICS])
ax.set_xlim(-0.5, len(METRICS) - 0.5)
ax.set_ylim(0, 0.8)
ax.set_ylabel("Score")
ax.grid(True, axis="y", color="0.85", linewidth=0.5)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper right", handletextpad=0.4)
save_tiff(fig1, os.path.join(BASE, "Fig_CV_metrics_dotplot.tiff"))
save_png(fig1, os.path.join(BASE, "_preview_dotplot.png"))

# ---------- (2) LINE CHART (대안): x=fold, 5개 지표를 색+선스타일+마커로 구분 ----------
styles = {
    "f2":        dict(color=OK["blue"],       ls="-",         marker="v", lw=2.0, label="F2"),
    "roc_auc":   dict(color=OK["orange"],     ls="--",        marker="o", lw=1.3, label="ROC-AUC"),
    "pr_auc":    dict(color=OK["green"],      ls=":",         marker="s", lw=1.3, label="PR-AUC"),
    "precision": dict(color=OK["vermillion"], ls="-.",        marker="^", lw=1.3, label="Precision"),
    "recall":    dict(color=OK["purple"],     ls=(0, (5, 1)), marker="D", lw=1.3, label="Recall"),
}
foldnum = [fr["fold"] for fr in folds]
fig2, ax2 = plt.subplots(figsize=(84 * MM, 72 * MM), layout="constrained")
for key, st in styles.items():
    vals = [fr[key] for fr in folds]
    ax2.plot(foldnum, vals, color=st["color"], linestyle=st["ls"],
             marker=st["marker"], linewidth=st["lw"], markersize=4.2,
             markerfacecolor="white", markeredgewidth=0.9, label=st["label"])
ax2.set_xlabel("Fold")
ax2.set_ylabel("Score")
ax2.set_xticks(foldnum)
ax2.set_ylim(0, 0.8)
ax2.grid(True, color="0.85", linewidth=0.5)
ax2.set_axisbelow(True)
ax2.legend(ncol=3, loc="center", bbox_to_anchor=(0.5, 0.50),
           frameon=True, edgecolor="0.8", framealpha=1.0,
           columnspacing=1.0, handletextpad=0.5)
save_tiff(fig2, os.path.join(BASE, "Fig_CV_metrics_lines.tiff"))
save_png(fig2, os.path.join(BASE, "_preview_lines.png"))

# ---------- (3) CONFUSION MATRIX (5-fold pooled; fold별 val-선택 임계치 적용) ----------
TP = sum(fr["tp"] for fr in folds)
FN = sum(fr["fn"] for fr in folds)
FP = sum(fr["fp"] for fr in folds)
TN = sum(fr["tn"] for fr in folds)
cm = np.array([[TP, FN], [FP, TN]], dtype=float)
cm_row = cm / cm.sum(axis=1, keepdims=True)  # 행(실제 클래스) 기준 정규화
cell_lab = [["TP", "FN"], ["FP", "TN"]]
fig3, ax3 = plt.subplots(figsize=(84 * MM, 70 * MM), layout="constrained")
ax3.imshow(cm_row, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for r in range(2):
    for c in range(2):
        frac = cm_row[r, c]
        ax3.text(c, r, f"{cell_lab[r][c]}\n{int(cm[r, c]):,}\n({frac*100:.1f}%)",
                 ha="center", va="center", fontsize=7.5,
                 color="white" if frac > 0.5 else "black")
ax3.set_xticks([0, 1])
ax3.set_xticklabels(["Predicted\npositive", "Predicted\nnegative"])
ax3.set_yticks([0, 1])
ax3.set_yticklabels(["Actual\npositive", "Actual\nnegative"])
ax3.tick_params(length=0)
for _s in ax3.spines.values():
    _s.set_visible(False)
save_tiff(fig3, os.path.join(BASE, "Fig_confusion_matrix.tiff"))
save_png(fig3, os.path.join(BASE, "_preview_confusion.png"))

for name in ["Fig_CV_metrics_dotplot.tiff", "Fig_CV_metrics_lines.tiff",
             "Fig_confusion_matrix.tiff"]:
    im = Image.open(os.path.join(BASE, name))
    print(f"{name}: mode={im.mode} px={im.size} dpi={im.info.get('dpi')} "
          f"size={im.size[0]/300*25.4:.1f}x{im.size[1]/300*25.4:.1f}mm")
print("DONE")
