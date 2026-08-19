# -*- coding: utf-8 -*-
"""EXP-010 per-fold 예측(predictions_fold{N}.npz) -> FSB 저널 규격 곡선 그림.

산출(전부 컬러 RGB, 300 dpi, 84mm 1-col TIFF(LZW), no title, English labels):
  Fig_ROC.tiff              fold별 ROC + 평균(±1SD 밴드) + chance
  Fig_PR.tiff               fold별 PR + pooled + baseline(prevalence)
  Fig_threshold_sweep.tiff  thresholds vs F2/Precision/Recall(test, fold 평균) + 선택 임계치
  Fig_calibration.tiff      calibration curve(pooled, quantile bins) + 대각선
  Fig_prob_dist.tiff        예측확률 분포 by class(로그 count) + 선택 임계치
  Fig_lift_gain.tiff        누적 gain(모델 vs random)

색은 색맹 친화(Okabe-Ito) 팔레트 + 선스타일 이중 구분(흑백 복사 대비).
입력: 같은 폴더의 predictions_fold1..5.npz (있는 fold만 사용).
검수용 PNG(_preview_*.png, 200 dpi)도 같이 저장 — 제출용 아님.

실행(PowerShell): $env:PYTHONUTF8="1"; python "make_curve_figs.py"
"""
import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

BASE = os.path.dirname(os.path.abspath(__file__))

# Okabe-Ito 색맹 안전 팔레트(8색)
OK = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7",
}

# Arial 우선(없으면 기본 sans-serif 폴백)
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


# ---------- 데이터 로드(있는 fold만) ----------
files = sorted(glob.glob(os.path.join(BASE, "predictions_fold*.npz")))
assert files, "predictions_fold*.npz 가 없습니다. 재실행으로 fold별 npz 를 먼저 생성하세요."
D = [dict(np.load(f, allow_pickle=True)) for f in files]
nf = len(D)
print(f"loaded {nf} fold(s): {[os.path.basename(f) for f in files]}")

y_all = np.concatenate([d["y_test"] for d in D])
p_all = np.concatenate([d["y_prob_test"] for d in D]).astype(float)
prevalence = float(y_all.mean())
bth = float(np.mean([float(d["best_th_f2"]) for d in D]))  # fold 평균 선택 임계치
print(f"pooled test n={len(y_all)} pos={int(y_all.sum())} prevalence={prevalence*100:.3f}% "
      f"selected_th(mean)={bth:.3f}")

# ---------- (1) ROC ----------
mean_fpr = np.linspace(0, 1, 200)
tprs, aucs = [], []
fig, ax = plt.subplots(figsize=(84 * MM, 80 * MM), layout="constrained")
for d in D:
    fpr, tpr, _ = roc_curve(d["y_test"], d["y_prob_test"])
    aucs.append(auc(fpr, tpr))
    it = np.interp(mean_fpr, fpr, tpr)
    it[0] = 0.0
    tprs.append(it)
    ax.plot(fpr, tpr, color="0.72", lw=0.7, zorder=1,
            label="Individual fold" if d is D[0] else None)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
if nf > 1:
    sd = np.std(tprs, axis=0)
    ax.fill_between(mean_fpr, np.clip(mean_tpr - sd, 0, 1), np.clip(mean_tpr + sd, 0, 1),
                    color=OK["skyblue"], alpha=0.35, zorder=0, label="± 1 SD")
ax.plot(mean_fpr, mean_tpr, color=OK["blue"], lw=1.8, zorder=3,
        label=f"Mean ROC (AUC = {np.mean(aucs):.3f} ± {np.std(aucs, ddof=1):.3f})")
ax.plot([0, 1], [0, 1], color="0.45", ls=(0, (4, 3)), lw=0.8, zorder=2, label="Chance")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.legend(loc="lower right", frameon=False, handletextpad=0.5)
save_tiff(fig, os.path.join(BASE, "Fig_ROC.tiff"))
save_png(fig, os.path.join(BASE, "_preview_ROC.png"))
plt.close(fig)

# ---------- (2) PR ----------
aps = []
fig, ax = plt.subplots(figsize=(84 * MM, 80 * MM), layout="constrained")
for d in D:
    pr, rc, _ = precision_recall_curve(d["y_test"], d["y_prob_test"])
    aps.append(average_precision_score(d["y_test"], d["y_prob_test"]))
    ax.plot(rc, pr, color="0.72", lw=0.7, zorder=1,
            label="Individual fold" if d is D[0] else None)
pr, rc, _ = precision_recall_curve(y_all, p_all)
ax.plot(rc, pr, color=OK["blue"], lw=1.5, zorder=3,
        label=f"Pooled (AP = {np.mean(aps):.3f} ± {np.std(aps, ddof=1):.3f})")
ax.axhline(prevalence, color=OK["vermillion"], ls=(0, (4, 3)), lw=1.0, zorder=2,
           label=f"Baseline ({prevalence*100:.2f}%)")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_xlim(0, 1)
ax.set_ylim(0, max(0.45, float(np.nanmax(pr)) * 1.05))
ax.legend(loc="upper right", frameon=False, handletextpad=0.5)
save_tiff(fig, os.path.join(BASE, "Fig_PR.tiff"))
save_png(fig, os.path.join(BASE, "_preview_PR.png"))
plt.close(fig)

# ---------- (3) THRESHOLD SWEEP (test, fold 평균) ----------
th = np.asarray(D[0]["thresholds"], dtype=float)
f2_m = np.mean([d["f2s_test"] for d in D], axis=0)
pr_m = np.mean([d["precisions_test"] for d in D], axis=0)
rc_m = np.mean([d["recalls_test"] for d in D], axis=0)
fig, ax = plt.subplots(figsize=(84 * MM, 66 * MM), layout="constrained")
ax.plot(th, f2_m, color=OK["blue"], ls="-", lw=2.0, label="F2")
ax.plot(th, rc_m, color=OK["orange"], ls="--", lw=1.3, label="Recall")
ax.plot(th, pr_m, color=OK["green"], ls="-.", lw=1.3, label="Precision")
ax.axvline(bth, color=OK["vermillion"], ls=":", lw=1.3, label=f"Selected th = {bth:.2f}")
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Score")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(True, color="0.88", linewidth=0.5)
ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False, handletextpad=0.5)
save_tiff(fig, os.path.join(BASE, "Fig_threshold_sweep.tiff"))
save_png(fig, os.path.join(BASE, "_preview_threshold_sweep.png"))
plt.close(fig)

# ---------- (4) CALIBRATION (pooled, quantile bins) ----------
nb = 10 if len(y_all) > 2000 else 5
frac_pos, mean_pred = calibration_curve(y_all, p_all, n_bins=nb, strategy="quantile")
fig, ax = plt.subplots(figsize=(84 * MM, 80 * MM), layout="constrained")
ax.plot([0, 1], [0, 1], color="0.45", ls=(0, (4, 3)), lw=0.8, label="Perfectly calibrated")
ax.plot(mean_pred, frac_pos, color=OK["blue"], marker="o", ms=3.6, lw=1.3,
        markerfacecolor="white", markeredgewidth=1.0, label="Model")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed positive fraction")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(loc="upper left", frameon=False, handletextpad=0.5)
save_tiff(fig, os.path.join(BASE, "Fig_calibration.tiff"))
save_png(fig, os.path.join(BASE, "_preview_calibration.png"))
plt.close(fig)

# ---------- (5) PREDICTED-PROBABILITY DISTRIBUTION by class ----------
bins = np.linspace(0, 1, 51)
fig, ax = plt.subplots(figsize=(84 * MM, 66 * MM), layout="constrained")
ax.hist(p_all[y_all == 0], bins=bins, color=OK["skyblue"], edgecolor=OK["blue"],
        linewidth=0.4, alpha=0.7, log=True, label=f"Negative (n={int((y_all==0).sum()):,})")
ax.hist(p_all[y_all == 1], bins=bins, histtype="step", color=OK["vermillion"], lw=1.6,
        log=True, label=f"Positive (n={int((y_all==1).sum()):,})")
ax.axvline(bth, color="0.30", ls=":", lw=1.1, label=f"Selected th = {bth:.2f}")
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Count (log scale)")
ax.set_xlim(0, 1)
ax.legend(loc="upper center", frameon=False, handletextpad=0.5)
save_tiff(fig, os.path.join(BASE, "Fig_prob_dist.tiff"))
save_png(fig, os.path.join(BASE, "_preview_prob_dist.png"))
plt.close(fig)

# ---------- (6) CUMULATIVE GAIN ----------
order = np.argsort(-p_all)
ys = y_all[order]
frac_pop = np.arange(1, len(ys) + 1) / len(ys)
frac_pos = np.cumsum(ys) / ys.sum()
fig, ax = plt.subplots(figsize=(84 * MM, 80 * MM), layout="constrained")
ax.plot(frac_pop, frac_pos, color=OK["blue"], lw=1.6, label="Model")
ax.plot([0, 1], [0, 1], color="0.45", ls=(0, (4, 3)), lw=0.8, label="Random")
ax.set_xlabel("Fraction of samples (ranked by score)")
ax.set_ylabel("Fraction of positives captured")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(True, color="0.88", linewidth=0.5)
ax.set_axisbelow(True)
ax.legend(loc="lower right", frameon=False, handletextpad=0.5)
save_tiff(fig, os.path.join(BASE, "Fig_lift_gain.tiff"))
save_png(fig, os.path.join(BASE, "_preview_lift_gain.png"))
plt.close(fig)

# ---------- 검증 ----------
print(f"\n[check] fold AUC = {[round(a,3) for a in aucs]}  mean={np.mean(aucs):.3f}±{np.std(aucs, ddof=1):.3f}")
print(f"[check] fold AP  = {[round(a,3) for a in aps]}  mean={np.mean(aps):.3f}±{np.std(aps, ddof=1):.3f}")
# 누적 이득(cumulative gain) 분위수별 양성 포착률 + lift — 본문/캡션 수치 재현용
# frac_pop(상위 비율), frac_pos(포착된 양성 비율)는 위 (6)번 블록에서 점수 내림차순으로 계산됨.
print("[check] cumulative gain (positives captured by top-q% of scored samples):")
n_total = len(frac_pop)
for q in (0.05, 0.10, 0.20, 0.30, 0.50):
    idx = int(np.ceil(q * n_total)) - 1          # 상위 q% 마지막 표본의 0-based 인덱스
    capture = frac_pos[idx]                       # 그 지점까지 누적 포착한 양성 비율
    lift = capture / q                            # 무작위 선별(=q) 대비 농축 배수
    print(f"         top {q*100:4.0f}% -> {capture*100:5.1f}% positives captured  (lift {lift:.2f}x)")
for name in ["Fig_ROC.tiff", "Fig_PR.tiff", "Fig_threshold_sweep.tiff",
             "Fig_calibration.tiff", "Fig_prob_dist.tiff", "Fig_lift_gain.tiff"]:
    p = os.path.join(BASE, name)
    if os.path.exists(p):
        im = Image.open(p)
        print(f"{name}: mode={im.mode} px={im.size} dpi={im.info.get('dpi')} "
              f"size={im.size[0]/300*25.4:.1f}x{im.size[1]/300*25.4:.1f}mm")
print("DONE")
