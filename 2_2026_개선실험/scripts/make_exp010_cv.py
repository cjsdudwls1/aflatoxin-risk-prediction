"""
EXP-010_cv5_hybrid : EXP-009 설정(Conv1D+BiLSTM, dropout 0.1, ts_aug v2 fix8, SMOTE 제거)을
그대로 두고 데이터 분할만 5-fold Stratified CV 로 바꾼다.

EXP-009 의 patch_exp009 = patch_exp003 + patch_exp008 + patch_exp004 인데,
patch_exp004 는 main 의 단일-split 호출부(anchor: "val_size=0.15, test_size=0.15,")에
augment 를 심으므로, main 을 CV 루프로 교체하면 그 anchor 가 사라진다.
=> 본 스크립트는 patch_exp003 + patch_exp008(모델 구조)만 재사용하고,
   - train_and_evaluate 의 pos_weight raw override hook (patch_exp004 Step1) 은 직접 삽입
   - preprocess 를 fold 인덱스 기반으로 일반화 (scaler/Target_Mean 은 각 fold train 만으로 fit -> 누수 차단)
   - main 을 StratifiedKFold(k=5) 루프로 교체 (augment 는 각 fold train 에만 적용)
   를 cell 25 통째 문자열 치환으로 수행한다.

사용:  python scripts/make_exp010_cv.py
검증:  patch 후 cell 25 를 compile() 로 SyntaxError 체크.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from apply_exp_patches import patch_exp003, patch_exp008, find_cell_containing  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
NB_PATH = ROOT / "experiments" / "EXP-010_cv5_hybrid" / "notebook.ipynb"


# ---------------------------------------------------------------------------
#  (b) preprocess 시그니처 : fold 인덱스 인자 추가
# ---------------------------------------------------------------------------
OLD_SIG = (
    "def preprocess(X_ts, X_lm, y, n_features=10, timesteps=50, random_state=42,\n"
    "               val_size=0.15, test_size=0.15):"
)
NEW_SIG = (
    "def preprocess(X_ts, X_lm, y, n_features=10, timesteps=50, random_state=42,\n"
    "               val_size=0.15, test_size=0.15,\n"
    "               train_idx=None, val_idx=None, test_idx=None):"
)

# ---------------------------------------------------------------------------
#  (c) split 블록 : 단일 split 또는 fold mode 분기
# ---------------------------------------------------------------------------
OLD_SPLIT = (
    "    # ----- (2) split 2단계 -----\n"
    "    # 1차: temp / test (test_size 분리)\n"
    "    X_lm_temp_df, X_lm_test_df, X_ts_temp, X_ts_test, y_temp, y_test = train_test_split(\n"
    "        X_lm, X_ts_arr, y, test_size=test_size, stratify=y, random_state=random_state\n"
    "    )\n"
    "    # 2차: train / val (val_size 를 temp 기준 상대비율로 환산)\n"
    "    val_relative = val_size / (1.0 - test_size)\n"
    "    X_lm_train_df, X_lm_val_df, X_ts_train, X_ts_val, y_train, y_val = train_test_split(\n"
    "        X_lm_temp_df, X_ts_temp, y_temp,\n"
    "        test_size=val_relative, stratify=y_temp, random_state=random_state\n"
    "    )"
)
NEW_SPLIT = (
    "    # ----- (2) split: 단일 split 또는 [EXP-010 CV] fold mode -----\n"
    "    if train_idx is not None and val_idx is not None and test_idx is not None:\n"
    "        # [EXP-010 CV] 외부 StratifiedKFold 인덱스로 직접 분할 (positional .iloc)\n"
    "        X_lm_train_df = X_lm.iloc[train_idx].reset_index(drop=True).copy()\n"
    "        X_lm_val_df   = X_lm.iloc[val_idx].reset_index(drop=True).copy()\n"
    "        X_lm_test_df  = X_lm.iloc[test_idx].reset_index(drop=True).copy()\n"
    "        X_ts_train = X_ts_arr[train_idx]\n"
    "        X_ts_val   = X_ts_arr[val_idx]\n"
    "        X_ts_test  = X_ts_arr[test_idx]\n"
    "        if hasattr(y, 'iloc'):\n"
    "            y_train = y.iloc[train_idx].reset_index(drop=True)\n"
    "            y_val   = y.iloc[val_idx].reset_index(drop=True)\n"
    "            y_test  = y.iloc[test_idx].reset_index(drop=True)\n"
    "        else:\n"
    "            _ya = np.asarray(y)\n"
    "            y_train = pd.Series(_ya[train_idx]); y_val = pd.Series(_ya[val_idx]); y_test = pd.Series(_ya[test_idx])\n"
    "    else:\n"
    "        # 1차: temp / test (test_size 분리)\n"
    "        X_lm_temp_df, X_lm_test_df, X_ts_temp, X_ts_test, y_temp, y_test = train_test_split(\n"
    "            X_lm, X_ts_arr, y, test_size=test_size, stratify=y, random_state=random_state\n"
    "        )\n"
    "        # 2차: train / val (val_size 를 temp 기준 상대비율로 환산)\n"
    "        val_relative = val_size / (1.0 - test_size)\n"
    "        X_lm_train_df, X_lm_val_df, X_ts_train, X_ts_val, y_train, y_val = train_test_split(\n"
    "            X_lm_temp_df, X_ts_temp, y_temp,\n"
    "            test_size=val_relative, stratify=y_temp, random_state=random_state\n"
    "        )"
)

# ---------------------------------------------------------------------------
#  (a) train_and_evaluate : pos_weight raw override hook (patch_exp004 Step1)
# ---------------------------------------------------------------------------
OLD_POSW = "    pos_weight = calculate_pos_weight(y_train)\n"
NEW_POSW = (
    "    pos_weight = calculate_pos_weight(y_train)\n"
    "    # [EXP-010 CV / EXP-004 v2 fix#3] augment 로 인한 pos_weight silent dilution 차단 — raw 강제\n"
    "    if 'EXP004_RAW_POS_WEIGHT' in globals():\n"
    "        pos_weight = float(globals()['EXP004_RAW_POS_WEIGHT'])\n"
    "        print(f'[EXP-010 CV] pos_weight raw 강제 fix: {pos_weight:.2f}')\n"
)

# ---------------------------------------------------------------------------
#  (d) main : StratifiedKFold(k=5) 루프로 교체
#       - augment 함수(_ts_augment_positives_v2)는 루프 안에서 정의 + 각 fold train 에만 적용
#       - 각 fold: train/val/test 3분할, val->test threshold isolation 유지
# ---------------------------------------------------------------------------
MAIN_RE = re.compile(
    r'# ={2,}\n#  Main Execution\n# ={2,}\nif __name__ == "__main__":.*\Z',
    re.DOTALL,
)
NEW_MAIN = '''# ===============================
#  Main Execution (EXP-010: 5-fold Stratified CV, EXP-009 config)
# ===============================
if __name__ == "__main__":
    import json as _json, os as _os, gc as _gc
    from sklearn.model_selection import StratifiedKFold, train_test_split as _tts
    from sklearn.metrics import (
        roc_auc_score as _ra, average_precision_score as _ap,
        fbeta_score as _fb, precision_score as _ps, recall_score as _rs,
        confusion_matrix as _cm,
    )

    # [EXP-004 v2 fix#1,#2,#4,#5,#6,#7,#8] 양성 시계열 augment
    #   scale-first(+-5%) -> noise(sigma=0.02), no-wrap edge-pad time-shift, X_tab num-cols jitter
    def _ts_augment_positives_v2(Xts, Xtab, y, n_copies=2, seed=42, n_tab_num=4):
        rng = np.random.RandomState(seed)
        y = np.asarray(y)
        pos = (y == 1)
        Xts_pos = Xts[pos]; Xtab_pos = Xtab[pos]; y_pos = y[pos]
        n_tab_total = Xtab_pos.shape[1]; n_onehot = n_tab_total - n_tab_num
        aug_ts, aug_tab, aug_y = [], [], []
        for _k in range(n_copies):
            scale = rng.uniform(0.95, 1.05, (Xts_pos.shape[0], 1, Xts_pos.shape[2])).astype(Xts_pos.dtype)
            x_scaled = Xts_pos * scale
            noise = rng.normal(0, 0.02, Xts_pos.shape).astype(Xts_pos.dtype)
            x_aug = x_scaled + noise
            shift = rng.randint(-2, 3, size=Xts_pos.shape[0])
            for r, s in enumerate(shift):
                if s > 0:
                    pad = np.tile(x_aug[r, 0:1, :], (s, 1))
                    x_aug[r] = np.concatenate([pad, x_aug[r, :-s, :]], axis=0)
                elif s < 0:
                    sa = -s
                    pad = np.tile(x_aug[r, -1:, :], (sa, 1))
                    x_aug[r] = np.concatenate([x_aug[r, sa:, :], pad], axis=0)
            aug_ts.append(x_aug)
            xtab_aug = Xtab_pos.copy()
            tab_noise = rng.normal(0, 0.02, (Xtab_pos.shape[0], n_tab_num)).astype(Xtab_pos.dtype)
            xtab_aug[:, n_onehot:] = xtab_aug[:, n_onehot:] + tab_noise
            aug_tab.append(xtab_aug); aug_y.append(y_pos)
        Xts_new = np.concatenate([Xts] + aug_ts, axis=0)
        Xtab_new = np.concatenate([Xtab] + aug_tab, axis=0)
        y_new = np.concatenate([y] + aug_y, axis=0)
        idx = rng.permutation(len(y_new))
        return Xts_new[idx], Xtab_new[idx], y_new[idx]

    # ----- 인덱스 정합 (positional .iloc 위해 reset) -----
    X_lm_cv = X_lm.reset_index(drop=True) if hasattr(X_lm, "reset_index") else pd.DataFrame(X_lm).reset_index(drop=True)
    y_cv = y.reset_index(drop=True) if hasattr(y, "reset_index") else pd.Series(np.asarray(y))
    full_idx = np.arange(len(y_cv))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    for fold_i, (trainval_idx, test_idx) in enumerate(skf.split(full_idx, y_cv)):
        print("\\n" + "=" * 70)
        print(f"[FOLD {fold_i + 1}/5]  trainval={len(trainval_idx)}, test={len(test_idx)}")
        print("=" * 70)
        set_seeds(42 + fold_i)
        globals()["CV_FOLD_IDX"] = fold_i + 1  # [fix] npz fold별 저장 (predictions_fold{N}.npz)

        # trainval -> train/val (val ~= 전체의 15% => trainval(80%) 기준 test_size=0.1875)
        tr_idx, vl_idx = _tts(
            trainval_idx, test_size=0.1875,
            stratify=y_cv.iloc[trainval_idx], random_state=42 + fold_i,
        )

        (X_ts_tr, X_tab_tr, y_tr,
         X_ts_vl, X_tab_vl, y_vl,
         X_ts_te, X_tab_te, y_te) = preprocess(
            X_ts, X_lm_cv, y_cv, n_features=10, timesteps=50,
            random_state=42 + fold_i,
            train_idx=tr_idx, val_idx=vl_idx, test_idx=test_idx,
        )

        # [EXP-004 v2 fix#3] augment 전 raw pos_weight 저장 (train_and_evaluate hook 이 사용)
        _raw_neg = int((np.asarray(y_tr) == 0).sum()); _raw_pos = int((np.asarray(y_tr) == 1).sum())
        globals()["EXP004_RAW_POS_WEIGHT"] = float(_raw_neg) / float(_raw_pos) if _raw_pos > 0 else 1.0
        print(f"[EXP-010 CV] fold raw pos_weight: {globals()['EXP004_RAW_POS_WEIGHT']:.2f} (neg={_raw_neg}, pos={_raw_pos})")

        # augment train only (val/test 는 절대 augment 안 함)
        X_ts_tr, X_tab_tr, y_tr = _ts_augment_positives_v2(X_ts_tr, X_tab_tr, y_tr, n_copies=2, seed=42, n_tab_num=4)
        print(f"[EXP-010 CV] augmented train: shape={np.asarray(y_tr).shape}, pos={int(np.asarray(y_tr).sum())}")

        model, y_prob_test, _bdf, best_th = train_and_evaluate(
            X_ts_tr, X_tab_tr, y_tr,
            X_ts_vl, X_tab_vl, y_vl,
            X_ts_te, X_tab_te, y_te,
            epochs=50, batch_size=256, loss_type="weighted_bce",
        )

        y_te_arr = np.asarray(y_te)
        y_pred_te = (np.asarray(y_prob_test) >= float(best_th)).astype(int)
        tn, fp, fn, tp = _cm(y_te_arr, y_pred_te).ravel()
        fr = {
            "fold": fold_i + 1,
            "n_train": int(len(y_tr)), "n_val": int(len(y_vl)), "n_test": int(len(y_te_arr)),
            "test_pos": int(y_te_arr.sum()),
            "best_threshold": float(best_th),
            "roc_auc": float(_ra(y_te_arr, y_prob_test)),
            "pr_auc": float(_ap(y_te_arr, y_prob_test)),
            "f2": float(_fb(y_te_arr, y_pred_te, beta=2, zero_division=0)),
            "precision": float(_ps(y_te_arr, y_pred_te, zero_division=0)),
            "recall": float(_rs(y_te_arr, y_pred_te, zero_division=0)),
            "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
        }
        fold_results.append(fr)
        print(f"[FOLD {fold_i + 1}] F2={fr['f2']:.4f} ROC-AUC={fr['roc_auc']:.4f} PR-AUC={fr['pr_auc']:.4f} "
              f"P={fr['precision']:.4f} R={fr['recall']:.4f} th={fr['best_threshold']:.2f} "
              f"(tp={tp},fn={fn},fp={fp},tn={tn})")
        del model; _gc.collect()

    # ===== 집계 =====
    cv_df = pd.DataFrame(fold_results)
    metric_cols = ["best_threshold", "roc_auc", "pr_auc", "f2", "precision", "recall"]
    summary = cv_df[metric_cols].agg(["mean", "std", "median", "min", "max"])
    print("\\n" + "=" * 70)
    print("[EXP-010] 5-fold CV summary (EXP-009 config: Conv1D+BiLSTM, dropout 0.1, ts_aug v2)")
    print("=" * 70)
    print(cv_df.to_string(index=False))
    print("\\n" + summary.to_string())

    _save = "/work/outputs" if _os.path.isdir("/work/outputs") else "."
    cv_df.to_csv(_os.path.join(_save, "cv_results.csv"), index=False)
    summary.to_csv(_os.path.join(_save, "cv_summary.csv"))
    with open(_os.path.join(_save, "cv_results.json"), "w", encoding="utf-8") as _f:
        _json.dump(
            {
                "fold_results": fold_results,
                "summary": {m: {s: float(summary.loc[s, m]) for s in summary.index} for m in metric_cols},
                "config": "EXP-009 hybrid (Conv1D k5+k3+BN x2 -> BiLSTM; dropout 0.1; ts_aug v2 fix8); SMOTE 제거",
                "cv": "StratifiedKFold k=5, per-fold train/val/test (val->test threshold isolation)",
            },
            _f, ensure_ascii=False, indent=2,
        )

    # fold 별 metric plot
    plt.figure(figsize=(9, 5))
    for _m in ["f2", "roc_auc", "pr_auc", "precision", "recall"]:
        plt.plot(cv_df["fold"], cv_df[_m], marker="o", label=_m)
    plt.xlabel("Fold"); plt.ylabel("Metric")
    plt.title("EXP-010 5-fold CV metrics (EXP-009 config)")
    plt.xticks(cv_df["fold"]); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(_os.path.join(_save, "cv_metrics.png"), dpi=120, bbox_inches="tight")
    plt.show()
    print(f"\\n[saved] cv_results.csv / cv_summary.csv / cv_results.json / cv_metrics.png -> {_save}")
'''

# ---------------------------------------------------------------------------
#  (e) [fix] predictions.npz 덮어쓰기 버그: CV 루프가 fold마다 train_and_evaluate 를
#       호출하는데 고정명 저장이라 마지막 fold 만 생존. fold 번호로 파일명 분기.
#       (이 블록은 train_and_evaluate 본문 = base 노트북 코드라 main CV 교체와 별개로 패치)
# ---------------------------------------------------------------------------
OLD_NPZ_BLOCK = (
    "    _save_dir = '/work/outputs' if _os.path.isdir('/work/outputs') else '.'\n"
    "    _np.savez(\n"
    "        _os.path.join(_save_dir, 'predictions.npz'),\n"
)
NEW_NPZ_BLOCK = (
    "    _save_dir = '/work/outputs' if _os.path.isdir('/work/outputs') else '.'\n"
    "    # [fix] CV 루프가 fold마다 호출 -> 고정명이면 덮어써짐. fold 번호로 파일명 분기.\n"
    "    _fold_idx = globals().get('CV_FOLD_IDX', None)\n"
    "    _npz_name = f'predictions_fold{_fold_idx}.npz' if _fold_idx is not None else 'predictions.npz'\n"
    "    _np.savez(\n"
    "        _os.path.join(_save_dir, _npz_name),\n"
)
OLD_NPZ_PRINT = '    print(f"[npz] saved predictions (val+test) to {_save_dir}/predictions.npz")'
NEW_NPZ_PRINT = '    print(f"[npz] saved predictions (val+test) to {_save_dir}/{_npz_name}")'


def main() -> int:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    log = []

    # 1) 모델 구조: EXP-003 (Conv1D x2 + BN x2) + EXP-008 (dropout 0.1)  — list 기반 기존 patch 재사용
    log += patch_exp003(nb)
    log += patch_exp008(nb)

    # 2) cell 25 통째 문자열 조작 (preprocess fold화 + pos_weight hook + main CV)
    ci = find_cell_containing(nb, "def preprocess(X_ts, X_lm, y, n_features=10")
    if ci < 0:
        print("[error] preprocess cell not found")
        return 1
    src = "".join(nb["cells"][ci]["source"])

    # (a) pos_weight raw override hook
    assert OLD_POSW in src, "pos_weight call-site anchor not found"
    src = src.replace(OLD_POSW, NEW_POSW, 1)
    log.append("[EXP-010] train_and_evaluate pos_weight raw override hook 삽입")

    # (b) preprocess 시그니처 + fold 인자
    assert OLD_SIG in src, "preprocess signature anchor not found"
    src = src.replace(OLD_SIG, NEW_SIG, 1)
    log.append("[EXP-010] preprocess 시그니처에 train_idx/val_idx/test_idx 추가")

    # (c) split 블록 -> fold mode 분기
    assert OLD_SPLIT in src, "preprocess split block anchor not found"
    src = src.replace(OLD_SPLIT, NEW_SPLIT, 1)
    log.append("[EXP-010] preprocess split 블록 -> fold mode 분기 (scaler/Target_Mean train-only fit 유지)")

    # (d) main -> 5-fold CV 루프
    assert MAIN_RE.search(src), "main block anchor not found"
    src = MAIN_RE.sub(lambda _m: NEW_MAIN, src)
    log.append("[EXP-010] main -> StratifiedKFold(k=5) CV 루프 (augment train-only, val->test isolation)")

    # (e) predictions.npz 덮어쓰기 -> fold별 파일명 (train_and_evaluate 본문, main 밖)
    assert OLD_NPZ_BLOCK in src, "npz savez block anchor not found"
    src = src.replace(OLD_NPZ_BLOCK, NEW_NPZ_BLOCK, 1)
    assert OLD_NPZ_PRINT in src, "npz print anchor not found"
    src = src.replace(OLD_NPZ_PRINT, NEW_NPZ_PRINT, 1)
    log.append("[EXP-010] predictions.npz -> fold별 predictions_fold{N}.npz (덮어쓰기 버그 fix)")

    nb["cells"][ci]["source"] = src.splitlines(keepends=True)

    # 3) 검증: cell 25 문법 컴파일
    joined = "".join(nb["cells"][ci]["source"])
    try:
        compile(joined, "<EXP-010 cell25>", "exec")
        log.append(f"[verify] cell#{ci} compile OK ({len(joined)} chars)")
    except SyntaxError as e:
        print(f"[error] cell#{ci} SyntaxError: {e}")
        print(f"  line {e.lineno}: {e.text}")
        return 2

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    for line in log:
        print(line)
    print(f"[saved] {NB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
