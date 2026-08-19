"""
EXP-002 ~ EXP-008 의 notebook.ipynb 를 각각 실험 의도에 맞게 patch.

사용:
    python scripts/apply_exp_patches.py EXP-002_focal_loss
    python scripts/apply_exp_patches.py --all

각 patch 는 notebook JSON 의 cell source(list[str]) 를 직접 수정.
실패하면 [error] 메시지와 함께 종료.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
EXP_DIR = ROOT / "experiments"


def load_nb(exp_id: str) -> tuple[Path, dict]:
    p = EXP_DIR / exp_id / "notebook.ipynb"
    nb = json.loads(p.read_text(encoding="utf-8"))
    return p, nb


def save_nb(p: Path, nb: dict) -> None:
    p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def replace_in_source(src: list[str], find_str: str, repl_str: str) -> bool:
    """src 의 어느 line 에서 find_str 의 첫 출현을 substring 치환. 줄바꿈/indent 보존."""
    for i, line in enumerate(src):
        if find_str in line:
            src[i] = line.replace(find_str, repl_str, 1)
            return True
    return False


def replace_exact_line(src: list[str], target_line: str, repl_line: str) -> bool:
    """src 에서 정확히 target_line 과 동일한 element 를 찾아 repl_line 으로 교체."""
    for i, line in enumerate(src):
        if line == target_line:
            src[i] = repl_line
            return True
    return False


def insert_after_in_source(src: list[str], anchor_str: str, insert_block: str) -> bool:
    """anchor_str 가 들어간 line 바로 다음 위치에 insert_block(여러줄) 삽입."""
    for i, line in enumerate(src):
        if anchor_str in line:
            new_lines = insert_block.split("\n")
            for j, nl in enumerate(new_lines):
                if j < len(new_lines) - 1:
                    new_lines[j] = nl + "\n"
            src[i + 1:i + 1] = new_lines
            return True
    return False


def find_cell_containing(nb: dict, marker: str) -> int:
    """source 에 marker 가 포함된 첫 code cell 의 index 반환. 없으면 -1."""
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        joined = "".join(cell.get("source", []))
        if marker in joined:
            return i
    return -1


# ============================================================
# EXP-002_focal_loss : loss_type='weighted_bce' → 'focal'
# ============================================================
def patch_exp002(nb: dict) -> list[str]:
    log = []
    ci = find_cell_containing(nb, "model, y_prob_test, baseline_df, best_th_f2 = train_and_evaluate(")
    if ci < 0:
        raise RuntimeError("call site cell not found")
    src = nb["cells"][ci]["source"]
    ok = replace_exact_line(src, "        loss_type='weighted_bce'\n", "        loss_type='focal'\n")
    if not ok:
        raise RuntimeError("exact call-site line not found")
    log.append(f"[EXP-002] cell#{ci} call site loss_type 'weighted_bce' -> 'focal'")
    return log


# ============================================================
# EXP-003_conv1d_bilstm : ts branch 앞에 Conv1D 두 layer 추가
# ============================================================
def patch_exp003(nb: dict) -> list[str]:
    log = []
    ci = find_cell_containing(nb, "def build_hybrid_model(")
    if ci < 0:
        raise RuntimeError("build_hybrid_model cell not found")
    src = nb["cells"][ci]["source"]
    anchor = "    ts_input = layers.Input(shape=(n_timestamps, n_features_ts), name='ts_input')\n"
    block = (
        "    # [EXP-003] Conv1D 두 layer 로 local pattern 추출 (kernel 5 → 3, causal padding)\n"
        "    x = layers.Conv1D(64, kernel_size=5, padding='causal', activation='relu',\n"
        "                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(ts_input)\n"
        "    x = layers.BatchNormalization()(x)\n"
        "    x = layers.Conv1D(64, kernel_size=3, padding='causal', activation='relu',\n"
        "                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)\n"
        "    x = layers.BatchNormalization()(x)\n"
    )
    if not insert_after_in_source(src, anchor.strip(), block):
        raise RuntimeError("ts_input anchor not found")
    # 기존 첫 BiLSTM 의 input 을 ts_input → x 로 redirect (continuation line 의 triple `)))`)
    for i, line in enumerate(src):
        if "kernel_regularizer=tf.keras.regularizers.l2(0.001)))(ts_input)" in line:
            src[i] = line.replace("(ts_input)", "(x)")
            log.append(f"[EXP-003] first BiLSTM input ts_input -> x (cell#{ci} line {i})")
            break
    log.append(f"[EXP-003] cell#{ci} 에 Conv1D x2 + BN x2 삽입")
    return log


# ============================================================
# EXP-004_ts_augmentation v2 (audit 결과 8개 결함 fix)
#   v1 결함 (각 항목 fix 됨):
#     [#1] X_tab 다양성 0 → cell-level prior inflation (FIX: num cols 에 small jitter)
#     [#2] np.roll wrap-around → 검사 직전 신호가 begin 으로 이동 (FIX: edge-padding)
#     [#3] pos_weight silent 4x dilution (254→50) (FIX: train_and_evaluate hook 으로 raw 강제)
#     [#4] 시계열-tabular conditional joint distribution 파괴 (FIX: X_tab num cols 도 jitter)
#     [#5] physical bound 위반 risk (FIX: noise/scale 강도 ↓, n_copies ↓)
#     [#6] scale 곱셈 순서 모호 (FIX: scale 먼저 → noise 추가, effective budget 일정)
#     [#7] over-emphasis (n_copies=4) (FIX: n_copies=2)
#     [#8] 진단 신호 부족 (FIX: unique row + num stats + fingerprint print)
# ============================================================
def patch_exp004(nb: dict) -> list[str]:
    log = []

    # ---- Step 1: train_and_evaluate 정의 셀에 pos_weight override hook 삽입 (fix #3) ----
    ci_te = find_cell_containing(nb, "def train_and_evaluate(X_ts_train, X_tab_train, y_train")
    if ci_te < 0:
        raise RuntimeError("train_and_evaluate cell not found")
    src_te = nb["cells"][ci_te]["source"]
    anchor_te = "pos_weight = calculate_pos_weight(y_train)"
    override_block = (
        "    # [EXP-004 v2 fix #3] augment 로 인한 pos_weight silent dilution 차단 — raw 비율 강제 fix\n"
        "    if 'EXP004_RAW_POS_WEIGHT' in globals():\n"
        "        pos_weight = float(globals()['EXP004_RAW_POS_WEIGHT'])\n"
        "        print(f'[EXP-004 v2] pos_weight raw 강제 fix: {pos_weight:.2f}')\n"
    )
    if not insert_after_in_source(src_te, anchor_te, override_block):
        raise RuntimeError("pos_weight anchor in train_and_evaluate not found")
    log.append(f"[EXP-004 v2] cell#{ci_te} train_and_evaluate 에 pos_weight override hook 삽입 (fix #3)")

    # ---- Step 2: preprocess 호출 셀에 augment v2 + raw pos_weight 저장 삽입 ----
    ci = find_cell_containing(nb, ") = preprocess(")
    if ci < 0:
        raise RuntimeError("preprocess call site not found")
    src = nb["cells"][ci]["source"]
    anchor = "val_size=0.15, test_size=0.15,"
    block = (
        "    )\n"
        "\n"
        "    # [EXP-004 v2 fix #3] augment 전 raw pos_weight 저장 (train_and_evaluate hook 이 사용)\n"
        "    _raw_neg = int((y_train == 0).sum())\n"
        "    _raw_pos = int((y_train == 1).sum())\n"
        "    EXP004_RAW_POS_WEIGHT = float(_raw_neg) / float(_raw_pos) if _raw_pos > 0 else 1.0\n"
        "    print(f'[EXP-004 v2] augment 전 raw pos_weight: {EXP004_RAW_POS_WEIGHT:.2f} (neg={_raw_neg}, pos={_raw_pos})')\n"
        "\n"
        "    # [EXP-004 v2] TS augment (scale-first → noise, no-wrap edge padding) + X_tab num-cols jitter\n"
        "    def _ts_augment_positives_v2(Xts, Xtab, y, n_copies=2, seed=42, n_tab_num=4):\n"
        "        rng = np.random.RandomState(seed)\n"
        "        pos = (y == 1)\n"
        "        Xts_pos = Xts[pos]\n"
        "        Xtab_pos = Xtab[pos]\n"
        "        y_pos = y[pos]\n"
        "        n_tab_total = Xtab_pos.shape[1]\n"
        "        n_onehot = n_tab_total - n_tab_num\n"
        "        aug_ts, aug_tab, aug_y = [], [], []\n"
        "        for k in range(n_copies):\n"
        "            # fix #5,#6: scale 먼저 (±5%) → noise (σ=0.02), 강도 ↓ 으로 physical bound 위반 ↓\n"
        "            scale = rng.uniform(0.95, 1.05, (Xts_pos.shape[0], 1, Xts_pos.shape[2])).astype(Xts_pos.dtype)\n"
        "            x_scaled = Xts_pos * scale\n"
        "            noise = rng.normal(0, 0.02, Xts_pos.shape).astype(Xts_pos.dtype)\n"
        "            x_aug = x_scaled + noise\n"
        "            # fix #2: time-shift edge padding (no np.roll wrap-around)\n"
        "            shift = rng.randint(-2, 3, size=Xts_pos.shape[0])\n"
        "            for r, s in enumerate(shift):\n"
        "                if s > 0:\n"
        "                    pad = np.tile(x_aug[r, 0:1, :], (s, 1))\n"
        "                    x_aug[r] = np.concatenate([pad, x_aug[r, :-s, :]], axis=0)\n"
        "                elif s < 0:\n"
        "                    sa = -s\n"
        "                    pad = np.tile(x_aug[r, -1:, :], (sa, 1))\n"
        "                    x_aug[r] = np.concatenate([x_aug[r, sa:, :], pad], axis=0)\n"
        "            aug_ts.append(x_aug)\n"
        "            # fix #1,#4: X_tab num cols (last 4: Target_Mean + 3 일수카운트) 만 small jitter, one-hot 그대로\n"
        "            xtab_aug = Xtab_pos.copy()\n"
        "            tab_noise = rng.normal(0, 0.02, (Xtab_pos.shape[0], n_tab_num)).astype(Xtab_pos.dtype)\n"
        "            xtab_aug[:, n_onehot:] = xtab_aug[:, n_onehot:] + tab_noise\n"
        "            aug_tab.append(xtab_aug)\n"
        "            aug_y.append(y_pos)\n"
        "        Xts_new = np.concatenate([Xts] + aug_ts, axis=0)\n"
        "        Xtab_new = np.concatenate([Xtab] + aug_tab, axis=0)\n"
        "        y_new = np.concatenate([y] + aug_y, axis=0)\n"
        "        idx = rng.permutation(len(y_new))\n"
        "        return Xts_new[idx], Xtab_new[idx], y_new[idx]\n"
        "\n"
        "    # fix #7: n_copies 4 → 2 (over-emphasis 완화)\n"
        "    X_ts_train, X_tab_train, y_train = _ts_augment_positives_v2(\n"
        "        X_ts_train, X_tab_train, y_train, n_copies=2, seed=42, n_tab_num=4)\n"
        "\n"
        "    # fix #8: 진단 — cell-level prior inflation + augment fingerprint\n"
        "    _pos_idx = (y_train == 1)\n"
        "    _xtab_pos = X_tab_train[_pos_idx]\n"
        "    _onehot_pos = _xtab_pos[:, :_xtab_pos.shape[1]-4]\n"
        "    _num_pos = _xtab_pos[:, _xtab_pos.shape[1]-4:]\n"
        "    _unique_onehot = len(np.unique(_onehot_pos, axis=0))\n"
        "    _unique_full = len(np.unique(_xtab_pos, axis=0))\n"
        "    print(f'[EXP-004 v2] augmented train shape: {y_train.shape}, pos={int(y_train.sum())}')\n"
        "    print(f'[EXP-004 v2] X_tab pos one-hot unique rows: {_unique_onehot} / {int(_pos_idx.sum())} (낮을수록 cell-level prior 위험)')\n"
        "    print(f'[EXP-004 v2] X_tab pos FULL unique rows (after num jitter): {_unique_full} / {int(_pos_idx.sum())} (=pos면 jitter OK)')\n"
        "    print(f'[EXP-004 v2] X_tab pos num cols mean: {_num_pos.mean(axis=0).round(4).tolist()}')\n"
        "    print(f'[EXP-004 v2] X_tab pos num cols std:  {_num_pos.std(axis=0).round(4).tolist()}')\n"
        "    import hashlib as _hl\n"
        "    _fp_ts = _hl.sha256(X_ts_train.tobytes()).hexdigest()[:16]\n"
        "    _fp_tab = _hl.sha256(X_tab_train.tobytes()).hexdigest()[:16]\n"
        "    print(f'[EXP-004 v2] X_ts_train fingerprint (sha256[:16]): {_fp_ts}')\n"
        "    print(f'[EXP-004 v2] X_tab_train fingerprint (sha256[:16]): {_fp_tab}')\n"
    )
    for i, line in enumerate(src):
        if anchor in line:
            for j in range(i, min(i + 5, len(src))):
                if src[j].strip().startswith(")"):
                    new_lines = block.split("\n")
                    new_lines = [nl + ("\n" if k < len(new_lines) - 1 else "") for k, nl in enumerate(new_lines)]
                    src[j:j + 1] = new_lines
                    log.append(f"[EXP-004 v2] cell#{ci} preprocess 호출 종료 ) 위치에 augmentation v2 block 삽입 (fix #1,#2,#4,#5,#6,#7,#8)")
                    return log
    raise RuntimeError("preprocess closing ) not found")


# ============================================================
# EXP-005_multi_seed_ensemble : train_and_evaluate 5회 + 예측 평균
# ============================================================
def patch_exp005(nb: dict) -> list[str]:
    log = []
    ci = find_cell_containing(nb, "model, y_prob_test, baseline_df, best_th_f2 = train_and_evaluate(")
    if ci < 0:
        raise RuntimeError("call site cell not found")
    src = nb["cells"][ci]["source"]
    old_call = "    model, y_prob_test, baseline_df, best_th_f2 = train_and_evaluate(\n"
    start = -1
    for i, line in enumerate(src):
        if "model, y_prob_test, baseline_df, best_th_f2 = train_and_evaluate(" in line:
            start = i
            break
    if start < 0:
        raise RuntimeError("call assignment not found")
    end = start
    for k in range(start, min(start + 12, len(src))):
        if src[k].rstrip("\n").endswith(")"):
            end = k
            break
    new_block = (
        "    # [EXP-005] 5-seed ensemble: train_and_evaluate 를 5회 호출, 예측 평균\n"
        "    import tensorflow as _tf_ens\n"
        "    from sklearn.metrics import fbeta_score as _fbeta\n"
        "    _seeds = [42, 43, 44, 45, 46]\n"
        "    _test_probs, _val_probs = [], []\n"
        "    _last_baseline = None\n"
        "    for _sd in _seeds:\n"
        "        _tf_ens.keras.utils.set_random_seed(_sd)\n"
        "        np.random.seed(_sd)\n"
        "        print(f'\\n[ensemble seed={_sd}]')\n"
        "        _mdl, _ypt, _bdf, _bth = train_and_evaluate(\n"
        "            X_ts_train, X_tab_train, y_train,\n"
        "            X_ts_val,  X_tab_val,  y_val,\n"
        "            X_ts_test, X_tab_test, y_test,\n"
        "            epochs=50, batch_size=256,\n"
        "            loss_type='weighted_bce'\n"
        "        )\n"
        "        _test_probs.append(np.asarray(_ypt).flatten())\n"
        "        _ypv = _mdl.predict({'ts_input': X_ts_val, 'tab_input': X_tab_val}, verbose=0)\n"
        "        _val_probs.append(np.asarray(_ypv).flatten())\n"
        "        _last_baseline = _bdf\n"
        "    y_prob_test = np.mean(_test_probs, axis=0)\n"
        "    _ens_val = np.mean(_val_probs, axis=0)\n"
        "    _ths = np.arange(0.01, 1.00, 0.01)\n"
        "    _f2s = [_fbeta(y_val, (_ens_val >= _th).astype(int), beta=2) for _th in _ths]\n"
        "    best_th_f2 = float(_ths[int(np.argmax(_f2s))])\n"
        "    print(f'[EXP-005] ensemble best_th_f2 from val = {best_th_f2:.3f}')\n"
        "    model = None\n"
        "    baseline_df = _last_baseline\n"
    )
    new_lines = new_block.split("\n")
    new_lines = [nl + ("\n" if k < len(new_lines) - 1 else "") for k, nl in enumerate(new_lines)]
    src[start:end + 1] = new_lines
    log.append(f"[EXP-005] cell#{ci} train_and_evaluate 호출 lines {start}-{end} → 5-seed ensemble 로 교체")
    return log


# ============================================================
# EXP-006_enhanced_tabular : tabular 에 rolling stats 7개 컬럼 추가
# ============================================================
def patch_exp006(nb: dict) -> list[str]:
    log = []
    ci = find_cell_containing(nb, "X_lm = df[['INSPCT_PURPS_NAME'")
    if ci < 0:
        raise RuntimeError("X_lm assignment cell not found")
    src = nb["cells"][ci]["source"]
    anchor = "X_lm = df[['INSPCT_PURPS_NAME'"
    block = (
        "# [EXP-006] tabular feature 추가: 시계열 prefix 별 rolling mean/std/trend 통계\n"
        "_ts_block = X_ts.values.reshape(len(X_ts), -1, 50) if hasattr(X_ts, 'values') else X_ts.reshape(len(X_ts), -1, 50)\n"
        "# 각 prefix(10개) 의 마지막 14일 mean, std, trend(diff mean) → 30개 feature\n"
        "_last14 = _ts_block[:, :, -14:]\n"
        "_mean14 = _last14.mean(axis=2)\n"
        "_std14 = _last14.std(axis=2)\n"
        "_trend14 = np.diff(_last14, axis=2).mean(axis=2)\n"
        "_extra = np.concatenate([_mean14, _std14, _trend14], axis=1)\n"
        "_extra_cols = (\n"
        "    [f'_ts_mean14_{i}' for i in range(_mean14.shape[1])] +\n"
        "    [f'_ts_std14_{i}' for i in range(_std14.shape[1])] +\n"
        "    [f'_ts_trend14_{i}' for i in range(_trend14.shape[1])]\n"
        ")\n"
        "import pandas as _pd_ext\n"
        "_extra_df = _pd_ext.DataFrame(_extra, columns=_extra_cols, index=df.index)\n"
    )
    for i, line in enumerate(src):
        if anchor in line:
            new_lines = block.split("\n")
            new_lines = [nl + ("\n" if k < len(new_lines) - 1 else "") for k, nl in enumerate(new_lines)]
            src[i:i] = new_lines
            log.append(f"[EXP-006] cell#{ci} X_lm 정의 앞에 rolling stats(30 feature) 생성 블록 삽입")
            break
    else:
        raise RuntimeError("X_lm anchor not found")
    for i, line in enumerate(src):
        if "X_lm = df[[" in line and "INSPCT_PURPS_NAME" in line:
            start = i
            while not src[start].rstrip("\n").endswith("]]"):
                start += 1
                if start >= len(src):
                    raise RuntimeError("X_lm close bracket not found")
            src[start] = src[start].rstrip("\n").rstrip("]")
            src[start] = src[start].rstrip("]") + "]]\n"
            src[start] = src[start].replace("]]", "]]")
            insert = "X_lm = _pd_ext.concat([X_lm, _extra_df], axis=1)\n"
            src.insert(start + 1, insert)
            log.append(f"[EXP-006] X_lm 에 _extra_df concat (cell#{ci} line {start + 1})")
            break
    return log


# ============================================================
# EXP-007_transformer_encoder : BiLSTM → Transformer encoder (4head, 2 layer)
# ============================================================
def patch_exp007(nb: dict) -> list[str]:
    log = []
    ci = find_cell_containing(nb, "def build_hybrid_model(")
    if ci < 0:
        raise RuntimeError("build_hybrid_model cell not found")
    src = nb["cells"][ci]["source"]
    start = -1
    end = -1
    for i, line in enumerate(src):
        if "ts_input = layers.Input(shape=(n_timestamps, n_features_ts), name='ts_input')" in line:
            start = i
        if start >= 0 and "att_out = AttentionLayer()(x)" in line:
            end = i
            break
    if start < 0 or end < 0:
        raise RuntimeError("TS branch boundaries not found")
    new_block = (
        "    ts_input = layers.Input(shape=(n_timestamps, n_features_ts), name='ts_input')\n"
        "    # [EXP-007] Transformer encoder (4-head MultiHeadAttention x 2 block + FFN)\n"
        "    x = layers.Dense(64, activation=None)(ts_input)  # project to d_model=64\n"
        "    # positional encoding (sinusoidal 간이 구현)\n"
        "    _pos = tf.cast(tf.range(tf.shape(x)[1]), tf.float32)[None, :, None]\n"
        "    _dim = tf.cast(tf.range(64), tf.float32)[None, None, :]\n"
        "    _angle = _pos / tf.pow(10000.0, (2 * (_dim // 2)) / 64.0)\n"
        "    _pe_sin = tf.sin(_angle[..., 0::2])\n"
        "    _pe_cos = tf.cos(_angle[..., 1::2])\n"
        "    _pe = tf.concat([_pe_sin, _pe_cos], axis=-1)\n"
        "    x = x + _pe\n"
        "    for _blk in range(2):\n"
        "        _att = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=dropout)(x, x)\n"
        "        x = layers.LayerNormalization()(x + _att)\n"
        "        _ff = layers.Dense(128, activation='relu')(x)\n"
        "        _ff = layers.Dense(64)(_ff)\n"
        "        x = layers.LayerNormalization()(x + _ff)\n"
        "    att_out = layers.GlobalAveragePooling1D()(x)\n"
    )
    new_lines = new_block.split("\n")
    new_lines = [nl + ("\n" if k < len(new_lines) - 1 else "") for k, nl in enumerate(new_lines)]
    src[start:end + 1] = new_lines
    log.append(f"[EXP-007] cell#{ci} TS branch lines {start}-{end} → Transformer encoder 로 교체")
    return log


# ============================================================
# EXP-008_dropout_low : dropout 0.3 → 0.1
# ============================================================
def patch_exp008(nb: dict) -> list[str]:
    log = []
    ci = find_cell_containing(nb, "def build_hybrid_model(")
    if ci < 0:
        raise RuntimeError("build_hybrid_model cell not found")
    src = nb["cells"][ci]["source"]
    ok = replace_in_source(src, "dropout=0.3, learning_rate=1e-4,", "dropout=0.1, learning_rate=1e-4,")
    if not ok:
        raise RuntimeError("dropout=0.3 not found in signature")
    log.append(f"[EXP-008] cell#{ci} build_hybrid_model dropout 0.3 -> 0.1")
    return log


# ============================================================
# EXP-009_hybrid_conv1d_dropout_tsaug : EXP-003 + EXP-008 + EXP-004 v2 합성
#   적용 순서:
#     1) patch_exp003 : build_hybrid_model 의 TS branch 에 Conv1D x2 + BN x2 삽입,
#                       첫 BiLSTM input ts_input → x redirect
#     2) patch_exp008 : build_hybrid_model 시그니처의 dropout 0.3 → 0.1
#     3) patch_exp004 : preprocess 호출 종료 ) 위치에 v2 augment block 삽입 +
#                       train_and_evaluate 에 raw pos_weight override hook 삽입
#   003 과 008 은 동일 cell 의 서로 다른 위치 (body vs signature) 만 건드림 → 충돌 없음.
#   004 는 별도 cell (preprocess / train_and_evaluate) → 003·008 과 격리.
# ============================================================
def patch_exp009(nb: dict) -> list[str]:
    log = []
    log.extend(patch_exp003(nb))
    log.extend(patch_exp008(nb))
    log.extend(patch_exp004(nb))
    log.append("[EXP-009] hybrid: EXP-003 (Conv1D+BiLSTM) + EXP-008 (dropout 0.1) + EXP-004 v2 (ts_aug fix8) 모두 적용 완료")
    return log


PATCHES = {
    "EXP-002_focal_loss": patch_exp002,
    "EXP-003_conv1d_bilstm": patch_exp003,
    "EXP-004_ts_augmentation": patch_exp004,
    "EXP-005_multi_seed_ensemble": patch_exp005,
    "EXP-006_enhanced_tabular": patch_exp006,
    "EXP-007_transformer_encoder": patch_exp007,
    "EXP-008_dropout_low": patch_exp008,
    "EXP-009_hybrid_conv1d_dropout_tsaug": patch_exp009,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_id", nargs="?", help="실험 ID (생략 시 --all 와 결합)")
    ap.add_argument("--all", action="store_true", help="모든 EXP-002~008 처리")
    args = ap.parse_args()

    if args.all:
        targets = list(PATCHES.keys())
    elif args.exp_id:
        targets = [args.exp_id]
    else:
        ap.error("exp_id 또는 --all 필요")

    failures = []
    for exp_id in targets:
        if exp_id not in PATCHES:
            print(f"[skip ] {exp_id}: 등록된 patch 없음")
            failures.append(exp_id)
            continue
        try:
            p, nb = load_nb(exp_id)
            log = PATCHES[exp_id](nb)
            save_nb(p, nb)
            for line in log:
                print(line)
            print(f"[saved] {p}")
        except Exception as e:
            print(f"[error] {exp_id}: {e}")
            failures.append(exp_id)

    if failures:
        print(f"\n[summary] 실패 {len(failures)}/{len(targets)}: {failures}")
        return 1
    print(f"\n[summary] 성공 {len(targets)}/{len(targets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
