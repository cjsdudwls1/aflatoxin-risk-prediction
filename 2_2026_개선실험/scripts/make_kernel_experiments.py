# -*- coding: utf-8 -*-
"""
커널 크기 실험 노트북 생성기 (교수 과제 #2)
==========================================
기준(base) = EXP-011_cv5_same_padding (현재 최종 모델, same padding).
이 노트북의 시계열 분기 Conv1D 커널만 수술적으로 바꿔 4개 변형 노트북을 만든다.

변형:
  EXP-012_kernel7      : 1번째 Conv1D 커널 5 -> 7   (2번째는 3 유지)
  EXP-013_kernel9      : 1번째 Conv1D 커널 5 -> 9
  EXP-014_kernel15     : 1번째 Conv1D 커널 5 -> 15
  EXP-015_multibranch_7_15 : 7일(급성)·15일(만성) Conv1D 병렬 -> concat -> 3일 mixing conv
                              (분석 근거: 습도 분리력이 W=1 과 W>=21 양 끝에서 강한 U자형
                               => 단일 중간 커널은 최악, 짧은+긴 커널 동시 필요)

생성물: experiments/<EXP_ID>/notebook.ipynb  (미실행, 출력 제거된 clean 노트북)
실행:   $env:EXP_ID="EXP-012_kernel7"; python -m modal run src/modal_run.py::run_notebook
검증:   각 변형은 build_hybrid_model 셀을 compile() 로 SyntaxError 체크.
"""
from __future__ import annotations
import copy
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
BASE_NB = ROOT / "experiments" / "EXP-011_cv5_same_padding" / "notebook.executed.ipynb"
EXP_DIR = ROOT / "experiments"

# Conv1D 블록 원본 (EXP-011 cell25, 442~447 라인) — 정확 일치로 치환
OLD_BLOCK = (
    "    x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu',\n"
    "                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(ts_input)\n"
    "    x = layers.BatchNormalization()(x)\n"
    "    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu',\n"
    "                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)\n"
    "    x = layers.BatchNormalization()(x)\n"
)
OLD_COMMENT = ("    # [EXP-003] Conv1D 두 layer 로 local pattern 추출 (kernel 5 → 3, same padding)"
               " — EXP-011 정정: causal->same (닫힌 50일 창 many-to-one 분류라 BiLSTM 양방향성과 일관)\n")
OLD_CONFIG = '"config": "EXP-009 hybrid (Conv1D k5+k3+BN x2 -> BiLSTM; dropout 0.1; ts_aug v2 fix8); SMOTE 제거"'


def single_kernel_block(k: int) -> str:
    return (
        f"    x = layers.Conv1D(64, kernel_size={k}, padding='same', activation='relu',\n"
        "                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(ts_input)\n"
        "    x = layers.BatchNormalization()(x)\n"
        "    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu',\n"
        "                      kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)\n"
        "    x = layers.BatchNormalization()(x)\n"
    )


MULTIBRANCH_BLOCK = (
    "    # [EXP-015] 다중 커널 병렬: 7일(급성 변동)·15일(만성/계절 envelope) 필터 동시 적용 후 결합\n"
    "    reg = tf.keras.regularizers.l2(0.001)\n"
    "    b_short = layers.Conv1D(64, kernel_size=7, padding='same', activation='relu', kernel_regularizer=reg)(ts_input)\n"
    "    b_short = layers.BatchNormalization()(b_short)\n"
    "    b_long = layers.Conv1D(64, kernel_size=15, padding='same', activation='relu', kernel_regularizer=reg)(ts_input)\n"
    "    b_long = layers.BatchNormalization()(b_long)\n"
    "    x = layers.Concatenate(axis=-1)([b_short, b_long])\n"
    "    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=reg)(x)\n"
    "    x = layers.BatchNormalization()(x)\n"
)

VARIANTS = {
    "EXP-012_kernel7":  {"type": "single", "k": 7,
                         "config": "kernel grid: Conv1D k7+k3 (base EXP-011 k5+k3); same padding"},
    "EXP-013_kernel9":  {"type": "single", "k": 9,
                         "config": "kernel grid: Conv1D k9+k3 (base EXP-011 k5+k3); same padding"},
    "EXP-014_kernel15": {"type": "single", "k": 15,
                         "config": "kernel grid: Conv1D k15+k3 (base EXP-011 k5+k3); same padding"},
    "EXP-015_multibranch_7_15": {"type": "multi",
                         "config": "multi-kernel: parallel Conv1D k7 & k15 -> concat -> k3 mix; same padding"},
}


def strip_outputs(nb: dict) -> None:
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None


def find_model_cell(nb: dict) -> int:
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") == "code" and "def build_hybrid_model(" in "".join(c.get("source", [])):
            return i
    raise RuntimeError("build_hybrid_model cell not found")


def patch_one(name: str, spec: dict) -> int:
    nb = json.loads(BASE_NB.read_text(encoding="utf-8"))
    strip_outputs(nb)
    ci = find_model_cell(nb)
    S = "".join(nb["cells"][ci]["source"])

    if OLD_BLOCK not in S:
        print(f"[error] {name}: OLD_BLOCK 미발견 — base 구조 변동")
        return 2

    if spec["type"] == "single":
        k = spec["k"]
        S = S.replace(OLD_BLOCK, single_kernel_block(k), 1)
        new_cmt = (f"    # [{name}] Conv1D kernel grid: 1번째 커널 5 -> {k} (2번째 3 유지), same padding"
                   f" — 분석 근거 reports/커널크기_설계근거.md\n")
        S = S.replace(OLD_COMMENT, new_cmt, 1)
    else:
        S = S.replace(OLD_BLOCK, MULTIBRANCH_BLOCK, 1)
        new_cmt = (f"    # [{name}] 다중 커널 병렬(7일+15일) — 분석 근거 reports/커널크기_설계근거.md\n")
        S = S.replace(OLD_COMMENT, new_cmt, 1)

    # config 문자열 갱신(추적성)
    new_config = f'"config": "{spec["config"]}"'
    S = S.replace(OLD_CONFIG, new_config, 1)

    # compile 검증
    try:
        compile(S, f"<{name} cell{ci}>", "exec")
    except SyntaxError as e:
        print(f"[error] {name}: SyntaxError line {e.lineno}: {e.text}")
        return 3

    nb["cells"][ci]["source"] = S.splitlines(keepends=True)

    out_dir = EXP_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nb = out_dir / "notebook.ipynb"
    executed = out_dir / "notebook.executed.ipynb"
    if executed.exists():
        print(f"[skip] {name}: 이미 실행 결과 존재 ({executed.name}) — notebook.ipynb만 갱신")
    out_nb.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    size = out_nb.stat().st_size
    print(f"[ok] {name}: cell#{ci} patched, compile OK, saved notebook.ipynb ({size:,} B)")
    return 0


def main() -> int:
    if not BASE_NB.exists():
        print(f"[error] base notebook 없음: {BASE_NB}")
        return 1
    rc = 0
    for name, spec in VARIANTS.items():
        rc |= patch_one(name, spec)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
