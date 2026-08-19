# -*- coding: utf-8 -*-
"""
CNN 제거(ablation) 실험 노트북 생성기 — EXP-018_no_cnn
=====================================================
목적: 논문 채택 모델(EXP-012, Conv1D 2층 + BiLSTM 2층 + Attention)에서
      **시계열 분기의 Conv1D 두 층만 제거**하고 나머지(BiLSTM+Attention, 정형 분기,
      가중 BCE, 시계열 증강 v2, 5-fold CV, per-fold threshold)는 100% 동일하게 두어
      CNN의 순수 기여를 분리한다.

기준(base) = EXP-011_cv5_same_padding/notebook.executed.ipynb
  (커널·깊이 절제실험과 동일한 base. Conv1D 블록만 수술적으로 치환.)

수술 내용:
  build_hybrid_model 셀의 Conv1D 두 층(+BatchNorm) 블록을 한 줄 passthrough 로 교체.
      (제거 전)  x = Conv1D(64,k5)(ts_input); BN; Conv1D(64,k3)(x); BN
      (제거 후)  x = ts_input        # BiLSTM 이 원시 50x10 시계열을 직접 입력
  => 그다음 `x = Bidirectional(LSTM(64,...))(x)` 가 원시 시계열을 직접 받는다.

검증: 검증된 make_kernel_experiments.py 의 상수/헬퍼를 그대로 재사용(바이트 동일성).
      치환 후 셀을 compile() 로 SyntaxError 체크.

생성물: experiments/EXP-018_no_cnn/notebook.ipynb (미실행, clean)
실행:   $env:EXP_ID="EXP-018_no_cnn"; python -m modal run src/modal_run.py::run_notebook
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# 검증된 커널 생성기에서 base 경로/상수/헬퍼를 그대로 차용 (동일성 보장)
from make_kernel_experiments import (  # noqa: E402
    BASE_NB, EXP_DIR, OLD_BLOCK, OLD_COMMENT, OLD_CONFIG,
    strip_outputs, find_model_cell,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NAME = "EXP-018_no_cnn"

# Conv1D 두 층(+BN) 제거 -> passthrough 한 줄
NEW_BLOCK = "    x = ts_input  # [EXP-018_no_cnn] Conv1D 제거: BiLSTM이 원시 50x10 시계열을 직접 입력\n"

NEW_COMMENT = (
    "    # [EXP-018_no_cnn] CNN 제거 ablation: Conv1D 두 층 제거 -> 시계열 분기 = BiLSTM+Attention only.\n"
    "    #   base EXP-011, Conv1D 외 전 구성(BiLSTM 64/32, attention, 정형 Dense 64/32, 가중 BCE,\n"
    "    #   ts_aug v2, 5-fold CV, per-fold threshold, dropout 0.1, L2) 동일. CNN 순수 기여 분리용.\n"
)

NEW_CONFIG = ('"config": "EXP-018_no_cnn ablation: Conv1D 전층 제거 (BiLSTM+Attention only); '
              'base EXP-011; dropout 0.1; ts_aug v2 fix8; SMOTE 제거"')


def main() -> int:
    if not BASE_NB.exists():
        print(f"[error] base notebook 없음: {BASE_NB}")
        return 1

    nb = json.loads(BASE_NB.read_text(encoding="utf-8"))
    strip_outputs(nb)
    ci = find_model_cell(nb)
    S = "".join(nb["cells"][ci]["source"])

    if OLD_BLOCK not in S:
        print(f"[error] {NAME}: OLD_BLOCK 미발견 — base 구조 변동. 중단.")
        return 2
    assert S.count(OLD_BLOCK) == 1, f"OLD_BLOCK count != 1: {S.count(OLD_BLOCK)}"

    # 핵심 치환: Conv1D 블록 제거
    S = S.replace(OLD_BLOCK, NEW_BLOCK, 1)

    # 주석/설정 갱신(추적성). 없으면 no-op + 경고.
    cmt_ok = OLD_COMMENT in S
    S = S.replace(OLD_COMMENT, NEW_COMMENT, 1) if cmt_ok else S
    cfg_ok = OLD_CONFIG in S
    S = S.replace(OLD_CONFIG, NEW_CONFIG, 1) if cfg_ok else S

    # 멱등성/안전성 사후검증
    assert "x = ts_input" in S, "passthrough 미적용"
    assert OLD_BLOCK not in S, "OLD_BLOCK 잔존"
    assert "layers.Conv1D" not in S, "Conv1D 잔존 — 제거 실패"
    assert "ts_input)(" not in S  # 잘못된 호출 형태 방지

    # compile 검증
    try:
        compile(S, f"<{NAME} cell{ci}>", "exec")
    except SyntaxError as e:
        print(f"[error] {NAME}: SyntaxError line {e.lineno}: {e.text}")
        return 3

    nb["cells"][ci]["source"] = S.splitlines(keepends=True)

    out_dir = EXP_DIR / NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    executed = out_dir / "notebook.executed.ipynb"
    if executed.exists():
        print(f"[error] {NAME}: 이미 실행 결과 존재 ({executed.name}) — 다른 EXP_ID 사용 권장. 중단.")
        return 4
    out_nb = out_dir / "notebook.ipynb"
    out_nb.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    size = out_nb.stat().st_size

    print(f"[ok] {NAME}: cell#{ci} patched (Conv1D 제거), compile OK")
    print(f"     comment_updated={cmt_ok}, config_updated={cfg_ok}")
    print(f"     saved: {out_nb} ({size:,} B)")
    print(f"     Conv1D in cell: {'layers.Conv1D' in S}  (False 여야 정상)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
