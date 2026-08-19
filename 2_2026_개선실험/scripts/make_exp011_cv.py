"""
EXP-011_cv5_same_padding : EXP-010_cv5_hybrid 와 모델/데이터/CV 평가 프로토콜이 완전히 동일하고,
하이브리드 모델 시계열 분기의 Conv1D 두 층 padding 만 'causal' -> 'same' 로 바꾼 변형.

[방법론 근거]
본 과제는 검사일 '직전 50일'이라는 닫힌(이미 전부 관측된) 과거 창을 통째로 읽어
단일 라벨(부적합 여부)을 예측하는 many-to-one 시퀀스 분류다.
- causal padding 의 목적은 "t 시점 출력이 t 시점까지의 입력만 보게" 강제하는 것 = 스텝별
  forecasting/streaming 모델에서만 의미가 있다.
- 그러나 바로 뒤의 Bidirectional LSTM 이 각 시점 표현을 앞뒤(미래 시점 포함) 모두에
  의존시키므로, causal padding 이 지키려던 '시점별 인과성'은 즉시 무효화된다.
  (예측 시점 = 검사일 이후 데이터는 입력 창에 없으므로 데이터 누수는 아님 — 결과는 유효.)
=> causal 은 BiLSTM 과 의도가 상충하는 dead constraint. 닫힌 창 분류에 일관된 'same' 으로 정정한다.

원본(EXP-003 -> EXP-009 -> EXP-010)의 patch_exp003 는 역사적 기록이므로 건드리지 않고,
EXP-010 의 최종 실행본 notebook.ipynb 를 복제해 padding 만 surgical 하게 치환한다.

사용:  python scripts/make_exp011_cv.py
검증:  치환은 정확히 2회(두 Conv1D), 그 셀 전체를 compile() 로 SyntaxError 체크.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
SRC_NB = ROOT / "experiments" / "EXP-010_cv5_hybrid" / "notebook.ipynb"
DST_DIR = ROOT / "experiments" / "EXP-011_cv5_same_padding"
DST_NB = DST_DIR / "notebook.ipynb"

OLD = "padding='causal'"
NEW = "padding='same'"
# 코드와 일치하도록 주석의 'causal padding' 표현도 동기화(추적성, code-reviewer LOW 정리).
CMT_OLD = "(kernel 5 → 3, causal padding)"
CMT_NEW = "(kernel 5 → 3, same padding) — EXP-011 정정: causal->same (닫힌 50일 창 many-to-one 분류라 BiLSTM 양방향성과 일관)"


def main() -> int:
    if not SRC_NB.exists():
        print(f"[error] base notebook 없음: {SRC_NB}")
        return 1
    nb = json.loads(SRC_NB.read_text(encoding="utf-8"))

    # 1) build_hybrid_model 정의 셀 찾기 (= cell 25 mega-cell: AttentionLayer/preprocess/main 포함)
    ci = -1
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        if "def build_hybrid_model(" in "".join(cell.get("source", [])):
            ci = i
            break
    if ci < 0:
        print("[error] build_hybrid_model cell not found")
        return 2

    # 2) 그 셀의 source list element 에서 causal -> same (코드 정확히 2회: 두 Conv1D) + 주석 동기화(1회)
    src = nb["cells"][ci]["source"]
    n = m = 0
    for j, line in enumerate(src):
        new_line = line
        if OLD in new_line:
            n += new_line.count(OLD)
            new_line = new_line.replace(OLD, NEW)
        if CMT_OLD in new_line:
            m += new_line.count(CMT_OLD)
            new_line = new_line.replace(CMT_OLD, CMT_NEW)
        src[j] = new_line
    print(f"[swap] cell#{ci} (build_hybrid_model): code \"{OLD}\"->\"{NEW}\" x{n}; comment sync x{m}")
    if n != 2:
        print(f"[error] 코드 padding 정확히 2회를 기대했으나 {n}회 — 중단(원본 구조 변동 가능성)")
        return 3
    if m != 1:
        print(f"[error] 주석 동기화 정확히 1회를 기대했으나 {m}회 — 중단(주석 형태 변동 가능성)")
        return 5

    # 3) 파일 전체(소스+출력)에 causal 잔존 없는지 확인
    if OLD in json.dumps(nb, ensure_ascii=False):
        print(f"[warn] \"{OLD}\" 가 다른 위치(출력 등)에 남아있음 — 확인 권장")

    # 4) 편집한 셀 전체 compile 검증 (AttentionLayer + build_hybrid_model + preprocess + main CV)
    joined = "".join(src)
    try:
        compile(joined, "<EXP-011 cell25>", "exec")
        print(f"[verify] cell#{ci} compile OK ({len(joined)} chars)")
    except SyntaxError as e:
        print(f"[error] cell#{ci} SyntaxError: {e}  (line {e.lineno}: {e.text})")
        return 4

    # 5) 저장 (EXP-010 과 동일한 json 포맷: ensure_ascii=False, indent=1 -> diff 최소화)
    DST_DIR.mkdir(parents=True, exist_ok=True)
    DST_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[saved] {DST_NB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
