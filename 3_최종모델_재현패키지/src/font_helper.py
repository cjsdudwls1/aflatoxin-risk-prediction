"""한글 폰트 헬퍼 — matplotlib 한글 폰트 자동 설정.

사용 예:
    from font_helper import setup_korean_font
    setup_korean_font()

import 자체에는 부수 효과가 없다. setup_korean_font() 호출 시에만 rcParams를 변경한다.
"""

from __future__ import annotations

import platform
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib import font_manager


_FONT_CANDIDATES = {
    "Windows": ["Malgun Gothic", "맑은 고딕", "Gulim", "Batang"],
    "Darwin": ["AppleGothic", "Apple SD Gothic Neo"],
    "Linux": ["NanumGothic", "Noto Sans CJK KR", "UnDotum"],
}


def _resolve_font() -> Optional[str]:
    system = platform.system()
    candidates = _FONT_CANDIDATES.get(system, [])
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def setup_korean_font(verbose: bool = True) -> Optional[str]:
    """matplotlib에 한글 폰트와 음수 부호 설정 적용.

    Returns
    -------
    str or None
        실제로 적용된 폰트 이름. 한글 폰트를 못 찾으면 None (그래도 unicode_minus는 적용).
    """
    font_name = _resolve_font()

    plt.rcParams["axes.unicode_minus"] = False

    if font_name:
        plt.rcParams["font.family"] = font_name
        if verbose:
            print(f"[font_helper] 한글 폰트 적용: {font_name} (axes.unicode_minus=False)")
        return font_name

    if verbose:
        system = platform.system()
        candidates = _FONT_CANDIDATES.get(system, [])
        print(
            f"[font_helper] 경고: {system} 시스템에서 한글 폰트 미발견. "
            f"후보={candidates}. 그래프의 한글이 깨져 보일 수 있습니다."
        )
    return None


if __name__ == "__main__":
    applied = setup_korean_font(verbose=True)
    raise SystemExit(0 if applied else 1)
