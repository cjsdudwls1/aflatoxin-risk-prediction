# -*- coding: utf-8 -*-
"""공통 입출력 유틸(단일 진실원천).

save_csv 가 그동안 5개 모듈에 똑같이 복붙돼 있었다(DRY 위반). 여기로 모아
한 곳만 고치면 모든 저장 동작이 일관되게 바뀌도록 한다.

encoding='utf-8-sig': 앞에 BOM(바이트순서표식)을 붙여 엑셀이 한글을 깨지
않고 읽게 하는 인코딩. (BOM 없는 utf-8 은 엑셀에서 한글이 깨져 보인다.)
"""


def save_csv(df, out_path):
    """DataFrame 을 CSV 로 저장하고 경로를 돌려준다(엑셀 한글 호환)."""
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    return out_path
