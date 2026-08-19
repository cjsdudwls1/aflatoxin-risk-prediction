# -*- coding: utf-8 -*-
"""
토양 화학성 수집 (SoilEnviron/SoilExam V2).

두 가지 조회 방식:
  - by='bjd' : 법정동코드(STDG_CD) -> 해당 법정동 내 전체 시료 (getSoilExamList)
  - by='pnu' : 필지 PNU 코드(PNU_CD) -> 특정 필지 (getSoilExam)

원본 CSV 와의 필드 차이(실측 확정, 2026-06 호출):
  API Stdg_Cd -> 원본 BJD_Code,  API ELCD -> 원본 SELC(전기전도도).
  그 외 11개 컬럼(Any_Year, Exam_Day, Exam_Type, PNU_Nm, ACID, VLDPHA,
  VLDSIA, OM, POSIFERT_MG, POSIFERT_K, POSIFERT_CA)은 이름·의미 동일.
  VLDSIA(유효규산)는 결측(None) 가능.
"""
import pandas as pd
from .http_util import get_xml
from .config import load_config
from .mapping import map_soil_item, SOIL_MODEL_COLS
from .io_util import save_csv

SOIL_BJD_URL = 'https://apis.data.go.kr/1390802/SoilEnviron/SoilExam/V2/getSoilExamList'
SOIL_PNU_URL = 'https://apis.data.go.kr/1390802/SoilEnviron/SoilExam/V2/getSoilExam'


def _paged(url, key_param, code, cfg, page_size):
    """페이징 자동 수집. items 가 page_size 미만이면 종료."""
    key = cfg.key('agriweather')
    verify = cfg.default('verify_ssl', True)
    out, page = [], 1
    while True:
        params = {
            'serviceKey': key, 'Page_No': str(page), 'Page_Size': str(page_size),
            key_param: str(code),
        }
        _root, items = get_xml(url, params, verify=verify)
        if not items:
            break
        out.extend(items)
        if len(items) < page_size:
            break
        page += 1
    return out


def fetch_soil_by_bjd(bjd_code, cfg=None, page_size=200):
    """법정동코드(10자리 STDG_CD)로 토양 시료 raw 리스트."""
    cfg = cfg or load_config()
    return _paged(SOIL_BJD_URL, 'STDG_CD', bjd_code, cfg, page_size)


def fetch_soil_by_pnu(pnu_code, cfg=None, page_size=200):
    """필지 PNU 코드(19자리 PNU_CD)로 토양 시료 raw 리스트."""
    cfg = cfg or load_config()
    return _paged(SOIL_PNU_URL, 'PNU_CD', pnu_code, cfg, page_size)


def collect_soil(codes, by='bjd', cfg=None):
    """여러 코드 수집 -> 원본 CSV 컬럼명으로 매핑된 DataFrame."""
    cfg = cfg or load_config()
    fn = fetch_soil_by_bjd if by == 'bjd' else fetch_soil_by_pnu
    rows = []
    for c in codes:
        rows.extend(fn(c, cfg=cfg))
    if not rows:
        return pd.DataFrame(columns=SOIL_MODEL_COLS)
    return pd.DataFrame([map_soil_item(r) for r in rows])
