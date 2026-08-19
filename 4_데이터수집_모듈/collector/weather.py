# -*- coding: utf-8 -*-
"""
일별 농업기상 수집 (GnrlWeather/getWeatherMonDayList3).

배경: 원본 79컬럼 CSV 를 만든 상세 엔드포인트(InsttWeather/연별)는 폐기(404).
      생존 엔드포인트는 이 월(月) 단위 getWeatherMonDayList3 (18필드) 하나뿐.

산출:
  - collect_weather(...): API 원본 18필드 raw DataFrame
  - to_model_df(raw)    : 모델 핵심10 컬럼으로 매핑한 DataFrame (mapping.py 기준)

파라미터:
  search_Year(연), search_Month(2자리 월), obsr_Spot_Cd(관측소코드)
"""
import pandas as pd
from .http_util import get_xml, get_total_count
from .config import load_config
from .mapping import map_weather_item, MODEL_WEATHER_COLS, WEATHER_KEY_MAP
from .io_util import save_csv

WEATHER_URL = ('https://apis.data.go.kr/1390802/AgriWeather/WeatherObsrInfo/V3/'
               'GnrlWeather/getWeatherMonDayList3')


def fetch_weather_month(spot, year, month, cfg=None, page_size=40):
    """단일 관측소·연·월 일별 기상 raw(dict 리스트). 페이징 자동.

    page_size=40: 한 달은 최대 31일이고 관측소당 하루 1행이라 항상 1페이지로
    끝난다. (하루 다중행이 생기는 미래 변화에 대비해 페이징 루프는 유지)
    """
    cfg = cfg or load_config()
    key = cfg.key('agriweather')
    verify = cfg.default('verify_ssl', True)
    out, page = [], 1
    while True:
        params = {
            'serviceKey': key, 'Page_No': str(page), 'Page_Size': str(page_size),
            'search_Year': str(year), 'search_Month': f'{int(month):02d}',
            'obsr_Spot_Cd': str(spot),
        }
        root, items = get_xml(WEATHER_URL, params, verify=verify)
        if not items:
            break
        out.extend(items)
        total = get_total_count(root)
        if total is None or len(out) >= total or len(items) < page_size:
            break
        page += 1
    return out


def collect_weather(spots, year, months=None, cfg=None):
    """여러 관측소 × 여러 월 raw DataFrame(원본 18필드)."""
    cfg = cfg or load_config()
    months = months or list(range(1, 13))
    rows = []
    for spot in spots:
        for m in months:
            rows.extend(fetch_weather_month(spot, year, m, cfg=cfg))
    return pd.DataFrame(rows)


def to_model_df(df_raw):
    """raw 18필드 -> 모델 핵심10 컬럼(+키) 매핑본."""
    model_cols = list(WEATHER_KEY_MAP.values()) + MODEL_WEATHER_COLS
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=model_cols)
    recs = df_raw.to_dict('records')
    return pd.DataFrame([map_weather_item(r) for r in recs])
