# -*- coding: utf-8 -*-
"""
농업기상 관측소 목록 수집 (getObsrSpotList).

주의: 이 엔드포인트는 'station' 키(보고서 공개키)로만 동작.
      agriweather(NEW) 키로는 403 Forbidden 이 떨어진다 -> 키를 분리해 둠.

응답 주요 컬럼:
  Obsr_Spot_Code, Obsr_Spot_Nm, Do_Se_Code, Mgc_Code, Clmt_Zone_Code,
  Comm_Mthd_Code, Instl_La(위도), Instl_Lo(경도), Instl_Al(고도),
  Instl_Adres(설치주소), Obsr_Begin_Datetm(관측개시일)
  -> 60일 결합의 cKDTree 는 Instl_La/Instl_Lo 를 사용한다.
"""
import pandas as pd
from .http_util import get_xml, get_total_count
from .config import load_config
from .io_util import save_csv

STATION_URL = 'http://apis.data.go.kr/1390802/AgriWeather/getObsrSpotList'


def fetch_stations(cfg=None, page_size=300):
    """관측소 전체 목록 DataFrame. 페이징 자동.

    주의(과거 결함 수정): 예전엔 1페이지(page_size=300)만 받아, 관측소가 300곳을
    넘으면 301번째부터 조용히 누락됐다. 지금은 weather/soil 과 동일하게 totalCount
    까지 페이지를 돌려 누락이 없다. (현재 전국 약 217곳이라 사실상 1페이지지만,
    올해 관측망이 늘어도 안전.)
    """
    cfg = cfg or load_config()
    key = cfg.key('station')
    verify = cfg.default('verify_ssl', True)
    out, page = [], 1
    while True:
        params = {'serviceKey': key, 'Page_Size': str(page_size),
                  'Page_No': str(page)}
        root, items = get_xml(STATION_URL, params, verify=verify)
        if not items:
            break
        out.extend(items)
        total = get_total_count(root)
        if total is None or len(out) >= total or len(items) < page_size:
            break
        page += 1
    df = pd.DataFrame(out)
    if 'no' in df.columns:
        df = df.drop(columns=['no'])
    return df


def save_stations(df, out_path):
    return save_csv(df, out_path)
