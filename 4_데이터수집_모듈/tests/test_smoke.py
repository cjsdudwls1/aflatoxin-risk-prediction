# -*- coding: utf-8 -*-
"""
스모크 검수 (네트워크 불필요).

확인 항목:
  1) 전 모듈 + scipy import
  2) 매핑 함수 단위(기상/토양) 정확성
  3) 60일 결합의 flatten 순서·윈도우 경계(합성 데이터)
  4) 지오코딩은 좌표가 있으면 건너뛰는지

실행:  python tests/test_smoke.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import pandas as pd
from collector import combine, geocode, mapping


def test_imports():
    import collector.stations, collector.weather, collector.soil  # noqa
    import collector.http_util, collector.config                  # noqa
    from scipy.spatial import cKDTree  # noqa  (결합 핵심 의존성)
    print('[import] 전 모듈 + scipy OK')


def test_mapping_units():
    w = mapping.map_weather_item({'stn_Cd': 'X', 'date': '2024-06-01',
                                  'temp': '11.2', 'rn': '0.5', 'hum': '80.3'})
    assert w['obsr_Spot_Code'] == 'X'
    assert w['date_Time'] == '2024-06-01'
    assert w['tmprt_150'] == '11.2'   # temp
    assert w['afp'] == '0.5'          # rn
    assert w['hd_150'] == '80.3'      # hum
    # 키2 + 핵심10 컬럼이 정확히 생성되는지
    assert set(['obsr_Spot_Code', 'date_Time'] + mapping.MODEL_WEATHER_COLS) == set(w)

    s = mapping.map_soil_item({'No': '1', 'Stdg_Cd': '2671025021',
                               'ELCD': '0.58', 'ACID': '7.2'})
    assert 'No' not in s              # 순번 제거
    assert s['BJD_Code'] == '2671025021'  # Stdg_Cd 개명
    assert s['SELC'] == '0.58'            # ELCD 개명
    assert s['ACID'] == '7.2'             # 나머지 보존
    print('[mapping] 기상/토양 단위 매핑 OK')


def test_combine_synthetic():
    wcols = combine.MODEL_WEATHER_COLS
    stations = pd.DataFrame({'Obsr_Spot_Code': ['AAA', 'BBB'],
                             'Instl_La': [37.0, 35.0],     # 위도
                             'Instl_Lo': [127.0, 129.0]})  # 경도
    # AAA 65일 연속, 값 = 일차*100 + 항목번호 (식별용)
    dates = pd.date_range('2024-05-01', periods=65, freq='D')
    rows = []
    for i, d in enumerate(dates):
        r = {'obsr_Spot_Code': 'AAA', 'date_Time': d.strftime('%Y-%m-%d')}
        for j, c in enumerate(wcols):
            r[c] = float(i * 100 + j)
        rows.append(r)
    weather = pd.DataFrame(rows)
    samples = pd.DataFrame({'x_coord': [127.01], 'y_coord': [37.01],
                            'REQEST_DE': ['2024-06-30']})

    res = combine.combine_60day(samples, stations, weather, window=60, k=20)
    i_end = (pd.Timestamp('2024-06-30') - pd.Timestamp('2024-05-01')).days  # 60
    i_start = i_end - 59                                                    # 1
    assert res.shape[1] == 3 + 600, res.shape          # 시료3 + 평탄화600
    assert res['tmprt_150_00'].iloc[0] == float(i_end * 100)      # _00=채취일
    assert res['tmprt_150_59'].iloc[0] == float(i_start * 100)    # _59=60일전
    assert res['afp_00'].iloc[0] == float(i_end * 100 + 6)        # afp=7번째
    print(f'[combine] 합성 결합 shape={res.shape}, _00=채취일/_59=60일전 경계 정확 OK')


def test_geocode_skip():
    df = pd.DataFrame({'x_coord': [127.0], 'y_coord': [37.0], 'addr': ['아무주소']})
    out = geocode.attach_coords(df.copy(), addr_col='addr')  # 키 없이도 통과해야
    assert 'x_coord' in out.columns and len(out) == 1
    print('[geocode] 좌표 존재시 건너뜀 OK')


if __name__ == '__main__':
    test_imports()
    test_mapping_units()
    test_combine_synthetic()
    test_geocode_skip()
    print('SMOKE OK')
