# -*- coding: utf-8 -*-
"""
좌표·일자 기반 '검사별 60일 농업기상 시계열' 결합.

원본 보고서(2. 생성코드정리)의 핵심 로직을 모듈로 충실히 재현한다:
  1) 관측소 위경도로 cKDTree 구성 (좌표순서 = [위도 Instl_La, 경도 Instl_Lo])
  2) 각 시료(채취일 REQEST_DE, 좌표)마다 최근접 k개 관측소 탐색
  3) 그 중 '채취일 기준 직전 60일이 결측없이 채워진' 가장 가까운 관측소 선택
  4) 그 관측소의 60일 기상(10컬럼)을 600개 컬럼으로 평탄화(flatten)

용어 풀이:
  - cKDTree: 좌표 최근접이웃을 빠르게 찾는 자료구조(K-D 트리). 모든 관측소와
    거리를 일일이 재지 않고 가까운 후보부터 바로 찾아준다.
  - flatten(평탄화): 60일 × 10항목 2차원 표를 한 줄(600칸)로 펼치는 것.
    컬럼명 규칙 {항목}_{일차2자리}, 일차 59=가장 과거, 00=채취일.
  - rolling(window=60).count(): 각 날짜에서 '직전 60개 행'의 비결측 개수.
    이게 60이면 그 종료일 기준 60일치가 빠짐없이 있다는 뜻.

주의(원본과 동일한 좌표 규약):
  task_coords = [y_coord(위도), x_coord(경도)] 순서로 트리에 질의한다.
"""
import numpy as np
import pandas as pd
from datetime import timedelta
from scipy.spatial import cKDTree
from .mapping import MODEL_WEATHER_COLS
from .io_util import save_csv

JOIN_KEYS = ['x_coord', 'y_coord', 'REQEST_DE']


def flattened_cols(window=60, weather_cols=None):
    """600개(=window×len) 평탄화 컬럼명. 원본 FLATTENED_COLS 와 동일 순서."""
    weather_cols = weather_cols or MODEL_WEATHER_COLS
    return [f'{c}_{d:02d}' for d in range(window - 1, -1, -1) for c in weather_cols]


def _prepare_weather(df_weather, weather_cols, window):
    """날씨 DF -> (관측소·일자 인덱싱 DF, 유효 60일 윈도우 집합)."""
    w = df_weather[['obsr_Spot_Code', 'date_Time'] + weather_cols].copy()
    w.dropna(subset=['obsr_Spot_Code', 'date_Time'], inplace=True)
    w['obsr_Spot_Code'] = w['obsr_Spot_Code'].astype(str)
    w['date_Time'] = pd.to_datetime(w['date_Time'])
    for c in weather_cols:
        w[c] = pd.to_numeric(w[c], errors='coerce')
    w.sort_values(['obsr_Spot_Code', 'date_Time'], inplace=True)

    wt = w.set_index('date_Time')
    rc = (wt.groupby('obsr_Spot_Code')[weather_cols[0]]
            .rolling(window=window).count().reset_index())
    ok = rc[rc[weather_cols[0]] == window]
    valid = set(zip(ok['obsr_Spot_Code'], ok['date_Time']))

    w_idx = w.set_index(['obsr_Spot_Code', 'date_Time']).sort_index()
    return w_idx, valid


def _best_spots(tasks, tree, code_map, valid, k):
    """각 task 의 최근접 k개 중 60일이 유효한 가장 가까운 관측소 1개씩."""
    coords = tasks[['y_coord', 'x_coord']].astype(float).values
    dist, idx = tree.query(coords, k=k)
    if k == 1:                       # k=1 이면 1차원 반환 -> 2차원화
        idx = idx[:, None]
        dist = dist[:, None]
    n = len(tasks)
    cand = pd.DataFrame({
        'task_id': np.arange(n).repeat(k),
        'neighbor_rank': np.tile(np.arange(k), n),
        'spot_index': idx.ravel(),
        'distance': dist.ravel(),
    })
    cand['Obsr_Spot_Code'] = cand['spot_index'].map(code_map)
    cand = tasks.reset_index(drop=True).merge(cand, left_index=True, right_on='task_id')
    cand['is_valid'] = [(c, d) in valid for c, d in
                        zip(cand['Obsr_Spot_Code'], cand['REQEST_DE'])]
    vc = cand[cand['is_valid']]
    if vc.empty:
        return vc
    vc = vc.sort_values(['task_id', 'neighbor_rank'])
    return vc.drop_duplicates('task_id', keep='first')


def combine_60day(df_samples, df_stations, df_weather,
                  window=60, k=20, weather_cols=None):
    """시료(주소·날짜) + 관측소 + 날씨 -> 60일 평탄화 피처 결합 DataFrame.

    df_samples : x_coord, y_coord, REQEST_DE 포함(좌표 없으면 geocode 선행)
    df_stations: Obsr_Spot_Code, Instl_La(위도), Instl_Lo(경도)
    df_weather : obsr_Spot_Code, date_Time, + weather_cols(기본 모델 핵심10)
    """
    weather_cols = weather_cols or MODEL_WEATHER_COLS
    fcols = flattened_cols(window, weather_cols)

    spots = df_stations[['Obsr_Spot_Code', 'Instl_La', 'Instl_Lo']].dropna().copy()
    spots['Instl_La'] = spots['Instl_La'].astype(float)
    spots['Instl_Lo'] = spots['Instl_Lo'].astype(float)
    spots = spots.reset_index(drop=True)
    tree = cKDTree(spots[['Instl_La', 'Instl_Lo']].values)
    code_map = spots['Obsr_Spot_Code'].astype(str)

    w_idx, valid = _prepare_weather(df_weather, weather_cols, window)

    tasks = df_samples[JOIN_KEYS].drop_duplicates().reset_index(drop=True)
    tasks['x_coord'] = tasks['x_coord'].astype(float)
    tasks['y_coord'] = tasks['y_coord'].astype(float)
    tasks['REQEST_DE'] = pd.to_datetime(tasks['REQEST_DE'].astype(str))

    best = _best_spots(tasks, tree, code_map, valid, k)

    recs = []
    for _, row in best.iterrows():
        spot, end = row['Obsr_Spot_Code'], row['REQEST_DE']
        start = end - timedelta(days=window - 1)
        try:
            wd = w_idx.loc[(spot, start):(spot, end), weather_cols]
        except KeyError:
            # 그 관측소에 해당 날짜구간 행이 아예 없음(정상적 누락) -> 건너뜀.
            # KeyError 만 잡는다: dtype 손상 등 예기치 못한 에러는 묻지 않고 드러낸다.
            continue
        if len(wd) == window:
            fd = dict(zip(fcols, wd.values.ravel()))
            fd['x_coord'] = row['x_coord']
            fd['y_coord'] = row['y_coord']
            fd['REQEST_DE'] = row['REQEST_DE']
            recs.append(fd)

    out = df_samples.copy()
    out['x_coord'] = out['x_coord'].astype(float)
    out['y_coord'] = out['y_coord'].astype(float)
    out['REQEST_DE'] = pd.to_datetime(out['REQEST_DE'].astype(str))
    if not recs:
        return out
    df_feats = pd.DataFrame(recs)
    return out.merge(df_feats, on=JOIN_KEYS, how='left')
