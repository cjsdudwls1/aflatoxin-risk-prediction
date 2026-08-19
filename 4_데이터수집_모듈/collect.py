# -*- coding: utf-8 -*-
"""
아플라톡신 데이터 수집 CLI.

서브커맨드:
  stations   관측소 목록 수집
  weather    일별 농업기상 수집(연/월/관측소 지정)
  soil       토양 화학성 수집(법정동 STDG_CD / 필지 PNU_CD)
  combine    이미 수집된 CSV들로 60일 결합만 수행
  pipeline   주소+날짜 CSV 하나로 (지오코딩→관측소→기상→60일결합) 전과정 자동

예시:
  python collect.py stations
  python collect.py weather --year 2024 --months 6,7 --spots 026002D003
  python collect.py weather --year 2024 --all-spots
  python collect.py soil --codes 2671025021 --by bjd
  python collect.py combine --samples 시료.csv --stations 관측소.csv --weather 기상.csv
  python collect.py pipeline --samples 시료.csv --addr-col 주소 --out 결합결과.csv
"""
import argparse
import os
from collections import defaultdict
from datetime import timedelta

import pandas as pd

from collector.config import load_config
from collector import stations as st_mod
from collector import weather as wx_mod
from collector import soil as soil_mod
from collector import geocode as geo_mod
from collector import combine as cmb_mod


# ----------------------------- 공통 헬퍼 -----------------------------
def _out_path(cfg, name):
    out_dir = cfg.default('output_dir', 'outputs')
    # 상대경로면 '실행 위치'가 아니라 이 모듈 폴더 기준으로 고정한다.
    # (어디서 python collect.py 를 돌려도 항상 모듈/outputs 에 저장)
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_dir)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, name)


def _parse_list(s):
    return [x.strip() for x in str(s).split(',') if x.strip()]


def _window_k(args, cfg):
    """--window/--k 를 안 주면 config defaults(window_size/n_neighbors)를 쓴다."""
    window = args.window if args.window is not None else int(cfg.default('window_size', 60))
    k = args.k if args.k is not None else int(cfg.default('n_neighbors', 20))
    return window, k


def _load_samples_with_coords(args, cfg):
    """시료 CSV 로드 + 좌표 확보(없으면 지오코딩)."""
    df = pd.read_csv(args.samples, dtype=str)
    if 'REQEST_DE' not in df.columns:
        raise SystemExit("입력 CSV 에 'REQEST_DE'(채취일) 컬럼이 필요합니다.")
    has_xy = 'x_coord' in df.columns and 'y_coord' in df.columns
    if not has_xy:
        if not args.addr_col:
            raise SystemExit('좌표(x_coord,y_coord)가 없으면 --addr-col 로 주소 컬럼을 '
                             '지정해 지오코딩하세요.')
        df = geo_mod.attach_coords(df, addr_col=args.addr_col, cfg=cfg)
    return df


def _needed_year_months(dates, window):
    """시료 채취일들로부터 '직전 window일'이 걸치는 (연,월) 집합."""
    s = set()
    for d in dates:
        d = pd.Timestamp(d)
        cur = (d - timedelta(days=window - 1)).replace(day=1)
        while cur <= d:
            s.add((cur.year, cur.month))
            cur = cur + pd.offsets.MonthBegin(1)
    return sorted(s)


# ----------------------------- 서브커맨드 -----------------------------
def cmd_stations(args, cfg):
    df = st_mod.fetch_stations(cfg=cfg)
    out = args.out or _out_path(cfg, '농업기상정보_관측소.csv')
    st_mod.save_stations(df, out)
    print(f'[stations] {len(df)}건 저장 -> {out}')


def cmd_weather(args, cfg):
    if args.all_spots:
        spots = st_mod.fetch_stations(cfg=cfg)['Obsr_Spot_Code'].astype(str).tolist()
    else:
        spots = _parse_list(args.spots) if args.spots else []
    if not spots:
        raise SystemExit('--spots(쉼표구분) 또는 --all-spots 가 필요합니다.')
    months = [int(m) for m in _parse_list(args.months)] if args.months else None
    raw = wx_mod.collect_weather(spots, args.year, months=months, cfg=cfg)
    out_raw = args.out_raw or _out_path(cfg, f'농업기상_raw_{args.year}.csv')
    out_model = args.out_model or _out_path(cfg, f'농업기상정보_{args.year}.csv')
    wx_mod.save_csv(raw, out_raw)
    wx_mod.save_csv(wx_mod.to_model_df(raw), out_model)
    print(f'[weather] raw {len(raw)}건 -> {out_raw}')
    print(f'[weather] model -> {out_model}')


def cmd_soil(args, cfg):
    if args.codes_file:
        codes = pd.read_csv(args.codes_file, header=None, dtype=str)[0].tolist()
    else:
        codes = _parse_list(args.codes) if args.codes else []
    if not codes:
        raise SystemExit('--codes(쉼표구분) 또는 --codes-file 이 필요합니다.')
    df = soil_mod.collect_soil(codes, by=args.by, cfg=cfg)
    out = args.out or _out_path(cfg, '토양화학성.csv')
    soil_mod.save_csv(df, out)
    print(f'[soil] {len(df)}건 저장 -> {out}')


def cmd_combine(args, cfg):
    samples = _load_samples_with_coords(args, cfg)
    df_st = pd.read_csv(args.stations, dtype={'Obsr_Spot_Code': str})
    df_wx = pd.read_csv(args.weather, dtype={'obsr_Spot_Code': str})
    window, k = _window_k(args, cfg)
    res = cmb_mod.combine_60day(samples, df_st, df_wx, window=window, k=k)
    out = args.out or _out_path(cfg, '결합_60일.csv')
    cmb_mod.save_csv(res, out)
    print(f'[combine] 결과 shape={res.shape} -> {out}')


def cmd_pipeline(args, cfg):
    samples = _load_samples_with_coords(args, cfg)
    samples['REQEST_DE'] = pd.to_datetime(samples['REQEST_DE'].astype(str))
    window, k = _window_k(args, cfg)

    df_st = st_mod.fetch_stations(cfg=cfg)
    spots = df_st['Obsr_Spot_Code'].astype(str).tolist()

    yms = _needed_year_months(samples['REQEST_DE'], window)
    print(f'[pipeline] 관측소 {len(spots)}곳 · 필요 월 {len(yms)}개 수집 시작')

    by_year = defaultdict(list)
    for (y, m) in yms:
        by_year[y].append(m)
    raw_parts = []
    for y, ms in sorted(by_year.items()):
        part = wx_mod.collect_weather(spots, y, months=sorted(ms), cfg=cfg)
        raw_parts.append(part)
        print(f'   {y}년 {sorted(ms)} -> {len(part)}건')
    raw = pd.concat(raw_parts, ignore_index=True) if raw_parts else pd.DataFrame()
    df_wx = wx_mod.to_model_df(raw)

    res = cmb_mod.combine_60day(samples, df_st, df_wx, window=window, k=k)
    out = args.out or _out_path(cfg, '파이프라인_결합_60일.csv')
    cmb_mod.save_csv(res, out)
    print(f'[pipeline] 최종 shape={res.shape} -> {out}')


# ----------------------------- 파서 -----------------------------
def build_parser():
    p = argparse.ArgumentParser(description='아플라톡신 데이터 수집 모듈 CLI')
    p.add_argument('--config', default=None, help='config.yaml 경로(기본: 모듈 루트)')
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('stations', help='관측소 목록 수집')
    s.add_argument('--out')
    s.set_defaults(func=cmd_stations)

    w = sub.add_parser('weather', help='일별 농업기상 수집')
    w.add_argument('--year', type=int, required=True)
    w.add_argument('--months', help='쉼표구분 월(예: 6,7). 생략시 1~12')
    w.add_argument('--spots', help='쉼표구분 관측소코드')
    w.add_argument('--all-spots', action='store_true', help='관측소 전체 수집')
    w.add_argument('--out-raw')
    w.add_argument('--out-model')
    w.set_defaults(func=cmd_weather)

    so = sub.add_parser('soil', help='토양 화학성 수집')
    so.add_argument('--codes', help='쉼표구분 코드(STDG_CD 또는 PNU_CD)')
    so.add_argument('--codes-file', help='코드 1열 CSV 파일 경로')
    so.add_argument('--by', choices=['bjd', 'pnu'], default='bjd',
                    help='bjd=법정동(STDG_CD), pnu=필지(PNU_CD)')
    so.add_argument('--out')
    so.set_defaults(func=cmd_soil)

    c = sub.add_parser('combine', help='수집된 CSV들로 60일 결합')
    c.add_argument('--samples', required=True, help='좌표/주소 + REQEST_DE CSV')
    c.add_argument('--stations', required=True, help='관측소 CSV')
    c.add_argument('--weather', required=True, help='모델스키마 기상 CSV')
    c.add_argument('--addr-col', help='좌표 없을때 지오코딩할 주소 컬럼명')
    c.add_argument('--window', type=int, default=None,
                   help='60일 결합 윈도우(미지정시 config window_size=60)')
    c.add_argument('--k', type=int, default=None,
                   help='최근접 관측소 후보 수(미지정시 config n_neighbors=20)')
    c.add_argument('--out')
    c.set_defaults(func=cmd_combine)

    pl = sub.add_parser('pipeline', help='주소+날짜 CSV 한 번에 처리')
    pl.add_argument('--samples', required=True, help='주소(또는 좌표) + REQEST_DE CSV')
    pl.add_argument('--addr-col', help='좌표 없을때 지오코딩할 주소 컬럼명')
    pl.add_argument('--window', type=int, default=None,
                    help='60일 결합 윈도우(미지정시 config window_size=60)')
    pl.add_argument('--k', type=int, default=None,
                    help='최근접 관측소 후보 수(미지정시 config n_neighbors=20)')
    pl.add_argument('--out')
    pl.set_defaults(func=cmd_pipeline)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    args.func(args, cfg)


if __name__ == '__main__':
    main()
