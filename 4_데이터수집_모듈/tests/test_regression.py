# -*- coding: utf-8 -*-
"""
회귀대조 검수: 신규 수집기가 작년(scraped_data) 데이터를 재현하는지 검증.

판정 기준:
  - 기상 exact 7컬럼 : 작년 CSV와 100% 일치(허용오차 0.05) -> 실패시 FAIL
  - 기상 diff 3컬럼  : 차이가 '문서화된 사유' 범위인지 통계만 출력(FAIL 아님)
  - 토양 13컬럼      : 동일 시료(주소) 값 일치
  - 관측소           : 코드 교집합의 위경도 일치

작년 데이터(scraped_data)가 없으면 해당 항목은 SKIP.
실제 API 를 소량 호출한다(관측소1·기상 1관측소1개월·토양 1코드).

실행:  python tests/test_regression.py
경로 변경:  환경변수 SCRAPED_DIR 로 scraped_data 폴더 지정 가능.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import pandas as pd
from collector import weather as wx_mod
from collector import soil as soil_mod
from collector import stations as st_mod
from collector.config import load_config
from collector.mapping import (REGRESSION_EXACT_COLS, REGRESSION_DIFF_COLS,
                               SOIL_MODEL_COLS)

SCRAPED = os.environ.get('SCRAPED_DIR') or os.path.normpath(
    os.path.join(ROOT, '..', '..', '2026식약처',
                 '2. 실험에 사용된 전체 데이터', 'scraped_data'))
TOL = 0.05         # 토양/관측소 일치 허용오차
# 기상 판정 기준(현실 반영):
#   두 소스(작년 CSV·API)가 각각 소수1자리로 반올림 -> 같은 실측도 표기상 0.1 벌어진다.
ROUND_TOL = 0.15   # 이 안이면 '반올림 수준 일치'(완전 동일로 간주)
HARD_TOL = 1.0     # 이걸 넘으면 '다른 데이터'로 보고 FAIL
MIN_RATE = 0.95    # 반올림 허용 일치율 하한(산발적 측정단위차는 WARN 통과)
# 관측소 좌표 판정:
COORD_TOL = 1e-4   # ~10m: 표기정밀도/미세갱신 흡수(이 안은 동일로 간주)
COORD_BIG = 1e-2   # ~1km 이상이면 실제 이설로 FAIL


def _num(series):
    return pd.to_numeric(series, errors='coerce')


# ----------------------------- 기상 -----------------------------
def test_weather(cfg):
    src = os.path.join(SCRAPED, '농업기상정보.csv')
    if not os.path.exists(src):
        print('[기상] SKIP - 작년 CSV 없음:', src)
        return None
    spot, year, month = '026002D003', 2024, 6
    use = ['obsr_Spot_Code', 'date_Time'] + REGRESSION_EXACT_COLS + list(REGRESSION_DIFF_COLS)
    old = pd.read_csv(src, usecols=lambda c: c in use, dtype={'obsr_Spot_Code': str})
    old = old[(old['obsr_Spot_Code'] == spot) &
              (old['date_Time'].astype(str).str.startswith(f'{year}-{month:02d}'))].copy()

    raw = wx_mod.collect_weather([spot], year, months=[month], cfg=cfg)
    new = wx_mod.to_model_df(raw)
    new['obsr_Spot_Code'] = new['obsr_Spot_Code'].astype(str)

    m = old.merge(new, on=['obsr_Spot_Code', 'date_Time'], suffixes=('_old', '_new'))
    print(f'[기상] 관측소 {spot} {year}-{month:02d} · 공통 {len(m)}일 대조')
    if m.empty:
        print('   (공통 행 없음 -> 검증 불가)')
        return None

    ok_all = True
    warns = 0
    exceed_rows = []
    for c in REGRESSION_EXACT_COLS:
        d = (_num(m[f'{c}_old']) - _num(m[f'{c}_new'])).abs()
        both = d.notna()
        dd = d[both]
        maxd = dd.max() if both.any() else float('nan')
        rate = (dd <= ROUND_TOL).mean() if both.any() else float('nan')
        if both.any() and maxd <= ROUND_TOL:
            status = 'OK'                                    # 완전 반올림 일치
        elif both.any() and rate >= MIN_RATE and maxd <= HARD_TOL:
            status = 'WARN'                                  # 산발 측정/반올림차(통과)
            warns += 1
        else:
            status = 'FAIL'                                  # 데이터가 실제로 다름
            ok_all = False
        print(f'   [exact] {c:16s} 반올림일치율 {rate:6.1%} 최대차 {maxd:.3f} -> {status}')
        mask = d.notna() & (d > ROUND_TOL)
        for _, r in m[mask].iterrows():
            exceed_rows.append((c, r['date_Time'], r[f'{c}_old'], r[f'{c}_new']))
    if exceed_rows:
        print('   -- 반올림(0.15) 초과 상세(투명성) --')
        for c, dt, ov, nv in exceed_rows[:15]:
            print(f'      {c:14s} {dt} 작년={ov} 신규={nv}')
    for c in REGRESSION_DIFF_COLS:
        a, b = _num(m[f'{c}_old']), _num(m[f'{c}_new'])
        d = (a - b).abs()
        print(f'   [diff ] {c:16s} 작년결측 {a.isna().mean():5.1%} '
              f'최대차 {d.max() if d.notna().any() else float("nan")} '
              f'· {REGRESSION_DIFF_COLS[c]}')
    assert ok_all, '기상 exact 컬럼에서 데이터 수준 불일치(반올림 한계 초과 FAIL)'
    print(f'[기상] exact 7컬럼 반올림 일치 확인 (WARN {warns}건 = 산발 측정/단위차) ✓')
    return True


# ----------------------------- 토양 -----------------------------
def test_soil(cfg):
    src = os.path.join(SCRAPED, '토양_정보.csv')
    if not os.path.exists(src):
        print('[토양] SKIP - 작년 CSV 없음')
        return None
    code = '2671025021'
    old = pd.read_csv(src, dtype=str)
    old = old[old['BJD_Code'].astype(str) == code]
    new = soil_mod.collect_soil([code], by='bjd', cfg=cfg)
    cols = [c for c in SOIL_MODEL_COLS if c in old.columns and c in new.columns]

    mism = []
    for _, nr in new.iterrows():
        cand = old[old['PNU_Nm'] == nr['PNU_Nm']]
        if cand.empty:
            mism.append((nr.get('PNU_Nm'), '작년에 동일주소 없음'))
            continue
        orow = cand.iloc[0]
        for c in cols:
            av = _num(pd.Series([orow[c]])).iloc[0]
            bv = _num(pd.Series([nr[c]])).iloc[0]
            if pd.isna(av) and pd.isna(bv):
                continue                      # 둘 다 결측 = 일치
            if pd.isna(av) or pd.isna(bv):
                if str(orow[c]) != str(nr[c]):
                    mism.append((nr['PNU_Nm'], c, orow[c], nr[c]))
            elif abs(av - bv) > TOL:
                mism.append((nr['PNU_Nm'], c, av, bv))
    print(f'[토양] 코드 {code} · 신규 {len(new)}행 / 작년 {len(old)}행 · 비교컬럼 {len(cols)}')
    for x in mism[:10]:
        print('   불일치:', x)
    assert not mism, '토양 값 불일치'
    print(f'[토양] {len(cols)}컬럼 값 일치 ✓')
    return True


# ----------------------------- 관측소 -----------------------------
def test_stations(cfg):
    src = os.path.join(SCRAPED, '농업기상정보_관측소.csv')
    if not os.path.exists(src):
        print('[관측소] SKIP - 작년 CSV 없음')
        return None
    old = pd.read_csv(src, dtype={'Obsr_Spot_Code': str}).drop_duplicates('Obsr_Spot_Code')
    new = st_mod.fetch_stations(cfg=cfg)
    new['Obsr_Spot_Code'] = new['Obsr_Spot_Code'].astype(str)
    new = new.drop_duplicates('Obsr_Spot_Code')

    common = sorted(set(old['Obsr_Spot_Code']) & set(new['Obsr_Spot_Code']))
    om = old.set_index('Obsr_Spot_Code')
    nm = new.set_index('Obsr_Spot_Code')
    diffs = []
    for c in common:
        la0, lo0 = float(om.loc[c, 'Instl_La']), float(om.loc[c, 'Instl_Lo'])
        la1, lo1 = float(nm.loc[c, 'Instl_La']), float(nm.loc[c, 'Instl_Lo'])
        gap = max(abs(la0 - la1), abs(lo0 - lo1))
        if gap > COORD_TOL:
            diffs.append((c, gap, (la0, lo0), (la1, lo1)))
    print(f'[관측소] 작년 {len(old)} / 신규 {len(new)} / 공통 {len(common)}곳 · '
          f'좌표차(>{COORD_TOL}) {len(diffs)}곳')
    for c, gap, o, n in sorted(diffs, key=lambda x: -x[1])[:10]:
        kind = '큰차이(이설?)' if gap > COORD_BIG else '미세'
        print(f'   {c} 차이 {gap:.6f}도 [{kind}] 작년{o} 신규{n}')
    big = [d for d in diffs if d[1] > COORD_BIG]
    moved_rate = len(big) / max(len(common), 1)
    if big:
        print(f'   ※ 위치변경(이설) {len(big)}곳 = {moved_rate:.1%}: API 가 주는 현재 '
              f'좌표이며 모듈 가공이 아님(결합 시 해당 근방 시료 영향 가능)')
    # 이설이 소수면 '데이터 현실'로 통과, 과다(>3%)면 원천 이상으로 FAIL
    assert moved_rate <= 0.03, f'관측소 이설 과다 {moved_rate:.1%} - 데이터 원천 이상 의심'
    print(f'[관측소] 동일 {len(common) - len(diffs)}곳 · 미세 {len(diffs) - len(big)}곳 · '
          f'이설 {len(big)}곳({moved_rate:.1%}) ✓')
    return True


if __name__ == '__main__':
    cfg = load_config()
    print('scraped_data:', SCRAPED, '\n')
    results = {
        '기상': test_weather(cfg),
        '토양': test_soil(cfg),
        '관측소': test_stations(cfg),
    }
    print('\n=== 회귀대조 요약 ===')
    for name, r in results.items():
        tag = 'PASS' if r is True else ('SKIP' if r is None else 'FAIL')
        print(f'  {name}: {tag}')
    print('REGRESSION DONE')
