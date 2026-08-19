# -*- coding: utf-8 -*-
"""
농업기상정보 크롤링 재현 스크립트 (식정원 최종보고서 364~368p 코드 기반)

출처: 공공데이터포털(data.go.kr) > 국립농업과학원 농업기상 기본 관측데이터 조회 OpenAPI
      제공기관 코드 1390802 (AgriWeather)
   - getObsrSpotList                : 관측소 목록  -> 농업기상정보_관측소.csv
   - .../InsttWeather/getWeatherYearDayList : 지점·연도별 일 단위 기상 -> 농업기상정보.csv

사용법:
  python reproduce_crawl.py spots          # 관측소 목록만 크롤링/비교
  python reproduce_crawl.py weather 2024   # 특정 연도 기상(일부 지점) 샘플 크롤링

주의: 보고서 본문에 노출된 서비스키는 만료/차단될 수 있음.
      그 경우 data.go.kr에서 본인 키를 발급받아 SERVICE_KEY 교체.
"""
import sys, requests, pandas as pd
import xml.etree.ElementTree as ET
import urllib3
urllib3.disable_warnings()

SERVICE_KEY = '여기에_본인_서비스키_data.go.kr에서_발급'
SCRAPED = r'scraped_data'  # 원본 비교 대상 폴더 (이 스크립트와 같은 위치 기준)


def transform_to_df(response):
    """보고서 transform_to_df 재현: items/item 을 DataFrame 으로."""
    if response.status_code != 200 or 'xml' not in response.headers.get('Content-Type', ''):
        # 404/에러 페이지 등 XML 이 아닌 응답
        raise RuntimeError(f'API 오류 HTTP {response.status_code}: {response.text[:200]!r}\n'
                           f'  URL={response.url}\n'
                           f'  -> data.go.kr 활용신청/엔드포인트 경로/서비스키 확인 필요')
    root = ET.fromstring(response.content)
    items = root.find('.//items')
    if items is None:
        return pd.DataFrame()
    rows = [{c.tag: c.text for c in it} for it in items.findall('item')]
    return pd.DataFrame(rows)


def get_spot_list():
    params = {'serviceKey': SERVICE_KEY, 'Page_Size': '300', 'Page_No': '1', 'type': 'json'}
    r = requests.get('http://apis.data.go.kr/1390802/AgriWeather/getObsrSpotList',
                     params=params, verify=False, timeout=60)
    return transform_to_df(r)


def get_data_by_year(spot='026002D003', year=2024):
    params = {'serviceKey': SERVICE_KEY, 'Page_No': '1', 'Page_Size': '366',
              'search_Year': year, 'obsr_Spot_Code': spot}
    r = requests.get(
        'https://apis.data.go.kr/1390802/AgriWeather/WeatherObsrInfo/V3/InsttWeather/getWeatherYearDayList',
        params=params, verify=False, timeout=60)
    return transform_to_df(r)


def cmd_spots():
    df = get_spot_list().drop(columns=['no'], errors='ignore')
    print('[크롤링] 관측소', df.shape)
    try:
        orig = pd.read_csv(f'{SCRAPED}/농업기상정보_관측소.csv', dtype=str)
        same_codes = set(orig.Obsr_Spot_Code) == set(df.Obsr_Spot_Code)
        print(f'원본 {orig.shape} / 코드집합 일치: {same_codes}')
    except FileNotFoundError:
        print('원본 CSV 없음 — 비교 생략')
    df.to_csv('repro_농업기상정보_관측소.csv', index=False, encoding='utf-8-sig')
    print('saved -> repro_농업기상정보_관측소.csv')


def cmd_weather(year):
    spots = get_spot_list().Obsr_Spot_Code.values[:3]  # 샘플 3개 지점
    out = pd.concat([get_data_by_year(s, year).drop(columns=['no', 'obsr_Spot_Nm'], errors='ignore')
                     for s in spots], ignore_index=True)
    print(f'[크롤링] {year}년 {len(spots)}개 지점 기상', out.shape)
    print(out.head().to_string())
    out.to_csv(f'repro_농업기상정보_{year}_sample.csv', index=False, encoding='utf-8-sig')
    print(f'saved -> repro_농업기상정보_{year}_sample.csv')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'spots'
    if cmd == 'weather':
        cmd_weather(int(sys.argv[2]) if len(sys.argv) > 2 else 2024)
    else:
        cmd_spots()
