# -*- coding: utf-8 -*-
"""
주소 -> 좌표(경도 x_coord, 위도 y_coord) 지오코딩 (카카오 로컬 API).

언제 필요한가:
  60일 결합은 시료의 좌표(x_coord, y_coord)가 있어야 동작한다. 입력 CSV 에
  좌표가 이미 있으면 지오코딩은 건너뛴다(불필요). 좌표가 없고 주소만 있을
  때에만 카카오 API 로 변환한다.

좌표 규약(원본 보고서와 동일):
  x_coord = 경도(longitude) = 카카오 응답 x
  y_coord = 위도(latitude)  = 카카오 응답 y
  -> 60일 결합의 관측소 트리도 [위도, 경도] 순이라 규약이 일치한다.

원리(지오코딩):
  '서울시 강남구 ...' 같은 사람이 읽는 주소를 위경도 숫자로 바꾸는 작업.
  카카오 로컬 API 에 주소를 질의하면 후보 목록(documents)을 주고, 가장
  정확한 첫 후보의 x(경도)·y(위도)를 사용한다.
"""
import time
import requests
from requests.exceptions import HTTPError
from .config import load_config
from .io_util import save_csv

KAKAO_URL = 'https://dapi.kakao.com/v2/local/search/address.json'


def geocode_one(address, rest_key, session=None):
    """단일 주소 -> (x_coord 경도, y_coord 위도). 결과 없으면 (None, None)."""
    headers = {'Authorization': f'KakaoAK {rest_key}'}
    sess = session or requests
    r = sess.get(KAKAO_URL, params={'query': address}, headers=headers, timeout=20)
    r.raise_for_status()
    docs = r.json().get('documents', [])
    if not docs:
        return None, None
    d = docs[0]
    return float(d['x']), float(d['y'])


def attach_coords(df, addr_col, cfg=None, pause=0.03):
    """df 에 x_coord/y_coord 컬럼 추가. 이미 둘 다 있으면 그대로 반환.

    addr_col: 주소가 담긴 컬럼명.
    같은 주소는 한 번만 호출(중복 제거 후 매핑)하여 호출량을 줄인다.

    실패를 조용히 삼키지 않는다(과거 결함 수정):
      - 주소못찾음/통신오류 건수를 집계해 출력한다.
      - 401/403(키 오류)은 더 호출해도 같은 결과라 즉시 중단(RuntimeError).
      - 429(호출한도 초과)는 남은 주소를 멈추고 부분 결과를 남긴다.
      - 좌표를 못 얻은 시료 수도 경고한다(그 행은 60일 결합에서 기상피처가 빔).
    """
    if 'x_coord' in df.columns and 'y_coord' in df.columns:
        return df
    cfg = cfg or load_config()
    rest_key = cfg.key('kakao_rest')  # 비어있으면 config 에서 친절한 에러 발생
    sess = requests.Session()
    uniq = df[addr_col].dropna().drop_duplicates()
    table = {}
    fail_no_result = 0   # 주소를 못 찾음(정상적 실패)
    fail_error = 0       # 통신/응답 오류
    aborted = False
    for addr in uniq:
        try:
            xy = geocode_one(addr, rest_key, session=sess)
            table[addr] = xy
            if xy == (None, None):
                fail_no_result += 1
        except HTTPError as e:
            status = getattr(e.response, 'status_code', None)
            table[addr] = (None, None)
            fail_error += 1
            if status in (401, 403):
                raise RuntimeError(
                    f'[geocode] 카카오 키 오류(HTTP {status}). config.yaml 의 '
                    f'kakao_rest 키를 확인하세요. (지오코딩 중단)') from e
            if status == 429:
                print('[geocode] 카카오 호출한도 초과(HTTP 429) - 남은 주소 중단. '
                      '잠시 후 재시도하거나 좌표를 직접 제공하세요.')
                aborted = True
                break
        except Exception:
            table[addr] = (None, None)
            fail_error += 1
        if pause:
            time.sleep(pause)

    total = len(uniq)
    ok = total - fail_no_result - fail_error
    if fail_no_result or fail_error or aborted:
        print(f'[geocode] 고유주소 {total}건 중 성공 {ok} · 주소못찾음 '
              f'{fail_no_result} · 오류 {fail_error}'
              + (' · 한도초과로 중단' if aborted else ''))

    out = df.copy()
    out['x_coord'] = out[addr_col].map(lambda a: table.get(a, (None, None))[0])
    out['y_coord'] = out[addr_col].map(lambda a: table.get(a, (None, None))[1])
    n_missing = int(out['x_coord'].isna().sum())
    if n_missing:
        print(f'[geocode] 경고: 좌표 없는 시료 {n_missing}/{len(out)}행 '
              f'-> 이 행들은 60일 결합에서 기상피처가 비어(NaN) 남습니다.')
    return out
