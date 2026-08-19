# -*- coding: utf-8 -*-
"""
지오코딩 로직 검수 (네트워크·카카오키 불필요, 가짜 응답=mock 사용).

왜 mock 인가:
  실제 카카오 호출은 키가 있어야 하고 외부 통신이라 검증이 불안정하다.
  그래서 '카카오가 이런 JSON 을 준다면' 하는 가짜 응답을 끼워넣어,
  우리 코드(응답 파싱·중복제거·오류분기)가 그 응답을 올바로 처리하는지만
  떼어내 검증한다. -> 키 없이도 로직 정확성을 보장.

확인 항목:
  1) geocode_one 이 documents[0] 의 x(경도)/y(위도)를 올바로 뽑는지
  2) 결과가 없으면 (None, None)
  3) attach_coords 가 같은 주소를 1번만 호출(중복제거)하는지
  4) 401/403(키오류) -> 즉시 RuntimeError 로 중단
  5) 429(한도초과) -> 남은 주소 멈추고 부분결과 보존(예외 안던짐)
  6) 주소못찾음 -> 그 행 좌표 NaN (조용히 성공으로 위장하지 않음)
  7) 좌표가 이미 있으면 호출 없이 건너뜀

실행:  python tests/test_geocode.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import pandas as pd
from requests.exceptions import HTTPError
from collector import geocode

_ORIG_SESSION = geocode.requests.Session   # 끝에서 원복


class FakeResp:
    """requests 응답 흉내: status_code / raise_for_status / json 만 구현."""
    def __init__(self, payload=None, status=200):
        self._payload = payload or {}
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            e = HTTPError(f'HTTP {self.status_code}')
            e.response = self      # geocode 가 e.response.status_code 를 본다
            raise e

    def json(self):
        return self._payload


class FakeSession:
    """requests.Session 흉내. script(addr)->FakeResp. 호출주소를 calls 에 기록."""
    def __init__(self, script):
        self.script = script
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        addr = params['query']
        self.calls.append(addr)
        return self.script(addr)


class FakeCfg:
    """config 로더 흉내 - 더미 키를 주어 '키 비었음' 에러를 우회."""
    def key(self, name):
        return 'DUMMY_KEY'

    def default(self, name, d=None):
        return d


def _patch_session(fake):
    geocode.requests.Session = lambda: fake


# ----------------------------- 1·2: 단일 파싱 -----------------------------
def test_geocode_one_parse():
    sess = FakeSession(lambda a: FakeResp({'documents': [{'x': '127.5', 'y': '37.5'}]}))
    x, y = geocode.geocode_one('서울시 어딘가', 'DUMMY', session=sess)
    assert (x, y) == (127.5, 37.5), (x, y)
    assert x == 127.5 and y == 37.5      # x=경도, y=위도 규약
    print('[geocode] 단일주소 파싱 OK (x=경도 127.5, y=위도 37.5)')


def test_geocode_one_empty():
    sess = FakeSession(lambda a: FakeResp({'documents': []}))
    assert geocode.geocode_one('없는주소', 'DUMMY', session=sess) == (None, None)
    print('[geocode] 결과없음 -> (None, None) OK')


# ----------------------------- 3: 중복제거 -----------------------------
def test_attach_dedup():
    df = pd.DataFrame({'addr': ['A동', 'A동', 'B동']})      # A 중복 1건
    table = {'A동': {'x': '127.0', 'y': '37.0'},
             'B동': {'x': '128.0', 'y': '36.0'}}
    sess = FakeSession(lambda a: FakeResp({'documents': [table[a]]}))
    _patch_session(sess)
    out = geocode.attach_coords(df.copy(), addr_col='addr', cfg=FakeCfg(), pause=0)
    assert out['x_coord'].tolist() == [127.0, 127.0, 128.0], out['x_coord'].tolist()
    assert out['y_coord'].tolist() == [37.0, 37.0, 36.0]
    assert sess.calls == ['A동', 'B동'], sess.calls   # 중복 A 는 단 1번만 호출
    print('[geocode] 중복주소 1회호출+좌표매핑 OK (3행이지만 호출 2회)')


# ----------------------------- 4: 키오류 즉시중단 -----------------------------
def test_attach_401_abort():
    df = pd.DataFrame({'addr': ['키오류주소', '다음주소']})
    sess = FakeSession(lambda a: FakeResp(status=401))
    _patch_session(sess)
    try:
        geocode.attach_coords(df.copy(), addr_col='addr', cfg=FakeCfg(), pause=0)
        assert False, '401 인데 RuntimeError 가 발생하지 않았다'
    except RuntimeError as e:
        assert '401' in str(e), str(e)
    assert sess.calls == ['키오류주소'], sess.calls   # 첫 키오류에서 바로 멈춤
    print('[geocode] 키오류(401) 즉시 중단(RuntimeError) OK - 둘째주소 호출 안함')


# ----------------------------- 5: 한도초과 부분결과 -----------------------------
def test_attach_429_partial():
    df = pd.DataFrame({'addr': ['정상동', '한도초과동', '도달못함동']})

    def script(a):
        if a == '정상동':
            return FakeResp({'documents': [{'x': '127.0', 'y': '37.0'}]})
        if a == '한도초과동':
            return FakeResp(status=429)
        return FakeResp({'documents': [{'x': '9', 'y': '9'}]})   # 도달하면 안됨

    sess = FakeSession(script)
    _patch_session(sess)
    out = geocode.attach_coords(df.copy(), addr_col='addr', cfg=FakeCfg(), pause=0)
    g = out.set_index('addr')
    assert g.loc['정상동', 'x_coord'] == 127.0           # 중단 전 성공분 보존
    assert pd.isna(g.loc['도달못함동', 'x_coord'])        # 중단으로 미처리
    assert '도달못함동' not in sess.calls, sess.calls     # 429 에서 break
    print('[geocode] 한도초과(429) 중단+부분결과 보존 OK (정상동 유지, 이후 미호출)')


# ----------------------------- 6: 주소못찾음 -----------------------------
def test_attach_no_result():
    df = pd.DataFrame({'addr': ['있는동', '없는동']})

    def script(a):
        return (FakeResp({'documents': [{'x': '127.0', 'y': '37.0'}]}) if a == '있는동'
                else FakeResp({'documents': []}))

    sess = FakeSession(script)
    _patch_session(sess)
    out = geocode.attach_coords(df.copy(), addr_col='addr', cfg=FakeCfg(), pause=0)
    g = out.set_index('addr')
    assert g.loc['있는동', 'x_coord'] == 127.0
    assert pd.isna(g.loc['없는동', 'x_coord'])    # 못찾음 = 거짓좌표 대신 NaN
    print('[geocode] 주소못찾음 -> 해당행 좌표 NaN OK (조용한 위장 없음)')


# ----------------------------- 7: 좌표있으면 건너뜀 -----------------------------
def test_attach_skip_when_coords():
    df = pd.DataFrame({'x_coord': [127.0], 'y_coord': [37.0], 'addr': ['아무']})
    # 호출되면 안되므로 일부러 예외나는 세션을 끼움
    _patch_session(FakeSession(lambda a: (_ for _ in ()).throw(
        AssertionError('좌표가 있는데 카카오를 호출했다'))))
    out = geocode.attach_coords(df.copy(), addr_col='addr', cfg=FakeCfg(), pause=0)
    assert out['x_coord'].iloc[0] == 127.0 and len(out) == 1
    print('[geocode] 좌표존재시 건너뜀 OK (카카오 호출 0회, 키 불필요)')


if __name__ == '__main__':
    try:
        test_geocode_one_parse()
        test_geocode_one_empty()
        test_attach_dedup()
        test_attach_401_abort()
        test_attach_429_partial()
        test_attach_no_result()
        test_attach_skip_when_coords()
        print('GEOCODE OK')
    finally:
        geocode.requests.Session = _ORIG_SESSION   # 전역 원복
