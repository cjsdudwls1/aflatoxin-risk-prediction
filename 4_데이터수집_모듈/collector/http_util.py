# -*- coding: utf-8 -*-
"""
공공데이터 OpenAPI 공통 HTTP 유틸.

- 재시도(네트워크 오류/일시적 5xx)
- XML 파싱 -> items(list of dict)
- 404/401/403 은 즉시 실패(재시도 무의미)

원리: 공공데이터포털 API 는 정상 응답을 XML(<response><body><items><item>...)
      로 주고, 활용신청 누락/키 오류/폐기 엔드포인트는 HTML 또는 4xx 를 준다.
      Content-Type 에 'xml' 이 없으면 비정상으로 간주한다.

SSL 검증(verify):
  기본 True(정상 인증서 검증). data.go.kr 은 정상 HTTPS 인증서를 제공하므로
  검증을 켜는 게 옳다(켜야 중간자 변조를 탐지). 사내 프록시/방화벽이 인증서를
  가로채는 폐쇄망에서만 config 의 verify_ssl: false 로 끌 수 있다(그때만 경고 억제).
"""
import time
import requests
import urllib3
import xml.etree.ElementTree as ET

# verify_ssl: false 로 끈 환경에서만 의미. 기본(True)이면 경고가 애초에 없다.
urllib3.disable_warnings()

DEFAULT_TIMEOUT = 60
MAX_RETRY = 3
RETRY_WAIT = 2.0
HARD_FAIL_STATUS = (400, 401, 403, 404)  # 재시도해도 동일한 영구 오류

# 본문(HTTP 200) 안에 들어오는 결과코드 헤더 — API 마다 대소문자가 다르다.
#   토양: <Result_Code>  /  기상·관측소: <result_Code>  /  표준: <returnReasonCode>
RESULT_CODE_TAGS = ('resultCode', 'result_Code', 'Result_Code', 'returnReasonCode')
RESULT_MSG_TAGS = ('resultMsg', 'result_Msg', 'Result_Msg', 'returnAuthMsg')
TOTAL_COUNT_TAGS = ('totalCount', 'total_Count', 'Total_Count')
SUCCESS_CODES = {'200', '00', '000', '0000', '0', 'NORMAL_SERVICE', 'NORMAL SERVICE'}


class ApiError(RuntimeError):
    """API 응답이 비정상(비XML/에러코드/폐기 엔드포인트)일 때."""


def get_xml(url, params, timeout=DEFAULT_TIMEOUT, max_retry=MAX_RETRY, verify=True):
    """
    GET 후 XML 파싱. (root_element, items_list) 반환.
    items_list 는 [{태그: 값, ...}, ...].
    영구 오류는 ApiError 즉시 raise, 일시 오류는 재시도.
    verify: SSL 인증서 검증(기본 True). 폐쇄망에서만 False 로.
    """
    last_err = None
    for attempt in range(1, max_retry + 1):
        try:
            r = requests.get(url, params=params, verify=verify, timeout=timeout)
        except requests.RequestException as e:
            last_err = e
            time.sleep(RETRY_WAIT)
            continue

        ct = r.headers.get('Content-Type', '')
        if r.status_code in HARD_FAIL_STATUS:
            raise ApiError(
                f'HTTP {r.status_code} (영구오류) url={r.url}\n'
                f'  body={r.text[:200]!r}\n'
                f'  -> 엔드포인트 폐기/활용신청 누락/서비스키 권한 확인 필요'
            )
        if r.status_code != 200 or 'xml' not in ct:
            last_err = ApiError(f'HTTP {r.status_code} CT={ct!r} body={r.text[:200]!r}')
            time.sleep(RETRY_WAIT)
            continue

        try:
            root = ET.fromstring(r.content)
        except ET.ParseError as e:
            last_err = ApiError(f'XML 파싱 실패: {e} body={r.text[:200]!r}')
            time.sleep(RETRY_WAIT)
            continue

        check_result(root)  # HTTP 200 이어도 본문 결과코드가 오류면 즉시 raise
        return root, parse_items(root)

    if isinstance(last_err, Exception):
        raise last_err
    raise ApiError('알 수 없는 오류로 응답을 얻지 못했습니다.')


def parse_items(root):
    """root 에서 .//items/item 을 dict 리스트로."""
    items = root.find('.//items')
    if items is None:
        return []
    return [{c.tag: c.text for c in it} for it in items.findall('item')]


def _find_text_ci(root, names):
    """대소문자 무시하고 첫 일치 태그의 text 반환(없으면 None)."""
    low = {n.lower() for n in names}
    for el in root.iter():
        if el.tag.lower() in low:
            return el.text
    return None


def check_result(root):
    """본문 결과코드 검증. 성공코드가 아니면 ApiError.

    원리: 공공데이터 API 는 HTTP 200 이어도 <Result_Code>400</Result_Code>
    처럼 본문 헤더에 오류를 담아 보낼 때가 있다. 이를 잡지 않으면 '0건'을
    정상으로 오인하게 되므로(빈 결과를 성공으로 착각) 명시적으로 실패시킨다.
    """
    code = _find_text_ci(root, RESULT_CODE_TAGS)
    if code is None:
        return  # 결과코드 헤더가 아예 없는 응답은 통과(파싱 단계서 이미 검증)
    if code.strip().upper() in SUCCESS_CODES:
        return
    msg = _find_text_ci(root, RESULT_MSG_TAGS) or ''
    raise ApiError(f'API 본문 오류: 코드={code} 메시지={msg!r}')


def get_total_count(root):
    """페이징용 totalCount(int) 또는 None. (API별 대소문자 차이 흡수)"""
    t = _find_text_ci(root, TOTAL_COUNT_TAGS)
    try:
        return int(t) if t is not None else None
    except (ValueError, TypeError):
        return None
