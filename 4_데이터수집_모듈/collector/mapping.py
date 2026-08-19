# -*- coding: utf-8 -*-
"""
[단일 진실원천] 농업기상 API 필드 <-> 모델 핵심 컬럼 매핑

배경:
  원본 보고서의 79컬럼 농업기상정보.csv 는 상세 엔드포인트(InsttWeather/연별
  오퍼레이션)에서 수집됐으나, 해당 엔드포인트는 2026 현재 전부 404(폐기)됨.
  생존한 일별 엔드포인트는 GnrlWeather/getWeatherMonDayList3 (18필드) 하나뿐.

검증 방법:
  같은 관측소(026002D003)·같은 날(2024-07-01)의 원본 CSV 값과 API 응답 값을
  직접 대조하여 매핑을 확정함. (tmp/probe_map.py, probe_s2.py 참조)

매핑 신뢰도(30일 회귀대조로 확정 · 026002D003 · 2024-06):
  OK    : 반올림 수준 일치 (7종). 작년 CSV·API 가 각각 소수1자리로 반올림하여
          같은 실측이라도 ±0.1 산발차가 생긴다. afp(강수)는 큰비 날 0.5단위차 1건.
  APPROX: 정의/결측 차이가 있는 같은 물리량 (습도/일조/토양수분 3종)
"""

# API 응답 필드명 -> 모델이 쓰는 핵심 컬럼명
# (key=API필드, value=(모델컬럼, 한글의미, 신뢰도))
WEATHER_FIELD_MAP = {
    'temp':        ('tmprt_150',     '1.5m 평균기온',   'OK'),
    'hghst_Artmp': ('tmprt_150Top',  '1.5m 최고기온',   'OK'),
    'lowst_Artmp': ('tmprt_150Lwet', '1.5m 최저기온',   'OK'),
    'hum':         ('hd_150',        '1.5m 상대습도',   'APPROX'),  # 78.6 vs 78.5 (반올림차)
    'wind':        ('arvlty_300',    '풍속',            'OK'),
    'max_Wind':    ('arvlty_300Top', '최대풍속',        'OK'),
    'rn':          ('afp',           '강수량',          'OK'),     # 대체로 일치, 큰비 날 0.5단위차 가능
    'sun_Time':    ('sunshn_Time',   '일조시간',        'APPROX'),  # 원본CSV 91.3% 결측, API는 값 제공
    'srqty':       ('solrad_Qy',     '일사량',          'OK'),
    'soil_Wt':     ('soil_Mitr_10',  '토양수분(근사)',  'APPROX'),  # 36.8 vs 27.7, API는 깊이별 없이 단일값
}

# API 키 식별 필드 -> 모델 키 컬럼
WEATHER_KEY_MAP = {
    'stn_Cd': 'obsr_Spot_Code',
    'date':   'date_Time',
}

# 모델이 60일 결합에 사용하는 핵심 기상 컬럼 순서 (보고서 WEATHER_COLS 동일)
MODEL_WEATHER_COLS = [
    'tmprt_150', 'tmprt_150Top', 'tmprt_150Lwet', 'hd_150',
    'arvlty_300', 'arvlty_300Top', 'afp', 'sunshn_Time', 'solrad_Qy', 'soil_Mitr_10',
]

# 회귀대조에서 '반올림 수준(±0.1)'으로 일치하는 컬럼(검수 기준).
# 주의: 완전 동일이 아니라 소수1자리 반올림 산발차가 있음(afp 는 0.5단위차 1건 관측).
REGRESSION_EXACT_COLS = [
    'tmprt_150', 'tmprt_150Top', 'tmprt_150Lwet',
    'arvlty_300', 'arvlty_300Top', 'afp', 'solrad_Qy',
]
# 차이가 존재하여 회귀대조에서 사유를 명시하는 컬럼
REGRESSION_DIFF_COLS = {
    'hd_150':       '반올림 수준 미세차 (API hum 소수1자리 차이)',
    'sunshn_Time':  '원본 CSV 91.3% 결측 / API sun_Time 은 값 제공 → 모듈이 더 충실',
    'soil_Mitr_10': 'API soil_Wt 는 깊이별 미구분 단일값 → 10cm 정의와 불일치(근사)',
}


def map_weather_item(item: dict) -> dict:
    """API item(dict, 태그->값) 을 모델 스키마(키+핵심10컬럼) 로 변환."""
    out = {}
    for api_f, model_c in WEATHER_KEY_MAP.items():
        out[model_c] = item.get(api_f)
    for api_f, (model_c, _kor, _conf) in WEATHER_FIELD_MAP.items():
        out[model_c] = item.get(api_f)
    return out


# =====================================================================
# 토양 (SoilEnviron/SoilExam V2)
# =====================================================================
# 실측(2026-06, 부산 기장군 시료)으로 확정한 응답 14필드 중,
# 원본 CSV 와 이름이 다른 것은 2개뿐. 나머지는 그대로 보존한다.
SOIL_FIELD_MAP = {
    'Stdg_Cd': 'BJD_Code',  # 법정동코드(法定洞 코드): 행정구역 식별 10자리
    'ELCD':    'SELC',      # 전기전도도(電氣傳導度): 토양 속 염류 농도 지표(EC)
}

# 토양 최종 컬럼 순서(원본 CSV 기준). PNU 조회시 BJD_Code 대신 PNU_Cd 가 들어온다.
SOIL_MODEL_COLS = [
    'BJD_Code', 'Any_Year', 'Exam_Day', 'Exam_Type', 'PNU_Nm',
    'ACID', 'VLDPHA', 'VLDSIA', 'OM',
    'POSIFERT_MG', 'POSIFERT_K', 'POSIFERT_CA', 'SELC',
]


def map_soil_item(item: dict) -> dict:
    """토양 API item 을 원본 CSV 컬럼명으로 변환.

    - 'No'(페이지 순번)은 모델에 불필요하므로 제거
    - Stdg_Cd -> BJD_Code, ELCD -> SELC 만 개명, 나머지는 그대로
    - getSoilExam(PNU) 응답의 PNU_Cd 는 매핑표에 없으므로 그대로 보존
    """
    out = {}
    for k, v in item.items():
        if k == 'No':
            continue
        out[SOIL_FIELD_MAP.get(k, k)] = v
    return out
