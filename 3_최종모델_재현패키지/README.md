# 🌾 아플라톡신 검출 예측 모델 — 재현 패키지 (EXP-011 최종본)

> 곡물·견과류 등 식품의 **아플라톡신 부적합(초과 검출)** 여부를,
> 검사 직전 **50일치 농업기상 시계열 + 검사 메타정보**로 예측하는 딥러닝 모델입니다.
> 이 폴더 하나만 받으면 **최종 모델(EXP-011)을 처음부터 다시 학습**해 동일한 결과를 재현할 수 있습니다.

---

## 📌 한눈에 보기

| 항목 | 내용 |
|------|------|
| **재현 대상** | EXP-011 (`Conv1D padding: causal → same` 적용한 최종 모델) |
| **모델 구조** | Conv1D → BiLSTM → Attention 하이브리드 (시계열) + Dense (표형 정보) |
| **데이터** | 80,883건 × 613컬럼 (양성=부적합 316건, 약 0.39% 극단 불균형) |
| **검증 방식** | 5-폴드 교차검증(StratifiedKFold), 폴드마다 train/val/test 분리 |
| **학습 환경** | Modal 클라우드 GPU(NVIDIA L4) 1장, 약 **33분** 소요 |
| **핵심 성능(Pooled)** | F2 **0.229** · ROC-AUC **0.657** · Recall **0.237** · Precision **0.199** |

> 💡 **Pooled(통합) 성능**이란? 5개 폴드의 test 예측을 **전부 한데 모아 한 번에** 지표를 계산한 값입니다.
> 폴드별로 따로 재서 평균 내는 방식보다, 실제 운영에서 전체 데이터를 한 번에 판정하는 상황에 더 가깝습니다.

---

## 🗂️ 폴더 구조

```
재현패키지_아플라톡신_EXP-011/
├── README.md                     ← 지금 보는 문서
│
├── data/
│   └── processed/
│       └── df_enhanced.parquet   ← 모델 입력 데이터 (51MB, 80,883×613) ★완전 재현의 출발점
│
├── src/                          ← 실행 인프라
│   ├── modal_run.py              ← Modal GPU에서 노트북을 실행하고 결과를 자동 저장
│   ├── requirements.txt          ← GPU 이미지에 설치될 패키지 목록(버전 고정)
│   ├── font_helper.py            ← 그래프 한글 폰트 도우미
│   ├── collect_env.py            ← 실행 환경 정보 수집(기록용)
│   ├── new_experiment.py         ← (참고) 새 실험 폴더 생성기
│   └── update_leaderboard.py     ← (참고) 실험 성적표 갱신기
│
├── experiments/
│   └── EXP-011_cv5_same_padding/
│       └── notebook.ipynb        ← ★재학습에 올라갈 입력 노트북(출력 비워둔 정본)
│
├── expected_results/             ← 「정답지」 — 내가 돌렸을 때 나온 결과 (대조용)
│   ├── cv_results.json           ← 폴드별 상세 지표
│   ├── cv_summary.csv            ← 폴드 요약
│   ├── cv_results.csv
│   ├── cv_metrics.png            ← 폴드별 지표 그림
│   ├── run_metrics.json          ← 실행 환경/GPU 정보
│   ├── execute.log               ← 실제 실행 로그
│   ├── notebook.executed.ipynb   ← 출력까지 포함된 실행본(참고용)
│   └── figures/                  ← 논문/보고서용 그림 + 예측값(npz) + 그림 생성 스크립트
│
└── full_pipeline/                ← 「데이터를 처음부터 만들고 싶을 때」 참고 자료
    ├── notebooks/                ← 전처리→특성공학→모델링 전체 파이프라인 노트북
    │   ├── 1.데이터 전처리+2.EDA.executed.ipynb
    │   ├── 2. 특성공학.executed.ipynb
    │   ├── 3.모델링.executed.ipynb
    │   └── 1.5 대상기간 선정.ipynb
    └── data_collection/          ← 원본 데이터 수집(공공 API) 방법
        ├── reproduce_crawl.py    ← 농업기상 OpenAPI 크롤링 재현 스크립트
        └── 데이터_생성_코드정리.md  ← LIMS+기상 결합 전 과정 코드 정리
```

---

## 🚀 빠른 시작 — 완전 재현 (권장 경로)

> 이 패키지의 **메인 용도**입니다. 동봉된 `df_enhanced.parquet`로 모델을 **5-폴드 전부 다시 학습**합니다.
> GPU가 필요하며, 여기서는 설치가 간편한 **Modal 클라우드 GPU**를 사용합니다.

### 0단계. 준비물
- Python 3.11 (로컬)
- [Modal](https://modal.com) 계정 (무료 가입, 신용카드 등록 시 소액 무료 크레딧 제공)
- 인터넷 연결

### 1단계. 로컬에 Modal 설치 + 로그인
무거운 ML 패키지(TensorFlow 등)는 **로컬에 설치할 필요가 없습니다.** Modal이 GPU 서버 안에서
`src/requirements.txt`를 보고 자동으로 환경을 만들어 줍니다. 로컬엔 `modal`만 있으면 됩니다.

```powershell
# Windows PowerShell
pip install modal
python -m modal setup          # 브라우저로 로그인 (또는: python -m modal token new)
```

### 2단계. 학습 실행
이 폴더(`재현패키지_아플라톡신_EXP-011`)에서 그대로 실행합니다.

```powershell
# Windows PowerShell
$env:EXP_ID  = "EXP-011_cv5_same_padding"
$env:CHANGED = "Conv1D padding: causal -> same"
$env:NOTES   = "EXP-011 재현 실행"
$env:BASE_EXP= "EXP-010_cv5_hybrid"
python -m modal run src/modal_run.py::run_notebook
```

```bash
# macOS / Linux (bash)
EXP_ID=EXP-011_cv5_same_padding \
CHANGED="Conv1D padding: causal -> same" \
NOTES="EXP-011 재현 실행" \
BASE_EXP=EXP-010_cv5_hybrid \
python -m modal run src/modal_run.py::run_notebook
```

**무슨 일이 일어나나요?**
1. 로컬의 `experiments/EXP-011_cv5_same_padding/notebook.ipynb`(입력 코드)와
   `data/processed/df_enhanced.parquet`(데이터)를 GPU 서버로 업로드합니다.
2. GPU(L4) 위에서 노트북을 처음부터 끝까지 실행합니다 — **5개 폴드를 차례로 학습**(약 33분).
3. 결과가 다시 로컬 `experiments/EXP-011_cv5_same_padding/` 폴더로 내려옵니다.

### 3단계. 결과 확인
실행이 끝나면 `experiments/EXP-011_cv5_same_padding/` 안에 아래가 생깁니다.

| 파일 | 설명 |
|------|------|
| `notebook.executed.ipynb` | 출력 셀까지 포함된 실행 결과 노트북 |
| `cv_results.json` | 폴드별 ROC-AUC / F2 / 혼동행렬 등 |
| `execute.log` | 실행 로그 (GPU 정보 포함) |
| `manifest.json` | 실험 메타정보 요약 |

➡️ 새로 나온 `cv_results.json`을 **`expected_results/cv_results.json`(정답지)** 과 비교하세요.
숫자가 (거의) 같으면 재현 성공입니다. → 아래 [재현 성공 판정 기준](#-재현-성공-판정-기준) 참고.

---

## 🎯 기대 결과 (정답지)

### Pooled 성능 — 우리 보고서/논문의 대표 수치
5개 폴드 test 예측을 모두 합쳐(n = 80,883, 양성 316건) 한 번에 계산한 값입니다.

| 지표 | 값 | 의미(쉽게) |
|------|----|-----------|
| **F2-score** | **0.229** | Recall을 4배 더 중시한 종합 점수(놓침을 더 싫어함) |
| **ROC-AUC** | **0.657** | 양성/음성을 점수로 줄 세웠을 때 순서가 맞을 확률(0.5=찍기) |
| **PR-AUC** | **0.110** | 극단 불균형에서 정밀도·재현율 균형(기준선 ≈ 0.0039) |
| **Recall(재현율)** | **0.237** | 실제 부적합 316건 중 **75건 탐지**(241건은 놓침) |
| **Precision(정밀도)** | **0.199** | 부적합 경보 377건 중 **75건이 적중**(302건은 헛경보) |

**Pooled 혼동행렬**

|              | 예측: 부적합 | 예측: 적합 |
|--------------|------------|-----------|
| **실제: 부적합** | TP = 75 | FN = 241 |
| **실제: 적합**   | FP = 302 | TN = 80,265 |

> 0.39%라는 극단적 불균형(1000건 중 약 4건만 부적합)에서, "전부 적합"으로 찍으면 Recall은 0입니다.
> 이 모델은 **실제 부적합의 약 1/4을 건져 올리면서** 헛경보를 통제하는 균형점을 학습했습니다.

### 폴드별 성능 (변동성 참고용)
`expected_results/cv_results.json` 요약 — 폴드마다 따로 계산한 값의 평균입니다.

| 지표 | 폴드 평균 | 표준편차 | 최소~최대 |
|------|----------|---------|----------|
| ROC-AUC | 0.660 | 0.049 | 0.619 ~ 0.738 |
| F2 | 0.231 | 0.044 | 0.175 ~ 0.283 |
| PR-AUC | 0.124 | 0.038 | 0.091 ~ 0.180 |
| Recall | 0.238 | 0.042 | 0.172 ~ 0.270 |
| Precision | 0.242 | 0.103 | 0.106 ~ 0.354 |

> **Pooled와 폴드평균이 왜 조금 다른가요?** 폴드별 양성 수가 60여 건으로 매우 적어,
> 폴드마다 지표가 출렁입니다(특히 Precision). 그래서 **대표값은 Pooled**를 쓰고,
> 폴드평균은 "얼마나 흔들리는지"를 보는 보조 지표로만 봅니다.

---

## ✅ 재현 성공 판정 기준

GPU 학습은 시드를 고정해도(`PYTHONHASHSEED=42`, `TF_DETERMINISTIC_OPS=1`) cuDNN/하드웨어 차이로
**소수점 아래 미세한 차이**가 날 수 있습니다. 아래 정도면 **재현 성공**으로 봅니다.

- ✔ Pooled ROC-AUC가 **0.64 ~ 0.68** 범위
- ✔ Pooled F2가 **0.20 ~ 0.26** 범위
- ✔ 혼동행렬 TP가 **60 ~ 90** 사이, 전반적 경향(소수 탐지·다수 놓침)이 동일
- ✔ 5개 폴드 모두 정상 학습 완료(`execute.log`에 오류 없음)

> 숫자가 완전히 동일하지 않아도 정상입니다. **경향과 범위**가 맞으면 같은 모델을 재현한 것입니다.

---

## 🧠 모델이 어떻게 작동하나요? (쉽게)

입력은 두 갈래입니다.

1. **시계열 가지** — 검사 직전 **50일치 날씨**(기온·습도·일사량·결로·강수·토양수분 등 10종)
2. **표형 가지** — 검사 종류, 식품 유형, 산지, 검사 목적 등 **표(table) 형태 정보**

시계열 가지는 세 단계를 거칩니다.

- **Conv1D (1차원 합성곱)** — 날씨의 **짧은 구간 패턴**을 잡습니다.
  예: "며칠간 고온다습이 이어진 구간". 커널(작은 창)을 시간축으로 훑으며 국소 모양을 요약합니다.
  - 🔑 **EXP-011의 핵심 변경**: 패딩을 `causal`(과거만 봄) → `same`(앞뒤 모두 봄)으로 바꿨습니다.
    이번 문제는 "이미 끝난 닫힌 50일 창"을 통째로 보고 한 번 판정하는 구조라, 과거만 보게 막는
    `causal` 제약은 불필요한 족쇄였습니다. 뒤따르는 BiLSTM이 어차피 양방향이라 방향을 맞춰준 것입니다.
- **BiLSTM (양방향 LSTM)** — 50일 흐름을 **앞에서 뒤로, 뒤에서 앞으로** 모두 읽어 시간적 맥락을 학습합니다.
- **Attention (어텐션)** — 50일 중 **어느 날이 결정에 중요한지** 가중치를 매겨 핵심 시점에 집중합니다.

두 가지(시계열 요약 + 표형 정보)를 합친 뒤 마지막 Dense 층이 **부적합 확률**을 출력합니다.
극단 불균형 대응을 위해 양성에 가중치를 크게 준 손실함수(weighted BCE, `pos_weight ≈ 293`)를 씁니다.

---

## 🔬 처음부터 재현하기 — 원본 데이터 수집·가공 (심화)

> ⚠️ 이 절은 **선택**입니다. `df_enhanced.parquet`가 이미 동봉돼 있어 완전 재현에는 필요 없습니다.
> "데이터 자체를 0부터 다시 만들고 싶다"면 아래를 따르세요. 자료는 `full_pipeline/`에 있습니다.

### 데이터 계보 (어디서 와서 어떻게 합쳐지나)

```
[원본 ①] LIMS 검사결과 Excel 여러 개   ──┐  (식약처/식정원 내부 데이터)
                                          │
[원본 ②] 농업기상 OpenAPI (공공데이터)  ──┼─▶ 좌표·검사일 기준 60일 기상 결합
[원본 ③] 검사주소 → 좌표 지오코딩       ──┘        │
                                                   ▼
              df_통합_LIMS_기상정보_일조포함_결합_60일_X.pkl / _y.pkl
                                                   │
       (NB1) 1.데이터 전처리+2.EDA  →  df_fixed.pkl.gz
                                                   │
       (NB2) 2.특성공학           →  df_enhanced.parquet  ★ 이 패키지에 동봉된 파일
                                                   │
       (NB3) 3.모델링 (= EXP-011)  →  학습·평가
```

### 원본 데이터 출처

| 원본 | 출처 / 받는 방법 |
|------|-----------------|
| **① 아플라톡신 검사결과 (LIMS)** | 식약처/식품안전정보원 내부 검사 시스템(LIMS)에서 추출한 Excel. **공개 데이터가 아니므로 기관을 통해 확보해야 합니다.** 가공 방법은 `full_pipeline/data_collection/데이터_생성_코드정리.md` 참조 |
| **② 농업기상정보** | **공공데이터포털 [data.go.kr](https://www.data.go.kr)** > 국립농업과학원 **농업기상 기본 관측데이터 조회 OpenAPI**(제공기관 코드 **1390802**, AgriWeather). 활용신청 후 본인 **서비스키** 발급 → `full_pipeline/data_collection/reproduce_crawl.py`로 크롤링 |
| **③ 주소→좌표** | 검사 주소를 지오코딩하여 위도/경도로 변환(주소-좌표 DB). 변환 후 검사지점에서 가장 가까운 농업기상 관측소를 cKDTree로 매칭 |

**농업기상 OpenAPI 두 가지 엔드포인트**
- `getObsrSpotList` → 관측소(지점) 목록
- `.../InsttWeather/getWeatherYearDayList` → 지점·연도별 일 단위 기상 (변수 10종)

**기상변수 10종**: 평균기온(`tmprt_150`)·최고기온(`tmprt_150Top`)·최저기온(`tmprt_150Lwet`)·습도(`hd_150`)·
결로(`arvlty_300`,`arvlty_300Top`)·강수(`afp`)·일조시간(`sunshn_Time`)·일사량(`solrad_Qy`)·토양수분(`soil_Mitr_10`)

크롤링 재현(서비스키만 본인 것으로 교체):
```bash
cd full_pipeline/data_collection
python reproduce_crawl.py spots          # 관측소 목록 크롤링/비교
python reproduce_crawl.py weather 2024   # 특정 연도 기상 샘플 크롤링
```

### 📝 데이터 재수집 시 알아둘 차이점 (중요)

다른 시점에 원본 데이터를 **다시 수집해 대조**한 결과입니다. 대부분 일치하며, 차이는 아래뿐입니다.
원본 데이터는 시간이 지나면 **제공기관 쪽에서 사후 보정·갱신**될 수 있어 생기는 자연스러운 현상입니다.

- **[일별기상]** 2024년 7월을 제외하면 **전 항목 일치**. 2024년 7월 **140건만 미세 차이**(기온 0.1℃·습도 0.1% 수준)
  → 제공기관의 **사후 품질보정으로 추정**.
  - 대조 세팅: 데이터 양이 많아 ▲7개 변수(평균·최고·최저기온, 습도, 일사량, 결로시간, 강수량) ▲5개 지점만
    ▲1·7월 데이터만 = 약 **2.4만 셀** 대조.
- **[관측소]** 지점코드 **전부 일치**, 지점명·기후대·관측개시일 **오류 0건**. 주소 3건·좌표 1곳·고도 1건만 다름
  → 수집 이후 제공기관이 **지점정보를 갱신한 것으로 추정**.
- **[토양검정]** 강릉, 부산 기장 **2개 지역만 대조**. 결과: **2022년도 데이터 없음**, 그 외 전부 일치
  (토양검정은 **최근 4개년도 윈도우만 유지**되는 것으로 보임).
- **[황사]** 기상청 포털 서버 에러로 회원가입 20회 넘게 실패 → **다음에 재시도 예정**(현재 미대조).

> 이 미세 차이는 모델 입력(`df_enhanced.parquet`)에 사실상 영향을 주지 않는 수준입니다.
> 완전한 동일 재현을 원하면 **동봉된 `df_enhanced.parquet`를 그대로 사용**하시면 됩니다.

---

## 🖼️ (선택) 그림만 다시 그리기

GPU 재학습 없이도, 이미 저장된 예측값(`expected_results/figures/predictions*.npz`)으로
논문/보고서용 그림(ROC, PR, 보정곡선, 임계값 스윕 등)을 재생성할 수 있습니다.

```bash
cd expected_results/figures
python make_curve_figs.py     # ROC/PR/lift-gain/calibration/prob_dist/threshold_sweep
python make_journal_figs.py   # 추가 저널용 그림
```
> 한글 그래프가 깨지면 `fonts-nanum`(Linux) 또는 맑은 고딕(Windows) 설치 후 다시 실행하세요.

---

## 🔧 문제 해결 / FAQ

**Q. `python -m modal run` 했더니 "이미 실행된 결과가 있습니다(FileExistsError)"가 떠요.**
A. `experiments/EXP-011_cv5_same_padding/`에 `notebook.executed.ipynb`가 이미 생긴 경우입니다(이전 실행 흔적).
   그 폴더의 실행 산출물(`notebook.executed.ipynb`, `execute.log`, `run_metrics.json`, `manifest.json`)만
   지우고 다시 실행하거나, `EXP_ID`를 `EXP-011_repro` 같은 새 이름으로 바꿔 실행하세요.

**Q. "입력 데이터가 없습니다(df_enhanced.parquet)" 오류가 나요.**
A. 반드시 **이 패키지 최상위 폴더에서** `python -m modal run src/modal_run.py::run_notebook`을 실행하세요.
   `modal_run.py`는 `src/`의 부모를 프로젝트 루트로 보고 `data/processed/df_enhanced.parquet`를 찾습니다.

**Q. Modal 말고 내 GPU(로컬/서버)에서 돌리고 싶어요.**
A. `src/requirements.txt`로 환경을 만든 뒤(`pip install -r src/requirements.txt`),
   `experiments/EXP-011_cv5_same_padding/notebook.ipynb`를 Jupyter로 직접 실행하면 됩니다.
   단, 노트북은 같은 폴더에 `df_enhanced.pkl.gz`(또는 parquet)가 있다고 가정하므로 경로만 맞춰 주세요.
   `requirements.txt`는 **TensorFlow 2.15.1 / numpy 1.26.4 / scikit-learn 1.5.2**로 버전을 고정해 두었습니다(호환성 중요).

**Q. 결과 숫자가 정답지와 완전히 똑같지 않아요.**
A. 정상입니다. 위 [재현 성공 판정 기준](#-재현-성공-판정-기준)의 범위 안이면 성공입니다.

**Q. 33분보다 오래/짧게 걸려요.**
A. GPU 종류·대기열에 따라 달라집니다. L4 기준 약 33분이며, 더 빠른 GPU면 단축됩니다.

---

## 📎 핵심 사양 요약

- **실험 ID**: `EXP-011_cv5_same_padding` (base: `EXP-010_cv5_hybrid`, 변경점: `Conv1D padding: causal → same`)
- **데이터**: `df_enhanced.parquet` — 80,883행 × 613열, 양성 316건(0.39%)
- **시계열 입력**: 50 timestep × 10 변수 / **표형 입력**: 인코딩된 메타 특성
- **교차검증**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, 폴드마다 train/val/test 분리(임계값은 val→test 격리)
- **손실/불균형**: weighted BCE(`pos_weight ≈ neg/pos ≈ 293`), 양성만 시계열 증강(`n_copies=2`)
- **학습**: epochs 50, batch 256, Adam(lr 1e-4, clipnorm 1.0), EarlyStopping/ReduceLROnPlateau(`val_pr_auc` 기준)
- **재현성 환경변수**: `PYTHONHASHSEED=42`, `TF_DETERMINISTIC_OPS=1`, `TF_CUDNN_DETERMINISTIC=1`
- **실행 환경**: Modal GPU NVIDIA L4, Python 3.11, 약 1,988초(≈33분)

---

*문의나 막히는 부분이 있으면 `expected_results/execute.log`(실제 실행 로그)와
`full_pipeline/data_collection/데이터_생성_코드정리.md`(데이터 가공 전 과정)를 함께 참고하세요.* 🍀
