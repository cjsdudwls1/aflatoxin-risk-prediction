# Figure captions — EXP-010 (5-fold CV, FSB journal format)

저널 규정상 그림 자체에는 제목을 넣지 않고, 본문 끝 **Figure captions** 섹션에 캡션을 모은다.
아래는 제출용 영문 캡션(번호는 논문에서 선택·확정 시 부여) + 한글 한 줄 설명.
모든 그림: TIFF, 컬러(RGB), 300 dpi, 84.0 mm(1-column). 색은 색맹 친화(Okabe-Ito) 팔레트 + 선스타일/마커 이중 구분(흑백 복사 대비).

> 통계 표기 일관성: ROC-AUC = 0.662 ± 0.050, PR-AUC(AP) = 0.129 ± 0.035 는 cv_results.json
> 요약값(표본표준편차 ddof=1)과 동일하게 그림 legend 에 표기했다.

---

## CV 요약 (둘 중 택1 권장)

**`Fig_CV_metrics_dotplot.tiff`** *(권장)*
> Five-fold cross-validation performance of the hybrid Conv1D–BiLSTM–attention model. For each metric (F2, ROC-AUC, PR-AUC, precision, and recall), open blue circles denote the five individual fold scores and red diamonds denote the mean ± standard deviation across folds; the value above each diamond is the mean.

한글: 5개 지표별로 fold 점 5개(파란 빈 원) + 평균±표준편차(주홍 다이아몬드). CV 결과를 한 장에 요약하는 메인 그림.

**`Fig_CV_metrics_lines.tiff`** *(대안)*
> Per-fold values of the five evaluation metrics across the five cross-validation folds. Color, line style, and marker distinguish the metrics (see legend).

한글: x축이 fold(1–5), 5개 지표를 색·선스타일·마커로 구분한 선도표. 도트플롯의 대안.

---

## 진단 곡선

**`Fig_ROC.tiff`**
> Receiver operating characteristic (ROC) curves over the five cross-validation folds. Thin gray lines, individual folds; light blue band, ± 1 standard deviation about the mean; thick blue line, mean ROC (area under the curve, AUC = 0.662 ± 0.050); dashed diagonal, chance level.

한글: fold별 ROC(회색) + 평균(굵은 파랑) + ±1SD 음영(연하늘) + 무작위 대각선.

**`Fig_PR.tiff`**
> Precision–recall curves over the five cross-validation folds. Thin gray lines, individual folds; thick blue line, curve pooled over all five test folds (average precision, AP = 0.129 ± 0.035); dashed red line, baseline positive prevalence (0.39%).

한글: fold별 PR(회색) + pooled(굵은 파랑) + 양성비율 baseline(주홍 점선). 극불균형에서 ROC보다 정직한 그림.

**`Fig_threshold_sweep.tiff`**
> Test-set F2 (blue), recall (orange), and precision (green) as a function of the decision threshold, averaged over the five folds. The dotted vertical line marks the mean F2-optimal threshold (0.98) determined on the validation sets.

한글: 임계치에 따른 F2(파랑)/재현율(주황)/정밀도(초록), test 5fold 평균 + 선택 임계치(0.98) 표시. 왜 임계치가 높은지 보여줌.

**`Fig_prob_dist.tiff`**
> Distribution of predicted probabilities for negative (n = 80,567; light blue bars) and positive (n = 316; orange outline) samples pooled over the five test folds, on a logarithmic count scale. The dotted vertical line marks the selected decision threshold (0.98).

한글: 클래스별 예측확률 분포(로그 카운트). 양성이 0 근처와 0.99 근처로 갈리는 = 극단 high-tail 에서만 양성을 잡는 모델 특성을 가장 잘 보여줌.

**`Fig_lift_gain.tiff`**
> Cumulative gain curve pooled over the five cross-validation test folds. Samples are ranked by predicted probability (x-axis); the y-axis shows the fraction of all positive samples captured. The dashed diagonal is the random-selection baseline.

한글: 점수 상위부터 선별 시 양성 포착 비율(누적 이득). 실무적 선별 효용 — 상위 20%에서 양성 ~48% 포착.

**`Fig_calibration.tiff`** *(우선순위 낮음)*
> Reliability (calibration) curve pooled over the five cross-validation test folds using quantile binning. Open circles, observed positive fraction versus mean predicted probability per bin; dashed diagonal, perfect calibration.

한글: 예측확률 보정도. 극불균형이라 확률이 전반적으로 낮게(보수적) 나오는 걸 보여주나 정보량은 적음 — 보충 그림(supplementary) 후보.

---

## 5-fold pooled 혼동행렬

**`Fig_confusion_matrix.tiff`**
> Confusion matrix pooled over the five cross-validation test folds (n = 80,883; 316 positive samples), obtained with the per-fold F2-optimal threshold selected on the validation set. Cells are shaded by row-normalized (per true class) proportion (blue colormap); counts and within-class percentages are annotated.

한글: 5fold 합산 혼동행렬(TP 80 / FN 236 / FP 264 / TN 80,303). 행(실제 클래스) 기준 음영(파란 계열).
