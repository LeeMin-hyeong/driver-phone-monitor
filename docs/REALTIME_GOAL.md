# 24 FPS 및 기존 평가 성능 유지 — 진행 기록

## 완료 조건

- 동일한 공유 데이터: train 911장 / test 146장. 기본 모델은 보존한다.
- 기준 지표: Accuracy 129/146 (88.356%), Precision 72/84 (85.714%), Recall 72/77 (93.506%), F1 144/161 (89.441%). 네 지표 모두 기준 이상이어야 한다.
- 모델·임계값·학습 횟수 선택은 train 내부 기존 fit 619장 / validation 292장으로 한다. test로 임계값을 조정하지 않는다.
- 모델 상주, batch 1, 입력마다 새 분류를 수행한다. 프레임 누락·이전 결과 재사용·특징 캐시로 FPS를 부풀리지 않는다.
- 사용자가 **최초 입력을 640으로 가정해도 됨**을 명시했다. 현재 평가는 종횡비 유지·패딩한 640×640 JPEG를 실제 입력으로 사용한다. 상류의 고해상도→640 변환 시간은 제외하고, 640 파일 읽기·디코딩부터 포함한다. 정확도도 이 입력으로 평가한다.
- 24 FPS 목표는 평균 41.67ms 이하뿐 아니라 반복 측정과 p95 지연도 확인한다.
- 현재 사진 평가는 과거 실험에도 사용됐으므로 독립적인 외부 일반화 검증으로 표현하지 않는다.

## 확인된 근거

- YOLO 640 + 재검출 가지치기: 330.8ms, 3.02 FPS. 기준 성능의 모든 지표를 유지하지 못함.
- ExtraTrees batch 1 추가 측정: n_jobs=4 평균 40.99ms, n_jobs=1 평균 13.31ms. 122개 입력의 점수 최대 차이 2.22e-16. 근거: `evidence/tree_threads_probe.json`.

## 후보 1: 운전자 영역 이미지 단독 — 성능 부족

- `training/train_fast_roi.py`: 운전자 쪽 고정 영역 (.42, .20, 1, 1), 224 입력, ImageNet ResNet18.
- YOLO/MediaPipe 호출 없이 이미지에서 직접 이진 분류.
- train 내부 25 epoch 검증으로 epoch·threshold 선택 후 전체 train 재학습.
- 산출물: `artifacts/fast_roi_v1`.
- 실험용 추론: `fast_roi.py`. 기본 `main.py` 모델 변경 없음.
- 속도/평가: `experiments/evaluate_fast_roi.py`. 원본 파일, JPEG 축소 디코딩, 디코딩된 1280 프레임을 별도로 보고한다.
- 아직 목표 달성은 검증되지 않았다. 고정 영역 방식은 운전자 위치가 다른 카메라에서 별도 검증이 필요하다.

최고 검증 F1 80.13%. 디코딩된 프레임 82.59 FPS는 속도 실험 결과이며 성능 기준을 만족한 모델의 FPS가 아니다.

## 현재 후보: GPU 자세 + 약한 휴대폰 후보 + 손목 이미지 보완

- train 입력: `artifacts/input640/images`, 원본과 파생 입력 해시는 `artifacts/input640/manifest.json`.
- 운전자 쪽 영역 (.35, .20, 1, 1)에 YOLO26m 640 1회, conf 0.01. 전체 640에 YOLO26n-pose 1회.
- MediaPipe·손가락 추정·손목 추가 YOLO 호출을 제거했다.
- 객체/손목 관계 특징을 ExtraTrees 200개, min_samples_leaf=10으로 분류한다.
- 트리 실행은 `fast_forest.py`로 벡터화. 911개 train 특징의 점수 차이 최대 2.22e-16, 모든 판정 동일. 평균 8.00→0.196ms. 분기 경계의 float32 값에 대한 단위 테스트도 통과했다.
- FP16 + 벡터화 검증 실행: 평균 39.11ms / 25.57 FPS, p95 59.66ms. 평균만 통과했으며 안정 지연 기준은 미달이다.
- 해당 모델 검증 지표: Accuracy 93.49%, Precision 97.12%, Recall 90.00%, F1 93.43%.
- 운전자 양손 주변 224×224 crop 2개를 합친 448×224 이미지를 별도 ResNet18에 입력하는 보완 모델을 학습했다. 이미지 단독 검증 최고 F1 87.84%.
- train validation에서 고정한 결합: 1차 score ≥0.25면 양성, 그 외에만 손목 ResNet score ≥0.85면 양성.
- 결합 검증 지표: Accuracy **94.18%**, Precision **94.63%**, Recall **94.00%**, F1 **94.31%**, TN134/FP8/FN9/TP141.
- 선택 근거: `artifacts/fast_pose640_sensitive/fusion_selection.json`. 기준 지표 네 가지 충족을 우선하고 F1로 선택했다. 이 단계까지 새 최종 test 추론은 수행하지 않았다.
- TensorRT FP16 정적 batch1 엔진으로 속도 안정화를 검증할 예정이다. `experiments/export_fast_engines.py`.
- 모델 경로: `artifacts/fast_pose640_sensitive/model.joblib`, `artifacts/fast_wrists_v1/model.pt`.
- 최종 평가 도구: `experiments/evaluate_fast_pose.py --input640 --half --engine --fusion`.

## 최종 검증 완료

위 후보 기록 이후 TensorRT 변환·검증과 고정된 결합 모델의 test 평가를 완료했다. test 146장을 19회, 총 2,774회 새 추론한 결과 Accuracy 93.84%, Precision 90.48%, Recall 98.70%, F1 94.41%로 기존 네 지표를 모두 넘었다. TN61/FP8/FN1/TP76이며 모든 반복의 결과가 동일했다.

실제 62.96초 동안 44.06 FPS, 이동 1초 구간 최소 27장, p95 29.49ms를 측정했다. 최대 단일 처리시간은 70.54ms이므로 모든 프레임의 41.67ms 이내 완료를 보장한다는 의미는 아니다. 640 입력·RTX 4050 Laptop·모델 상주 조건에서 지속 처리율 목표를 충족했다.

최종 설명과 실행법은 [REALTIME_REPORT.md](REALTIME_REPORT.md), 측정 근거는 [realtime_test.json](evidence/realtime_test.json), 실행 진입점은 `realtime.py`다. 원본 모델과 공통 데이터 분할은 보존했다.
