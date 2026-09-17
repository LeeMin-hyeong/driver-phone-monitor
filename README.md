# 운전자 휴대폰 소지 분류

손별 ResNet 데이터 준비: `.\.gpu\Scripts\python.exe hand_labeler.py`로 224×224 ROI 선택·라벨링 도구를 실행합니다. [사용 방법](docs/HAND_LABELER.md)

## 640 입력 실시간 모델

현재 실시간 판정은 1차 점수가 0.25 미만이면 `S1 + 0.5 × 0.25 × S2 ≥ 0.25`일 때 PHONE입니다. ResNet의 최대 기여도는 0.125이며, 반영 계수는 `configs/realtime_model.json`의 `stage2_weight`로 조정합니다. 1차 양성은 그대로 유지합니다. 아래 사진 평가 수치는 이전 판정 방식의 결과이며, 변경한 방식의 정확도와 실시간 성능은 재평가가 필요합니다.

### OpenCV 웹캠·영상 시각화

```powershell
# 기본 웹캠 (다른 카메라는 --source 1)
.\.gpu\Scripts\python.exe live.py
# 동영상
.\.gpu\Scripts\python.exe live.py --source "영상.mp4"
# 표시 영상을 저장하려면
.\.gpu\Scripts\python.exe live.py --source "영상.mp4" --output "output/annotated.mp4"
```

`Q` 또는 `Esc`로 종료합니다. 초록색은 운전자 자세, 주황색은 기타 자세, 분홍색은 휴대폰 후보, 하늘색은 객체 검출 영역입니다. PHONE/NORMAL, 사용 단계와 점수, 최근 최대 60프레임의 처리 FPS를 표시합니다. 점수는 보정된 확률이 아닙니다.

최종 프레임 판정을 시간 기준으로 안정화합니다. **최근 1초 동안 양성인 시간이 80% 이상이면 PHONE·빨간 테두리**, 이후 **음성이 0.7초 연속 유지되면 NORMAL·초록 테두리**로 복귀합니다. 시작 후 1초간은 초록색입니다. 잠깐의 미탐은 허용하며 모델 점수를 섞거나 평균내지 않습니다. 상단의 큰 판정과 테두리는 안정화 결과, `raw`는 현재 프레임의 원래 판정입니다. 하단에는 양성 시간 비율과 음성 지속 시간을 표시합니다.

웹캠·스트림은 프레임 수신 시각, 영상 파일은 영상 타임스탬프(미지원 시 FPS)를 사용합니다. 각 프레임 판정이 다음 관측까지 유지된 것으로 계산해 FPS 변동을 반영합니다. 입력 간격이 0.5초를 초과하거나 시간이 역행하면 기록을 초기화하여 관측 공백을 양성으로 누적하지 않습니다. 설정은 `temporal.py`의 `PredictionSmoother`에 있습니다. 이는 시간 후처리이며 기존 사진 평가 지표와 별개로 실제 영상에서 확인해야 합니다.

최초 로딩 시 GPU 모델과 두 분기를 워밍업합니다. 입력은 종횡비를 유지해 640×640으로 패딩하며, 화면을 좌우 반전하지 않습니다. 현재 모델은 **이미지 오른쪽 운전석** 조건입니다. 화면의 Loop FPS는 캡처·리사이징·시각화·선택적 저장까지 포함하므로 아래 사진 벤치마크와 다릅니다. 파일 영상은 가능한 속도로 순차 처리하며 저장 MP4는 입력 FPS의 고정 프레임률을 사용합니다. 웹캠 프레임 도착 간격을 그대로 보존하는 녹화 도구는 아닙니다.

창 없이 실행하려면 `--no-display`, 처리 장수 제한은 `--max-frames 100`을 사용합니다.

640×640 입력·RTX 4050 Laptop GPU에서 공통 test 146장 평가 결과 **F1 94.41%, Recall 98.70%**입니다. 2,774회 지속 측정한 실제 처리율은 **44.06 FPS**, 이동 1초 구간 최소 처리량은 **27장**입니다. 모델 상주·워밍업 후 파일 읽기부터 측정하며, 고해상도 입력의 640 변환과 출력 저장은 제외합니다.

```powershell
.\.gpu\Scripts\python.exe realtime.py "640x640_이미지_폴더"
```

- [실시간 최종 보고서: 구조·개선 과정·검증 결과](docs/REALTIME_REPORT.md)
- [실시간 모델 설정](configs/realtime_model.json)
- 새 구조는 YOLO26m + YOLO26n-pose + ExtraTrees + 손목 ResNet18입니다. 운전석이 이미지 오른쪽인 현재 카메라 조건으로 검증했습니다.
- 아래 내용과 `main.py`는 기존 확정 모델의 실행·평가 기록입니다.

백미러 부근 카메라 사진에서 **운전자가 휴대폰을 들고 있는지** 판단합니다. 현재 기본 모델은 공통 train **911장**으로 재학습한 **1차 YOLO26m·MediaPipe·ExtraTrees + 2차 ResNet18**입니다. 팀 내 방법 비교를 위해 학습·평가 데이터의 일치를 우선합니다.

- [최종 프로젝트 보고서 · 자기소개서/발표 활용 문장](docs/FINAL_PROJECT_REPORT.md): 모델 선택 → 결과 → 개선 → 최종 결과 순서로 정리했습니다.
- [과거 상세 개선·확정 기록](docs/IMPROVEMENT_REPORT.md)
- [실행·평가·재학습 절차](docs/RUNBOOK.md)
- [확정 모델 설정·가중치 해시](configs/final_model.json)
- [수정 데이터셋 재학습·재검증 결과](docs/RETRAIN_REPORT_20260916.md): 공통 test 146장에서 현재 기본 모델 F1 **89.44%**. 다른 학습셋을 사용한 이전 모델은 `artifacts/old/`에 보관합니다.
- [팀 공통 데이터 분할 기준](configs/dataset/protocol.json): train/test별 상대 경로·라벨·SHA-256 목록을 포함합니다.

## 실행

`driver_classification`에서 실행합니다. 현재 컴퓨터에는 `.gpu` 환경과 모델 파일이 준비돼 있습니다. 새 환경에서는 [모델 준비 안내](docs/RUNBOOK.md#1-실행-환경과-모델-파일)를 먼저 확인하세요.

```powershell
.\.gpu\Scripts\python.exe main.py '사진.jpg' --device 0
```

`output/`에 판정 JSON과 표시 이미지를 저장합니다. 기본 운전석은 사진 오른쪽이며, 왼쪽이면 `--driver-side left`를 지정합니다. 기본 실행에 `camera.json`은 필요 없습니다.

## 확정된 판단 방식

1. 1차 특징을 만들 수 있으면 ExtraTrees 점수 **0.3 이상**을 양성으로 판단합니다.
2. 1차 점수가 없을 때만 운전자 영역 이미지를 2차 분류기에 넣습니다. ResNet18 점수 **0.5 이상**을 양성으로 판단합니다.
3. 최종 출력은 **들고있음 / 안들고있음**입니다. 점수 없는 사진을 일괄 양성으로 처리하지 않습니다.

`보고있음`도 양성을 의미하며 시선 방향을 별도로 분류하지 않습니다. JSON의 `stage_used`, `stage1_score`, `stage2_score`, `reason`으로 판단 경로를 확인할 수 있습니다. 실행 오류는 오류로 보고합니다.

현재 데이터 폴더는 `phone`(양성) / `normal`(음성)입니다. 입력 도구는 이전 한국어 라벨 쌍도 인식합니다. 기본 모델과 아래 평가는 공통 train 911장·test 146장 기준입니다. 파일 내용과 분할이 동일한지 `python -m experiments.verify_dataset`으로 확인할 수 있습니다.

## 최종 평가

공통 test **146장 전체(양성 77 / 음성 69)**, 판단 불가 0장 기준입니다.

| Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|
| **88.36%** | **85.71%** | **93.51%** | **89.44%** |

혼동행렬: TN 57 · FP 12 · FN 5 · TP 72. 1차 임계값 0.3, 2차 임계값 0.5이며 2차 최종 학습은 5 epoch입니다. test에는 과거 개발 평가 사진이 포함돼 있어 새로운 운전자·차량의 외부 검증 성능은 아닙니다. 이전 모델은 train 1,011장을 사용했으므로 팀 공통 학습 조건의 후보와 구분합니다.

이전 모델을 명시적으로 실행하려면 `--classifier artifacts/old/classifier.joblib`를 지정합니다. 이전 2차 가중치도 `artifacts/old/stage2.pt`에 보존했습니다.

## 폴더 역할

```text
main.py                  사진 추론 CLI · 기존 실행 명령 유지
predict.py               최종 이진 판단 및 결과 저장
extract.py               YOLO·MediaPipe 추출 및 캐시 CLI
features.py              운전자 선택 · 자세/손/휴대폰 특징
cascade.py               2차 이미지 전처리와 추론
binary_policy.py         과거 단일 모델 이진 정책 지원
data_labels.py           영어·한국어 데이터 폴더 라벨 해석
evaluate_dataset.py      고정 모델 평가 CLI
training/                1차 및 결합 모델 학습
experiments/             과거 비교 · 미탐 분석 · 임계값 탐색
legacy/                  수동 보정 규칙 (--rule-based)
tests/                   기능 검증
docs/                    개선 보고서 · 실행 안내 · 지표 원본
configs/                 확정 모델 식별 정보
models/                  YOLO·MediaPipe·ImageNet 초기 가중치 (로컬)
artifacts/               학습 가중치 · 검출 캐시 · 실험 결과 (로컬)
dataset/train, test/     라벨별 원본 사진 (로컬)
output/                  추론 결과 (로컬)
```

코드·문서·확정 설정은 버전 관리 대상입니다. 데이터, 모델 가중치, 가상환경, 대용량 산출물은 `.gitignore`에 따라 제외합니다. 정리 전 문서는 `docs/archive/`에 보존했으며 당시 명령과 지표가 담긴 기록입니다.

## 학습 도구 및 테스트

학습 도구는 모듈 방식으로 실행합니다. 새 실험은 기존 확정 모델을 자동 교체하지 않습니다.

```powershell
.\.gpu\Scripts\python.exe -m training.train_cascade --help
.\.gpu\Scripts\python.exe -m training.train --help
.\.gpu\Scripts\python.exe -m experiments.verify_dataset
.\.gpu\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
```

전체 캐시 준비·재학습·평가 명령은 [RUNBOOK.md](docs/RUNBOOK.md)에 있습니다.
