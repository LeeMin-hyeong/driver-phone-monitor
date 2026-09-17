# 운전자 휴대폰 사진 분류

## 현재 기본 설정: 1차 특징 분류 + 2차 이미지 분류

`main.py`는 현재 `dataset/train`으로 학습한 두 단계 분류기를 사용합니다. 1차는 YOLO26m·MediaPipe 특징의 ExtraTrees이며 점수 **0.3 이상**을 양성으로 판단합니다. **1차 점수가 없는 경우만** ImageNet 사전학습 ResNet18을 우리 train 이미지로 파인튜닝한 2차 모델로 전달하며, 2차 점수 **0.5 이상**을 양성으로 판단합니다. 최종 출력은 `들고있음` / `안들고있음`입니다. 실행 오류는 오류로 보고합니다.

2차 입력은 운전자 사람 박스를 여유 있게 확장한 영역이며, 사람 박스가 없으면 화면 오른쪽 60%를 운전석 영역으로 가정합니다(`--driver-side left`는 왼쪽 60%). 자세 추정은 2차 입력 생성에 필요하지 않습니다. 이 영역 가정은 카메라별 검증이 필요합니다. 입력은 종횡비를 유지해 320×320으로 맞춥니다.

train 내부 촬영 번호 묶음으로 731장 학습 / 280장 검증을 분리해 학습 횟수와 임계값을 선택했습니다. 이후 train 전체로 두 모델을 각각 새로 학습했고 test로 설정을 조정하지 않았습니다. 최종 이미지 모델은 2 epoch 학습했습니다. `artifacts/cascade_20260916/REPORT.md`에서 비교 지표와 한계를 확인할 수 있습니다.

```powershell
# 현재 기본 모델로 추론
.\.gpu\Scripts\python.exe main.py '사진.jpg' --device 0
# 새 실험: 기존 검출 캐시를 사용하므로 먼저 train/test 캐시를 준비해야 합니다.
.\.gpu\Scripts\python.exe train_cascade.py --output artifacts/cascade_repeat
# 고정된 두 단계 모델 평가
.\.gpu\Scripts\python.exe evaluate_dataset.py --dataset dataset/test --classifier artifacts/cascade_20260916/classifier.joblib --output artifacts/cascade_20260916/verified --device 0
```

`stage_used`, `stage1_score`, `stage2_score`, `fallback_used`가 추론 JSON에 기록됩니다. 2차가 사용되면 `positive_score`는 2차 점수입니다. 점수들은 보정된 확률이 아닙니다. 기본 bundle의 `cascade.checkpoint`는 학습한 2차 가중치의 절대 경로를 참조하므로 모델을 다른 컴퓨터로 옮길 때 함께 갱신해야 합니다. 아래는 이전 실행 설정의 기록입니다.

## 이전 설정: 점수 결측을 양성으로 처리하는 이진 분류

`dataset/train` 학습 모델을 사용하며, train의 교차검증 F1으로 선택한 임계값은 **0.5265758057645732**입니다. 점수가 임계값 이상이면 `들고있음`, 미만이면 `안들고있음`입니다. 운전자 식별 실패 등으로 점수가 없으면 **양성으로 처리**합니다. 이는 휴대폰 검출 성공을 뜻하지 않으며 `fallback_used=true`, `positive_score=null`, 실패 사유를 JSON에 남깁니다. 손상된 입력이나 실행 오류는 오류로 보고합니다.

`main.py` 기본 실행에 적용합니다. `--rule-based`는 별도의 기존 규칙 모드입니다. 최적점은 train 교차검증에서 선택했으며 test로 조정하지 않았습니다. 전체 테스트 결과 및 정책 비교는 `artifacts/train_test_20260916/binary/REPORT.md`에 있습니다. 아래 판단 보류 구간 설명은 이전 실행 설정의 기록입니다.

## dataset/train 학습 및 dataset/test 평가 (2026-09-16)

각 분할의 `들고있음`은 양성, `안들고있음`은 음성입니다. YOLO26m·MediaPipe를 고정하고 최종 ExtraTrees 분류기를 train만으로 새로 학습합니다. test에서는 임계값을 조정하지 않습니다.

```powershell
.\.gpu\Scripts\python.exe extract.py --dataset dataset/train --positive 들고있음 --negative 안들고있음 --output artifacts/train_test_20260916/train --device 0
.\.gpu\Scripts\python.exe train.py --artifacts artifacts/train_test_20260916/train --positive 들고있음 --negative 안들고있음
.\.gpu\Scripts\python.exe evaluate_dataset.py --dataset dataset/test --classifier artifacts/train_test_20260916/train/classifier.joblib --output artifacts/train_test_20260916/test --device 0
```

결과는 `artifacts/train_test_20260916/REPORT.md`, 사진별 결과는 `test/predictions.csv`에 저장합니다. `train/evaluation.json`은 학습 내부 교차검증이고 `test/evaluation.json`이 요청한 테스트 지표입니다. 기본 실행 모델은 자동으로 교체하지 않으며 새 모델로 추론하려면 `main.py`에 `--classifier artifacts/train_test_20260916/train/classifier.joblib`를 지정합니다. 아래 기존 dataset 명령은 분할 전 폴더 구조의 과거 실험입니다.

## 현재 기본 실행: 자동 랜드마크 + 데이터 기반 분류

현재 특징 버전은 **v4**이며 기본 검출기는 **YOLO26m**입니다. 사람 검출로 지정한 운전석 영역에 신뢰할 수 있는 골격이 없으면 `driver_pose_missing`으로 판단을 유보합니다. 골격이 한 개라는 이유로 동승자를 운전자로 대체하지 않습니다. 연결된 손의 손바닥 중심 및 다섯 손가락 끝과 휴대폰 박스 사이 거리를 추가했습니다. 손을 연결하지 못하면 거리 결측값과 `hand_found=0`을 함께 전달합니다.

v4 학습은 기존 dataset만 사용하며 새 데이터의 라벨은 학습에 넣지 않습니다. 학습 모델·지표는 `artifacts/m_v4`, 새 데이터 재평가는 `artifacts/new_dataset_m_v4`에 저장합니다. 기존 모델은 `artifacts/baseline_v3/classifier.joblib`에 보존합니다. 이전 평가 사진의 오류를 보고 설계를 바꿨으므로 새 데이터 재평가는 개발 평가로 해석해야 합니다.

```powershell
.\.gpu\Scripts\python.exe extract.py --output artifacts/m_v4 --device 0
.\.gpu\Scripts\python.exe train.py --artifacts artifacts/m_v4
.\.gpu\Scripts\python.exe evaluate_dataset.py --classifier artifacts/m_v4/classifier.joblib --output artifacts/new_dataset_m_v4 --device 0
```

**`camera.json` 없이 실행됩니다.** MediaPipe Pose Landmarker(몸 33점)와 Hand Landmarker(손 21점)를 자동 추출하고, YOLO26n으로 휴대폰·컵·물병을 검출합니다. 운전자 손목 주변을 확대해 작은 물체를 다시 검출한 뒤, 어깨 기준으로 정규화한 팔·얼굴·손 특징과 휴대폰 소유권 거리를 ExtraTrees 분류기에 입력합니다.

```powershell
# driver_classification 폴더에서
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py '.\dataset\보고있음\IMG_3261(1).jpg'
```

`output` 폴더에 JPG와 JSON이 저장됩니다. **초록 골격이 운전자**, 주황은 다른 사람, 빨간 사각형은 휴대폰입니다. JSON의 `label`은 `보고있음` / `안보고있음` / `판단 불가`입니다. 점수는 보정된 확률이 아니며, 0.35~0.65 사이 또는 운전자 후보가 비슷하면 판단을 유보합니다. `evidence`에는 손목별 휴대폰 거리와 다른 사람과의 거리 비교가 들어갑니다.

제공 폴더의 라벨을 학습하므로 시선을 직접 측정하거나 물리적인 파지를 증명하는 모델은 아닙니다. 예를 들어 전방을 보면서 휴대폰을 들고 있는 장면도 `보고있음` 라벨로 학습됩니다. 컵·물병·거치대·휴대폰을 가린 손은 여전히 오분류 원인이 됩니다.

### 운전자 자동 선택의 가정

- 제공 사진에 맞춰 **사진 오른쪽의 앞자리**를 운전석으로 가정합니다. YOLO 사람 박스의 크기·하단 위치로 앞자리 후보를 고른 뒤 운전석 방향의 후보를 선택합니다.
- 해당 사람 박스 안의 어깨 중점과 골격을 연결합니다. 골격이 없거나 연결 후보가 모호하면 판단을 유보합니다. 외모나 얼굴 신원은 사용하지 않습니다.
- 반전 사진 또는 운전석이 왼쪽인 사진은 `--driver-side left`를 지정하세요.
- YOLO 사람 검출까지 운전자를 놓치거나 사람이 크게 겹치는 경우, 카메라의 좌석 방향 가정이 맞지 않는 경우에는 좌석을 잘못 고를 수 있습니다. 이번 방지는 모든 배치의 운전자 신원 확인을 보장하지 않습니다.

### 재학습 및 전체 사진 평가

아래 모델 준비 절차로 YOLO/MediaPipe 가중치를 마련한 뒤 실행합니다. 현재 작업 폴더에는 모델과 학습 결과가 준비되어 있습니다.

```powershell
.\.venv\Scripts\python.exe extract.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe report.py
.\.venv\Scripts\python.exe -m unittest discover -s . -p 'test_*.py' -v
```

- `보고있음`, `안보고있음` 두 폴더만 사용합니다. `버릴사진`과 dataset 바로 아래 미분류 사진은 사용하지 않습니다.
- SHA-256 중복 제거 후 IMG 번호 100단위 묶음을 통째로 제외하는 개발 교차검증을 합니다. 파일명은 평가 묶음 생성에만 쓰며 분류기 입력에는 포함하지 않습니다.
- 특징을 개선할 때 오분류를 확인했으므로, 이 결과는 **손대지 않은 최종 테스트 성능이 아닙니다**. 동일 인물·실내가 여러 묶음에 포함됩니다.
- [EVALUATION.md](EVALUATION.md)에 실제 지표와 한계가 정리됩니다. `artifacts/all_images.csv`는 두 폴더 모든 사진의 평가 결과입니다.
- 실행 모델은 평가 후 전체 데이터로 다시 학습합니다. 학습 사진을 단일 실행한 점수는 평가 점수가 아닙니다. JSON의 `present_in_training_dataset`로 구분합니다.
- 캐시와 실행 모델은 `artifacts`, 신경망 가중치는 `models`에 저장하며 Git에서는 제외합니다. 새 복제본에서는 다운로드와 재학습이 필요합니다. `joblib` 파일은 직접 학습한 신뢰할 수 있는 파일만 사용하세요.

### 새 데이터셋 평가 (재학습 없음)

```powershell
.\.venv\Scripts\python.exe evaluate_dataset.py --dataset '.\new dataset'
```

`들고있음`을 양성, `안들고있음`을 음성으로 처리하고 `버릴사진`은 제외합니다. 기존 모델과 임계값을 유지하며, 결과는 `artifacts/new_dataset/evaluation.json`과 `predictions.csv`에 저장합니다. 전체 파일·중복 제거·기존 학습 파일 중복 제외 지표를 각각 기록합니다. `overall`은 랜드마크 추출 성공 사진을 0.5 기준으로 모두 분류한 결과, `selective`는 실행 모드의 판단 유보를 제외한 결과입니다. 실패 건수와 판정률을 함께 확인하세요.

### YOLO26m 비교 실행

`models/yolo26m.pt`에 공식 가중치를 준비한 뒤, 기존 분류기를 고정하고 검출기만 변경해 평가합니다.

```powershell
.\.venv\Scripts\python.exe evaluate_dataset.py --yolo-model models/yolo26m.pt --output artifacts/new_dataset_yolo26m
.\.venv\Scripts\python.exe compare_detectors.py
```

결과는 `artifacts/new_dataset_yolo26m/REPORT.md`와 `comparison.json`에 저장됩니다. 캐시에는 검출 가중치의 SHA-256을 기록해 다른 모델의 검출 결과 재사용을 방지합니다. `compare_detectors.py`는 기존 n 평가와 이미지·라벨·최종 분류기·임계값이 같은지 확인합니다. 여기의 F1/Recall은 최종 소지 분류 지표이며, 박스 정답이 필요한 검출 mAP와 다릅니다.

### 실행 환경

GPU 실행은 CUDA PyTorch가 설치된 `.gpu` 환경에서 사용합니다. YOLO는 기본적으로 사용 가능한 CUDA GPU를 자동 선택하고, MediaPipe는 CPU에서 실행됩니다. `--device 0`으로 GPU를 명시할 수 있습니다.

```powershell
.\.gpu\Scripts\python.exe main.py '.\new dataset\들고있음\IMG_7086.jpg' --yolo-model models/yolo26m.pt --device 0
.\.gpu\Scripts\python.exe evaluate_dataset.py --yolo-model models/yolo26m.pt --device 0 --output artifacts/new_dataset_yolo26m_gpu
.\.gpu\Scripts\python.exe compare_detectors.py --m-output artifacts/new_dataset_yolo26m_gpu
```

`.gpu`는 CUDA PyTorch 2.14.0+cu130을 설치하고 현재 컴퓨터의 `.runtime`·`.venv` 패키지를 참조하는 로컬 환경입니다. 다른 컴퓨터에서는 새 환경에 `requirements.txt`와 공식 CUDA PyTorch를 설치하세요.

Python 3.12 환경에서 실제 추출·학습·추론을 검증합니다. Codex 실행 환경에서는 기존 `.venv`의 Windows Store Python 경로를 찾지 못해 별도 `.runtime` 실행기를 사용했습니다. 기존 `.venv`의 기본 Python 설정은 변경하지 않았으며 필요한 분류 라이브러리는 설치했습니다. 기존 실행기가 작동하지 않으면 `.\.runtime\Scripts\python.exe`로 같은 명령을 실행할 수 있습니다. `.runtime`은 이 작업 컴퓨터용이며 다른 컴퓨터로 복사하지 마세요.

## 기존 수동 보정 규칙 모드

백미러 부근에서 실내를 촬영한 **고정 카메라** 사진용입니다. YOLO26n으로 휴대폰을 검출하고 MediaPipe Pose / Hand Landmarker로 사람의 어깨·팔·손을 연결합니다. 입력 사진의 좌우를 임의로 반전하지 않습니다.

## 판단 흐름

1. 사진 전체에서 `yolo26n.pt`의 `cell phone`을 탐지합니다.
2. 전체 사진에서 최대 8명의 자세와 16개의 손을 탐지합니다. 동승자도 소유권 비교에 포함합니다.
3. 보정한 운전석 어깨 중심 영역과 어깨 너비 범위에 들어오는 사람이 정확히 한 명이어야 합니다. 화면 왼쪽/오른쪽만으로 운전자를 정하지 않습니다.
4. 어깨·팔꿈치·손목의 신뢰도가 충분한 사람에게 손을 연결합니다. 손목 거리는 어깨 너비로 나눠 비교하며, 두 후보가 비슷하면 연결하지 않습니다.
5. 손바닥과 휴대폰 사각형이 가깝고 손가락 끝이 2개 이상 근접하는지 확인합니다. 다른 사람의 손목도 휴대폰에 가까우면 확정을 보류합니다.

| status | 의미 |
|---|---|
| `holding` | 운전자에게 연결된 손과 휴대폰의 접촉 후보가 발견됨 |
| `not_holding` | 운전자 양손·팔이 확인되며, 검출된 휴대폰과 운전자 손의 접촉 근거 없음 |
| `unknown` | 운전자 식별 실패, 후보 중복, 손 가림, 소유권 겹침 등 |

`holding`은 단일 사진의 **2D 접촉 휴리스틱**이며 실제 파지를 증명하지 않습니다. 거치대 휴대폰 위에 손을 대는 경우도 오탐할 수 있습니다. `not_holding`도 휴대폰 검출 누락을 배제하지 못합니다. 임계값은 초기값이며 정확도 측정 결과가 아닙니다. 사람별 MediaPipe z 좌표는 공통 차량 깊이가 아니므로 앞자리/뒷자리 판정에 사용하지 않습니다.

## 설치 및 모델 준비

Python 3.11 또는 3.12 환경을 권장합니다. 프로젝트 폴더에서 실행합니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
New-Item -ItemType Directory -Force models
Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task' -OutFile models/pose_landmarker_full.task
Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' -OutFile models/hand_landmarker.task
```

YOLO 가중치는 첫 추론 시 Ultralytics가 다운로드합니다. 오프라인 환경에서는 `--yolo-model`, `--pose-model`, `--hand-model`에 준비한 파일 경로를 지정하세요. 의존성 하한은 설치 안내용이며 실제 모델 조합의 실행 검증이나 버전 고정을 의미하지 않습니다.

## 카메라 보정 및 실행

```powershell
.\.venv\Scripts\python main.py cabin.jpg --calibrate --config camera.json
.\.venv\Scripts\python main.py cabin.jpg --rule-based --config camera.json --output output
```

보정 창에서 **운전자 양어깨 중점이 움직일 수 있는 영역**을 드래그하고 Enter를 누릅니다. 몸 전체를 선택하지 마세요. 이어 원본 이미지 기준 양어깨 사이 픽셀 거리를 입력합니다. 설정은 그 거리의 75~130%를 초기 허용 범위로 저장합니다. 뒷좌석 사람이 후보에 들어오면 ROI와 최소 어깨 너비를 좁혀야 합니다. 몸을 기울이는 운전자와 뒷좌석 승객이 같은 영역·크기로 보이는 경우 이 규칙만으로 구분할 수 없습니다.

운전석 위치는 실제 사진으로 지정하므로 우핸들 차량과 반전 영상도 보정할 수 있습니다. 카메라 위치·화각·크롭·반전 여부가 바뀌면 다시 보정하세요. 같은 종횡비의 크롭 변경은 자동으로 감지할 수 없습니다.

`output/<사진이름>.json`에는 상태·사유·운전자 인덱스·휴대폰별 소유권 근거가 저장됩니다. JPG에는 운전자 팔(초록), 다른 사람 팔(파랑), 휴대폰(빨강), 운전석 영역(보라)이 표시됩니다. 같은 이름의 결과는 덮어씁니다.

## 검증

```powershell
python -m unittest discover -s . -p 'test_*.py' -v
```

가상 좌표 테스트는 소유권 규칙만 검증합니다. 실사진, 모델 가중치 및 추론 환경을 통한 정확도·속도 검증은 별도입니다. 실제 데이터에서는 운전자 양손 소지/귀 옆 통화/무릎 위, 조수석만 소지, 2열만 소지, 중앙으로 뻗은 팔, 거치대, 지갑, 야간·역광·가림을 포함하세요. 차량·촬영 세션 단위로 평가 데이터를 분리하고 `holding` 정밀도·재현율과 `unknown` 비율을 함께 측정하세요.

## 공식 문서

- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)
- [MediaPipe Pose Python](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)
- [MediaPipe Hand Python](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)
