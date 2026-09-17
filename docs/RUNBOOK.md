# 실행·평가·재학습 절차

모든 명령은 `driver_classification` 폴더에서 실행한다. 확정 구조와 개선 근거는 [개선 보고서](IMPROVEMENT_REPORT.md)를 참고한다.

## 1. 실행 환경과 모델 파일

현재 컴퓨터에서는 Python 3.12 기반 `.gpu` 환경에서 검증했다. YOLO/ResNet18은 CUDA, MediaPipe는 CPU를 사용한다. `.gpu`는 이 컴퓨터의 다른 가상환경 패키지를 참조하므로 그대로 다른 컴퓨터에 복사하는 설치 방식은 아니다.

- 일반 의존성: `requirements.txt`
- 실제 검증 버전: `requirements-tested.txt`
- 새 환경에는 장치에 맞는 CUDA PyTorch·torchvision을 먼저 설치한 뒤 일반 의존성을 설치한다. CPU PyTorch를 설치하면 `--device 0`을 사용할 수 없다.
- 새로 저장소를 복제한 경우 데이터와 가중치는 Git에 없으므로 별도로 준비해야 한다.

필요한 실행 파일:

```text
models/yolo26m.pt
models/pose_landmarker_full.task
models/hand_landmarker.task
artifacts/classifier.joblib
artifacts/retrain_20260916_v2/model/stage2.pt
```

학습에는 추가로 `models/resnet18-f37072fd.pth`가 필요하다. 사용한 공식 ImageNet 초기 가중치 URL은 `https://download.pytorch.org/models/resnet18-f37072fd.pth`이며 SHA-256은 최종 선택 기록에 있다.

bundle의 `cascade.checkpoint`는 2차 가중치의 절대 경로다. 다른 컴퓨터로 옮길 때 실제 경로에 맞춰 bundle을 갱신해야 한다. 이때 bundle 해시는 바뀌므로 원래 확정 bundle과 배포용 변경본을 구분한다. 출처를 신뢰하는 로컬 joblib 파일만 로드한다.

## 2. 확정 모델 실행

현재 기본 모델은 공통 train 911장으로 학습하고 test 146장으로 검증한 재학습 모델이다. 이전 모델은 `artifacts/old/classifier.joblib`, `artifacts/old/stage2.pt`에 있다. `--classifier artifacts/old/classifier.joblib`로만 명시적으로 선택한다.

```powershell
.\.gpu\Scripts\python.exe main.py '사진.jpg' --device 0
# 사진에서 운전석이 왼쪽이면
.\.gpu\Scripts\python.exe main.py '사진.jpg' --device 0 --driver-side left
```

`output/`에 JSON과 표시 이미지가 생성된다. 최종 `status`는 `positive` 또는 `negative`, `label`은 `들고있음` 또는 `안들고있음`이다.

| JSON 필드 | 의미 |
|---|---|
| `stage_used` | 최종 판단을 수행한 단계: 1 또는 2 |
| `stage1_score` | 1차 점수. 특징 생성 실패 시 null |
| `stage2_score` | 2차가 실행됐을 때만 점수 기록 |
| `positive_score` | 최종 판단에 사용한 단계의 점수 |
| `fallback_used` | 1차 특징 생성이 실패해 2차로 전달됐는지 |
| `reason` | 1차 판단 또는 2차 전달 원인 |

점수들은 보정된 확률이 아니다. 실행 오류는 이진 판정으로 감추지 않고 오류로 반환한다. 기본 실행에는 `camera.json`이 필요 없다. 수동 보정 규칙을 연구하려는 경우에만 `--calibrate`, `--rule-based`를 사용한다.

## 3. 고정 모델 평가

```powershell
.\.gpu\Scripts\python.exe evaluate_dataset.py --dataset dataset/test --classifier artifacts/classifier.joblib --output artifacts/final_recheck --device 0
```

현재 입력은 `phone`(양성), `normal`(음성) 하위 폴더다. 기존 `들고있음`/`안들고있음`, `보고있음`/`안보고있음`도 자동 인식한다. 여러 라벨 쌍이 함께 있으면 `--positive`, `--negative`를 모두 지정한다. 모든 사진의 최종 이진 결과로 계산한 `all_files.overall`과 `unique_images.overall`을 확인한다. 동일 파일 중복과 학습 이미지 중복도 기록한다.

평가 결과는 지정한 출력 폴더에 저장한다. 기존 확정 실험 기록을 덮어쓰지 않으려면 새 출력 폴더를 사용한다.

공통 데이터 기준은 train 911장·test 146장이며 현재 기본 모델도 이 기준이다. 초기 모델의 1,011장·140장 실험은 과거 기록이다. 다른 팀원도 파일·라벨·분할이 같은지 다음 명령으로 확인한다. 이 명령은 모델 없이 파일 해시와 라벨 목록을 검사한다.

```powershell
.\.gpu\Scripts\python.exe -m experiments.verify_dataset
```

기준 파일은 `configs/dataset/train.csv`, `test.csv`, `protocol.json`이다. CSV는 프로젝트 기준 상대 경로를 사용하므로 다른 PC에서도 사용할 수 있다. 데이터 원본은 별도로 공유해야 한다. test로 임계값을 조정하지 않고 test 전체를 분모로 사용한다. 내부 train/validation 분할도 일치시킬 경우 `configs/dataset/train_validation.json`의 해시 목록을 사용한다.

## 4. 재학습 실험

확정 모델을 자동 교체하지 않는다. 새로운 출력 폴더에서 실험한 뒤 별도로 채택 여부를 결정한다.

### 4.1 train/test의 검출 캐시 준비

```powershell
.\.gpu\Scripts\python.exe extract.py --dataset dataset/train --positive phone --negative normal --output artifacts/cascade_inputs/train --device 0
.\.gpu\Scripts\python.exe extract.py --dataset dataset/test --positive phone --negative normal --output artifacts/cascade_inputs/test --device 0
```

추출 단계는 라벨로 모델을 학습하지 않는 고정 검출 단계다. 현재 추출 CLI는 라벨 폴더 바로 아래의 JPG/JPEG/PNG 파일을 처리하므로 사진을 그 위치에 둔다. 동일 파일·검출기·추출 버전·장치 조건의 캐시는 재사용할 수 있다.

### 4.2 train 내부 검증 → 전체 train 재학습 → test

```powershell
.\.gpu\Scripts\python.exe -m training.train_cascade --source artifacts/cascade_inputs --output artifacts/cascade_new_run --epochs 8
```

기본 이미지 데이터 위치는 `dataset/train`, `dataset/test`이다. 학습에는 source 아래의 train manifest/cache, 최종 테스트에는 test cache가 필요하다. `locked_selection.json`이 이미 있는 실험 폴더는 재사용을 거부한다.

source 아래에 `test/manifest.json`이 있으면 최종 평가는 그 실행 시작 시점의 파일·라벨 목록을 사용한다. 이미지 해시가 바뀌면 중단한다. 수정 데이터 재학습 결과는 [재학습 보고서](RETRAIN_REPORT_20260916.md)에 있으며 기존 확정 실험과 별도로 보존했다.

절차:

1. train을 IMG 번호 100단위 묶음으로 나누고 묶음의 25%를 검증에 배정한다. 확정 실험에서는 731장/280장이었으며, 데이터가 달라지면 실제 장수도 달라진다.
2. 1차를 학습하고, 2차를 ImageNet 가중치에서 파인튜닝한다.
3. 검증 전체 F1으로 epoch와 두 단계 임계값을 선택해 고정한다.
4. 두 모델을 train 전체로 각각 새로 학습한다.
5. test를 평가한다. test를 이용한 추가 임계값 탐색은 수행하지 않는다.

학습 결과의 `classifier.joblib`를 `--classifier`로 지정하면 새 모델을 시험할 수 있다. 확정 모델은 별도로 보존한다.

1차만 학습하는 과거 비교 경로:

```powershell
.\.gpu\Scripts\python.exe -m training.train --artifacts artifacts/cascade_inputs/train --positive phone --negative normal
```

## 5. 진단 도구와 테스트

과거 실험을 분석하는 도구이며, 확정 모델의 자동 실행 경로에는 포함하지 않는다.

```powershell
.\.gpu\Scripts\python.exe -m experiments.analyze_misses --help
.\.gpu\Scripts\python.exe -m experiments.sweep_thresholds --help
.\.gpu\Scripts\python.exe -m experiments.optimize_binary --help
.\.gpu\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
```

`compare_detectors`, `compare_features`, `report`는 특정 과거 실험 산출물 구조를 전제로 한다. 실행 전에 해당 모듈의 입력·출력 위치를 확인한다. `experiments.report`의 새 보고서 출력은 `artifacts/development_report.md`이며 확정 보고서를 덮어쓰지 않는다.
