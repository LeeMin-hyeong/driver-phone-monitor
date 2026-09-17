# 손 주변 ROI 라벨링

프로젝트 폴더에서 실행합니다. GPU는 필요 없고 Python의 Tkinter와 Pillow를 사용합니다.

```powershell
.\.gpu\Scripts\python.exe hand_labeler.py
```

기본 원본은 `dataset/train`, 결과는 `dataset/hand_rois`입니다. 다른 원본과 저장 위치를 지정할 수 있습니다.

```powershell
.\.gpu\Scripts\python.exe hand_labeler.py --source "dataset/test" --output "dataset/hand_rois"
```

1. 원본의 손 주변을 클릭합니다. 노란 정사각형을 드래그하면 위치가 이동합니다.
2. 기본 선택 크기는 원본 좌표 기준 224×224입니다. 휠이나 슬라이더로 범위를 조정할 수 있으며 저장 이미지는 항상 224×224입니다. 작은 원본은 가능한 최대 정사각형을 확대합니다. 휴대폰이 잘리지 않도록 주변 여유를 포함하세요.
3. 미리보기를 보고 `phone`(1), `normal`(2), `uncertain`(3)을 선택합니다. 손 구분은 인물 기준이며 불명확하면 `unknown`을 유지합니다.
4. Enter 또는 저장 버튼을 누릅니다. 반대 손은 별도 선택·라벨로 저장합니다. 원본 라벨을 자동으로 복사하지 않습니다.
5. ‘검토 완료 후 다음’은 완료 여부를 기록합니다. ‘다음’은 완료 표시 없이 건너뜁니다. 재실행하면 같은 출력 폴더의 기록과 마지막 원본을 복구합니다. 저장하지 않은 선택 영역은 이동 시 사라집니다.

현재 원본의 저장 목록에서 샘플을 선택해 라벨·손 구분을 바꾼 뒤 수정 버튼을 누를 수 있습니다. 잘못 선택한 범위는 샘플 제외 후 다시 저장합니다. 제외한 PNG는 복구용으로 남지만 활성 라벨 목록에서는 빠집니다.

## 저장 데이터

- `crops/<id>.png`: RGB 224×224 이미지. 파일 이름은 라벨 수정과 무관하게 유지합니다.
- `annotations.json`: 활성 샘플의 라벨, target, 손 구분, 원본 절대 경로·SHA-256, 원본 크기, EXIF 방향 보정 후 좌표, 원본 split/subset, 검토 완료 목록.
- `phone=1`, `normal=0`, `uncertain=null`. 학습 시 uncertain은 제외합니다. 폴더의 PNG 전체 대신 JSON의 활성 records를 읽어야 합니다.
- 기존 공통 데이터는 원본 해시로 train/test 및 fit/validation을 상속합니다. 양손은 같은 분할을 유지합니다. 새 이미지의 split/subset은 unassigned이며 촬영 세션 단위로 분할한 뒤 사용하세요.

이 도구는 데이터 준비용입니다. 기존 모델이나 추론 방식, 학습 코드는 변경하지 않습니다. 한 출력 폴더에는 도구를 하나만 실행하세요.
