"""Single photograph inference and interactive camera calibration."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

from legacy.logic import Point, classify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--config', type=Path, default=Path('camera.json'))
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--pose-model', default='models/pose_landmarker_full.task')
    parser.add_argument('--hand-model', default='models/hand_landmarker.task')
    parser.add_argument('--yolo-model', default=str(Path(__file__).resolve().parent/'models/yolo26m.pt'))
    parser.add_argument('--device', default='auto', help='auto, cpu, or CUDA index (0)')
    parser.add_argument('--output', type=Path, default=Path('output'))
    parser.add_argument('--rule-based', action='store_true', help='기존 수동 보정/접촉 규칙 사용')
    parser.add_argument('--driver-side', choices=['right', 'left'], default='right', help='사진에서 운전석이 있는 방향')
    parser.add_argument('--classifier', type=Path, help='학습된 classifier.joblib 경로')
    args = parser.parse_args()
    if not args.calibrate and not args.rule_based:
        try:
            from predict import predict_image
            predict_image(args.image, args.output, args.driver_side, args.classifier,
                          args.yolo_model, args.device)
        except (FileNotFoundError, ModuleNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        return
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError as exc:
        parser.error(f'{exc.name} 패키지가 없습니다. 실행 중인 Python: {sys.executable}\n'
                     '프로젝트 폴더에서 .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt 실행 후\n'
                     '.\\.venv\\Scripts\\python.exe main.py 명령으로 실행하세요.')
    if not args.calibrate and not args.config.is_file():
        parser.error(f'카메라 보정 파일이 없습니다: {args.config.resolve()}\n'
                     '먼저 같은 명령에 --calibrate 옵션을 추가하여 운전석을 보정하세요.\n'
                     '보정이 끝나면 --calibrate 없이 다시 실행하세요.')
    image = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        parser.error('이미지를 읽을 수 없습니다.')
    height, width = image.shape[:2]
    if args.calibrate:
        # Display coordinates map back to the original image using normalized ratios.
        factor = min(1.0, 1200 / width, 800 / height)
        preview = cv2.resize(image, (round(width * factor), round(height * factor)))
        print('Select the DRIVER shoulder-center region, excluding other seats; press Enter.')
        x, y, w, h = cv2.selectROI('Driver shoulder-center region', preview)
        cv2.destroyAllWindows()
        if not w or not h:
            parser.error('보정이 취소되었습니다.')
        sw = float(input('Driver shoulder width in ORIGINAL pixels (left to right shoulder): '))
        if not 0 < sw <= width:
            parser.error('어깨 너비가 잘못되었습니다.')
        ph, pw = preview.shape[:2]
        config = dict(driver_shoulder_roi=[x/pw, y/ph, (x+w)/pw, (y+h)/ph],
                      min_shoulder_width_ratio=sw/width*0.75,
                      max_shoulder_width_ratio=min(1.0, sw/width*1.3),
                      aspect_ratio=width/height, landmark_confidence=0.6)
        args.config.parent.mkdir(parents=True, exist_ok=True)
        args.config.write_text(json.dumps(config, indent=2), encoding='utf-8')
        print(f'Saved {args.config}')
        return
    config = json.loads(args.config.read_text(encoding='utf-8'))
    roi = config['driver_shoulder_roi']
    if (len(roi) != 4 or not 0 <= roi[0] < roi[2] <= 1 or not 0 <= roi[1] < roi[3] <= 1
            or not 0 < config['min_shoulder_width_ratio'] < config['max_shoulder_width_ratio'] <= 1):
        parser.error('운전석 ROI 또는 어깨 너비 설정이 잘못되었습니다.')
    if abs(width / height - config['aspect_ratio']) > 0.02:
        parser.error('보정 사진과 종횡비가 다릅니다. 다시 보정하세요.')
    for model in (args.pose_model, args.hand_model):
        if not Path(model).is_file():
            parser.error(f'MediaPipe 모델이 없습니다: {model}. README를 참고하세요.')
    import mediapipe as mp
    from ultralytics import YOLO
    detector = YOLO(args.yolo_model)
    phone_ids = [i for i, name in detector.names.items() if name == 'cell phone']
    if not phone_ids:
        parser.error('YOLO 모델에 cell phone 클래스가 없습니다.')
    prediction = detector.predict(image, classes=phone_ids, conf=0.25, imgsz=1280, verbose=False)[0]
    phones = [dict(box=b.xyxy[0].tolist(), confidence=float(b.conf[0])) for b in prediction.boxes]
    vision = mp.tasks.vision
    base = mp.tasks.BaseOptions
    with ExitStack() as stack:
        pose_model = stack.enter_context(vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(base_options=base(model_asset_path=args.pose_model), num_poses=8)))
        hand_model = stack.enter_context(vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(base_options=base(model_asset_path=args.hand_model), num_hands=16)))
        rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        poses = [[Point(p.x*width, p.y*height, min(p.visibility, p.presence)) for p in landmarks]
                 for landmarks in pose_model.detect(rgb).pose_landmarks]
        hands = [[Point(p.x*width, p.y*height) for p in landmarks]
                 for landmarks in hand_model.detect(rgb).hand_landmarks]
    result = classify(poses, hands, phones, config, width, height)
    result.update(image=str(args.image), pose_count=len(poses), hand_count=len(hands))
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / args.image.stem
    target.with_suffix('.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    for i, pose in enumerate(poses):
        color = (0, 255, 0) if i == result['driver_index'] else (255, 160, 0)
        for a, b in ((11, 12), (11, 13), (13, 15), (12, 14), (14, 16)):
            if min(pose[a].confidence, pose[b].confidence) >= config.get('landmark_confidence', 0.6):
                cv2.line(image, (int(pose[a].x), int(pose[a].y)), (int(pose[b].x), int(pose[b].y)), color, 3)
    for phone in phones:
        x1, y1, x2, y2 = map(int, phone['box'])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.rectangle(image, (int(roi[0]*width), int(roi[1]*height)),
                  (int(roi[2]*width), int(roi[3]*height)), (255, 0, 255), 2)
    cv2.putText(image, result['status'], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    ok, encoded = cv2.imencode('.jpg', image)
    if not ok:
        raise RuntimeError('결과 이미지 인코딩 실패')
    encoded.tofile(target.with_suffix('.jpg'))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
