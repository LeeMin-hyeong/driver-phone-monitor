"""Extract/cache label-independent YOLO26m + MediaPipe observations."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import time
from data_labels import resolve_labels

ROOT = Path(__file__).resolve().parent
EXTRACTOR_VERSION = 3
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.cache'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache'))


class Extractor:
    def __init__(self, driver_side='right', yolo_model=None, device='auto'):
        import torch
        import mediapipe as mp
        from ultralytics import YOLO
        self.mp = mp
        self.driver_side = driver_side
        self.device = ('0' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
        self.stack = ExitStack()
        model_path = Path(yolo_model or ROOT / 'models/yolo26m.pt')
        self.yolo = YOLO(str(model_path))
        self.yolo_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        self.pose = self.stack.enter_context(mp.tasks.vision.PoseLandmarker.create_from_options(
            mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_buffer=(ROOT / 'models/pose_landmarker_full.task').read_bytes()),
                num_poses=8, min_pose_detection_confidence=0.35, min_pose_presence_confidence=0.35)))
        self.hand = self.stack.enter_context(mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_buffer=(ROOT / 'models/hand_landmarker.task').read_bytes()),
                num_hands=16, min_hand_detection_confidence=0.35, min_hand_presence_confidence=0.35)))

    def close(self):
        self.stack.close()

    def extract(self, path):
        import cv2
        import numpy as np
        from PIL import Image, ImageOps
        # Honor EXIF rotation before all models; coordinates refer to upright pixels.
        with Image.open(path) as source:
            rgb = np.asarray(ImageOps.exif_transpose(source).convert('RGB'))
        h, w = rgb.shape[:2]
        factor = min(1, 1280 / max(h, w))
        rgb = cv2.resize(rgb, (round(w * factor), round(h * factor)))
        h, w = rgb.shape[:2]
        result = self.yolo.predict(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                   classes=[0, 39, 41, 67], conf=0.12, imgsz=1280, verbose=False, device=self.device)[0]
        boxes = [dict(cls=int(b.cls[0]), confidence=float(b.conf[0]),
                      box=(b.xyxy[0].cpu().numpy() / [w, h, w, h]).tolist()) for b in result.boxes]
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        poses = [[[p.x, p.y, p.z, min(p.visibility, p.presence)] for p in pose]
                 for pose in self.pose.detect(mp_image).pose_landmarks]
        hands = [[[p.x, p.y, p.z] for p in hand] for hand in self.hand.detect(mp_image).hand_landmarks]
        observation = dict(width=w, height=h, boxes=boxes, poses=poses, hands=hands, extractor_version=EXTRACTOR_VERSION,
                           yolo_sha256=self.yolo_sha256, yolo_device=self.device)
        self.refine(rgb, observation)
        return observation

    def refine(self, rgb, observation):
        """Zoom at automatically selected driver's wrists to find small objects."""
        import cv2
        import numpy as np
        from features import features
        _, info = features(observation, self.driver_side)
        if 'driver_index' not in info:
            return
        p = observation['poses'][info['driver_index']]
        h,w = rgb.shape[:2]
        shoulder = np.linalg.norm((np.asarray(p[11][:2])-p[12][:2])*[w,h])
        radius = max(80, shoulder * 0.7)
        for wrist in (15,16):
            if p[wrist][3] < 0.35:
                continue
            cx,cy = p[wrist][0]*w,p[wrist][1]*h
            x1,y1,x2,y2 = max(0,int(cx-radius)),max(0,int(cy-radius)),min(w,int(cx+radius)),min(h,int(cy+radius))
            if x2-x1<20 or y2-y1<20:
                continue
            crop=rgb[y1:y2,x1:x2]
            result=self.yolo.predict(cv2.cvtColor(crop,cv2.COLOR_RGB2BGR),classes=[39,41,67],
                                     conf=0.15,imgsz=640,verbose=False,device=self.device)[0]
            for b in result.boxes:
                coordinates=(b.xyxy[0].cpu().numpy()+[x1,y1,x1,y1])/[w,h,w,h]
                observation['boxes'].append(dict(cls=int(b.cls[0]),confidence=float(b.conf[0]),
                                                  box=coordinates.tolist(),source='wrist_crop'))
            image=self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=np.ascontiguousarray(crop))
            for hand in self.hand.detect(image).hand_landmarks:
                observation['hands'].append([[(q.x*(x2-x1)+x1)/w,(q.y*(y2-y1)+y1)/h,q.z] for q in hand])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'dataset/train')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts')
    parser.add_argument('--yolo-model',type=Path,default=ROOT/'models/yolo26m.pt')
    parser.add_argument('--device',default='auto')
    parser.add_argument('--positive')
    parser.add_argument('--negative')
    args = parser.parse_args()
    args.positive, args.negative = resolve_labels(
        [p.name for p in args.dataset.iterdir() if p.is_dir()], args.positive, args.negative)
    cache = args.output / 'cache'
    cache.mkdir(parents=True, exist_ok=True)
    manifest = []
    for label in (args.positive, args.negative):
        for path in sorted((args.dataset / label).iterdir()):
            if path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest.append(dict(path=str(path.resolve()), label=label, sha256=digest))
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    engine = Extractor(yolo_model=args.yolo_model,device=args.device)
    start = time.monotonic()
    try:
        for i, item in enumerate(manifest[:args.limit]):
            target = cache / (item['sha256'] + '.json')
            previous = json.loads(target.read_text()) if target.exists() else None
            if (previous is None or previous.get('extractor_version') != EXTRACTOR_VERSION
                    or previous.get('yolo_sha256')!=engine.yolo_sha256 or previous.get('yolo_device')!=engine.device):
                observation = engine.extract(item['path'])
                temporary = target.with_suffix('.tmp')
                temporary.write_text(json.dumps(observation), encoding='utf-8')
                temporary.replace(target)
            if i % 10 == 0:
                print(f'{i+1}/{len(manifest)} elapsed={time.monotonic()-start:.1f}s', flush=True)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
