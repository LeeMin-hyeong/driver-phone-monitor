"""OpenCV camera/video display for the verified 640 driver classifier."""
import argparse
from collections import deque
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch

from extract import ROOT
from realtime import load_engine
from temporal import PredictionSmoother

WINDOW = 'Driver phone detection - Q / ESC to quit'
EDGES = ((5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),
         (11,12),(11,13),(13,15),(12,14),(14,16))


def prepare_frame(bgr):
    """Preserve aspect ratio with the same bilinear padding as evaluation."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (640,640):
        rgb = np.asarray(ImageOps.pad(Image.fromarray(rgb), (640,640),
                         method=Image.Resampling.BILINEAR, color=(114,114,114)))
    return rgb


def draw_result(rgb, result, fps, inference_ms, stability):
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    vis = result['visualization']
    def point(p):
        return tuple(np.clip(np.rint(np.asarray(p[:2])*640),0,639).astype(int))
    roi = vis['roi']
    cv2.rectangle(frame,point(roi[:2]),point(roi[2:]),(255,200,70),1)
    for i,pose in enumerate(vis['keypoints']):
        color = (100,240,70) if i == vis['driver_index'] else (50,180,255)
        for a,b in EDGES:
            if min(pose[a][2],pose[b][2]) >= .4:
                cv2.line(frame,point(pose[a]),point(pose[b]),color,2,cv2.LINE_AA)
        for p in pose[5:]:
            if p[2] >= .4: cv2.circle(frame,point(p),3,color,-1)
        if i == vis['driver_index'] and pose[5][2] >= .4:
            cv2.putText(frame,'DRIVER',point(pose[5]),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
    for box in vis['boxes']:
        if box['cls'] == 67:
            cv2.rectangle(frame,point(box['box'][:2]),point(box['box'][2:]),(90,60,255),2)
    cv2.rectangle(frame,(0,0),(640,87),(30,25,20),-1)
    alert = bool(stability['prediction'])
    color = (90,90,255) if alert else (100,240,70)
    text = 'PHONE' if alert else 'NORMAL'
    raw = 'PHONE' if result['prediction'] else 'NORMAL'
    cv2.putText(frame,f"{text}  raw: {raw} / stage {result['stage_used']}",(12,28),
                cv2.FONT_HERSHEY_SIMPLEX,.8,color,2,cv2.LINE_AA)
    s2 = '-' if result['stage2_score'] is None else f"{result['stage2_score']:.3f}"
    for y,line in [(52,f"S1 {result['stage1_score']:.3f} / S2 {s2}  (not probabilities)"),
                   (76,f"Loop {fps:.1f} FPS | inference {inference_ms:.1f} ms")]:
        cv2.putText(frame,line,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.48,(235,235,235),1,cv2.LINE_AA)
    border_color = (0,0,255) if alert else (0,220,0)
    cv2.rectangle(frame,(4,4),(635,635),border_color,8)
    cv2.rectangle(frame,(8,602),(632,631),(30,25,20),-1)
    status = 'ALERT' if alert else 'OK'
    cv2.putText(frame,f"{status} | 1s vote {stability['positive_ratio']:.0%} | normal {stability['negative_seconds']:.1f}/0.7s",
                (16,623),cv2.FONT_HERSHEY_SIMPLEX,.55,border_color,1,cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='0',help='Camera index, video path, or stream URL')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/realtime_model.json')
    parser.add_argument('--output',type=Path,help='Optional annotated MP4 (constant FPS)')
    parser.add_argument('--no-display',action='store_true',help='Process without a GUI window')
    parser.add_argument('--max-frames',type=int,default=0,help='0 processes until EOF or quit')
    args = parser.parse_args()
    if args.max_frames < 0: parser.error('--max-frames must be nonnegative')
    if args.output and Path(args.source).is_file() and Path(args.source).resolve() == args.output.resolve():
        parser.error('--output must differ from the input video')
    if not torch.cuda.is_available(): parser.error('CUDA GPU required')
    torch.set_num_threads(1)
    config = json.loads(args.config.read_text(encoding='utf-8'))
    engine = load_engine(config)
    # Warm both stages without opening the camera during model initialization.
    threshold = engine.bundle['threshold']
    try:
        engine.bundle['threshold'] = float('inf')
        for _ in range(8): engine.score_frame(np.full((640,640,3),114,dtype=np.uint8))
    finally:
        engine.bundle['threshold'] = threshold
    torch.cuda.synchronize()
    source = int(args.source) if args.source.isdecimal() else args.source
    cap = cv2.VideoCapture(source)
    writer = None
    count = 0
    durations = deque(maxlen=60)
    total = 0.
    smoother = PredictionSmoother()
    last_video_time = -1.
    try:
        if not cap.isOpened(): raise RuntimeError(f'Cannot open input: {args.source}')
        file_input = not isinstance(source,int) and Path(source).is_file()
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if isinstance(source,int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            rate = cap.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(rate) or rate <= 0: rate = 30.
            writer = cv2.VideoWriter(str(args.output),cv2.VideoWriter_fourcc(*'mp4v'),rate,(640,640))
            if not writer.isOpened(): raise RuntimeError(f'Cannot write: {args.output}')
        if not args.no_display: cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL)
        while not args.max_frames or count < args.max_frames:
            start = time.perf_counter()
            ok,bgr = cap.read()
            if not ok:
                if count == 0: raise RuntimeError('Input opened but no frames could be read')
                if isinstance(source,int): print('Camera stopped delivering frames.')
                break
            frame_time = time.perf_counter()
            if file_input:
                # Use video time, not how fast this machine processes the file.
                video_time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.
                if not np.isfinite(video_time) or video_time <= last_video_time:
                    if not np.isfinite(source_fps) or source_fps <= 0:
                        raise RuntimeError('Video timestamps and FPS unavailable for the 1-second timer')
                    video_time = max(count/source_fps,last_video_time+1/source_fps)
                last_video_time = video_time
                frame_time = video_time
            rgb = prepare_frame(bgr)
            torch.cuda.synchronize()
            tick = time.perf_counter()
            result = engine.score_frame(rgb,visualize=True)
            torch.cuda.synchronize()
            inference_ms = (time.perf_counter()-tick)*1000
            fps = len(durations)/sum(durations) if durations else 0.
            stability = smoother.update(bool(result['prediction']),frame_time)
            display = draw_result(rgb,result,fps,inference_ms,stability)
            if writer is not None: writer.write(display)
            quit_requested = False
            if not args.no_display:
                cv2.imshow(WINDOW,display)
                quit_requested = cv2.waitKey(1)&0xff in (27,ord('q'),ord('Q'))
                if cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE) < 1: quit_requested = True
            elapsed = time.perf_counter()-start
            durations.append(elapsed); total += elapsed; count += 1
            if quit_requested: break
    finally:
        cap.release()
        if writer is not None: writer.release()
        if not args.no_display: cv2.destroyAllWindows()
    print(json.dumps(dict(frames=count,loop_fps=count/total if total else 0,
                         scope='Capture, resize, inference, overlay, optional display/write; warmup excluded'),indent=2))


if __name__ == '__main__':
    main()
