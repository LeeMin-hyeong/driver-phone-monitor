"""Resident-model inference for 640x640 driver images using a verified realtime profile."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from extract import ROOT
from fast_pose import FastPose


def read640(path):
    image=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError('Cannot decode image: '+str(path))
    if image.shape[:2]!=(640,640):
        raise ValueError('Realtime profile requires an already prepared 640x640 input: '+str(path))
    return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


def load_engine(config):
    for relative,digest in config['sha256'].items():
        path=ROOT/relative
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Model hash mismatch: '+relative)
    engine=FastPose(ROOT/config['stage1_model'],half=True,vectorized=True,engine=True,
                    wrist_checkpoint=ROOT/config['stage2_model'],wrist_threshold=config['stage2_threshold'])
    engine.bundle['threshold']=config['stage1_threshold']
    return engine


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path,help='640x640 image or directory of images')
    p.add_argument('--config',type=Path,default=ROOT/'configs/realtime_model.json')
    p.add_argument('--output',type=Path,default=ROOT/'output/realtime_predictions.jsonl')
    args=p.parse_args()
    if not torch.cuda.is_available():p.error('CUDA GPU is required for this realtime profile')
    if not args.config.is_file():p.error('Verified realtime profile is not available yet')
    config=json.loads(args.config.read_text(encoding='utf-8'))
    files=sorted(x for x in args.input.iterdir() if x.suffix.lower() in ('.jpg','.jpeg','.png')) if args.input.is_dir() else [args.input]
    if not files:p.error('No images found')
    torch.set_num_threads(1)
    start=time.perf_counter();engine=load_engine(config);torch.cuda.synchronize()
    load_ms=(time.perf_counter()-start)*1000
    first=read640(files[0])
    start=time.perf_counter()
    # Exercise the fallback too, even when the first image is stage-one positive.
    threshold=engine.bundle['threshold']
    try:
        engine.bundle['threshold']=float('inf')
        for _ in range(8):engine.score_frame(first)
    finally:
        engine.bundle['threshold']=threshold
    torch.cuda.synchronize();warmup_ms=(time.perf_counter()-start)*1000
    args.output.parent.mkdir(parents=True,exist_ok=True)
    times=[]
    with args.output.open('w',encoding='utf-8') as f:
        for path in files:
            torch.cuda.synchronize();start=time.perf_counter()
            result=engine.score_frame(read640(path))
            torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000
            times.append(elapsed)
            result.update(image=str(path.resolve()),label='phone' if result['prediction'] else 'normal',
                          elapsed_ms=elapsed,score_is_calibrated_probability=False)
            f.write(json.dumps(result,ensure_ascii=False)+'\n')
    print(json.dumps(dict(n=len(times),mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)),
                          fps=1000/float(np.mean(times)),load_ms=load_ms,warmup_ms=warmup_ms,
                          scope='Read/decode + fresh prediction per 640x640 image; resident models; output writes excluded',
                          output=str(args.output)),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
