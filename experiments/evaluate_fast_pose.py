"""Uncached original-file latency and accuracy of a locked fast pose model."""
import argparse
import csv
import hashlib
import json
import random
import time
import numpy as np
import torch
import cv2
from extract import ROOT
from fast_pose import FastPose
from frame_io import read_rgb_gpu
from training.train_cascade import metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',default='artifacts/fast_pose_roi_model')
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--input640',action='store_true')
    p.add_argument('--half',action='store_true')
    p.add_argument('--sklearn',action='store_true',help='Disable vectorized forest execution')
    p.add_argument('--engine',action='store_true')
    p.add_argument('--fusion',action='store_true')
    p.add_argument('--min-seconds',type=float,default=0,help='Continue full repeats until measured processing time reaches this duration')
    args=p.parse_args()
    torch.set_num_threads(1)
    run=ROOT/args.run
    locked=json.loads((run/'locked_selection.json').read_text())
    modelpath=run/('validation.joblib' if args.split=='validation' else 'model.joblib')
    fusion=json.loads((run/'fusion_selection.json').read_text()) if args.fusion else None
    wrist_path=ROOT/fusion['image']/('validation.pt' if args.split=='validation' else 'model.pt') if fusion else None
    engine=FastPose(modelpath,half=args.half,vectorized=not args.sklearn,engine=args.engine,
                    wrist_checkpoint=wrist_path,wrist_threshold=fusion['best']['image_threshold'] if fusion else .85)
    if fusion:
        assert engine.bundle['model'].min_samples_leaf==fusion['best']['leaf']
        engine.bundle['threshold']=fusion['best']['box_threshold']
    rows=list(csv.DictReader((ROOT/('configs/dataset/train.csv' if args.split=='validation' else 'configs/dataset/test.csv')).open(encoding='utf-8-sig')))
    if args.split=='validation':
        hashes=set(json.loads((ROOT/'configs/dataset/train_validation.json').read_text())['validation_hashes'])
        rows=[r for r in rows if r['sha256'] in hashes]
    assert not set(engine.bundle['train_hashes'])&{r['sha256'] for r in rows}
    input_manifest=ROOT/'artifacts/input640/manifest.json'
    prepared={r['source_sha256']:r for r in json.loads(input_manifest.read_text(encoding='utf-8'))} if args.input640 else {}
    for r in rows:
        assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
        if args.input640:
            item=prepared[r['sha256']]
            assert hashlib.sha256((ROOT/item['input']).read_bytes()).hexdigest()==item['input_sha256']
    rng=random.Random(42);rng.shuffle(rows)
    def read(r):
        if args.input640:
            path=ROOT/'artifacts/input640/images'/(r['sha256']+'.jpg')
            return cv2.cvtColor(cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        return read_rgb_gpu(ROOT/r['path'])
    for r in rows[:8]:engine.score_frame(read(r))
    results=[]
    repeat=0
    measured_seconds=0.
    wall_start=time.perf_counter()
    while repeat<args.repeats or measured_seconds<args.min_seconds:
        order=rows.copy();rng.shuffle(order)
        for i,r in enumerate(order):
            path=ROOT/r['path']
            torch.cuda.synchronize();start=time.perf_counter()
            rgb=read(r)
            decoded=time.perf_counter()
            result=engine.score_frame(rgb)
            torch.cuda.synchronize();end=time.perf_counter()
            measured_seconds+=end-start
            results.append(dict(path=str(path),sha256=r['sha256'],target=int(r['target']),repeat=repeat,
                                total_ms=(end-start)*1000,decode_ms=(decoded-start)*1000,
                                inference_ms=(end-decoded)*1000,start_offset_ms=(start-wall_start)*1000,
                                end_offset_ms=(end-wall_start)*1000,**result))
            if (i+1)%50==0:print(f'repeat {repeat+1} {i+1}/{len(rows)}',flush=True)
        repeat+=1
    wall_seconds=time.perf_counter()-wall_start
    first=[r for r in results if r['repeat']==0]
    quality=metrics([r['target'] for r in first],[r['prediction'] for r in first])
    repeat_metrics=[]
    actual_repeats=repeat
    for index in range(actual_repeats):
        rr=[r for r in results if r['repeat']==index]
        repeat_metrics.append(metrics([r['target'] for r in rr],[r['prediction'] for r in rr]))
    times=np.array([r['total_ms'] for r in results])
    ends=np.array([r['end_offset_ms'] for r in results])
    starts=np.concatenate(([0.],ends[ends<=wall_seconds*1000-1000]))
    rolling=np.searchsorted(ends,starts+1000,side='left')-np.searchsorted(ends,starts,side='right')
    reference=dict(accuracy=129/146,precision=72/84,recall=72/77,f1=144/161)
    report=dict(split=args.split,metrics=quality,metrics_by_repeat=repeat_metrics,reference=reference,
        model_sha256=hashlib.sha256(modelpath.read_bytes()).hexdigest(),
        mean_ms=float(times.mean()),p95_ms=float(np.percentile(times,95)),max_ms=float(times.max()),fps=1000/float(times.mean()),
        fraction_under_41_67_ms=float(np.mean(times<=1000/24)),
        decode_mean_ms=float(np.mean([r['decode_ms'] for r in results])),
        inference_mean_ms=float(np.mean([r['inference_ms'] for r in results])),
        gpu=torch.cuda.get_device_name(),repeats=actual_repeats,measured_processing_seconds=measured_seconds,
        wall_seconds=wall_seconds,wall_fps=len(results)/wall_seconds,min_rolling_1s_fps=int(rolling.min()),
        input640=args.input640,half=args.half,vectorized=not args.sklearn,engine=args.engine,fusion=args.fusion,
        wrist_sha256=hashlib.sha256(wrist_path.read_bytes()).hexdigest() if wrist_path else None,
        input_manifest_sha256=hashlib.sha256(input_manifest.read_bytes()).hexdigest() if args.input640 else None,
        engine_sha256={name:hashlib.sha256((ROOT/'models'/name).read_bytes()).hexdigest() for name in ['yolo26m.engine','yolo26n-pose.engine']} if args.engine else None,
        protocol=('Prepared 640x640 JPEG read/decode every call; upstream resize excluded under user-authorized 640 input assumption. ' if args.input640 else 'Original JPEG read every call; CUDA JPEG decode, resize to max1280. ')+
        'One object and one pose prediction, CPU features and classifier. Resident models, batch1, CPU threads1, GPU synchronized, 8 warmups. No decoded-frame or feature cache. OS filesystem cache not flushed. Hash verification and result writes excluded.',
        quality_gate=all(all(m[k]>=v for k,v in reference.items()) for m in repeat_metrics),
        latency_gate=bool(times.mean()<=1000/24 and np.percentile(times,95)<=1000/24 and rolling.min()>=24 and len(results)/wall_seconds>=24))
    report['goal_proven']=args.split=='test' and report['quality_gate'] and report['latency_gate']
    suffix='_fp16' if args.half else ''
    if not args.sklearn:suffix+='_vectorized'
    if args.engine:suffix+='_trt'
    if args.fusion:suffix+='_fusion'
    (run/f'{args.split}_runtime{suffix}.json').write_text(json.dumps(report,indent=2))
    with (run/f'{args.split}_runtime{suffix}.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
