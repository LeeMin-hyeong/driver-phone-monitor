"""Evaluate a locked fast ROI candidate, separately reporting file and frame latency."""
import argparse
import csv
import hashlib
import json
import random
import time
import numpy as np
from PIL import Image, ImageOps
import torch
from extract import ROOT
from fast_roi import FastROI
from training.train_cascade import metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='artifacts/fast_roi_v1')
    parser.add_argument('--split',choices=['train','test'],default='train')
    args=parser.parse_args()
    run=ROOT/args.run
    locked=json.loads((run/'locked_selection.json').read_text())
    torch.set_num_threads(1)
    engine=FastROI(run/'model.pt')
    rows=list(csv.DictReader((ROOT/f'configs/dataset/{args.split}.csv').open(encoding='utf-8-sig')))
    if args.split=='train':
        # Timing probe only: refit model has seen all these samples.
        random.Random(42).shuffle(rows)
        rows=rows[:40]
    else:
        assert not set(locked['training_hashes'])&{r['sha256'] for r in rows}
    modes=['original_file','reduced_jpeg_file','decoded_1280_frame']
    rng=random.Random(42);rng.shuffle(rows)
    for mode in modes:
        for r in rows[:5]:
            engine.score_file(ROOT/r['path'],reduced=mode=='reduced_jpeg_file')
    results=[]
    for i,r in enumerate(rows):
        path=ROOT/r['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
        with Image.open(path) as source:
            frame=ImageOps.exif_transpose(source).convert('RGB')
        frame.thumbnail((1280,1280),Image.Resampling.BILINEAR)
        order=modes.copy();rng.shuffle(order)
        for mode in order:
            for repeat in range(3):
                torch.cuda.synchronize();start=time.perf_counter()
                s=engine.score_frame(frame) if mode=='decoded_1280_frame' else engine.score_file(path,reduced=mode=='reduced_jpeg_file')
                torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
                results.append(dict(path=str(path),sha256=r['sha256'],target=int(r['target']),mode=mode,
                                    repeat=repeat,score=s,prediction=int(s>=engine.threshold),ms=ms))
        if (i+1)%20==0: print(f'{i+1}/{len(rows)}',flush=True)
    report=dict(split=args.split,model_sha256=hashlib.sha256((run/'model.pt').read_bytes()).hexdigest(),
                threshold=engine.threshold,gpu=torch.cuda.get_device_name(),
                scope='Resident model; batch1 FP32; CPU threads1; 5 warmups; 3 repeats/image; GPU synchronized; decoded frame excludes camera decode and initial resize; file modes include read/decode. Train subset is latency probe, not generalization evidence.',modes={})
    for mode in modes:
        rr=[r for r in results if r['mode']==mode]
        first=[r for r in rr if r['repeat']==0]
        times=np.array([r['ms'] for r in rr])
        report['modes'][mode]=dict(**metrics([r['target'] for r in first],[r['prediction'] for r in first]),
            mean_ms=float(times.mean()),median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),
            max_ms=float(times.max()),fps=1000/float(times.mean()),fraction_under_41_67_ms=float(np.mean(times<=1000/24)))
    target=run/f'{args.split}_evaluation.json'
    target.write_text(json.dumps(report,indent=2))
    with (run/f'{args.split}_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
