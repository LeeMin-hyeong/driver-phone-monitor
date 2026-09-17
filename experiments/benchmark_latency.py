"""Measure uncached, batch-1 GPU inference with resident models and CUDA sync."""
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from extract import ROOT, Extractor
from features import features
from cascade import ImageFallback


def main():
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    output = ROOT/'artifacts/latency_20260916'
    output.mkdir(exist_ok=True)
    with (ROOT/'artifacts/retrain_20260916_v2/model/predictions.csv').open(encoding='utf-8-sig') as f:
        pool = list(csv.DictReader(f))
    rng = random.Random(42)
    rows = rng.sample([r for r in pool if r['stage_used']=='1'],24) + rng.sample([r for r in pool if r['stage_used']=='2'],6)
    rng.shuffle(rows)
    started = time.perf_counter()
    bundle = joblib.load(ROOT/'artifacts/classifier.joblib')
    engine = Extractor(device='0')
    fallback = ImageFallback(bundle['cascade'], '0')
    torch.cuda.synchronize()
    load_seconds = time.perf_counter()-started
    timings = {}

    def instrument(obj, name, key):
        original = getattr(obj,name)
        def wrapped(*args, **kwargs):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            result = original(*args, **kwargs)
            torch.cuda.synchronize()
            timings[key] = timings.get(key,0)+(time.perf_counter()-begin)*1000
            return result
        setattr(obj,name,wrapped)
    instrument(engine.yolo,'predict','yolo_ms')
    instrument(engine.pose,'detect','pose_ms')
    instrument(engine.hand,'detect','hands_ms')

    def infer(row):
        timings.clear()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        obs = engine.extract(row['path'])
        after_extract = time.perf_counter()
        f,_ = features(obs)
        stage = 2 if f is None else 1
        if stage == 1:
            score = float(bundle['model'].predict_proba([[f[k] for k in bundle['feature_names']]])[0,1])
        else:
            score = fallback.score(row['path'],obs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        return dict(path=row['path'],stage=stage,score=score,total_ms=(end-begin)*1000,
                    extract_ms=(after_extract-begin)*1000,decision_ms=(end-after_extract)*1000,**timings)

    measured=[]
    try:
        # Warm both branches, CUDA kernels, and full/wrist-crop detector shapes.
        warm = [next(r for r in pool if r['stage_used']==s) for s in ('1','2')]
        begin = time.perf_counter()
        for row in warm*2:
            infer(row)
        warm_seconds = time.perf_counter()-begin
        for i,row in enumerate(rows):
            measured.append(infer(row))
            print(f'{i+1}/{len(rows)} stage={measured[-1]["stage"]} total_ms={measured[-1]["total_ms"]:.1f}',flush=True)
    finally:
        engine.close()

    def summarize(items):
        values = [r['total_ms'] for r in items]
        return dict(n=len(items),mean_ms=float(np.mean(values)),median_ms=float(np.median(values)),
                    p95_ms=float(np.percentile(values,95)),max_ms=max(values),min_ms=min(values),
                    serial_fps=1000/float(np.mean(values)),under_33_33_ms=sum(v<=1000/30 for v in values),
                    component_mean_ms={k:float(np.mean([r.get(k,0) for r in items])) for k in
                                       ('extract_ms','yolo_ms','pose_ms','hands_ms','decision_ms')})
    report = dict(gpu=torch.cuda.get_device_name(0),torch_version=torch.__version__,torch_cpu_threads=4,
                  model_sha256=hashlib.sha256((ROOT/'artifacts/classifier.joblib').read_bytes()).hexdigest(),
                  load_seconds=load_seconds,warmup_seconds=warm_seconds,warmup_images=4,
                  overall=summarize(measured),stage1=summarize([r for r in measured if r['stage']==1]),
                  stage2=summarize([r for r in measured if r['stage']==2]),
                  protocol='30 test photos stratified by route (24 stage1, 6 stage2), seed42; serial batch1; resident models; no feature cache; original file read/decode included; CUDA synchronized; output rendering/writing and process imports excluded. Includes synchronization instrumentation overhead. Sample latency, not a hard deadline guarantee. Current main.py reloads models per image and will be slower.')
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (output/'per_image.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(measured[0]));writer.writeheader();writer.writerows(measured)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
