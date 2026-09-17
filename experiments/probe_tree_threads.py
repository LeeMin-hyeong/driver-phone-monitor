"""Measure batch-one ExtraTrees only; weights and on-disk model stay unchanged."""
import csv
import json
import time
import random
import joblib
import numpy as np
from extract import ROOT
from features import features


def main():
    bundle=joblib.load(ROOT/'artifacts/classifier.joblib')
    model=bundle['model']
    samples=[]
    for r in csv.DictReader((ROOT/'configs/dataset/test.csv').open(encoding='utf-8-sig')):
        obs=json.loads((ROOT/'artifacts/yolo_optimization_20260916/640_prune'/(r['sha256']+'.json')).read_text())
        f,_=features(obs)
        if f is not None:
            samples.append(np.asarray([[f[k] for k in bundle['feature_names']]],dtype=np.float32))
    times={1:[],4:[]};diffs=[]
    for jobs in times:
        model.n_jobs=jobs
        for x in samples[:5]:
            model.predict_proba(x)
    rng=random.Random(42)
    for x in samples:
        order=[1,4];rng.shuffle(order);scores={}
        for jobs in order:
            model.n_jobs=jobs
            start=time.perf_counter()
            scores[jobs]=model.predict_proba(x)
            times[jobs].append((time.perf_counter()-start)*1000)
        diffs.append(float(np.max(np.abs(scores[1]-scores[4]))))
    report=dict(scope='ExtraTrees predict_proba only; cached 640_prune features; batch1; five warmups per setting; randomized order; model not saved',
                n=len(samples),max_score_difference=max(diffs),
                results={str(j):dict(mean_ms=float(np.mean(v)),median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95))) for j,v in times.items()})
    (ROOT/'docs/evidence/tree_threads_probe.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
