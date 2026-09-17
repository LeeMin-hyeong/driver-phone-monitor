"""Verify unchanged forest scores and benchmark all training feature rows."""
import csv
import json
import time
import joblib
import numpy as np
from extract import ROOT
from fast_forest import BatchOneForest
from training.train_fast_pose import combined_features


def main():
    run=ROOT/'artifacts/fast_pose640_sensitive'
    bundle=joblib.load(run/'model.joblib')
    reference=bundle['model'];fast=BatchOneForest(reference)
    rows=list(csv.DictReader((ROOT/'configs/dataset/train.csv').open(encoding='utf-8-sig')))
    differences=[];times={};scores=[]
    xx=[]
    for r in rows:
        obs=json.loads((ROOT/'artifacts/single_yolo640_roi_input640_conf0.01/train'/(r['sha256']+'.json')).read_text())
        pose=json.loads((ROOT/'artifacts/fast_pose_input640/train'/(r['sha256']+'.json')).read_text())
        f,_=combined_features(obs,pose)
        xx.append(np.asarray([[f.get(k,-1) for k in bundle['feature_names']]],dtype=np.float32))
    for name,model in [('sklearn',reference),('vectorized',fast)]:
        for x in xx[:10]:model.predict_proba(x)
        values=[];elapsed=[]
        for x in xx:
            start=time.perf_counter();values.append(model.predict_proba(x));elapsed.append((time.perf_counter()-start)*1000)
        scores.append(np.asarray(values))
        times[name]=dict(mean_ms=float(np.mean(elapsed)),p95_ms=float(np.percentile(elapsed,95)))
    diff=float(np.max(np.abs(scores[0]-scores[1])))
    assert diff<1e-12
    assert np.array_equal(scores[0][:,0,1]>=bundle['threshold'],scores[1][:,0,1]>=bundle['threshold'])
    report=dict(n=len(rows),max_probability_difference=diff,all_predictions_equal=True,times=times,
                note='All 911 train feature rows, learned trees unchanged; this is numerical equivalence and CPU timing, not a generalization test.')
    (run/'forest_equivalence.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
