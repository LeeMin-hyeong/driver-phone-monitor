"""Fit fast detector + GPU pose features on the shared train validation split."""
import csv
import argparse
import json
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from extract import ROOT
from features import features
from training.train_fast_boxes import box_features
from training.train_cascade import metrics

MAP={0:0,2:1,5:2,7:3,8:4,11:5,12:6,13:7,14:8,15:9,16:10,23:11,24:12}


def combined_features(obs,pose):
    mapped=[]
    for p in pose['keypoints']:
        q=[[0.,0.,0.,0.] for _ in range(33)]
        for dst,src in MAP.items():q[dst]=[p[src][0],p[src][1],0.,p[src][2]]
        mapped.append(q)
    observation=dict(obs,poses=mapped,hands=[])
    f,info=features(observation)
    g={'box_'+k:v for k,v in box_features(obs).items()}
    g['pose_found']=int(f is not None)
    # Only real COCO keypoints and wrist-object relations, no invented fingers.
    if f is not None:
        g.update({k:float(v) for k,v in f.items() if not k.startswith(('p19_','p20_'))
                  and not any(part in k for part in ('finger','palm','tip','hand'))})
    return g,info


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',default='artifacts/single_yolo640/train')
    parser.add_argument('--output',default='artifacts/fast_pose_model')
    parser.add_argument('--pose-cache',default='artifacts/fast_pose/train')
    args=parser.parse_args()
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((ROOT/'configs/dataset/train.csv').open(encoding='utf-8-sig')))
    split=json.loads((ROOT/'configs/dataset/train_validation.json').read_text())
    fs=[]
    for r in rows:
        obs=json.loads((ROOT/args.cache/(r['sha256']+'.json')).read_text())
        pose=json.loads((ROOT/args.pose_cache/(r['sha256']+'.json')).read_text())
        f,_=combined_features(obs,pose);fs.append(f)
    names=sorted(set().union(*(f.keys() for f in fs)))
    x=np.asarray([[f.get(k,-1) for k in names] for f in fs]);y=np.array([int(r['target']) for r in rows])
    fit=[i for i,r in enumerate(rows) if r['sha256'] in set(split['fit_hashes'])]
    val=[i for i,r in enumerate(rows) if r['sha256'] in set(split['validation_hashes'])]
    best=None;history=[]
    for leaf in (2,5,10):
        model=ExtraTreesClassifier(n_estimators=200,min_samples_leaf=leaf,max_features=.8,class_weight='balanced',random_state=42,n_jobs=1)
        model.fit(x[fit],y[fit]);s=model.predict_proba(x[val])[:,1]
        np.save(out/f'validation_scores_leaf{leaf}.npy',s)
        for t in np.arange(.1,.901,.025):
            m=dict(leaf=leaf,threshold=float(t),**metrics(y[val],s>=t));history.append(m)
            if best is None or (m['f1'],m['recall'])>(best['f1'],best['recall']):best=m
    locked=dict(best=best,feature_names=names,history=history,train_count=len(rows),detector_roi=obs.get('detector_roi',[0,0,1,1]),detector_conf=obs.get('detector_conf',.12),
                train_hashes=[r['sha256'] for r in rows],validation_pose_missing=sum(not fs[i]['pose_found'] for i in val),
                selection='Shared train validation F1, ties recall. No test used. YOLO26m640 plus YOLO26n-pose640, no MediaPipe or wrist crops.')
    (out/'locked_selection.json').write_text(json.dumps(locked,indent=2))
    model=ExtraTreesClassifier(n_estimators=200,min_samples_leaf=best['leaf'],max_features=.8,class_weight='balanced',random_state=42,n_jobs=1)
    model.fit(x[fit],y[fit])
    joblib.dump(dict(model=model,feature_names=names,threshold=best['threshold'],train_hashes=[rows[i]['sha256'] for i in fit],detector_roi=locked['detector_roi'],detector_conf=locked['detector_conf']),out/'validation.joblib')
    model.fit(x,y)
    joblib.dump(dict(model=model,feature_names=names,threshold=best['threshold'],train_hashes=locked['train_hashes'],detector_roi=locked['detector_roi'],detector_conf=locked['detector_conf']),out/'model.joblib')
    print(json.dumps(best),flush=True)


if __name__=='__main__':main()
