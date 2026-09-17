"""Screen a single-detector geometry classifier on train validation only."""
import argparse
import csv
import json
import math
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from extract import ROOT
from features import driver_person_box
from training.train_cascade import metrics


def box_features(obs):
    # Never use wrist crop results: runtime requires only one detector call.
    boxes=[b for b in obs['boxes'] if b.get('source')!='wrist_crop']
    driver=driver_person_box(boxes,'right')
    f={'driver_found':int(driver is not None)}
    x1,y1,x2,y2=driver or [.42,.20,1,1]
    w,h=max(x2-x1,.01),max(y2-y1,.01)
    f.update(driver_x=x1,driver_y=y1,driver_w=w,driver_h=h)
    people=[b for b in boxes if b['cls']==0]
    f['people_count']=len(people)
    for cls,name in ((67,'phone'),(39,'bottle'),(41,'cup')):
        objects=[b for b in boxes if b['cls']==cls]
        f[name+'_count']=len(objects)
        def rank(b):
            a,c,d,e=b['box'];cx,cy=(a+d)/2,(c+e)/2
            dx=max(x1-cx,0,cx-x2)/w;dy=max(y1-cy,0,cy-y2)/h
            return b['confidence']*math.exp(-4*(dx+dy))
        objects.sort(key=rank,reverse=True)
        for i in range(4):
            if i>=len(objects):
                values=[-1]*10
            else:
                b=objects[i];a,c,d,e=b['box'];cx,cy=(a+d)/2,(c+e)/2
                overlap=max(0,min(d,x2)-max(a,x1))*max(0,min(e,y2)-max(c,y1))/max((d-a)*(e-c),1e-6)
                other_overlap=[]
                for p in people:
                    if p['box']==driver:continue
                    px,py,qx,qy=p['box']
                    other_overlap.append(max(0,min(d,qx)-max(a,px))*max(0,min(e,qy)-max(c,py))/max((d-a)*(e-c),1e-6))
                values=[b['confidence'],(cx-x1)/w,(cy-y1)/h,(d-a)/w,(e-c)/h,overlap,
                        max(other_overlap,default=0),rank(b),cx,cy]
            for k,v in enumerate(values):f[f'{name}_{i}_{k}']=v
    return f


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',default='artifacts/retrain_20260916_v2/inputs/train/cache')
    parser.add_argument('--output',default='artifacts/fast_boxes_screen1280')
    args=parser.parse_args()
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((ROOT/'configs/dataset/train.csv').open(encoding='utf-8-sig')))
    split=json.loads((ROOT/'configs/dataset/train_validation.json').read_text())
    fs=[box_features(json.loads((ROOT/args.cache/(r['sha256']+'.json')).read_text())) for r in rows]
    names=sorted(fs[0]);x=np.asarray([[f[k] for k in names] for f in fs]);y=np.array([int(r['target']) for r in rows])
    fit=[i for i,r in enumerate(rows) if r['sha256'] in set(split['fit_hashes'])]
    val=[i for i,r in enumerate(rows) if r['sha256'] in set(split['validation_hashes'])]
    best=None;history=[]
    for leaf in (2,5,10):
        m=ExtraTreesClassifier(n_estimators=200,min_samples_leaf=leaf,max_features=.8,class_weight='balanced',random_state=42,n_jobs=1)
        m.fit(x[fit],y[fit]);s=m.predict_proba(x[val])[:,1]
        np.save(out/f'validation_scores_leaf{leaf}.npy',s)
        for t in np.arange(.1,.901,.025):
            r=dict(leaf=leaf,threshold=float(t),**metrics(y[val],s>=t))
            history.append(r)
            if best is None or (r['f1'],r['recall'])>(best['f1'],best['recall']):best=r
    report=dict(best=best,history=history,cache=args.cache,feature_names=names,
                note='Train validation only; no MediaPipe, no wrist crops; 1280 screening is not 640 deployment evidence.')
    (out/'validation.json').write_text(json.dumps(report,indent=2))
    np.save(out/'validation_targets.npy',y[val])
    print(json.dumps(best),flush=True)


if __name__=='__main__':main()
