"""Paired test-set ablation; does not modify the production extractor/model."""
import csv
import hashlib
import json
import random
import time

import joblib
import numpy as np
import torch

from extract import ROOT, Extractor
from features import features
from cascade import ImageFallback

OUT = ROOT/'artifacts/yolo_optimization_20260916'
VARIANTS = ('baseline', '640', 'prune', '640_prune')


class ExperimentExtractor(Extractor):
    def refine(self, rgb, observation):
        import cv2
        _, info = features(observation, self.driver_side)
        if 'driver_index' not in info:
            return
        p = observation['poses'][info['driver_index']]
        h,w = rgb.shape[:2]
        shoulder = np.linalg.norm((np.asarray(p[11][:2])-p[12][:2])*[w,h])
        radius = max(80, shoulder*.7)
        evidence = {e['wrist']:e for e in info['evidence']}
        for wrist in (15,16):
            if p[wrist][3] < .35:
                continue
            cx,cy = p[wrist][0]*w,p[wrist][1]*h
            x1,y1,x2,y2 = max(0,int(cx-radius)),max(0,int(cy-radius)),min(w,int(cx+radius)),min(h,int(cy+radius))
            if x2-x1<20 or y2-y1<20:
                continue
            crop=rgb[y1:y2,x1:x2]
            e=evidence[wrist]
            skip = ('prune' in self.variant and e['phone_confidence'] >= .5
                    and e['phone_distance'] <= .25
                    and e['competitor_distance']-e['phone_distance'] >= .1)
            if skip:
                self.timings['skipped_calls'] += 1
            else:
                result=self.yolo.predict(cv2.cvtColor(crop,cv2.COLOR_RGB2BGR),classes=[39,41,67],
                                         conf=.15,imgsz=640,verbose=False,device=self.device)[0]
                for b in result.boxes:
                    coordinates=(b.xyxy[0].cpu().numpy()+[x1,y1,x1,y1])/[w,h,w,h]
                    observation['boxes'].append(dict(cls=int(b.cls[0]),confidence=float(b.conf[0]),
                                                     box=coordinates.tolist(),source='wrist_crop'))
            image=self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=np.ascontiguousarray(crop))
            for hand in self.hand.detect(image).hand_landmarks:
                observation['hands'].append([[(q.x*(x2-x1)+x1)/w,(q.y*(y2-y1)+y1)/h,q.z] for q in hand])


def main():
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((ROOT/'configs/dataset/test.csv').open(encoding='utf-8-sig')))
    for r in rows:
        r['path']=str(ROOT/r['path'])
        assert hashlib.sha256(open(r['path'],'rb').read()).hexdigest()==r['sha256']
    bundle=joblib.load(ROOT/'artifacts/classifier.joblib')
    engine=ExperimentExtractor(device='0')
    fallback=ImageFallback(bundle['cascade'],'0')
    original=engine.yolo.predict

    def yolo(*args,**kwargs):
        full=kwargs['imgsz']==1280
        # Ultralytics retains predictor kwargs between calls; reset every call.
        kwargs['rect']=True
        if full and '640' in engine.variant:
            kwargs.update(imgsz=640,rect=False)
        torch.cuda.synchronize()
        start=time.perf_counter()
        result=original(*args,**kwargs)
        torch.cuda.synchronize()
        key='full_yolo_ms' if full else 'crop_yolo_ms'
        engine.timings[key]+=(time.perf_counter()-start)*1000
        engine.timings['yolo_calls']+=1
        return result
    engine.yolo.predict=yolo

    def infer(r,variant):
        engine.variant=variant
        engine.timings=dict(full_yolo_ms=0.,crop_yolo_ms=0.,yolo_calls=0,skipped_calls=0)
        torch.cuda.synchronize()
        start=time.perf_counter()
        obs=engine.extract(r['path'])
        extracted=time.perf_counter()
        f,info=features(obs)
        stage=2 if f is None else 1
        if stage==1:
            score=float(bundle['model'].predict_proba([[f[k] for k in bundle['feature_names']]])[0,list(bundle['model'].classes_).index(1)])
        else:
            score=fallback.score(r['path'],obs)
        threshold=bundle['cascade'][f'stage{stage}_threshold']
        torch.cuda.synchronize()
        end=time.perf_counter()
        result=dict(variant=variant,path=r['path'],sha256=r['sha256'],target=int(r['target']),
                    prediction=int(score>=threshold),score=score,stage=stage,reason=info.get('reason',''),
                    total_ms=(end-start)*1000,extract_ms=(extracted-start)*1000,
                    decision_ms=(end-extracted)*1000,**engine.timings)
        return result,obs

    results=[]
    rng=random.Random(42)
    rng.shuffle(rows)
    try:
        for v in VARIANTS:
            for r in rows[:4]:
                infer(r,v)
        print('Warmup complete; 146 photos x 4 variants',flush=True)
        for i,r in enumerate(rows):
            order=list(VARIANTS)
            rng.shuffle(order)
            for v in order:
                result,obs=infer(r,v)
                results.append(result)
                cache=OUT/v
                cache.mkdir(exist_ok=True)
                (cache/(r['sha256']+'.json')).write_text(json.dumps(obs),encoding='utf-8')
            if (i+1)%10==0:
                print(f'{i+1}/{len(rows)}',flush=True)
                write_results(results,bundle)
    finally:
        engine.close()
    write_results(results,bundle)


def write_results(results,bundle):
    summaries={}
    for v in VARIANTS:
        rr=[r for r in results if r['variant']==v]
        tp=sum(r['target']==1 and r['prediction']==1 for r in rr)
        tn=sum(r['target']==0 and r['prediction']==0 for r in rr)
        fp=sum(r['target']==0 and r['prediction']==1 for r in rr)
        fn=sum(r['target']==1 and r['prediction']==0 for r in rr)
        means={k:float(np.mean([r[k] for r in rr])) for k in ('total_ms','extract_ms','decision_ms','full_yolo_ms','crop_yolo_ms','yolo_calls')}
        summaries[v]=dict(n=len(rr),tp=tp,tn=tn,fp=fp,fn=fn,accuracy=(tp+tn)/len(rr),
                          precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1),
                          means=means,fps=1000/means['total_ms'],p95_ms=float(np.percentile([r['total_ms'] for r in rr],95)),
                          yolo_calls=sum(r['yolo_calls'] for r in rr),skipped_calls=sum(r['skipped_calls'] for r in rr),
                          stage2_count=sum(r['stage']==2 for r in rr))
    with (OUT/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    (OUT/'summary.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),
        model_sha256=hashlib.sha256((ROOT/'artifacts/classifier.joblib').read_bytes()).hexdigest(),
        protocol='Frozen model and thresholds; all 146 test photos; random variant order per image; 4 warmups per variant; no cache; resident models; synchronized GPU; excludes cache/output writes. No retraining. Pruning rule fixed before evaluation.',
        resize='Only initial YOLO call: imgsz=640, rect=False (aspect-preserving square letterbox); MediaPipe and crop inputs unchanged.',
        pruning='Skip wrist YOLO if initial nearest-phone confidence >= .5, distance <= .25 shoulder widths, competitor distance margin >= .1; keep all hand inference.',
        variants=summaries),indent=2),encoding='utf-8')
    print({v:(round(s['fps'],2),round(s['f1'],4),s['yolo_calls']) for v,s in summaries.items()},flush=True)


if __name__=='__main__':
    main()
