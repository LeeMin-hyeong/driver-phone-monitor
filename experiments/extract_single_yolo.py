"""Cache exactly one 640 YOLO call per training photo, without pose or hand models."""
import csv
import argparse
import hashlib
import json
import numpy as np
import cv2
import torch
from PIL import Image, ImageOps
from extract import ROOT
from ultralytics import YOLO


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--driver-roi',action='store_true')
    parser.add_argument('--input640',action='store_true')
    parser.add_argument('--conf',type=float,default=.12)
    args=parser.parse_args()
    torch.set_num_threads(1)
    model=YOLO(str(ROOT/'models/yolo26m.pt'))
    folder='single_yolo640'+('_roi' if args.driver_roi else '')+('_input640' if args.input640 else '')
    if args.conf!=.12:folder+='_conf'+str(args.conf)
    out=ROOT/'artifacts'/folder/'train';out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((ROOT/'configs/dataset/train.csv').open(encoding='utf-8-sig')))
    digest=hashlib.sha256((ROOT/'models/yolo26m.pt').read_bytes()).hexdigest()
    for i,r in enumerate(rows):
        path=ROOT/r['path'];dest=out/(r['sha256']+'.json')
        if dest.exists():continue
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
        input_path=ROOT/'artifacts/input640/images'/(r['sha256']+'.jpg') if args.input640 else path
        with Image.open(input_path) as im:rgb=np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
        h,w=rgb.shape[:2];factor=min(1,1280/max(h,w))
        rgb=cv2.resize(rgb,(round(w*factor),round(h*factor)));h,w=rgb.shape[:2]
        roi=[.35,.20,1.,1.] if args.driver_roi else [0.,0.,1.,1.]
        x1,y1,x2,y2=[round(v*s) for v,s in zip(roi,[w,h,w,h])]
        detector_image=rgb[y1:y2,x1:x2]
        result=model.predict(cv2.cvtColor(detector_image,cv2.COLOR_RGB2BGR),imgsz=640,rect=False,conf=args.conf,classes=[0,39,41,67],device=0,verbose=False)[0]
        boxes=[dict(cls=int(b.cls[0]),confidence=float(b.conf[0]),box=((b.xyxy[0].cpu().numpy()+[x1,y1,x1,y1])/[w,h,w,h]).tolist()) for b in result.boxes]
        dest.write_text(json.dumps(dict(width=w,height=h,boxes=boxes,yolo_sha256=digest,imgsz=640,rect=False,detector_roi=roi,detector_conf=args.conf,source_sha256=r['sha256'])))
        if (i+1)%50==0:print(f'{i+1}/{len(rows)}',flush=True)
    print('Complete',flush=True)


if __name__=='__main__':main()
