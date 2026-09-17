"""Extract official YOLO26n pose keypoints for training-only realtime experiments."""
import csv
import argparse
import json
import hashlib
import numpy as np
import torch
import cv2
from PIL import Image, ImageOps
from extract import ROOT
from ultralytics import YOLO


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input640',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    weight=ROOT/'models/yolo26n-pose.pt'
    model=YOLO(str(weight));digest=hashlib.sha256(weight.read_bytes()).hexdigest()
    out=ROOT/('artifacts/fast_pose_input640/train' if args.input640 else 'artifacts/fast_pose/train');out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader((ROOT/'configs/dataset/train.csv').open(encoding='utf-8-sig')))
    for i,r in enumerate(rows):
        dest=out/(r['sha256']+'.json')
        if dest.exists():continue
        path=ROOT/r['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
        input_path=ROOT/'artifacts/input640/images'/(r['sha256']+'.jpg') if args.input640 else path
        with Image.open(input_path) as im:rgb=np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
        h,w=rgb.shape[:2];factor=min(1,1280/max(h,w))
        rgb=cv2.resize(rgb,(round(w*factor),round(h*factor)));h,w=rgb.shape[:2]
        result=model.predict(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),imgsz=640,rect=False,conf=.12,device=0,verbose=False)[0]
        keypoints=result.keypoints.data.cpu().numpy()
        keypoints[:,:,:2]/=[w,h]
        dest.write_text(json.dumps(dict(keypoints=keypoints.tolist(),width=w,height=h,model_sha256=digest,imgsz=640)))
        if (i+1)%50==0:print(f'{i+1}/{len(rows)}',flush=True)
    print('Complete',flush=True)


if __name__=='__main__':main()
