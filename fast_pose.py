"""Experimental single-pass object + GPU pose realtime classifier."""
import cv2
import numpy as np
import joblib
import time
import torch
from PIL import Image
from extract import ROOT
from ultralytics import YOLO
from training.train_fast_pose import combined_features
from fast_forest import BatchOneForest


class FastPose:
    def __init__(self,model_path=None,half=False,vectorized=True,engine=False,wrist_checkpoint=None,wrist_threshold=.85):
        self.half=half
        self.bundle=joblib.load(model_path or ROOT/'artifacts/fast_pose_model/model.joblib')
        self.classifier=BatchOneForest(self.bundle['model']) if vectorized else self.bundle['model']
        suffix='.engine' if engine else '.pt'
        self.detector=YOLO(str(ROOT/'models'/('yolo26m'+suffix)),task='detect')
        self.pose=YOLO(str(ROOT/'models'/('yolo26n-pose'+suffix)),task='pose')
        self.wrist_model=None
        self.wrist_threshold=wrist_threshold
        if wrist_checkpoint:
            from cascade import new_model,PREPROCESS
            state=torch.load(wrist_checkpoint,map_location='cpu',weights_only=True)
            assert state['input_kind']=='wrists'
            self.wrist_model=new_model().cuda().eval()
            self.wrist_model.load_state_dict(state['state_dict'])
            self.wrist_preprocess=PREPROCESS

    @torch.inference_mode()
    def score_frame(self,rgb,profile=False,visualize=False):
        """EXIF-upright RGB ndarray; normalized coordinates in input frame space."""
        h,w=rgb.shape[:2]
        stamps=[time.perf_counter()]
        def stamp():
            if profile:torch.cuda.synchronize()
            stamps.append(time.perf_counter())
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        roi=self.bundle.get('detector_roi',[0,0,1,1])
        x1,y1,x2,y2=[round(v*s) for v,s in zip(roi,[w,h,w,h])]
        det=self.detector.predict(bgr[y1:y2,x1:x2],imgsz=640,rect=False,conf=self.bundle.get('detector_conf',.12),classes=[0,39,41,67],device=0,quantize=16 if self.half else 32,verbose=False)[0]
        stamp()
        pose=self.pose.predict(bgr,imgsz=640,rect=False,conf=.12,device=0,quantize=16 if self.half else 32,verbose=False)[0]
        stamp()
        boxes=[dict(cls=int(b[5]),confidence=float(b[4]),box=((b[:4]+[x1,y1,x1,y1])/[w,h,w,h]).tolist()) for b in det.boxes.data.cpu().numpy()]
        keypoints=pose.keypoints.data.cpu().numpy();keypoints[:,:,:2]/=[w,h]
        obs=dict(width=w,height=h,boxes=boxes)
        f,info=combined_features(obs,dict(keypoints=keypoints.tolist()))
        stamp()
        values=np.asarray([[f.get(k,-1) for k in self.bundle['feature_names']]],dtype=np.float32)
        s=float(self.classifier.predict_proba(values)[0,1])
        stamp()
        result=dict(score=s,prediction=int(s>=self.bundle['threshold']),pose_found=f['pose_found'],reason=info.get('reason',''),stage_used=1,stage1_score=s,stage2_score=None)
        if self.wrist_model is not None and not result['prediction']:
            from wrist_image import wrist_image
            crop=wrist_image(Image.fromarray(rgb),keypoints.tolist(),info.get('driver_index'))
            x=self.wrist_preprocess(crop).unsqueeze(0).cuda()
            image_score=float(self.wrist_model(x).softmax(1)[0,1])
            result.update(score=image_score,prediction=int(image_score>=self.wrist_threshold),stage_used=2,stage2_score=image_score)
        if visualize:
            result['visualization']=dict(boxes=boxes,keypoints=keypoints.tolist(),
                                         driver_index=info.get('driver_index'),roi=roi)
        if profile:result.update({k:(b-a)*1000 for k,a,b in zip(['detector_ms','pose_ms','features_ms','forest_ms'],stamps,stamps[1:])})
        return result
