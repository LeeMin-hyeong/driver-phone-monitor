"""Plot actual TensorRT detections and pose coordinates on 640 test inputs."""
import json,hashlib,zipfile
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from extract import ROOT
from realtime import load_engine,read640
from training.train_fast_pose import combined_features
from wrist_image import wrist_image
OUT=ROOT/'docs/assets/realtime640'
FONT='C:/Windows/Fonts/malgun.ttf';BOLD='C:/Windows/Fonts/malgunbd.ttf'
def font(n,b=False):return ImageFont.truetype(BOLD if b else FONT,n)
def label(d,xy,t,c,size=16):
 box=d.textbbox(xy,t,font=font(size,True));d.rectangle((box[0]-3,box[1]-3,box[2]+3,box[3]+3),fill='#142134');d.text(xy,t,font=font(size,True),fill=c)
def main():
 torch.set_num_threads(1)
 config=json.loads((ROOT/'configs/realtime_model.json').read_text());engine=load_engine(config)
 rows=json.loads((OUT/'candidates.json').read_text(encoding='utf-8'))
 selected=[rows[0],rows[4],rows[9]];entries=[]
 for idx,r in enumerate(selected):
  path=ROOT/'artifacts/input640/images'/ (r['sha256']+'.jpg');rgb=read640(path);result=engine.score_frame(rgb)
  assert result['prediction']==int(r['prediction']) and result['stage_used']==int(r['stage_used'])
  det=engine.detector.predictor.results[0];pose=engine.pose.predictor.results[0]
  kp=pose.keypoints.data.cpu().numpy().copy();kp[:,:,:2]/=[640,640]
  roi=engine.bundle['detector_roi'];x1,y1,x2,y2=[round(v*640) for v in roi]
  boxes=[dict(cls=int(b[5]),confidence=float(b[4]),box=((b[:4]+[x1,y1,x1,y1])/640).tolist()) for b in det.boxes.data.cpu().numpy()]
  _,info=combined_features(dict(width=640,height=640,boxes=boxes),dict(keypoints=kp.tolist()))
  driver=info.get('driver_index');im=Image.fromarray(rgb);d=ImageDraw.Draw(im)
  # Object detector only sees its configured driver ROI. All phone candidates shown.
  d.rectangle((x1,y1,x2-1,y2-1),outline='#58cfff',width=2)
  for pi,p in enumerate(kp):
   color='#43f5a5' if pi==driver else '#ffbf59'
   for a,b in [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]:
    if min(p[a,2],p[b,2])>=.4:d.line((p[a,0]*640,p[a,1]*640,p[b,0]*640,p[b,1]*640),fill=color,width=3)
   for j in range(5,17):
    if p[j,2]>=.4:
     x,y=p[j,:2]*640;d.ellipse((x-4,y-4,x+4,y+4),fill=color)
   if pi==driver and p[5,2]>=.4:label(d,(max(5,min(530,p[5,0]*640)),max(85,p[5,1]*640-28)),'운전자',color)
  phones=[b for b in boxes if b['cls']==67]
  for b in phones:
   a,c,e,f=[v*640 for v in b['box']];d.rectangle((a,c,e,f),outline='#ff4c79',width=3)
  if phones:label(d,(238,140),f"휴대폰 후보 {len(phones)}개",'#ff7094',16)
  d.rectangle((0,0,640,68),fill='#142134')
  title=['01 양성 · 운전자 휴대폰 소지','02 음성 · 동승자 사용과 구분','03 1차 미탐 · ResNet 보완'][idx]
  d.text((15,8),title,font=font(23,True),fill='white')
  d.text((15,40),'640 입력  /  YOLO26m + YOLO26n-pose',font=font(15),fill='#bcd0e3')
  d.rectangle((0,560,640,640),fill='#142134')
  text=f"1차 {result['stage1_score']:.3f}"
  if result['stage_used']==2:text+=f"  →  ResNet {result['stage2_score']:.3f}"
  text+='  →  '+('양성' if result['prediction'] else '음성')
  d.text((12,565),text,font=font(18,True),fill='#43f5a5' if result['prediction'] else '#58cfff')
  d.text((12,593),'초록: 운전자 자세  주황: 기타 자세  분홍: 휴대폰 후보',font=font(14),fill='white')
  d.text((12,615),'하늘색: 검출 ROI  |  1차 기준 0.25 / ResNet 기준 0.85',font=font(13),fill='#bcd0e3')
  if idx==2:
   crop=wrist_image(Image.fromarray(rgb),kp.tolist(),driver)
   crop.save(OUT/'resnet_input_448x224.png')
   im.paste(crop.resize((224,112)),(392,100));d=ImageDraw.Draw(im)
   d.rectangle((391,99,616,212),outline='#58cfff',width=2)
   label(d,(394,77),'실제 양손목 crop','#58cfff',15)
  name=['01_positive.png','02_negative.png','03_resnet_fallback.png'][idx];im.save(OUT/name)
  entry=dict(asset=name,source=r['path'],source_sha256=r['sha256'],input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),target=int(r['target']),result=result,driver_index=driver,boxes=boxes,keypoints=kp.tolist())
  entries.append(entry)
 manifest=dict(config=config,image_size=[640,640],pose_visibility_threshold=.4,notes=['Coordinates are actual final TensorRT model output. No invented detections.','All phone candidates at detector threshold 0.01 are shown inside fixed ROI.','Stage 2 checks all first-stage negatives, not only unavailable scores.','ResNet scores and forest scores are not calibrated probabilities.'],assets=entries)
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 (OUT/'README.md').write_text('# 최종 640 입력 시각화\n\n640×640 실제 평가 이미지에 최종 TensorRT 객체 검출·자세 추정 좌표를 표시했습니다. 원본을 확대해 추론하지 않았습니다.\n\n- 01: 1차 양성\n- 02: 동승자가 휴대폰을 사용하지만 운전자는 음성. 1차 음성 후 ResNet도 음성.\n- 03: 1차 음성을 손목 ResNet이 양성으로 보완. 삽입 이미지는 실제 손목 crop이며 전체 448×224 원본도 제공합니다.\n\n초록: 운전자 자세, 주황: 기타 자세, 분홍: ROI 안의 모든 휴대폰 후보(conf ≥ 0.01), 하늘색: 고정 객체 검출 ROI. 자세 점은 신뢰도 0.4 이상만 표시합니다. ROI 바깥의 동승자 휴대폰 박스는 검출하지 않으므로 그리지 않았습니다. 검출 박스가 최종 양성을 뜻하지 않습니다.\n\n현재 구조에는 판단 불가 출력이 없으며, 1차 점수 0.25 미만은 ResNet으로 재검사합니다. 점수는 보정된 확률이 아닙니다.\n',encoding='utf-8')
 with zipfile.ZipFile(OUT/'realtime640_assets.zip','w',zipfile.ZIP_DEFLATED) as z:
  for n in ['01_positive.png','02_negative.png','03_resnet_fallback.png','resnet_input_448x224.png','manifest.json','README.md']:z.write(OUT/n,n)
 print([(e['asset'],e['result']) for e in entries])
if __name__=='__main__':main()

