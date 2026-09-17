"""Render factual model-output diagrams from the frozen test evaluation."""
import csv
import hashlib
import json
import argparse
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from features import features, driver_person_box

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/retrain_20260916_v2'
OUT = ROOT / 'docs/assets/approach'
FONT = 'C:/Windows/Fonts/malgun.ttf'
BOLD = 'C:/Windows/Fonts/malgunbd.ttf'


def font(n, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, n)


def records():
    rows = list(csv.DictReader((RUN / 'verified/predictions.csv').open(encoding='utf-8-sig')))
    for r in rows:
        r['obs'] = json.loads((RUN / 'verified/cache' / (r['sha256'] + '.json')).read_text(encoding='utf-8'))
        _, r['info'] = features(r['obs'])
    return rows


def photo(row):
    with Image.open(row['path']) as im:
        return ImageOps.exif_transpose(im).convert('RGB')


def candidates():
    rows = records()
    positive = [r for r in rows if r['target']=='1' and r['status']=='positive' and r['stage_used']=='1']
    negative = [r for r in rows if r['target']=='0' and r['status']=='negative' and r['stage_used']=='1']
    fallback = [r for r in rows if r['target']=='1' and r['status']=='positive' and r['stage_used']=='2']
    positive.sort(key=lambda r: -float(r['score']))
    negative.sort(key=lambda r: (-sum(b['cls']==67 for b in r['obs']['boxes']), float(r['score'])))
    chosen = positive[:6] + negative[:6] + fallback[:6]
    canvas = Image.new('RGB', (1500, 330*6), '#eef2f6')
    d = ImageDraw.Draw(canvas)
    for i,r in enumerate(chosen):
        x,y = (i%3)*500, (i//3)*330
        im=ImageOps.contain(photo(r),(490,265))
        canvas.paste(im,(x+(490-im.width)//2,y))
        d.text((x+8,y+267),f"{Path(r['path']).name[:13]} | y={r['target']} S{r['stage_used']} {float(r['score']):.3f}",font=font(18),fill='#132239')
        d.text((x+8,y+295),r['reason'] or 'landmarks',font=font(16),fill='#132239')
    OUT.mkdir(parents=True,exist_ok=True)
    canvas.save(OUT/'candidates.jpg')
    print(OUT/'candidates.jpg')
    print([(Path(r['path']).name,r['stage_used'],r['score']) for r in chosen])


BG = '#0d1728'
TEXT = '#eff5ff'
MUTED = '#aabbcf'
GREEN = '#55e3a4'
CYAN = '#64cfff'
ORANGE = '#ffbd69'
RED = '#ff6584'


def label(d, xy, text, color, size=23):
    f=font(size,True)
    box=d.textbbox(xy,text,font=f)
    d.rounded_rectangle((box[0]-8,box[1]-6,box[2]+8,box[3]+7),radius=5,fill=BG)
    d.text(xy,text,font=f,fill=color)


def rendered_photo(row, size):
    im=ImageOps.contain(photo(row),size)
    d=ImageDraw.Draw(im)
    w,h=im.size
    obs,info=row['obs'],row['info']
    driver=info.get('driver_index')
    for i,p in enumerate(obs['poses']):
        color=GREEN if i==driver else ORANGE
        for a,b in ((11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24)):
            if min(p[a][3],p[b][3])>=.4:
                d.line((p[a][0]*w,p[a][1]*h,p[b][0]*w,p[b][1]*h),fill=color,width=4)
        for j in (11,12,13,14,15,16,23,24):
            if p[j][3]>=.4:
                x,y=p[j][0]*w,p[j][1]*h
                d.ellipse((x-5,y-5,x+5,y+5),fill=color,outline=BG,width=1)
    # Show raw phone detections, including wrist-crop detections, without inventing landmarks.
    for b in obs['boxes']:
        if b['cls']==67:
            x1,y1,x2,y2=[v*s for v,s in zip(b['box'],(w,h,w,h))]
            d.rectangle((x1,y1,x2,y2),outline=RED,width=3)
    box=driver_person_box(obs['boxes'],'right')
    if box:
        x1,y1,x2,y2=[v*s for v,s in zip(box,(w,h,w,h))]
        d.rectangle((x1,y1,x2,y2),outline=CYAN,width=3)
        label(d,(max(10,x1+10),max(15,y1-35)),'운전자 후보' if driver is None else '운전자',CYAN)
    return im


def base(number,title,subtitle,row):
    im=Image.new('RGB',(1920,1080),BG)
    d=ImageDraw.Draw(im)
    d.text((64,38),f'APPROACH / {number}',font=font(22,True),fill=CYAN)
    d.text((64,84),title,font=font(48,True),fill=TEXT)
    d.text((66,154),subtitle,font=font(25),fill=MUTED)
    d.line((64,987,1856,987),fill='#314158',width=2)
    d.text((64,1012),f"실제 test 이미지 · {Path(row['path']).name} · 확정 모델 v2",font=font(20),fill=MUTED)
    d.text((1150,1012),'검출 좌표 기반 시각화 | 점수는 보정된 확률이 아님',font=font(20),fill=MUTED)
    return im


def legend(d,x,y):
    for color,text in ((CYAN,'운전자 영역'),(GREEN,'운전자 자세'),(ORANGE,'기타 검출 자세'),(RED,'휴대폰 검출')):
        d.line((x,y+14,x+36,y+14),fill=color,width=5)
        d.text((x+48,y),text,font=font(23),fill=MUTED)
        y+=42


def stage1_asset(row,positive):
    im=base('01' if positive else '02',
        '양성 · 운전자가 휴대폰을 든 경우' if positive else '음성 · 동승자의 휴대폰을 구분하는 경우',
        'YOLO26m 객체 검출 + MediaPipe 자세 → 운전자 관계 특징 → 1차 분류',row)
    d=ImageDraw.Draw(im)
    p=rendered_photo(row,(805,750))
    im.paste(p,(64+(805-p.width)//2,216+(750-p.height)//2))
    x=938
    d.text((x,230),'01   운전자 선택',font=font(31,True),fill=CYAN)
    d.text((x,286),'화면 오른쪽 사람 영역에 자세를 연결',font=font(26),fill=TEXT)
    d.text((x,360),'02   휴대폰과 손의 관계 분석',font=font(31,True),fill=CYAN)
    lines=['운전자 손과 휴대폰의 거리, 자세를 특징으로 사용',
           '동승자의 손·휴대폰 관계도 함께 비교']
    for j,line in enumerate(lines):
        d.text((x,416+42*j),line,font=font(26),fill=TEXT)
    d.rounded_rectangle((920,548,1856,724),radius=22,fill='#193b35' if positive else '#17334c')
    d.text((950,568),'1차 ExtraTrees → '+('양성 (phone)' if positive else '음성 (normal)'),font=font(36,True),fill=GREEN if positive else CYAN)
    relation='≥' if positive else '<'
    d.text((950,635),f"양성 점수 {float(row['score']):.3f}  {relation}  기준값 0.300",font=font(30),fill=TEXT)
    legend(d,x,768)
    return im


def fallback_asset(row):
    from cascade import driver_crop
    im=base('03','1차 판단 실패 → 파인튜닝 ResNet으로 보완',
            '운전자 자세가 없으면 1차 점수를 만들 수 없어, 운전자 이미지 영역을 2차 분류기에 전달',row)
    d=ImageDraw.Draw(im)
    d.text((64,224),'1. 운전자 자세 연결 실패',font=font(29,True),fill=ORANGE)
    p=rendered_photo(row,(580,636))
    im.paste(p,(64+(580-p.width)//2,280))
    d.text((64,928),'주황: 기타 자세  /  하늘색: 운전자 후보',font=font(21),fill=MUTED)
    d.line((656,560,713,560),fill=CYAN,width=5)
    d.polygon([(713,560),(696,550),(696,570)],fill=CYAN)
    d.text((738,224),'2. 운전자 영역 추출',font=font(29,True),fill=CYAN)
    crop=driver_crop(row['path'],row['obs'])
    im.paste(crop.resize((430,430)),(738,310))
    d.text((738,765),'실제 ResNet 입력: 320 × 320',font=font(24),fill=TEXT)
    d.text((738,806),'인물 박스 확장 → 크기 정규화',font=font(24),fill=MUTED)
    d.line((1190,560,1247,560),fill=CYAN,width=5)
    d.polygon([(1247,560),(1230,550),(1230,570)],fill=CYAN)
    d.text((1270,224),'3. ResNet18 이진 분류',font=font(29,True),fill=GREEN)
    d.rounded_rectangle((1260,310,1856,740),radius=22,fill='#182a40')
    for y,t,s,c in [(340,'ImageNet 사전학습',29,TEXT),(392,'→ train 데이터로 파인튜닝',27,TEXT),
                     (484,'최종 양성 (phone)',37,GREEN),(555,f"양성 점수 {float(row['score']):.3f}",30,TEXT),
                     (607,'기준값 0.500 이상',26,MUTED)]:
        d.text((1288,y),t,font=font(s,y==484),fill=c)
    d.text((1270,778),'실제 분기 사유',font=font(24,True),fill=ORANGE)
    d.text((1270,820),'driver_pose_missing',font=font(26),fill=TEXT)
    d.text((1270,866),'낮은 점수가 아닌, 점수 산출 불가',font=font(24),fill=MUTED)
    return im


def generate():
    rows=records()
    specs=[('01_positive.png','IMG_7564',True),('02_negative.png','IMG_7441',False),('03_resnet_fallback.png','IMG_7504',None)]
    OUT.mkdir(parents=True,exist_ok=True)
    entries=[]
    for filename,prefix,positive in specs:
        row=next(r for r in rows if Path(r['path']).name.startswith(prefix))
        assert (row['status']=='positive')==(row['target']=='1')
        assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()==row['sha256']
        if positive is None:
            assert row['stage_used']=='2' and row['info']['reason']=='driver_pose_missing'
            im=fallback_asset(row)
        else:
            assert row['stage_used']=='1' and row['info'].get('driver_index') is not None
            im=stage1_asset(row,positive)
        im.save(OUT/filename)
        entries.append(dict(asset=filename,source=str(Path(row['path']).relative_to(ROOT)),sha256=row['sha256'],
                            target=int(row['target']),prediction=row['status'],stage=int(row['stage_used']),
                            score=float(row['score']),threshold=.3 if row['stage_used']=='1' else .5,
                            reason=row['reason'] or 'landmark_classifier',driver_index=row['info'].get('driver_index'),
                            observation=str((RUN/'verified/cache'/(row['sha256']+'.json')).relative_to(ROOT))))
    metadata=dict(model_sha256=hashlib.sha256((ROOT/'artifacts/classifier.joblib').read_bytes()).hexdigest(),
                  evaluation='artifacts/retrain_20260916_v2/verified/predictions.csv',
                  size=[1920,1080],assets=entries,notes=[
                      'Actual saved evaluation scores; no fresh inference or synthetic photo content.',
                      'Pose edges require both endpoint visibility/presence >= 0.4; all raw phone boxes shown.',
                      'Other detected poses are orange; driver pose green; independent driver person box cyan.',
                      'These are selected correctly classified examples, not aggregate performance evidence.'])
    (OUT/'manifest.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'README.md').write_text('''# 접근방식 설명용 이미지 에셋

실제 평가셋 사진과 확정 모델 v2의 저장된 추론 결과로 만든 1920×1080 PNG 3장입니다.

| 이미지 | 설명 |
| --- | --- |
| [양성](01_positive.png) | 운전자와 동승자가 모두 휴대폰을 든 상황에서 운전자 양성 분류 |
| [음성](02_negative.png) | 동승자가 휴대폰을 들었지만 운전자는 음성 분류 |
| [ResNet 분기](03_resnet_fallback.png) | 운전자 자세 연결 실패 후 실제 운전자 crop을 ResNet18에 전달, 양성 분류 |

## 읽는 방법

- 하늘색: 독립적으로 선택한 운전자 인물 영역. 자세 연결 실패 시 운전자 후보로 표시합니다.
- 초록색: 선택된 운전자 자세. 주황색: 그 외 검출된 자세. 빨간색: 실제 휴대폰 검출 박스.
- 표시 좌표는 평가 캐시에서 가져왔으며 자세나 휴대폰 위치를 임의로 만들지 않았습니다.
- 운전자 자세를 선택할 수 없는 경우에만 ResNet으로 분기합니다. 1차 점수가 낮다는 이유로 분기하지 않습니다.
- ResNet 그림의 crop은 실제 320×320 전처리 결과를 설명용으로 확대했습니다.
- 양성 점수는 보정된 확률이나 검출 정확도를 뜻하지 않습니다.
- 실내 모의 운전 사진이며, 설명용으로 고른 정분류 사례입니다. 전체 성능 및 실제 차량 일반화의 증거로 해석하지 않습니다.

## 재생성

프로젝트 루트에서 `.\\.gpu\\Scripts\\python.exe -m experiments.create_explanation_assets` 실행.
원본 파일명·SHA256·점수·분기·관측 캐시 경로는 [manifest.json](manifest.json)에 기록했습니다.
''',encoding='utf-8')
    with zipfile.ZipFile(OUT/'approach_assets.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in [s[0] for s in specs]+['README.md','manifest.json']:
            z.write(OUT/name,name)
    print(json.dumps(entries,ensure_ascii=True,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidates',action='store_true')
    args=parser.parse_args()
    candidates() if args.candidates else generate()
