"""Diagnose cached false negatives; categories are proxies, not human box labels."""
import csv
import argparse
import json
from collections import Counter
from pathlib import Path
import joblib
from PIL import Image, ImageOps, ImageDraw
from extract import ROOT
from features import features, VERSION


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=ROOT/'artifacts/new_dataset_m_v4')
    base=parser.parse_args().base
    report=json.loads((base/'evaluation.json').read_text(encoding='utf-8'))
    if report['feature_version']!=VERSION:
        raise ValueError('이전 특징 버전의 결과입니다. 해당 버전의 기존 분석 보고서를 참고하세요.')
    rows=list(csv.DictReader((base/'predictions.csv').open(encoding='utf-8-sig')))
    details=[]
    groups={k:[] for k in ['A_no_confident_phone','B_weak_wrist_contact','C_contact_but_negative']}
    missing=[r for r in rows if r['target']=='1' and not r['score']]
    for r in rows:
        if r['target']!='1' or not r['score'] or float(r['score'])>=.5:
            continue
        o=json.loads((base/'cache'/(r['sha256']+'.json')).read_text())
        f,info=features(o)
        confident=[b for b in o['boxes'] if b['cls']==67 and b['confidence']>=.25]
        wrist=max((15,16),key=lambda w:f[f'w{w}_phone_contact'])
        contact=float(f[f'w{wrist}_phone_contact'])
        category='A_no_confident_phone' if not confident else ('B_weak_wrist_contact' if contact<.15 else 'C_contact_but_negative')
        record=dict(r,category=category,phone_boxes_ge025=len(confident),contact=contact,
                    best_wrist=wrist,phone_distance=float(f[f'w{wrist}_phone_distance']),
                    phone_confidence=float(f[f'w{wrist}_phone_confidence']),
                    ownership_margin=float(f[f'w{wrist}_ownership_margin']),
                    assigned_hands=int(f['w15_hand_found']+f['w16_hand_found']),
                    best_wrist_visibility=float(f[f'p{wrist}_visibility']),
                    driver_margin=info['driver_margin'])
        details.append(record); groups[category].append(record)
    bundle=joblib.load(ROOT/'artifacts/classifier.joblib')
    importances=sorted(zip(bundle['feature_names'],bundle['model'].feature_importances_),key=lambda pair:pair[1],reverse=True)
    summary=dict(false_negatives=len(details),positive_extraction_failures=len(missing),
                 categories={k:len(v) for k,v in groups.items()},
                 status_counts=dict(Counter(r['status'] for r in details)),
                 flags=dict(no_assigned_hands=sum(r['assigned_hands']==0 for r in details),
                            ambiguous_driver=sum(r['driver_margin']<.08 for r in details),
                            best_wrist_visibility_lt04=sum(r['best_wrist_visibility']<.4 for r in details),
                            negative_ownership_margin=sum(r['ownership_margin']<0 for r in details)),
                 top_training_importances=[dict(feature=n,importance=float(v)) for n,v in importances[:10]],
                 note='Phone box correctness and driver identity require visual review. These proxies are not verified causal counts.')
    summary['flags']={k:int(v) for k,v in summary['flags'].items()}
    selected=[]
    for key,items in groups.items():
        ordered=sorted(items,key=lambda r:r['path'])
        indices=sorted({round(i*(len(ordered)-1)/3) for i in range(4)}) if ordered else []
        selected.extend(ordered[i] for i in indices)
    sheet=Image.new('RGB',(1440,1080),'white')
    draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(selected):
        o=json.loads((base/'cache'/(r['sha256']+'.json')).read_text())
        with Image.open(r['path']) as source:
            im=ImageOps.exif_transpose(source).convert('RGB')
        im=ImageOps.contain(im,(355,320)); w,h=im.size; d=ImageDraw.Draw(im)
        for j,p in enumerate(o['poses']):
            for a,b in ((11,12),(11,13),(13,15),(12,14),(14,16)):
                if min(p[a][3],p[b][3])>=.4:
                    d.line((p[a][0]*w,p[a][1]*h,p[b][0]*w,p[b][1]*h),fill='lime' if j==int(r['driver_index']) else 'orange',width=2)
        for b in o['boxes']:
            if b['cls']==67 and b['confidence']>=.25:
                d.rectangle(tuple(v*s for v,s in zip(b['box'],[w,h,w,h])),outline='red',width=2)
        x,y=i%4*360,i//4*360; sheet.paste(im,(x,y))
        draw.text((x,y+322),r['category'][0]+' '+Path(r['path']).name,fill='black')
        draw.text((x,y+335),f"score={float(r['score']):.2f} contact={r['contact']:.2f} hands={r['assigned_hands']}",fill='black')
    sheet.save(base/'misses_12.jpg')
    summary['sample_files']=[dict(category=r['category'],name=Path(r['path']).name) for r in selected]
    (base/'miss_analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (base/'miss_analysis.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
