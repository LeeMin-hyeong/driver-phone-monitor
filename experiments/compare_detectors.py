"""Compare frozen-classifier results for n and m on identical evaluation images."""
import csv
import argparse
import json
import statistics
from pathlib import Path
from extract import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--m-output',type=Path,default=ROOT/'artifacts/new_dataset_yolo26m')
    args=parser.parse_args()
    dirs=[ROOT/'artifacts/new_dataset',args.m_output]
    reports=[json.loads((p/'evaluation.json').read_text(encoding='utf-8')) for p in dirs]
    rows=[{r['sha256']:r for r in csv.DictReader((p/'predictions.csv').open(encoding='utf-8-sig'))} for p in dirs]
    assert rows[0].keys()==rows[1].keys(), 'Different evaluation images'
    assert reports[0]['model_sha256']==reports[1]['model_sha256'], 'Different downstream classifiers'
    assert reports[0]['thresholds']==reports[1]['thresholds'], 'Different thresholds'
    assert all(rows[0][h]['target']==rows[1][h]['target'] for h in rows[0]), 'Different labels'
    improved=worsened=0
    for h in rows[0]:
        a,b=rows[0][h],rows[1][h]
        if not a['score'] or not b['score']:
            continue
        correct_a=(float(a['score'])>=.5)==bool(int(a['target']))
        correct_b=(float(b['score'])>=.5)==bool(int(b['target']))
        improved+=not correct_a and correct_b
        worsened+=correct_a and not correct_b
    proxies={}
    times=[]
    for name,base,items in zip(['n','m'],dirs,rows):
        stats=dict(positive_images=0,positive_images_with_phone=0)
        for h,r in items.items():
            path=base/'cache'/(h+'.json')
            if not path.exists():
                path=ROOT/'artifacts/cache'/(h+'.json')
            obs=json.loads(path.read_text())
            if name=='m' and 'extraction_seconds' in obs:
                times.append(obs['extraction_seconds'])
            if r['target']=='1':
                stats['positive_images']+=1
                stats['positive_images_with_phone']+=any(b['cls']==67 and b['confidence']>=.25 for b in obs['boxes'])
        proxies[name]=stats
    result=dict(n=reports[0]['all_files'],m=reports[1]['all_files'],corrected=improved,regressed=worsened,
                phone_presence_proxy=proxies,
                m_mean_extraction_seconds=statistics.mean(times) if times else None,
                caveat='Phone presence is not detector recall: passenger phones and false boxes can count; box ground truth is unavailable.')
    (dirs[1]/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# YOLO26n / YOLO26m 비교','','동일한 새 데이터 639장, 분류기·해상도·임계값 고정. 재학습 없음.','',
           '| 지표 | n | m | 차이 (%p) |','|---|---:|---:|---:|']
    for key in ['f1','accuracy','precision','recall']:
        a,b=result['n']['overall'][key],result['m']['overall'][key]
        lines.append(f'| {key} | {a:.2%} | {b:.2%} | {(b-a)*100:+.2f} |')
    lines+=['',f'오답→정답 {improved}장, 정답→오답 {worsened}장 (두 실행 모두 점수가 있는 사진).',
            '', '## 판단 유보 적용','','| 모델 | F1 | 정확도 | 정밀도 | 재현율 | 판정률 |','|---|---:|---:|---:|---:|---:|']
    for name in ['n','m']:
        a=result[name]; s=a['selective']
        lines.append(f"| {name} | {s['f1']:.2%} | {s['accuracy']:.2%} | {s['precision']:.2%} | {s['recall']:.2%} | {a['coverage']:.2%} |")
    lines+=['','검출 박스 정답이 없어 휴대폰 검출 자체의 mAP/Recall은 측정하지 않았습니다. 여기의 지표는 최종 운전자 소지 분류 성능입니다.','']
    (dirs[1]/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
