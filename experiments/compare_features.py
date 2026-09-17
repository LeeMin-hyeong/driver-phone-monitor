"""Compare v3/v4 including abstentions, so guard rejections do not inflate success."""
import csv
import json
from pathlib import Path
from collections import Counter
from extract import ROOT


def main():
    dirs=[ROOT/'artifacts/new_dataset_yolo26m_gpu',ROOT/'artifacts/new_dataset_m_v4']
    reports=[json.loads((p/'evaluation.json').read_text(encoding='utf-8')) for p in dirs]
    rows=[{r['sha256']:r for r in csv.DictReader((p/'predictions.csv').open(encoding='utf-8-sig'))} for p in dirs]
    assert rows[0].keys()==rows[1].keys()
    assert reports[0]['yolo_sha256']==reports[1]['yolo_sha256']
    assert reports[0]['thresholds']==reports[1]['thresholds']
    assert all(rows[0][h]['target']==rows[1][h]['target'] for h in rows[0])
    behavior=[]
    for table in rows:
        c=Counter()
        for r in table.values():
            y=int(r['target']); status=r['status']
            if status=='unknown':c['unknown_positive' if y else 'unknown_negative']+=1
            else:c[('TP' if y else 'FP') if status=='positive' else ('FN' if y else 'TN')]+=1
        positives=sum(int(r['target']) for r in table.values())
        behavior.append(dict(c,positive_capture_rate=c['TP']/max(positives,1),correct_decision_fraction=(c['TP']+c['TN'])/len(table)))
    common=[h for h in rows[0] if rows[0][h]['score'] and rows[1][h]['score']]
    common_metrics=[]
    for table in rows:
        c=Counter()
        for h in common:
            r=table[h];y=int(r['target']);positive=float(r['score'])>=.5
            c[('TP' if y else 'FP') if positive else ('FN' if y else 'TN')]+=1
        p=c['TP']/max(c['TP']+c['FP'],1);r=c['TP']/max(c['TP']+c['FN'],1)
        common_metrics.append(dict(n=len(common),accuracy=(c['TP']+c['TN'])/len(common),
                                   precision=p,recall=r,f1=2*p*r/max(p+r,1e-12)))
    flips=Counter()
    for h in common:
        a,b=rows[0][h],rows[1][h];y=int(a['target'])
        old=(float(a['score'])>=.5)==y;new=(float(b['score'])>=.5)==y
        flips['corrected' if not old and new else 'regressed' if old and not new else 'unchanged']+=1
    examples={}
    for name in ['IMG_7507.jpg','IMG_7742.jpg','IMG_7723.jpg']:
        examples[name]=[{k:r[k] for k in ['status','score','reason','driver_index']} for table in rows
                        for r in table.values() if Path(r['path']).name==name]
    result=dict(v3=reports[0]['all_files'],v4=reports[1]['all_files'],
                common_scored=len(common),common_changes=dict(flips),
                common_metrics_v3=common_metrics[0],common_metrics_v4=common_metrics[1],
                application_v3=behavior[0],application_v4=behavior[1],regression_examples=examples,
                caveat='Development re-evaluation after error-driven changes; classifier retrained only on original dataset with m features.')
    (dirs[1]/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# v4 변경 및 재평가','','YOLO26m 고정. 기존 학습 데이터로 m 특징 및 추가 손 특징을 재학습했다. 새 데이터 라벨은 학습에 넣지 않았다. 오류를 보고 설계를 바꾼 개발 재평가다.','',
           '| 지표 | v3 | v4 |','|---|---:|---:|']
    for k in ['f1','accuracy','precision','recall']:
        lines.append(f"| 점수 0.5 기준 {k} | {result['v3']['overall'][k]:.2%} | {result['v4']['overall'][k]:.2%} |")
    lines.append(f"| 점수 산출 사진 수 | {result['v3']['overall']['n']} | {result['v4']['overall']['n']} |")
    lines.append(f"| 실행 모드 판정률 | {result['v3']['coverage']:.2%} | {result['v4']['coverage']:.2%} |")
    lines.append(f"| 전체 양성 중 양성 확정 비율 | {behavior[0]['positive_capture_rate']:.2%} | {behavior[1]['positive_capture_rate']:.2%} |")
    lines.append(f"| 전체 사진 중 정답 확정 비율 (보류 포함 분모) | {behavior[0]['correct_decision_fraction']:.2%} | {behavior[1]['correct_decision_fraction']:.2%} |")
    lines+=['','v4 점수 미산출 사유: '+json.dumps(result['v4']['unscored_reasons'],ensure_ascii=False),
            '',f'두 모델 모두 점수가 있는 {len(common)}장에서: '+json.dumps(dict(flips)),
            '', '운전자 박스에 맞는 골격이 없으면 동승자로 대체하지 않고 판단 불가로 반환한다. 손바닥·손가락 특징이 추가되어도 검출·좌석 오류를 모두 해결하는 것은 아니다.',
            '', 'regression_examples는 알려진 동승자 대체 사례 3장의 변경 결과다. 전체 사진별 결과는 predictions.csv, 상세 수치는 comparison.json 참조.','']
    lines+=['## 공통으로 점수를 산출한 사진 비교','','| 지표 | v3 | v4 |','|---|---:|---:|']
    for k in ['f1','accuracy','precision','recall']:
        lines.append(f'| {k} | {common_metrics[0][k]:.2%} | {common_metrics[1][k]:.2%} |')
    (dirs[1]/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
