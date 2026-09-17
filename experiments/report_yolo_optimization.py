"""Summarize paired frozen-model optimization experiment."""
import csv
import json
from pathlib import Path
from experiments.analyze_yolo_optimization import ROOT, OUT, VARIANTS


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    rows=list(csv.DictReader((OUT/'predictions.csv').open(encoding='utf-8-sig')))
    by={v:{r['sha256']:r for r in rows if r['variant']==v} for v in VARIANTS}
    assert all(len(r)==146 for r in by.values())
    old={r['sha256']:r for r in csv.DictReader((ROOT/'artifacts/retrain_20260916_v2/verified/predictions.csv').open(encoding='utf-8-sig'))}
    reproduction=sum((r['prediction']=='1')!=(old[h]['status']=='positive') for h,r in by['baseline'].items())
    baseline=summary['variants']['baseline']
    names={'baseline':'기존 1280 + 재검출','640':'최초 640만','prune':'가지치기만','640_prune':'640 + 가지치기'}
    changes=[]
    for v in VARIANTS[1:]:
        fixed=[];broken=[];flipped=[]
        for h,r in by[v].items():
            b=by['baseline'][h]
            if b['prediction']!=r['prediction']:
                (fixed if r['prediction']==r['target'] else broken).append(Path(r['path']).name)
                flipped.append(dict(variant=v,filename=Path(r['path']).name,target=int(r['target']),
                                    baseline_prediction=int(b['prediction']),prediction=int(r['prediction']),
                                    baseline_score=float(b['score']),score=float(r['score']),
                                    baseline_stage=int(b['stage']),stage=int(r['stage'])))
        summary['variants'][v]['fixed']=fixed
        summary['variants'][v]['regressed']=broken
        changes.extend(flipped)
    summary['baseline_prediction_disagreements_with_verified']=reproduction
    evidence=ROOT/'docs/evidence/yolo_optimization_20260916.json'
    evidence.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'changed_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        if changes:
            writer=csv.DictWriter(f,fieldnames=list(changes[0]));writer.writeheader();writer.writerows(changes)
    lines=['# YOLO 입력 해상도 및 호출 가지치기 영향 분석','',
           '## 1. 비교 조건','',
           '- 평가셋 146장: 양성 77장, 음성 69장. 같은 사진에 네 설정을 모두 적용했습니다.',
           '- 기존 ExtraTrees·ResNet18 가중치와 임계값(0.3 / 0.5)을 고정했습니다. 재학습하지 않았습니다.',
           '- RTX 4050 Laptop GPU, 모델 상주, batch 1 순차 처리. 네 설정 각각 4회 워밍업 후 사진별 실행 순서를 무작위로 섞었습니다.',
           '- 원본 읽기·전처리·특징 추출·분류를 포함합니다. 모델 로딩, 결과 저장, 시각화 시간은 제외합니다. GPU 동기화 계측이 포함됩니다.',
           '- 모든 사진을 새로 추론했습니다. 한 번의 전체 실행에 대한 평균이며 반복 실행 안정성을 검증한 결과는 아닙니다.',
           '', '### 640 입력의 정의','',
           '최초 YOLO 호출만 `imgsz=640, rect=False`로 변경했습니다. 종횡비를 유지하고 여백을 채운 640×640 입력입니다. 사진을 정사각형으로 늘려 변형하지 않습니다. MediaPipe 원본과 손목 crop 입력은 그대로입니다. 기존 최초 YOLO는 `imgsz=1280, rect=True`입니다.',
           '', '### 가지치기 규칙','',
           '최초 검출 결과에서 각 운전자 손목에 대해 다음 세 조건을 모두 충족하면 해당 손목의 추가 YOLO 호출을 생략합니다. 규칙은 평가 전에 고정했습니다.',
           '', '- 선택된 휴대폰 검출 confidence ≥ 0.5',
           '- 손목–휴대폰 거리 ≤ 어깨 너비의 0.25배',
           '- 다른 사람 손목까지의 거리보다 운전자 손목 거리가 어깨 너비의 0.1배 이상 가까움',
           '', '손 랜드마크 추론은 유지합니다. 최초에 휴대폰을 놓친 경우에는 손목 확대 검출을 계속 수행합니다. 생략한 YOLO 호출에서 나올 수 있는 컵·병 검출도 함께 사라지므로 분류 특징이 바뀔 수 있습니다.',
           '', '## 2. 성능과 처리시간','',
           '| 설정 | ms/장 | FPS | 시간 감소 | Accuracy | Precision | Recall | F1 | FP | FN |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for v,s in summary['variants'].items():
        lines.append(f"| {names[v]} | {s['means']['total_ms']:.1f} | {s['fps']:.2f} | {100*(1-s['means']['total_ms']/baseline['means']['total_ms']):.1f}% | {100*s['accuracy']:.2f}% | {100*s['precision']:.2f}% | {100*s['recall']:.2f}% | {100*s['f1']:.2f}% | {s['fp']} | {s['fn']} |")
    lines += ['', '## 3. 호출과 단계별 시간','',
              '| 설정 | 최초 YOLO ms | 추가 YOLO 합계 ms | 전체 호출 수 | 생략 수 | 2차 분류 사진 수 | p95 ms |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for v,s in summary['variants'].items():
        lines.append(f"| {names[v]} | {s['means']['full_yolo_ms']:.1f} | {s['means']['crop_yolo_ms']:.1f} | {s['yolo_calls']} | {s['skipped_calls']} | {s['stage2_count']} | {s['p95_ms']:.1f} |")
    lines+=['','각 시간은 전체 146장 기준 평균입니다. 해상도에 따라 인물 검출 및 운전자 선택이 바뀌므로 손목 추가 검출 가능 여부와 ResNet 분기 수도 달라질 수 있습니다.',
            '', '## 4. 예측 변화','',f'재측정 baseline과 기존 확정 평가의 최종 예측 불일치: **{reproduction}장**.']
    for v in VARIANTS[1:]:
        s=summary['variants'][v]
        lines+=['',f"### {names[v]}",'',f"- 기존 오분류 → 정분류: {len(s['fixed'])}장 — "+(', '.join(s['fixed']) or '없음'),
                f"- 기존 정분류 → 오분류: {len(s['regressed'])}장 — "+(', '.join(s['regressed']) or '없음')]
    lines+=['','## 5. 결론과 변화 원인','',
            '- **640 입력:** 최초 YOLO 시간은 74.7 → 21.9ms로 약 70.7% 감소했지만 전체 시간 감소는 16.2%입니다. MediaPipe, 이미지 처리, 후단 분류 비용이 남아 있습니다.',
            '- **가지치기:** 기존 입력에서 43회(전체 호출의 11.1%, 추가 호출의 17.7%)를 생략했습니다. 최종 판정은 146장 모두 기존과 같았고 전체 시간은 3.4% 감소했습니다. 이 평가셋에서의 결과이며 모든 입력의 정확도 보존을 보장하지 않습니다.',
            '- **동시 적용:** 330.8ms/장, 3.02 FPS입니다. 640 단독 대비 평균 2.8ms(0.8%) 추가 절약이지만 p95는 497.7 → 586.5ms로 증가했습니다. 이 작은 평균 차이를 안정적인 추가 이득으로 단정하지 않습니다.',
            '- **성능 교환:** 640 계열은 미탐 5 → 3장, 오탐 12 → 15장입니다. Recall +2.60%p, Precision −2.57%p, F1 −0.28%p입니다.',
            '- 미탐 감소를 휴대폰 검출 능력 향상으로 단정할 수 없습니다. IMG_7537은 2차 → 1차로 분기가 바뀌었고, IMG_7746은 2차를 유지하면서 입력 crop을 결정하는 인물 검출이 바뀌었습니다. 반대로 IMG_7514와 IMG_7735도 2차 점수가 바뀌어 새로운 오탐이 됐습니다.',
            '- 현재 판정 유지가 우선이면 가지치기를 우선 검토하고, 미탐 감소와 속도가 우선이면 640을 다음 학습·validation 실험 후보로 삼을 수 있습니다. 현 실험에서 자동으로 기본 설정을 교체하지 않았습니다.',
            '- 두 변경만으로 30 FPS에는 도달하지 못했습니다. 최상의 평균도 33.3ms 목표의 약 9.9배입니다.',
            '', '## 6. 해석 범위','',
            '- 본 실험은 현재 학습 모델에 추론 최적화만 적용했을 때의 영향입니다. 변경한 특징으로 재학습했을 때의 성능은 측정하지 않았습니다.',
            '- 검출 박스 정답이 없으므로 휴대폰 검출 자체의 mAP·Recall은 계산하지 않았습니다. 표의 Recall은 최종 운전자 phone 이진 분류 지표입니다.',
            '- 이전 30장 평균 750.8ms와 직접 속도 향상률을 계산하지 않습니다. 데이터 구성과 실행 시점이 다르므로 이번 동일 실행의 baseline을 사용합니다.',
            '- test 결과로 가지치기 임계값을 반복 조정하지 않았습니다. 추가 최적화 규칙 선택은 train 내부 validation에서 하고 별도 평가로 확인해야 합니다.',
            '- 실험 전용 Extractor를 사용했으며 기본 추론 코드·최종 모델은 변경하지 않았습니다.',
            '', '## 7. 재현','',
            '프로젝트 루트에서:', '', '```powershell',
            '.\\.gpu\\Scripts\\python.exe -m experiments.analyze_yolo_optimization',
            '.\\.gpu\\Scripts\\python.exe -m experiments.report_yolo_optimization', '```','',
            '- 요약 근거: [JSON](evidence/yolo_optimization_20260916.json)',
            '- 상세 점수·시간: `artifacts/yolo_optimization_20260916/predictions.csv`',
            '- 예측 변경 목록: `artifacts/yolo_optimization_20260916/changed_predictions.csv`']
    path=ROOT/'docs/YOLO_OPTIMIZATION_REPORT.md'
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
