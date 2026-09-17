"""Publish a realtime profile only after checking the actual test CSV and timings."""
import csv
import hashlib
import json
import shutil
import numpy as np
from extract import ROOT
from training.train_cascade import metrics


def main():
    run=ROOT/'artifacts/fast_pose640_sensitive'
    stem='test_runtime_fp16_vectorized_trt_fusion'
    data=json.loads((run/(stem+'.json')).read_text())
    rows=list(csv.DictReader((run/(stem+'.csv')).open(encoding='utf-8-sig')))
    manifest=list(csv.DictReader((ROOT/'configs/dataset/test.csv').open(encoding='utf-8-sig')))
    expected={r['sha256']:int(r['target']) for r in manifest}
    assert len(expected)==146
    for repeat in range(data['repeats']):
        rr=[r for r in rows if int(r['repeat'])==repeat]
        assert len(rr)==146 and {r['sha256'] for r in rr}==set(expected)
        assert all(int(r['target'])==expected[r['sha256']] for r in rr)
        m=metrics([int(r['target']) for r in rr],[int(r['prediction']) for r in rr])
        assert all(m[k]>=v for k,v in data['reference'].items()),'Quality gate not met'
    times=np.array([float(r['total_ms']) for r in rows])
    assert times.sum()/1000>=60,'At least 60 seconds of measured processing required'
    assert times.mean()<=1000/24 and np.percentile(times,95)<=1000/24,'Latency gate not met'
    ends=np.array([float(r['end_offset_ms']) for r in rows])
    starts=np.concatenate(([0.],ends[ends<=data['wall_seconds']*1000-1000]))
    rolling=np.searchsorted(ends,starts+1000,side='left')-np.searchsorted(ends,starts,side='right')
    assert rolling.min()>=24 and len(rows)/data['wall_seconds']>=24,'Sustained throughput gate not met'
    assert data['input640'] and data['engine'] and data['fusion'] and data['vectorized']
    fusion=json.loads((run/'fusion_selection.json').read_text())
    stage1=run/'model.joblib';stage2=ROOT/fusion['image']/'model.pt'
    assert hashlib.sha256(stage1.read_bytes()).hexdigest()==data['model_sha256']
    assert hashlib.sha256(stage2.read_bytes()).hexdigest()==data['wrist_sha256']
    paths=[stage1,stage2,ROOT/'models/yolo26m.engine',ROOT/'models/yolo26n-pose.engine']
    hashes={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for name,digest in data['engine_sha256'].items():assert hashes['models/'+name]==digest
    config=dict(status='verified',name='driver_phone_realtime640',input_shape=[640,640],driver_side='right',cpu_threads=1,
                stage1_model=str(stage1.relative_to(ROOT)).replace('\\','/'),stage2_model=str(stage2.relative_to(ROOT)).replace('\\','/'),
                stage1_threshold=fusion['best']['box_threshold'],stage2_threshold=fusion['best']['image_threshold'],
                sha256=hashes,gpu=data['gpu'],evidence='docs/evidence/realtime_test.json',
                note='640x640 input, resident models, batch1; stages are newly trained on the shared 911-image train split. Legacy main.py remains available.')
    (ROOT/'configs/realtime_model.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(run/(stem+'.json'),ROOT/'docs/evidence/realtime_test.json')
    m=data['metrics'];base=data['reference'];cm=m['confusion_matrix']
    lines=['# 640 입력 실시간 운전자 휴대폰 분류 결과','',
           '## 1. 목표와 평가 조건','',
           '기존 네 평가 지표를 유지하면서 24 FPS 이상 처리하는 것을 목표로 했다. 사용자 승인 조건에 따라 시스템 최초 입력을 640×640 JPEG로 가정했다.',
           '', '- train 911장 / test 146장 유지. train 내부 fit 619장 / validation 292장으로 모델·임계값을 선택했다.',
           '- train/test 원본 해시 및 640 파생 입력 해시 중복 0건. 원본 사진과 라벨은 변경하지 않았다.',
           '- 고해상도→640 변환은 상류 작업으로 제외했다. 640 파일 읽기·디코딩, 검출, 자세 추정, 특징 계산, 최종 분류는 측정에 포함했다.',
           '- 모델 로딩·워밍업·결과 파일 저장은 제외했다. 입력마다 새 추론을 수행했으며 프레임 생략이나 예측 결과 캐시를 사용하지 않았다.',
           '- 과거 실험에 사용된 test이므로 독립적인 외부 차량 일반화 검증으로 해석하지 않는다.',
           '', '## 2. 최종 결과','', '| 지표 | 기존 확정 모델 | 실시간 모델 |','|---|---:|---:|']
    for k,label in [('accuracy','Accuracy'),('precision','Precision'),('recall','Recall'),('f1','F1')]:
        lines.append(f'| {label} | {100*base[k]:.2f}% | {100*m[k]:.2f}% |')
    lines += ['',f"- 혼동행렬: TN {cm[0][0]}, FP {cm[0][1]}, FN {cm[1][0]}, TP {cm[1][1]}.",
              f"- 평균 **{data['mean_ms']:.2f}ms / {data['fps']:.2f} FPS**, p95 **{data['p95_ms']:.2f}ms**, 최대 {data['max_ms']:.2f}ms.",
              f"- {data['repeats']}회 × 146장 = {len(rows):,}회 처리, 누적 측정 구간 {data['measured_processing_seconds']:.1f}초. 모든 반복에서 네 지표 기준을 충족했다.",
              f"- 루프 부가 비용까지 포함한 실제 처리율 {data['wall_fps']:.2f} FPS, 실제 시각에 기반한 이동 1초 구간의 최소 처리량 {data['min_rolling_1s_fps']}장/초.",
              f"- 환경: {data['gpu']}, batch 1, 모델 상주, CPU 스레드 1. 측정된 장비/조건의 결과이며 모든 프레임의 하드 실시간 마감 보장은 아니다.",
              '', '## 3. 개선 과정','',
              '| 변경 | 검증에서 확인한 내용 |','|---|---|',
              '| 이미지 단독 ResNet | 프레임 계산은 빠르지만 검증 F1 80.13%로 부족 |',
              '| 객체 위치만 사용 | 높은 Precision에 비해 휴대폰 미탐이 많음 |',
              '| GPU 자세 모델 + 약한 객체 후보 | 검증 F1 93.43%, Recall 90.00% |',
              '| 손목 이미지 보완 | 검증 Accuracy 94.18%, Precision 94.63%, Recall 94.00%, F1 94.31% |',
              '| 트리 계산 벡터화 | 학습된 트리 유지, 911개 입력 판정 동일, 평균 8.00→0.196ms |',
              '| TensorRT FP16 | 변환 후 검증 Accuracy 93.49%, Precision 93.38%, Recall 94.00%, F1 93.69% |',
              '', '## 4. 최종 구조','',
              '```mermaid','flowchart LR','    A[640 JPEG 입력] --> B[운전자 영역 YOLO26m]',
              '    A --> C[전체 화면 YOLO26n-pose]','    B --> D[객체·손목 관계 특징]',
              '    C --> D','    D --> E[벡터화 ExtraTrees]','    E -->|점수 ≥ 0.25| P[phone]',
              '    E -->|점수 < 0.25| F[운전자 양손 crop ResNet18]',
              '    F -->|점수 ≥ 0.85| P','    F -->|점수 < 0.85| N[normal]','```','',
              'MediaPipe 손가락 추정과 손목 YOLO 재검출을 제거했다. 두 손목 주변 crop을 한 이미지로 묶어 보완 분류한다. 이전 모델의 ‘점수 없음에만 ResNet’ 분기와 달리, 새 모델은 1차 음성 후보를 재검사한다.',
              '', '## 5. 실행','', '프로젝트 루트의 GPU 환경에서:', '', '```powershell',
              '.\\.gpu\\Scripts\\python.exe realtime.py "640x640_이미지_폴더"',
              '```','', '640×640 입력만 허용하며, 모델을 한 번 로딩하고 폴더의 이미지를 순차 처리한다. JSONL 결과와 로딩/워밍업/추론 시간을 출력한다.',
              '', '- 설정과 모델 해시: [realtime_model.json](../configs/realtime_model.json)',
              '- 평가 근거: [realtime_test.json](evidence/realtime_test.json)',
              '- 기존 방식은 `main.py`와 `artifacts/classifier.joblib`에 보존했다.',
              '- TensorRT 엔진은 해당 GPU/runtime용으로 생성했다. 다른 배포 환경에서는 `python -m experiments.export_fast_engines`로 재생성하고 정확도·지연을 다시 검증한다.',
              '- 검증 환경: PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130, Ultralytics 8.4.152, TensorRT 10.13.3.9. 내보내기에는 ONNX 1.19.1, onnxslim 0.1.96을 사용했다.',
              '- 이 환경에서는 PyTorch에 포함된 CUDA 13 런타임을 사용하고 `tensorrt-cu13`, `tensorrt-cu13-libs`, `tensorrt-cu13-bindings` 10.13.3.9를 `pip install --no-deps`로 설치했다. 이 설치 방식은 CUDA 런타임이 이미 준비된 환경에 한한다.',
              '', '## 6. 자기소개서·프로젝트 설명용 요약','',
              f'“객체 검출·자세 추정·이진 분류 파이프라인을 분석해 중복 검출을 제거하고, GPU 자세 모델과 손목 이미지 보완 분류를 결합했습니다. 학습된 트리의 판정을 유지하는 벡터화 실행과 TensorRT를 적용해 640 입력에서 {data["fps"]:.1f} FPS를 측정했으며, 동일한 146장 평가셋에서 F1 {100*m["f1"]:.2f}%, Recall {100*m["recall"]:.2f}%로 기존 지표를 유지했습니다.”']
    (ROOT/'docs/REALTIME_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(config,ensure_ascii=True,indent=2))


if __name__=='__main__':main()
