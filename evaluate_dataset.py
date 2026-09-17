"""Evaluate a frozen classifier on labeled images without fitting or changing it."""
import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from extract import ROOT, Extractor, EXTRACTOR_VERSION
from features import VERSION, features
from binary_policy import decide
from data_labels import resolve_labels


def summarize(rows, binary=False):
    valid = rows if binary else [r for r in rows if r['score'] is not None]
    certain = [r for r in valid if r['status'] != 'unknown']
    def metrics(items):
        if not items:
            return None
        y = [r['target'] for r in items]
        pred = [int(r['status'] == 'positive') if binary else int(r['score'] >= 0.5) for r in items]
        p, r, f, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
        return dict(n=len(items), accuracy=accuracy_score(y, pred), precision=p, recall=r, f1=f,
                    confusion_matrix=confusion_matrix(y, pred, labels=[0,1]).tolist())
    unscored=[r for r in rows if r['score'] is None]
    reasons=Counter(r.get('reason') or 'extraction_error' for r in unscored)
    return dict(total=len(rows), positives=sum(r['target'] for r in rows),
                unscored=len(unscored),unscored_reasons=dict(reasons),
                extraction_failures=sum(v for k,v in reasons.items() if k in ('extraction_error','no_reliable_body_landmarks')),
                unknown=len(rows)-len(certain),
                coverage=len(certain)/len(rows) if rows else None,
                overall=metrics(valid), selective=metrics(certain))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'dataset/test')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/evaluation')
    parser.add_argument('--positive')
    parser.add_argument('--negative')
    parser.add_argument('--yolo-model', type=Path, default=ROOT/'models/yolo26m.pt')
    parser.add_argument('--device', default='auto', help='auto, cpu, or CUDA index (0)')
    parser.add_argument('--classifier',type=Path,default=ROOT/'artifacts/classifier.joblib')
    args=parser.parse_args()
    args.positive,args.negative=resolve_labels(
        [p.name for p in args.dataset.iterdir() if p.is_dir()],args.positive,args.negative)
    import torch
    device=('0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if device!='cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA를 사용할 수 없습니다. CUDA 지원 PyTorch 환경으로 실행하세요.')
    print(f'YOLO device={device}; torch={torch.__version__}',flush=True)
    yolo_hash=hashlib.sha256(args.yolo_model.read_bytes()).hexdigest()
    model_path=args.classifier
    model_hash=hashlib.sha256(model_path.read_bytes()).hexdigest()
    bundle=joblib.load(model_path)
    policy=bundle.get('binary_policy')
    cascade_config=bundle.get('cascade')
    fallback_engine=None
    if bundle['feature_version'] != VERSION:
        raise ValueError('Classifier feature version mismatch')
    train_hashes=set(bundle.get('training_hashes',[]))
    args.output.mkdir(parents=True,exist_ok=True)
    cache=args.output/'cache'
    cache.mkdir(exist_ok=True)
    records=[]
    for label,target in ((args.negative,0),(args.positive,1)):
        folder=args.dataset/label
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix.lower() in ('.jpg','.jpeg','.png','.webp'):
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                records.append(dict(path=str(path.resolve()),label=label,target=target,sha256=digest,
                                    training_overlap=digest in train_hashes))
    labels={}
    for r in records:
        labels.setdefault(r['sha256'],set()).add(r['target'])
    conflicts={h for h,values in labels.items() if len(values)>1}
    if conflicts:
        (args.output/'label_conflicts.json').write_text(json.dumps([r for r in records if r['sha256'] in conflicts],ensure_ascii=False,indent=2),encoding='utf-8')
        raise ValueError('Identical photos have conflicting labels; see label_conflicts.json')
    print(f'Files={len(records)} unique={len(labels)} training_overlap={len(set(labels)&train_hashes)}',flush=True)
    model=bundle['model']
    engine=None
    scores={}
    rows=[]
    started=time.perf_counter()
    try:
        for i,record in enumerate(records):
            digest=record['sha256']
            if digest not in scores:
                target=cache/(digest+'.json')
                old_cache=ROOT/'artifacts/cache'/(digest+'.json')
                source=target if target.exists() else old_cache
                obs=json.loads(source.read_text()) if source.exists() else None
                try:
                    if (obs is None or obs.get('extractor_version')!=EXTRACTOR_VERSION
                            or obs.get('yolo_sha256')!=yolo_hash or obs.get('yolo_device')!=device):
                        if engine is None:
                            engine=Extractor(yolo_model=args.yolo_model,device=device)
                        extraction_started=time.perf_counter()
                        obs=engine.extract(record['path'])
                        obs['extraction_seconds']=time.perf_counter()-extraction_started
                        target.write_text(json.dumps(obs),encoding='utf-8')
                    f,info=features(obs)
                    score=None if f is None else float(model.predict_proba([[f[k] for k in bundle['feature_names']]])[0,list(model.classes_).index(1)])
                    first_score=score
                    stage=1
                    if cascade_config:
                        threshold=cascade_config['stage1_threshold']
                        if score is None:
                            if fallback_engine is None:
                                from cascade import ImageFallback
                                fallback_engine=ImageFallback(cascade_config, device)
                            score=fallback_engine.score(record['path'],obs)
                            threshold=cascade_config['stage2_threshold']
                            stage=2
                    unknown=score is None or 0.35<score<0.65 or info.get('driver_margin',0)<0.08
                    scores[digest]=dict(score=score,status='unknown' if unknown else ('positive' if score>=0.65 else 'negative'),
                                        driver_index=info.get('driver_index'),reason=info.get('reason',''),error='')
                    if policy:
                        scores[digest]['status']=decide(score, policy)
                    if cascade_config:
                        scores[digest].update(status='positive' if score>=threshold else 'negative',
                                              stage_used=stage,stage1_score=first_score,
                                              stage2_score=score if stage==2 else None)
                except Exception as exc:
                    if policy or cascade_config:
                        raise  # Runtime errors are not valid binary predictions.
                    scores[digest]=dict(score=None,status='unknown',driver_index=None,reason='extraction_error',error=str(exc))
            rows.append(dict(record,**scores[digest]))
            if i%25==0:
                print(f'{i+1}/{len(records)} processed; elapsed={time.perf_counter()-started:.0f}s',flush=True)
    finally:
        if engine is not None:
            engine.close()
    unique=list({r['sha256']:r for r in rows}.values())
    novel=[r for r in unique if not r['training_overlap']]
    overlap=[r for r in unique if r['training_overlap']]
    result=dict(dataset=str(args.dataset.resolve()),positive_label=args.positive,negative_label=args.negative,
                model_sha256=model_hash,model_retrained=False,feature_version=VERSION,
                yolo_model=str(args.yolo_model.resolve()),yolo_sha256=yolo_hash,
                yolo_device=device,torch_version=torch.__version__,
                elapsed_seconds=time.perf_counter()-started,
                thresholds=cascade_config or policy or dict(binary=0.5,negative_max=0.35,positive_min=0.65,driver_margin_min=0.08),
                stage2_count=sum(r.get('stage_used')==2 for r in rows),
                all_files=summarize(rows, bool(policy or cascade_config)),unique_images=summarize(unique, bool(policy or cascade_config)),
                unseen_exact_hashes=summarize(novel, bool(policy or cascade_config)),training_overlap=summarize(overlap, bool(policy or cascade_config)),
                note='Unseen means no exact file hash match. It does not prove different people, scenes, or absence of near-duplicates.')
    assert hashlib.sha256(model_path.read_bytes()).hexdigest()==model_hash, 'Model changed during evaluation'
    (args.output/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (args.output/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
