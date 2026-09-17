"""Deduplicate and evaluate landmarks on held-out capture-number blocks."""
import csv
import argparse
import json
from pathlib import Path
import re
from collections import Counter

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

from extract import ROOT, EXTRACTOR_VERSION
from features import features, VERSION
from data_labels import resolve_labels


def metrics(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    return dict(accuracy=accuracy_score(y, pred), precision=p, recall=r, f1=f,
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def main():
    import joblib
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,default=ROOT/'artifacts')
    parser.add_argument('--positive')
    parser.add_argument('--negative')
    args=parser.parse_args()
    artifacts=args.artifacts
    records = json.loads((artifacts/'manifest.json').read_text(encoding='utf-8'))
    args.positive,args.negative=resolve_labels([r['label'] for r in records],args.positive,args.negative)
    if {r['label'] for r in records} != {args.positive, args.negative}:
        raise ValueError('Manifest labels must match the positive and negative labels.')
    seen = {}
    rows, missing = [], []
    detector_hashes=set()
    for record in records:
        key = record['sha256']
        if key in seen:
            if seen[key] != record['label']:
                raise ValueError('Identical image has conflicting labels: '+record['path'])
            continue
        seen[key] = record['label']
        obs = json.loads((artifacts/'cache'/f'{key}.json').read_text())
        if obs.get('extractor_version') != EXTRACTOR_VERSION:
            raise ValueError('오래된 추출 캐시입니다. extract.py를 먼저 실행하세요.')
        detector_hashes.add(obs['yolo_sha256'])
        f, info = features(obs)
        if f is None:
            missing.append(dict(record,reason=info['reason']))
            continue
        # IMG numbers group adjacent bursts; both (1) copies stay in same block.
        # These are capture-block checks, NOT an unseen-driver/vehicle benchmark.
        number = int(re.search(r'\d+', Path(record['path']).stem)[0])
        rows.append(dict(record, features=f, info=info, group=number//100))
    if len(detector_hashes)!=1:
        raise ValueError('추출 캐시에 서로 다른 검출기가 섞여 있습니다.')
    names = sorted(rows[0]['features'])
    x = np.asarray([[r['features'][k] for k in names] for r in rows])
    y = np.asarray([int(r['label']==args.positive) for r in rows])
    groups = np.asarray([r['group'] for r in rows])
    probabilities = np.zeros(len(rows))
    folds = []
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        model = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, max_features=0.8,
                                    class_weight='balanced', random_state=42, n_jobs=4)
        model.fit(x[train], y[train])
        probabilities[test] = model.predict_proba(x[test])[:, list(model.classes_).index(1)]
        folds.append(dict(group=int(groups[test[0]]), n=len(test),
                          **metrics(y[test], probabilities[test]>=0.5)))
    prediction = probabilities >= 0.5
    # Abstention threshold is predeclared, not optimized on held-out outcomes.
    certain = ((probabilities <= 0.35) | (probabilities >= 0.65)) & np.asarray([r['info']['driver_margin']>=0.08 for r in rows])
    report = dict(total_files=len(records), unique_images=len(seen), valid_landmarks=len(rows),
                  missing_landmarks=len(missing), features_version=VERSION,
                  unscored_reasons=dict(Counter(r['reason'] for r in missing)),
                  validation='development leave-one-IMG-hundreds-block-out; feature design reviewed against errors; not untouched test',
                  overall=metrics(y, prediction), folds=folds,
                  selective=dict(coverage=float(certain.sum()/len(seen)),
                                 **metrics(y[certain], prediction[certain])),
                  caveat='Capture blocks can share actors and room. Not unseen-person or real-vehicle accuracy.',
                  unique_label_counts={label:sum(v==label for v in seen.values()) for label in (args.positive,args.negative)})
    (artifacts/'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (artifacts/'predictions.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path','label','group','probability','prediction','correct','driver_index'])
        writer.writeheader()
        for i,r in enumerate(rows):
            writer.writerow(dict(path=r['path'],label=r['label'],group=r['group'],probability=probabilities[i],
                                 prediction=args.positive if prediction[i] else args.negative,
                                 correct=bool(prediction[i]==y[i]),driver_index=r['info']['driver_index']))
    model.fit(x, y)
    joblib.dump(dict(model=model, feature_names=names, feature_version=VERSION,
                     training_hashes=sorted(seen),extractor_version=EXTRACTOR_VERSION,
                     positive_label=args.positive, negative_label=args.negative,
                     yolo_sha256=next(iter(detector_hashes))), artifacts/'classifier.joblib')
    lookup = {r['sha256']:i for i,r in enumerate(rows)}
    with (artifacts/'all_images.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['path','label','status','positive_score','correct','evaluation'])
        writer.writeheader()
        for record in records:
            i=lookup.get(record['sha256'])
            score=float(probabilities[i]) if i is not None else None
            status=(args.positive if prediction[i] else args.negative) if i is not None and certain[i] else '판단 불가'
            writer.writerow(dict(path=record['path'],label=record['label'],status=status,positive_score=score,
                                 correct=status==record['label'] if status!='판단 불가' else '',
                                 evaluation='out_of_fold_development'))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
