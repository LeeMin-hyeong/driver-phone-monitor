"""Select a binary policy from train OOF scores, then evaluate frozen test scores."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from binary_policy import decide


def measure(rows, policy):
    y = [r['target'] for r in rows]
    pred = [int(decide(r['score'], policy) == 'positive') for r in rows]
    p, r, f, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    return dict(n=len(y), accuracy=accuracy_score(y, pred), precision=p, recall=r, f1=f,
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    train_dir = args.run / 'train'
    bundle = joblib.load(train_dir / 'classifier.joblib')
    with (train_dir / 'all_images.csv').open(encoding='utf-8-sig', newline='') as f:
        train = [dict(target=int(r['label'] == bundle['positive_label']),
                      score=float(r['positive_score']) if r['positive_score'] else None) for r in csv.DictReader(f)]
    thresholds = sorted({0.0, 0.5, math.nextafter(1.0, math.inf), *(r['score'] for r in train if r['score'] is not None)})
    trials = []
    for fallback in ('negative', 'positive'):
        for threshold in thresholds:
            policy = dict(threshold=threshold, unscored_class=fallback)
            trials.append(dict(**policy, **measure(train, policy)))
    # Predeclared objective: F1, then recall, precision, and higher threshold.
    best = max(trials, key=lambda m: (m['f1'], m['recall'], m['precision'], m['threshold']))
    policy = {k: best[k] for k in ('threshold', 'unscored_class')}
    output = args.run / 'binary'
    output.mkdir(exist_ok=True)
    policy.update(selection='train out-of-fold F1 maximum; ties recall, precision, higher threshold')
    bundle['binary_policy'] = policy
    joblib.dump(bundle, output / 'classifier.joblib')
    with (output / 'train_sweep.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(trials[0]))
        writer.writeheader()
        writer.writerows(trials)
    # Test outcomes are read only after the operating point is fixed.
    with (args.run / 'test/predictions.csv').open(encoding='utf-8-sig', newline='') as f:
        raw = list(csv.DictReader(f))
    assert not (set(bundle['training_hashes']) & {r['sha256'] for r in raw})
    previous = json.loads((args.run / 'test/evaluation.json').read_text(encoding='utf-8'))
    assert previous['model_sha256'] == hashlib.sha256((train_dir / 'classifier.joblib').read_bytes()).hexdigest()
    test = [dict(target=int(r['target']), score=float(r['score']) if r['score'] else None) for r in raw]
    result = dict(policy=policy, train_oof=best, test=measure(test, policy),
                  test_baseline_0_5={fallback: measure(test, dict(threshold=.5, unscored_class=fallback)) for fallback in ('negative','positive')},
                  train_unscored=sum(r['score'] is None for r in train),
                  test_unscored=sum(r['score'] is None for r in test),
                  test_unknown=0, coverage=1.0, source_classifier_sha256=previous['model_sha256'])
    (output / 'policy_selection.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output / 'policy_predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path','target','score','status','fallback_used','reason'])
        writer.writeheader()
        for original, row in zip(raw, test):
            writer.writerow(dict(path=original['path'], **row, status=decide(row['score'], policy),
                                 fallback_used=row['score'] is None, reason=original['reason']))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
