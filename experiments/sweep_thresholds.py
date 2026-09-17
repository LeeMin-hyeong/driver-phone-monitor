"""Sweep cached prediction scores; leave unscored driver-guard cases unknown."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def metrics(scored, threshold, total_positives):
    tp = sum(y == 1 and s >= threshold for y, s in scored)
    fp = sum(y == 0 and s >= threshold for y, s in scored)
    fn = sum(y == 1 and s < threshold for y, s in scored)
    tn = len(scored) - tp - fp - fn
    return dict(threshold=threshold, tp=tp, fp=fp, tn=tn, fn=fn,
                precision=tp / (tp + fp) if tp + fp else 0,
                recall=tp / (tp + fn) if tp + fn else 0,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
                accuracy=(tp + tn) / len(scored) if scored else 0,
                whole_positive_capture=tp / total_positives if total_positives else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', type=Path)
    args = parser.parse_args()
    with args.predictions.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len({r['sha256'] for r in rows}) != len(rows):
        raise ValueError('Use unique-image predictions to avoid duplicate weighting.')
    scored = [(int(r['target']), float(r['score'])) for r in rows if r['score']]
    if not scored or any(y not in (0, 1) or not math.isfinite(s) for y, s in scored):
        raise ValueError('Expected finite scores and binary targets.')
    positives = sum(int(r['target']) for r in rows)
    candidates = sorted({0.0, 0.35, 0.5, 0.65, 1.0, *(s for _, s in scored)})
    sweep = [metrics(scored, t, positives) for t in candidates]
    best = max(sweep, key=lambda m: (m['recall'], m['precision'], m['threshold']))
    selected = {
        'baseline_0.65': metrics(scored, .65, positives),
        'baseline_0.5': metrics(scored, .5, positives),
        'maximum_recall': best,
        'maximum_recall_rounded_down': metrics(scored, math.floor(best['threshold'] * 1e6) / 1e6, positives),
        'maximum_f1': max(sweep, key=lambda m: (m['f1'], m['recall'], m['threshold'])),
    }
    for floor in (.8, .9, .95):
        eligible = [m for m in sweep if m['precision'] >= floor]
        selected[f'max_recall_precision_at_least_{floor}'] = max(
            eligible, key=lambda m: (m['recall'], m['precision'], m['threshold'])) if eligible else None
    result = dict(source=str(args.predictions.resolve()),
                  source_sha256=hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
                  total=len(rows), scored=len(scored), positives=positives,
                  unscored=len(rows) - len(scored),
                  unscored_positives=positives - sum(y for y, _ in scored),
                  policy='score >= threshold positive; lower scored values negative; no-score stays unknown',
                  limitation='Development-set threshold tuning, not independent test performance. Scores are uncalibrated.',
                  selected=selected)
    output = args.predictions.parent / 'threshold_analysis'
    output.mkdir(exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output / 'sweep.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
