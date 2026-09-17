"""Evaluate the current realtime policy on every shared test image, without tuning."""
import csv
import hashlib
import json

import cv2
import numpy as np
import torch

from extract import ROOT
from live import prepare_frame
from realtime import load_engine


def main():
    config = json.loads((ROOT/'configs/realtime_model.json').read_text(encoding='utf-8'))
    rows = list(csv.DictReader((ROOT/'configs/dataset/test.csv').open(encoding='utf-8-sig')))
    torch.set_num_threads(config.get('cpu_threads', 1))
    engine = load_engine(config)
    predictions = []
    confusion = np.zeros((2, 2), dtype=int)
    for i, row in enumerate(rows):
        data = (ROOT/row['path']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row['sha256'], row['path']
        bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError('Cannot decode ' + row['path'])
        result = engine.score_frame(prepare_frame(bgr))
        target = int(row['target'])
        confusion[target, result['prediction']] += 1
        predictions.append(dict(path=row['path'], sha256=row['sha256'], target=target, **result))
        if (i+1) % 25 == 0:
            print(f'Evaluated {i+1}/{len(rows)}', flush=True)
    tn, fp, fn, tp = map(int, confusion.ravel())
    report = dict(
        config=config, n=len(rows),
        accuracy=(tp+tn)/len(rows), precision=tp/(tp+fp) if tp+fp else 0,
        recall=tp/(tp+fn) if tp+fn else 0,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,
        confusion_matrix=confusion.tolist(), tn=tn, fp=fp, fn=fn, tp=tp,
        stage2_count=sum(r['stage_used'] == 2 for r in predictions),
        scope='Fresh inference on original shared test images using live.prepare_frame; no temporal smoothing or test tuning. Historical test set, not independent external validation.',
        predictions=predictions,
    )
    output = ROOT/'docs/evidence/realtime_weighted_05_test.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('config','predictions')}, indent=2))
    print(str(output))


if __name__ == '__main__':
    main()
