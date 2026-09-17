"""Select fusion using train validation predictions only, never test labels."""
import argparse
import json
import csv
import numpy as np
from extract import ROOT
from training.train_cascade import metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--boxes',default='artifacts/fast_boxes640')
    p.add_argument('--image',default='artifacts/fast_roi_v1')
    p.add_argument('--prefer-baseline',action='store_true')
    args=p.parse_args()
    box=ROOT/args.boxes;image=ROOT/args.image
    rows=list(csv.DictReader((ROOT/'configs/dataset/train.csv').open(encoding='utf-8-sig')))
    hashes=set(json.loads((ROOT/'configs/dataset/train_validation.json').read_text())['validation_hashes'])
    y=np.array([int(r['target']) for r in rows if r['sha256'] in hashes])
    c=np.load(image/'validation_scores.npy')
    best=None
    def rank(m):
        meets=(m['accuracy']>=129/146 and m['precision']>=72/84 and m['recall']>=72/77 and m['f1']>=144/161)
        return (meets if args.prefer_baseline else True,m['f1'],m['recall'])
    for leaf in (2,5,10):
        g=np.load(box/f'validation_scores_leaf{leaf}.npy')
        for a in np.arange(.1,.901,.025):
            for b in np.append(np.arange(.1,.901,.025),1.01):
                pred=(g>=a)|(c>=b)
                m=dict(rule='or',leaf=leaf,box_threshold=float(a),image_threshold=float(b),**metrics(y,pred))
                if best is None or rank(m)>rank(best):best=m
    report=dict(best=best,boxes=args.boxes,image=args.image,prefer_baseline=args.prefer_baseline,
                selection='Shared train validation: optional all-baseline-metrics feasibility first, then F1, ties recall; OR fusion of independently fit validation models; no test used')
    (box/'fusion_selection.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
