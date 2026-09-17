"""Train/validate on train only, refit both models, then evaluate test once."""
import argparse
import csv
import hashlib
import json
import random
import re
import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from cascade import driver_crop, new_model, PREPROCESS, NORMALIZE, IMAGE_SIZE
from extract import ROOT, EXTRACTOR_VERSION
from features import features, VERSION
from data_labels import resolve_labels


def seed_all():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def metrics(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    return dict(n=len(y), accuracy=accuracy_score(y, pred), precision=p, recall=r, f1=f,
                confusion_matrix=confusion_matrix(y, pred, labels=[0,1]).tolist())


class Crops(Dataset):
    def __init__(self, images, rows, indices, augment=False):
        self.images, self.rows, self.indices = images, rows, list(indices)
        self.transform = transforms.Compose([
            transforms.RandomAffine(5, translate=(.025,.025), scale=(.97,1.03), fill=114),
            transforms.ColorJitter(.15,.15,.1,.02), transforms.ToTensor(), NORMALIZE,
        ]) if augment else PREPROCESS

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        index = self.indices[i]
        return self.transform(self.images[index]), self.rows[index]['target']


def train_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    total = 0
    for x,y in loader:
        x,y = x.cuda(), y.cuda()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        scaler.step(optimizer)
        scaler.update()
        total += float(loss.detach())*len(y)
    return total/len(loader.dataset)


@torch.inference_mode()
def scores(model, loader):
    model.eval()
    return np.concatenate([model(x.cuda()).softmax(1)[:,1].cpu().numpy() for x,_ in loader])


def fit_stage1(rows, indices, names):
    usable = [i for i in indices if rows[i]['features'] is not None]
    model = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, max_features=.8,
                                class_weight='balanced', random_state=42, n_jobs=4)
    model.fit([[rows[i]['features'][k] for k in names] for i in usable], [rows[i]['target'] for i in usable])
    return model


def stage1_scores(model, rows, indices, names):
    return np.array([np.nan if rows[i]['features'] is None else
                     model.predict_proba([[rows[i]['features'][k] for k in names]])[0, list(model.classes_).index(1)]
                     for i in indices])


def prepare(rows, cache, crops, detector_hash):
    crops.mkdir(parents=True, exist_ok=True)
    images = []
    for i,row in enumerate(rows):
        path = Path(row['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Dataset image changed: '+str(path))
        obs = json.loads((cache/(row['sha256']+'.json')).read_text(encoding='utf-8'))
        assert obs['extractor_version'] == EXTRACTOR_VERSION and obs['yolo_sha256'] == detector_hash
        row['features'], info = features(obs)
        row['reason'] = info.get('reason', '')
        target = crops/(row['sha256']+'.png')
        if not target.exists():
            driver_crop(path, obs).save(target)
        with Image.open(target) as im:
            images.append(im.convert('RGB'))
        if i%100 == 0:
            print(f'prepare {i+1}/{len(rows)}', flush=True)
    return images


def training_setup(rows, indices, weights):
    seed_all()
    model = new_model(weights).cuda()
    optimizer = torch.optim.AdamW([
        dict(params=[p for n,p in model.named_parameters() if not n.startswith('fc.')], lr=1e-4),
        dict(params=model.fc.parameters(), lr=1e-3)], weight_decay=1e-4)
    counts = np.bincount([rows[i]['target'] for i in indices], minlength=2)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(len(indices)/(2*counts), dtype=torch.float32, device='cuda'))
    return model, optimizer, torch.amp.GradScaler('cuda'), criterion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'artifacts/train_test_20260916')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/cascade_20260916')
    parser.add_argument('--epochs', type=int, default=8)
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'CUDA required for this experiment'
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    assert not (output/'locked_selection.json').exists(), 'Use a new output for a new experiment'
    weights = ROOT/'models/resnet18-f37072fd.pth'
    weights_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert weights_hash.startswith('f37072fd')
    detector_hash = hashlib.sha256((ROOT/'models/yolo26m.pt').read_bytes()).hexdigest()
    rows = json.loads((args.source/'train/manifest.json').read_text(encoding='utf-8'))
    positive_label,negative_label=resolve_labels([r['label'] for r in rows])
    assert len({r['sha256'] for r in rows}) == len(rows)
    for row in rows:
        assert row['label'] in (positive_label,negative_label)
        row['target'] = int(row['label']==positive_label)
    groups = [int(re.search(r'\d+',Path(r['path']).stem)[0])//100 for r in rows]
    fit, val = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=42).split(rows, groups=groups))
    split = dict(fit_hashes=[rows[i]['sha256'] for i in fit], validation_hashes=[rows[i]['sha256'] for i in val],
                 fit_groups=sorted({groups[i] for i in fit}), validation_groups=sorted({groups[i] for i in val}),
                 note='Capture-number blocks, not verified independent people or sessions.')
    (output/'split.json').write_text(json.dumps(split,indent=2),encoding='utf-8')
    images = prepare(rows, args.source/'train/cache', output/'crops', detector_hash)
    names = sorted(next(r['features'] for r in rows if r['features'] is not None))
    stage1 = fit_stage1(rows, fit, names)
    s1 = stage1_scores(stage1, rows, val, names)
    missing = np.isnan(s1)
    y = np.array([rows[i]['target'] for i in val])
    print(f'fit={len(fit)} val={len(val)} val_fallback={int(missing.sum())}', flush=True)
    train_loader = DataLoader(Crops(images, rows, fit, True), batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(Crops(images, rows, val), batch_size=32, num_workers=0)
    model, optimizer, scaler, criterion = training_setup(rows, fit, weights)
    best, history = None, []
    thresholds = [.2,.3,.4,.5,.6,.7,.8]
    positive_baselines = [dict(threshold=t, **metrics(y, np.where(missing, True, s1>=t))) for t in thresholds]
    positive_baseline = max(positive_baselines, key=lambda m:(m['f1'],m['recall'],m['precision'],-abs(m['threshold']-.5)))
    for epoch in range(1,args.epochs+1):
        started = time.perf_counter()
        loss = train_epoch(model, train_loader, optimizer, scaler, criterion)
        s2 = scores(model, val_loader)
        candidates = []
        for t1 in thresholds:
            for t2 in thresholds:
                pred = np.where(missing, s2>=t2, s1>=t1)
                candidates.append(dict(epoch=epoch, stage1_threshold=t1, stage2_threshold=t2, **metrics(y,pred)))
        current = max(candidates, key=lambda m:(m['f1'],m['recall'],m['precision'],-abs(m['stage1_threshold']-.5)-abs(m['stage2_threshold']-.5)))
        current.update(loss=loss, seconds=time.perf_counter()-started,
                       stage2_alone=metrics(y,s2>=current['stage2_threshold']),
                       fallback_only=metrics(y[missing],s2[missing]>=current['stage2_threshold']))
        history.append(current)
        if best is None or (current['f1'],current['recall'],current['precision'])>(best['f1'],best['recall'],best['precision']):
            best = current.copy()
            torch.save(dict(state_dict=model.cpu().state_dict(), image_size=IMAGE_SIZE),output/'validation_stage2.pt')
            model.cuda()
        (output/'validation_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print(f'validation epoch={epoch} loss={loss:.4f} F1={current["f1"]:.4f} recall={current["recall"]:.4f} fallback_F1={current["fallback_only"]["f1"]:.4f} seconds={current["seconds"]:.1f}',flush=True)
    locked = dict(best=best, epochs=best['epoch'], routing='stage1 score missing only',
                  selection='validation cascade F1; ties recall, precision, thresholds nearer 0.5, then earlier epoch',
                  threshold_grid=thresholds, imagenet_sha256=weights_hash, seed=42,
                  image_size=IMAGE_SIZE, backbone='resnet18', fine_tune='all layers',
                  positive_fallback_validation_selected=positive_baseline,
                  validation_stage1_all_positive_fallback=metrics(y,np.where(missing,True,s1>=.5265758057645732)))
    (output/'locked_selection.json').write_text(json.dumps(locked,indent=2),encoding='utf-8')
    # No test paths, labels or predictions have been loaded above this line.
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    all_indices = list(range(len(rows)))
    stage1 = fit_stage1(rows, all_indices, names)
    model, optimizer, scaler, criterion = training_setup(rows, all_indices, weights)
    train_loader = DataLoader(Crops(images, rows, all_indices, True), batch_size=16, shuffle=True, num_workers=0)
    for epoch in range(1,locked['epochs']+1):
        started = time.perf_counter()
        loss = train_epoch(model, train_loader, optimizer, scaler, criterion)
        print(f'final train epoch={epoch}/{locked["epochs"]} loss={loss:.4f} seconds={time.perf_counter()-started:.1f}',flush=True)
    checkpoint = output/'stage2.pt'
    torch.save(dict(state_dict=model.cpu().state_dict(),image_size=IMAGE_SIZE),checkpoint)
    model.cuda().eval()
    cascade = dict(checkpoint=str(checkpoint.resolve()), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   stage1_threshold=best['stage1_threshold'],stage2_threshold=best['stage2_threshold'],
                   routing='missing_score',image_size=IMAGE_SIZE)
    bundle = dict(model=stage1, feature_names=names, feature_version=VERSION, extractor_version=EXTRACTOR_VERSION,
                  training_hashes=sorted(r['sha256'] for r in rows), yolo_sha256=detector_hash,
                  positive_label=positive_label,negative_label=negative_label,cascade=cascade)
    joblib.dump(bundle,output/'classifier.joblib')
    # Final held-out evaluation, after selection and refitting are complete.
    test_manifest=args.source/'test/manifest.json'
    test = []
    if test_manifest.exists():
        records=json.loads(test_manifest.read_text(encoding='utf-8'))
        test_positive,test_negative=resolve_labels([r['label'] for r in records])
        for row in records:
            assert row['label'] in (test_positive,test_negative)
            test.append(dict(path=row['path'],target=int(row['label']==test_positive),sha256=row['sha256']))
    else:
        test_positive,test_negative=resolve_labels([p.name for p in (ROOT/'dataset/test').iterdir() if p.is_dir()])
        for label,target in [(test_negative,0),(test_positive,1)]:
            for path in sorted((ROOT/'dataset/test'/label).rglob('*')):
                if path.suffix.lower() in ('.jpg','.jpeg','.png','.webp'):
                    test.append(dict(path=str(path.resolve()),target=target,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert not set(bundle['training_hashes']) & {r['sha256'] for r in test}
    assert len({r['sha256'] for r in test}) == len(test)
    test_images = prepare(test, args.source/'test/cache', output/'test_crops',detector_hash)
    test_s1 = stage1_scores(stage1,test,range(len(test)),names)
    test_s2 = scores(model,DataLoader(Crops(test_images,test,range(len(test))),batch_size=32,num_workers=0))
    test_y = np.array([r['target'] for r in test])
    test_missing = np.isnan(test_s1)
    pred = np.where(test_missing,test_s2>=best['stage2_threshold'],test_s1>=best['stage1_threshold'])
    report = dict(locked_selection=locked,train_total=len(rows),train_stage1_usable=sum(r['features'] is not None for r in rows),
                  test=metrics(test_y,pred),stage2_alone=metrics(test_y,test_s2>=best['stage2_threshold']),
                  fallback_only=metrics(test_y[test_missing],test_s2[test_missing]>=best['stage2_threshold']),
                  previous_binary=metrics(test_y,np.where(test_missing,True,test_s1>=.5265758057645732)),
                  validation_selected_positive_fallback=metrics(test_y,np.where(test_missing,True,test_s1>=positive_baseline['threshold'])),
                  same_stage1_threshold_positive_fallback=metrics(test_y,np.where(test_missing,True,test_s1>=best['stage1_threshold'])),
                  fallback_count=int(test_missing.sum()),unknown=0,coverage=1.,
                  caveat='Test photos participated in historical experiments; not untouched external validation. No test tuning in this run.')
    (output/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (output/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=['path','target','sha256','stage1_score','stage2_score','stage_used','prediction','reason'])
        writer.writeheader()
        for i,row in enumerate(test):
            writer.writerow(dict(path=row['path'],target=row['target'],sha256=row['sha256'],stage1_score=None if test_missing[i] else test_s1[i],
                                 stage2_score=test_s2[i],stage_used=2 if test_missing[i] else 1,prediction=int(pred[i]),reason=row['reason']))
    print('FINAL '+json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
