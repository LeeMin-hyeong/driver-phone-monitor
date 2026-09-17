"""Train a fixed-driver-ROI classifier, selecting solely on the shared train split."""
import argparse
import csv
import hashlib
import json
import time

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from extract import ROOT
from cascade import new_model, NORMALIZE
from training.train_cascade import seed_all, train_epoch, scores, metrics

ROI=(.42,.20,1.,1.)
SIZE=224


def crop(path):
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im).convert('RGB')
        w,h=im.size
        im=im.crop(tuple(round(v*s) for v,s in zip(ROI,(w,h,w,h))))
        return ImageOps.pad(im,(SIZE,SIZE),method=Image.Resampling.BILINEAR,color=(114,114,114))


class Data(Dataset):
    def __init__(self,images,rows,indices,augment=False):
        self.images,self.rows,self.indices=images,rows,list(indices)
        self.transform=transforms.Compose(([
            transforms.RandomAffine(8,translate=(.05,.05),scale=(.9,1.1),fill=114),
            transforms.RandomHorizontalFlip(),transforms.ColorJitter(.2,.2,.15,.03)
        ] if augment else [])+[transforms.ToTensor(),NORMALIZE])
    def __len__(self):
        return len(self.indices)
    def __getitem__(self,i):
        j=self.indices[i]
        return self.transform(self.images[j]),int(self.rows[j]['target'])


def setup(rows,indices):
    seed_all()
    model=new_model(ROOT/'models/resnet18-f37072fd.pth').cuda()
    opt=torch.optim.AdamW([
        dict(params=[p for n,p in model.named_parameters() if not n.startswith('fc.')],lr=1e-4),
        dict(params=model.fc.parameters(),lr=1e-3)],weight_decay=1e-4)
    counts=np.bincount([int(rows[i]['target']) for i in indices],minlength=2)
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(len(indices)/(2*counts),dtype=torch.float32,device='cuda'))
    return model,opt,torch.amp.GradScaler('cuda'),criterion


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=25)
    parser.add_argument('--output',default='artifacts/fast_roi_v1')
    parser.add_argument('--kind',choices=['roi','wrists'],default='roi')
    args=parser.parse_args()
    torch.set_num_threads(4)
    out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'locked_selection.json').exists(),'Do not overwrite a completed experiment'
    rows=list(csv.DictReader((ROOT/'configs/dataset/train.csv').open(encoding='utf-8-sig')))
    split=json.loads((ROOT/'configs/dataset/train_validation.json').read_text())
    fit=[i for i,r in enumerate(rows) if r['sha256'] in set(split['fit_hashes'])]
    val=[i for i,r in enumerate(rows) if r['sha256'] in set(split['validation_hashes'])]
    assert len(fit)+len(val)==len(rows) and not set(fit)&set(val)
    images=[];cache=out/'crops';cache.mkdir(exist_ok=True)
    for i,r in enumerate(rows):
        path=ROOT/r['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
        target=cache/(r['sha256']+'.png')
        if not target.exists():
            if args.kind=='roi':
                crop(path).save(target)
            else:
                from wrist_image import wrist_image
                from training.train_fast_pose import combined_features
                obs=json.loads((ROOT/'artifacts/single_yolo640_roi_input640_conf0.01/train'/(r['sha256']+'.json')).read_text())
                pose=json.loads((ROOT/'artifacts/fast_pose_input640/train'/(r['sha256']+'.json')).read_text())
                _,info=combined_features(obs,pose)
                with Image.open(ROOT/'artifacts/input640/images'/(r['sha256']+'.jpg')) as source:
                    wrist_image(source.convert('RGB'),pose['keypoints'],info.get('driver_index')).save(target)
        with Image.open(target) as im: images.append(im.convert('RGB'))
        if i%100==0: print(f'prepare {i}/{len(rows)}',flush=True)
    model,opt,scaler,criterion=setup(rows,fit)
    train_loader=DataLoader(Data(images,rows,fit,True),batch_size=32,shuffle=True,num_workers=0)
    val_loader=DataLoader(Data(images,rows,val),batch_size=32,num_workers=0)
    y=np.array([int(rows[i]['target']) for i in val])
    best=None;history=[]
    for epoch in range(1,args.epochs+1):
        start=time.perf_counter()
        loss=train_epoch(model,train_loader,opt,scaler,criterion)
        s=scores(model,val_loader)
        candidates=[dict(threshold=float(t),**metrics(y,s>=t)) for t in np.arange(.1,.901,.025)]
        current=max(candidates,key=lambda c:(c['f1'],c['recall'],c['precision'],-abs(c['threshold']-.5)))
        current.update(epoch=epoch,loss=loss,seconds=time.perf_counter()-start)
        history.append(current)
        if best is None or (current['f1'],current['recall'])>(best['f1'],best['recall']):
            best=current.copy()
            torch.save(dict(state_dict=model.state_dict(),image_size=SIZE,roi=ROI,input_kind=args.kind),out/'validation.pt')
            np.save(out/'validation_scores.npy',s)
        (out/'history.json').write_text(json.dumps(history,indent=2))
        print(json.dumps(current),flush=True)
    locked=dict(best=best,roi=ROI,image_size=SIZE,input_kind=args.kind,architecture='resnet18',seed=42,
                train_count=len(rows),fit_count=len(fit),validation_count=len(val),
                selection='Shared train validation F1, ties recall; no test data read',
                training_hashes=[r['sha256'] for r in rows])
    (out/'locked_selection.json').write_text(json.dumps(locked,indent=2))
    del model,opt,scaler
    torch.cuda.empty_cache()
    indices=list(range(len(rows)))
    model,opt,scaler,criterion=setup(rows,indices)
    loader=DataLoader(Data(images,rows,indices,True),batch_size=32,shuffle=True,num_workers=0)
    for epoch in range(1,best['epoch']+1):
        loss=train_epoch(model,loader,opt,scaler,criterion)
        print(f'refit {epoch}/{best["epoch"]} loss={loss:.4f}',flush=True)
    torch.save(dict(state_dict=model.state_dict(),image_size=SIZE,roi=ROI,threshold=best['threshold'],input_kind=args.kind),out/'model.pt')
    print('Complete: '+str(out),flush=True)


if __name__=='__main__':
    main()
