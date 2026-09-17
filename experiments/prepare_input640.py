"""Create deployment-like 640x640 JPEG inputs, preserving all original split labels."""
import csv
import hashlib
import json
from PIL import Image,ImageOps
from extract import ROOT


def main():
    out=ROOT/'artifacts/input640/images';out.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for split in ('train','test'):
        rows=list(csv.DictReader((ROOT/f'configs/dataset/{split}.csv').open(encoding='utf-8-sig')))
        for i,r in enumerate(rows):
            path=ROOT/r['path'];target=out/(r['sha256']+'.jpg')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
            if not target.exists():
                with Image.open(path) as im:
                    image=ImageOps.exif_transpose(im).convert('RGB')
                image=ImageOps.pad(image,(640,640),method=Image.Resampling.BILINEAR,color=(114,114,114))
                image.save(target,quality=95,subsampling=0)
            manifest.append(dict(split=split,source=r['path'],source_sha256=r['sha256'],input_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),input=str(target.relative_to(ROOT)),target=int(r['target'])))
            if (i+1)%100==0:print(f'{split} {i+1}/{len(rows)}',flush=True)
    (out.parent/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Complete',flush=True)


if __name__=='__main__':main()
