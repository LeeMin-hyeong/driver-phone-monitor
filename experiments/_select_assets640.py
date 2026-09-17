import csv
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
root=Path('.'); rows=[r for r in csv.DictReader(open('artifacts/fast_pose640_sensitive/test_runtime_fp16_vectorized_trt_fusion.csv',encoding='utf-8-sig')) if r['repeat']=='0']
groups=[[r for r in rows if r['target']=='1' and r['prediction']=='1' and r['stage_used']=='1'],[r for r in rows if r['target']=='0' and r['prediction']=='0'],[r for r in rows if r['target']=='1' and r['prediction']=='1' and r['stage_used']=='2']]
out=Path('docs/assets/realtime640');out.mkdir(parents=True,exist_ok=True)
canvas=Image.new('RGB',(1280,3*350),'white');d=ImageDraw.Draw(canvas)
font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',15)
selected=[]
for gi,g in enumerate(groups):
 g.sort(key=lambda r: (not any(x in r['path'] for x in ['IMG_7564','IMG_7441','IMG_7504']), -float(r['score']) if gi!=1 else float(r['score'])))
 for j,r in enumerate(g[:4]):
  canvas.paste(Image.open('artifacts/input640/images/'+r['sha256']+'.jpg').resize((320,320)),(320*j,gi*350))
  d.text((320*j+3,gi*350+320),f'{gi}/{j} '+Path(r['path']).name[:12]+f" s1={float(r['stage1_score']):.3f}",font=font,fill='black')
  selected.append(r)
canvas.save(out/'candidates.jpg')
import json
(out/'candidates.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8')
print('fallback count',len(groups[2]))
