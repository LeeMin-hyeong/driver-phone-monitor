"""Export fixed-shape FP16 TensorRT engines for the current RTX host."""
import hashlib
import json
import torch
import tempfile
import shutil
from pathlib import Path
from extract import ROOT
from ultralytics import YOLO


def main():
    torch.set_num_threads(1)
    results=[]
    # TensorRT's native ONNX parser cannot open this host's Korean user path.
    work=Path(tempfile.mkdtemp(prefix='driver_trt_',dir='C:/msys64/tmp'))
    for name in ('yolo26m','yolo26n-pose'):
        path=ROOT/'models'/(name+'.pt')
        staging=work/path.name
        shutil.copy2(path,staging)
        exported=YOLO(str(staging)).export(format='engine',imgsz=640,batch=1,dynamic=False,
                                     quantize=16,workspace=1,device=0,opset=17,simplify=True,verbose=False)
        target=ROOT/'models'/(name+'.engine')
        shutil.copy2(exported,target)
        results.append(dict(source=str(path.relative_to(ROOT)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            engine=str(target.relative_to(ROOT)),engine_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    import tensorrt
    report=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,tensorrt=tensorrt.__version__,
                input_shape=[1,3,640,640],precision='FP16',artifacts=results,
                note='Engines built for this GPU/runtime; rebuild for a different deployment target.')
    (ROOT/'artifacts/fast_pose640_sensitive/tensorrt_export.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
