"""Experimental fixed-camera driver classifier; not the default classifier."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms
from cascade import new_model, NORMALIZE


class FastROI:
    def __init__(self, checkpoint, device='cuda'):
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.device=torch.device(device)
        self.model=new_model().to(self.device).eval()
        self.model.load_state_dict(state['state_dict'])
        self.roi=state['roi'];self.size=state['image_size'];self.threshold=state['threshold']
        self.transform=transforms.Compose([transforms.ToTensor(),NORMALIZE])

    @torch.inference_mode()
    def score_frame(self,image):
        """Accept an EXIF-upright RGB PIL image, already decoded by caller."""
        w,h=image.size
        crop=image.crop(tuple(round(v*s) for v,s in zip(self.roi,(w,h,w,h))))
        crop=ImageOps.pad(crop,(self.size,self.size),method=Image.Resampling.BILINEAR,color=(114,114,114))
        x=self.transform(crop).unsqueeze(0).to(self.device)
        return float(self.model(x).softmax(1)[0,1])

    def score_file(self,path,reduced=False):
        if reduced:
            import cv2
            frame=cv2.imdecode(np.fromfile(Path(path),dtype=np.uint8),cv2.IMREAD_REDUCED_COLOR_2)
            if frame is None: raise ValueError('Cannot decode '+str(path))
            image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        else:
            with Image.open(path) as source:
                image=ImageOps.exif_transpose(source).convert('RGB')
        return self.score_frame(image)
