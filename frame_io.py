"""Decode original JPEGs on GPU and transfer only the resized RGB frame."""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import decode_jpeg


@torch.inference_mode()
def read_rgb_gpu(path,max_side=1280):
    # Read each original file on every invocation; no decoded-image cache.
    data=torch.frombuffer(bytearray(Path(path).read_bytes()),dtype=torch.uint8)
    frame=decode_jpeg(data,device='cuda',apply_exif_orientation=True)
    if frame.shape[0]!=3:
        raise ValueError('Expected a three-channel JPEG')
    _,h,w=frame.shape
    factor=min(1,max_side/max(h,w))
    if factor<1:
        frame=F.interpolate(frame.unsqueeze(0).float(),size=(round(h*factor),round(w*factor)),
                            mode='bilinear',align_corners=False).squeeze(0).round().clamp(0,255).to(torch.uint8)
    return np.ascontiguousarray(frame.permute(1,2,0).cpu().numpy())
