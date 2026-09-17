"""Two driver wrist crops for image-level phone classification."""
import numpy as np
from PIL import Image,ImageOps


def wrist_image(image,keypoints,driver_index):
    w,h=image.size
    canvas=Image.new('RGB',(448,224),(114,114,114))
    fallback=image.crop((round(.35*w),round(.2*h),w,h))
    for slot,j in enumerate((9,10)):
        patch=fallback
        if driver_index is not None:
            p=np.asarray(keypoints[driver_index])
            if p[j,2]>=.25:
                shoulder=np.linalg.norm((p[5,:2]-p[6,:2])*[w,h])
                radius=max(32,.7*shoulder)
                cx,cy=p[j,:2]*[w,h]
                box=(max(0,round(cx-radius)),max(0,round(cy-radius)),min(w,round(cx+radius)),min(h,round(cy+radius)))
                if box[2]>box[0] and box[3]>box[1]:patch=image.crop(box)
        patch=ImageOps.pad(patch,(224,224),method=Image.Resampling.BILINEAR,color=(114,114,114))
        canvas.paste(patch,(slot*224,0))
    return canvas
