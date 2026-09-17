"""Image-based fallback for photographs without usable driver landmarks."""
import hashlib
from pathlib import Path

from PIL import Image, ImageOps
import torch
from torchvision import transforms
from torchvision.models import resnet18

from features import driver_person_box

IMAGE_SIZE = 320
NORMALIZE = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
PREPROCESS = transforms.Compose([transforms.ToTensor(), NORMALIZE])


def driver_crop(path, observation, side='right'):
    box = driver_person_box(observation['boxes'], side)
    if box is None:
        box = [.4, 0, 1, 1] if side == 'right' else [0, 0, .6, 1]
    else:
        x1, y1, x2, y2 = box
        width, height = x2-x1, y2-y1
        box = [max(0, x1-.15*width), max(0, y1-.08*height),
               min(1, x2+.15*width), min(1, y2+.08*height)]
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    w, h = image.size
    image = image.crop(tuple(round(v*d) for v, d in zip(box, [w, h, w, h])))
    return ImageOps.pad(image, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.BILINEAR, color=(114,114,114))


def new_model(weights_path=None):
    model = resnet18(weights=None)
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model


class ImageFallback:
    def __init__(self, config, device='auto'):
        checkpoint = Path(config['checkpoint'])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != config['sha256']:
            raise ValueError('Stage 2 checkpoint hash mismatch')
        self.device = torch.device('cuda:0' if device == 'auto' and torch.cuda.is_available() else
                                   'cpu' if device in ('auto', 'cpu') else 'cuda:'+str(device))
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['image_size'] != IMAGE_SIZE:
            raise ValueError('Stage 2 preprocessing version mismatch')
        self.model = new_model()
        self.model.load_state_dict(state['state_dict'])
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def score(self, path, observation, side='right'):
        x = PREPROCESS(driver_crop(path, observation, side)).unsqueeze(0).to(self.device)
        return float(self.model(x).softmax(1)[0, 1])
