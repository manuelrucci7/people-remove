import numpy as np
from PIL import Image
import torch

def preprocess(img: Image, mask: Image, resolution: int) -> torch.Tensor:
    img = img.resize((resolution, resolution), Image.BICUBIC)
    mask = mask.resize((resolution, resolution), Image.NEAREST)
    img = np.array(img)
    mask = np.array(mask)[:, :, np.newaxis] // 255
    img = torch.Tensor(img).float() * 2 / 255 - 1
    mask = torch.Tensor(mask).float()
    img = img.permute(2, 0, 1).unsqueeze(0)
    mask = mask.permute(2, 0, 1).unsqueeze(0)
    x = torch.cat([mask - 0.5, img * mask], dim=1)
    return x

def resize(image, max_size, interpolation=Image.BICUBIC):
    w, h = image.size
    if w > max_size or h > max_size:
        resize_ratio = max_size / w if w > h else max_size / h
        image = image.resize((int(w * resize_ratio), int(h * resize_ratio)), interpolation)
    return image

def read_mask(mask, invert=False):
    mask = Image.fromarray(mask)#open(mask_path)
    mask = resize(mask, max_size=512, interpolation=Image.NEAREST)
    mask = np.array(mask)
    if len(mask.shape) == 3:
        if mask.shape[2] == 4:
            _r, _g, _b, _a = np.rollaxis(mask, axis=-1)
            mask = np.dstack([_a, _a, _a])
        elif mask.shape[2] == 2:
            _l, _a = np.rollaxis(mask, axis=-1)
            mask = np.dstack([_a, _a, _a])
        elif mask.shape[2] == 3:
            _r, _g, _b = np.rollaxis(mask, axis=-1)
            mask = np.dstack([_r, _r, _r])
    else:
        mask = np.dstack([mask, mask, mask])
    if invert:
        mask = 255 - mask
    mask[mask < 255] = 0
    return Image.fromarray(mask).convert("L")