import numpy as np
import gradio as gr
from detection import Detection
from segmentation import Segmentation
from inpainting import MiganInpainting
import os 
import cv2 

model_path = "models/migan_512_places2.pt"
inpainting = MiganInpainting(model_path)

# Read Image
conf_det = {
    "detection": {
        "model_path": "models/yolov8m.pt",
        "classes": {
            "person": 0.1,
        },
        "device": "cuda",
        "model_size_width": 1080,
        "model_size_height": 1920,
    }
}
det = Detection(conf_det["detection"])

conf_seg = {
    "detection": {
        "model_path": "models/yolov8m-seg.pt",
        "classes": {
            "person": 0.1,
        },
        "device": "cuda",
        "model_size_width": 1920,
        "model_size_height": 1080,
    }
}
seg = Segmentation(conf_seg["detection"])


filepath_folder = "videos/images"
names = os.listdir( filepath_folder)
for name in names:
    filename = os.path.join(filepath_folder, name)
    im = cv2.imread(filename, cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    #im_draw_list, mask_list = seg.detect(im)
    im_draw_list, mask_list = det.detect(im)
    
    im_draw = im_draw_list[0]
    mask = mask_list[0]
    
    #Inpainting
    result = inpainting.inpaint(im, mask)
    
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"videos/results/{name}", result)
    break