import numpy as np
import gradio as gr
from detection import Detection
from segmentation import Segmentation
from inpainting import MiganInpainting
import cv2 
import shutil
import os 
import subprocess

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

def process(im):
    
    print(im)

    # Detection
    print(im.shape)
    im_draw_list, mask_list = seg.detect(im)
    #im_draw_list, mask_list = det.detect(im)
    
    im_draw = im_draw_list[0]
    mask = mask_list[0]
    
    #Inpainting
    result = inpainting.inpaint(im, mask)
    
    return im_draw

def process_video(video):
    folder_tmp = "videos/results"
    if os.path.exists(folder_tmp):
        shutil.rmtree(folder_tmp)
    os.makedirs(folder_tmp)

    # Create a video capture object
    cap = cv2.VideoCapture(video)
    frames = []

    i= 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert each frame to grayscale
        #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #im_draw_list, mask_list = seg.detect(im)
        im_draw_list, mask_list = det.detect(frame)
        
        im_draw = im_draw_list[0]
        mask = mask_list[0]
        
        #Inpainting
        result = inpainting.inpaint(frame, mask)

        #result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"videos/results/rome_{i}.png", result)
        i = i +1
        frames.append(result)

    cap.release()

    video_path =  "output_video.mp4"
    if os.path.exists(video_path):
        os.remove(video_path)

    ffmpeg_command = f"ffmpeg -framerate 10 -i videos/results/rome_%d.png -c:v libx264 -pix_fmt yuv420p output_video.mp4"  
    subprocess.run(ffmpeg_command, shell=True)
    print("Completed")

    return 'output_video.mp4'


#demo = gr.Interface(process, gr.Image(), "image")
demo = gr.Interface(process_video, inputs=gr.Video(),  outputs=gr.Video())
demo.launch(share=True)




#ffmpeg -framerate 10 -i results/rome_%d.png -c:v libx264 -pix_fmt yuv420p output_video.mp4