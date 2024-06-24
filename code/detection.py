from ultralytics import YOLO
import numpy as np
import torch
import cv2

class Detection:
    def __init__(self, conf):
        self.device = conf["device"]
        self.model = YOLO(conf["model_path"])
        self.img_size = [conf["model_size_width"], conf["model_size_height"]]
        self.conf_thres = 0.1
        self.iou_thres = 0.5
        self.classes = conf["classes"]
        
    def detect(self, im):
        im_draw_list = []
        mask_list = []

        results = self.model(im, imgsz=self.img_size, conf=self.conf_thres, iou=self.iou_thres, device=torch.device(self.device), verbose=False)
        
        for i in range(0, len(results)):
            im_draw=im.copy()
            maskf = np.zeros(im.shape[:2], dtype=np.uint8)

            names = results[i].names
            scores = results[i].boxes.conf.cpu().numpy()
            class_ids = results[i].boxes.cls.cpu().numpy()
            boxes = results[i].boxes.xyxy.cpu().numpy()

            for box, score, class_id in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = box.astype(int)
                class_name = names[class_id]
                if class_name in self.classes and score >= self.classes[class_name]:
                    if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= im.shape[1] and y2 <= im.shape[0]:
                        #YOLO 
                        maskf[y1:y2, x1:x2] = 255
                                
                        cv2.putText(im_draw, f"{names[class_id]}: {score:.2f}", (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        cv2.rectangle(im_draw, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            im_draw_list.append(im_draw)
            mask_list.append(maskf)

        return im_draw_list, mask_list
    
if __name__ == "__main__":
    # Read Image
    filepath = "images/bus.jpg"
    conf = {
        "detection": {
            "model_path": "models/yolov8n.pt",
            "classes": {
                "person": 0.5,
            },
            "device": "cpu",
            "model_size_width": 640,
            "model_size_height": 640,
        }
    }
    im = cv2.imread(filepath, cv2.IMREAD_COLOR)

    det = Detection(conf["detection"])
    im_draw_list, mask_list = det.detect(im)
    im_draw = im_draw_list[0]
    mask = mask_list[0]
    #cv2.imshow("im_draw", im_draw)
    #cv2.imshow("mask", mask)
    cv2.imwrite("images/mask_det.jpg", mask)
    #cv2.waitKey(0)