import cv2
from inpainting import MiganInpainting
from segmentation import Segmentation
from detection import Detection

# Read Image
filepath = "images/bus.jpg"
conf = {
    "detection": {
        "model_path": "models/yolov8m.pt",
        "classes": {
            "person": 0.1,
        },
        "device": "cpu",
        "model_size": 1920,
    }
}
im = cv2.imread(filepath, cv2.IMREAD_COLOR)

det = Detection(conf["detection"])
model_path = "models/migan_512_places2.pt"
inpainting = MiganInpainting(model_path)

# Create a VideoCapture object
cap = cv2.VideoCapture('videos/rome-1920-1080-10fps-short.mp4')

# Check if camera opened successfully
if not cap.isOpened(): 
    print("Error opening video file")

# Read until video is completed
while(cap.isOpened()):
  
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # ---------------------------------
    # Segmentation
    im_draw_list, mask_list = det.detect(frame)
    im_draw = im_draw_list[0]
    mask = mask_list[0]

    #cv2.imshow('Frame', im_draw)
    #cv2.waitKey(0)
    # Inpainting
    result = inpainting.inpaint(frame, mask)
    # ---------------------------------
    
    # Display the resulting frame
    cv2.imwrite("images/rome.jpg", result)
    
    cv2.imshow('Frame', result)
    cv2.waitKey(0)

    # Press Q on keyboard to exit
    if cv2.waitKey(33) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()