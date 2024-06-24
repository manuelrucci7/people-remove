import cv2

# Create a VideoCapture object
cap = cv2.VideoCapture('videos/rome-1920-1080-10fps-short.mp4')

# Check if camera opened successfully
if not cap.isOpened(): 
    print("Error opening video file")

# Read until video is completed
i = 0
while(cap.isOpened()):
  
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

        cv2.imwrite(f"videos/rome_{i}.png", frame )
        i = i+1

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()