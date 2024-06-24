# people-remove

https://lightning.ai/ruccimanuel7/studios/people-remove-yolo-migan

Remove people from images using ultralitycs yolo and migan inpainting

## Setup

```
python3 -m venv env
source env/bin/activate
# Install cpu torch: https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install opencv-python
pip3 install ultralytics
```

```
cd code
python3 detection.py
```
