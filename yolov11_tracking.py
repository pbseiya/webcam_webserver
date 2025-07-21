# source = 'http://192.168.1.148:8080/video'
# source = 'https://youtu.be/LNwODJXcvt4'
source = '/mnt/f/project/qclab/clips/hips_ss_10_25.mp4'
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from ultralytics import YOLO

# Load a pretrained YOLOv11 model
model = YOLO('yolo11m.pt')  # Replace with your model path

# run inference on an source
# results = model.track(source=source, show=True, stream=True, conf=0.25, tracker='bytetrack.yaml')

# results = model.track(source=source, show=True)
# results = model.track(source=source, conf=0.3, iou=0.5, show=True)
results = model.track(source=source, conf=0.3, iou=0.5, show=True, tracker='bytetrack.yaml')  # with ByteTrack