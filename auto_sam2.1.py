# https://www.youtube.com/watch?v=M7xWw4Iodhg
# download sam2_b.pt from wget https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_b.pt

source = 'cats_dogs.jpg'
sam_model = 'sam2_b.pt'

from ultralytics.data.annotator import auto_annotate

auto_annotate(data=source, det_model='yolo11n.pt', sam_model=sam_model)