import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from ultralytics.data.annotator import auto_annotate

torch.serialization.add_safe_globals([YOLO])

# Load a model
model = YOLO('./face_det.pt', task='detect')

# Predict with the model
results = model("./teamrgb.jpg")  # predict on an image

classes: torch.Tensor = results[0].boxes.cls.int()
classes = classes.numpy()
classes = np.where(classes == 1)[0]
head_xyxy = results[0].boxes.xyxy.numpy()
head_boxes = head_xyxy[classes]

im = plt.imread('./teamrgb.jpg')
annotate = Annotator(im)
for box in head_boxes:
    annotate.box_label(box, label='head')
annotate.save()
out = Image.open('image.jpg')
out = out.convert('BGR')
out.save('correct.jpg')
