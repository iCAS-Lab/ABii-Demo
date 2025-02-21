"""
"""
################################################################################
import time
import sys
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from PySide6.QtCore import QObject, Signal
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
################################################################################


class Inference(QObject):

    update_num_heads = Signal(int)

    def __init__(
        self,
        mode: str = 'default_detection'
    ):
        super().__init__()
        # List of models
        self.head_detect = './src/models/face_det.pt'
        self.emotion_model = './src/models/emotion.pt'
        self.pose_model = './src/models/yolo11n-pose.pt'
        self.detect_model = './src/models/yolo11n.pt'

        self.predict_funcs = {
            'detect_emotion': self.detect_emotion,
            'pose': self.pose,
            'default_detection': self.detect
        }

        # Sanity check the modes
        if mode in list(self.predict_funcs.keys()):
            self.mode = mode
        else:
            raise Exception(f'Inference mode must be in:\n{self.valid_modes}')

        # YOLO Model
        self.head_detection = None
        self.emotion_classifier = None
        self.pose_detection = None
        self.default_detection = None
        self.load_models()

        # Original Image Size
        self.camera_resolutions = {
            0: (96, 96),
            1: (100, 100),
            2: (128, 128),
            3: (256, 256),
            4: (320, 320),
            5: (640, 360),
            6: (640, 480),
            7: (800, 600),
            8: (960, 540),
            9: (960, 720),
            10: (1024, 576),
            11: (1280, 720),
            12: (1280, 960),
            13: (1920, 1080),
            # 14: (2560, 1440),
            # 15: (3840, 2160)
        }
        self.current_camera_resolution = self.camera_resolutions[13]

        self.focus_box_color = (0, 128, 0)
        self.default_box_color = (0, 0, 128)

        # Setup Statistics Dictionary
        self.init_stats_dict()

    def reset_values(self):
        self.init_stats_dict()

    def change_camera_resolution(self, id: int):
        self.current_camera_resolution = self.camera_resolutions[id]
        self.reset_values()
        return self.current_camera_resolution

    def init_stats_dict(self):
        self.stats = {}

    def draw_bbox_on_image(self):
        pass

    def crop_image_by_relative_coords(self):
        pass

    def update_stats(self):
        pass

    def load_models(self):
        if self.mode == 'detect_emotion':
            del self.pose_detection
            del self.default_detection
            self.head_detection = YOLO(self.head_detect, task='detect')
            self.emotion_classifier = self.emotion_model
        elif self.mode == 'pose':
            del self.head_detection
            del self.emotion_classifier
            del self.default_detection
            self.pose_detection = YOLO(self.pose_model, task='pose')
        elif self.mode == 'default_detection':
            del self.head_detection
            del self.emotion_classifier
            del self.pose_detection
            self.default_detection = YOLO(self.detect_model,  task='detect')

    def detect_emotion(self, frame):
        outputs = self.head_detection(frame, verbose=False)
        classes: torch.Tensor = outputs[0].boxes.cls.int()
        classes = classes.numpy()
        classes = np.where(classes == 1)[0]

        # Convert types to numpy arrays
        head_xyxy = outputs[0].boxes.xyxy.numpy()
        head_xywh = outputs[0].boxes.xywh.numpy()

        # Only get the heads
        head_boxes_xyxy = head_xyxy[classes]
        head_boxes_xywh = head_xywh[classes]

        # Extract widths and heights from xywh
        head_boxes_w = head_boxes_xywh[:, 2]
        head_boxes_h = head_boxes_xywh[:, 3]

        # Compute Area and get the max area
        areas = head_boxes_w * head_boxes_h
        max_area_idx = np.argmax(areas)

        # Get the head that is closest
        fx1, fy1, fx2, fy2 = head_boxes_xyxy[max_area_idx]
        focus_frame = frame[int(fy1): int(fy2), int(fx1): int(fx2)]

        # Annotate all boxes onto the frame, set the closest to green border and
        # others to blue border
        annotate = Annotator(frame)
        for i, box in enumerate(head_boxes_xyxy):
            if i == max_area_idx:
                annotate.box_label(box, label='', color=self.focus_box_color)
            else:
                annotate.box_label(box, label='', color=self.default_box_color)

        # TODO: Get the emotion if spacebar is pressed

        # TODO: Sent emotion to ABii

        return annotate.im

    def detect(self, frame):
        outputs = self.default_detection(frame, verbose=False)
        class_dict = outputs[0].names
        classes: torch.Tensor = outputs[0].boxes.cls.int()
        classes = classes.numpy()
        confidence = outputs[0].boxes.conf
        # Convert types to numpy arrays
        boxes_xyxy = outputs[0].boxes.xyxy.numpy()

        # Annotate all boxes onto the frame, set the closest to green border and
        # others to blue border
        annotate = Annotator(frame)
        for i, box in enumerate(boxes_xyxy):
            annotate.box_label(
                box,
                label=f'{class_dict[classes[i]]} {confidence[i]:.2f}',
                color=self.default_box_color
            )

        return annotate.im

    def pose(self, frame):
        pose_results = self.pose_detection(frame, verbose=False)
        annotate = Annotator(frame)
        annotate.kpts(pose_results[0].keypoints.data[0])
        return annotate.im

    def predict(self, frame):
        return self.predict_funcs[self.mode](frame)
