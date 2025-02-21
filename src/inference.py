"""
"""
################################################################################
import time
import sys
import cv2
import matplotlib.pyplot as plt
import torch
import gc
import numpy as np
from PySide6.QtCore import QObject, Signal
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from comm import Comm
################################################################################


class Inference(QObject):

    update_num_heads = Signal(int)

    def __init__(self):
        super().__init__()
        # List of models
        self.head_model = './src/models/face_det.pt'
        self.emotion_model = './src/models/emotion.pt'
        self.pose_model = './src/models/yolo11n-pose.pt'
        self.detect_model = './src/models/yolo11n.pt'
        self.hand_model = './src/models/hand-pose.pt'

        self.predict_funcs = {
            'detect_head': self.detect_head,
            'pose': self.pose,
            'default_detection': self.detect,
            'detect_hand': self.detect_hand
        }

        # ABii Communication
        self.abii_comm = Comm(abii_connected=False)

        self.modes = list(self.predict_funcs.keys())
        self.num_modes = len(self.modes)
        self.mode_idx = 0
        self.mode = self.modes[self.mode_idx]

        self.use_max = True
        self.focus_idx = 0
        self.num_detections = 0
        self.focus_frame = None

        # YOLO Model
        self.head_detection = None
        self.emotion_classifier = None
        self.pose_detection = None
        self.default_detection = None
        self.hand_detection = None
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

    def next_detection(self):
        self.use_max = False
        self.focus_idx = (self.focus_idx + 1) % self.num_detections

    def prev_detection(self):
        self.use_max = False
        self.focus_idx = (self.focus_idx - 1) % self.num_detections

    def reset(self):
        self.use_max = True
        self.focus_idx = 0
        self.mode_idx = 0
        self.mode = self.modes[self.mode_idx]
        self.load_models()

    def cycle_models_up(self):
        self.mode_idx = (self.mode_idx + 1) % self.num_modes
        self.mode = self.modes[self.mode_idx]
        self.load_models()

    def cycle_models_down(self):
        self.mode_idx = (self.mode_idx - 1) % self.num_modes
        self.mode = self.modes[self.mode_idx]
        self.load_models()

    def load_models(self):
        if self.mode == 'detect_head':
            (self.pose_detection,
             self.default_detection,
             self.hand_detection) = (None, None, None)
            self.head_detection = YOLO(self.head_model, task='detect')
            self.emotion_classifier = self.emotion_model
        elif self.mode == 'pose':
            (self.head_detection,
             self.emotion_classifier,
             self.default_detection,
             self.hand_detection) = (None, None, None, None)
            self.pose_detection = YOLO(self.pose_model, task='pose')
        elif self.mode == 'default_detection':
            (self.head_detection,
             self.emotion_classifier,
             self.pose_detection,
             self.hand_detection) = (None, None, None, None)
            self.default_detection = YOLO(self.detect_model,  task='detect')
        elif self.mode == 'detect_hand':
            (self.head_detection,
             self.emotion_classifier,
             self.pose_detection,
             self.default_detection) = (None, None, None, None)
            self.hand_detection = YOLO(self.hand_model, task='pose')
        # TODO: Check that this actually reduces memory consumption
        gc.collect()

    def detect_head(self, frame):
        annotate = Annotator(frame)
        outputs = self.head_detection(frame, verbose=False)
        classes: torch.Tensor = outputs[0].boxes.cls.int()
        classes = classes.numpy()
        classes = np.where(classes == 1)[0]
        confidence = outputs[0].boxes.conf

        # Convert types to numpy arrays
        detections_xyxy = outputs[0].boxes.xyxy.numpy()
        detections_xywh = outputs[0].boxes.xywh.numpy()

        # Only get the heads
        head_boxes_xyxy = detections_xyxy[classes]
        head_boxes_xywh = detections_xywh[classes]

        # Update the number of detections
        self.num_detections = len(head_boxes_xyxy)

        if self.num_detections < 1:
            return annotate.im

        # Extract widths and heights from xywh
        head_boxes_w = head_boxes_xywh[:, 2]
        head_boxes_h = head_boxes_xywh[:, 3]

        # Compute Area and get the max area
        if self.use_max:
            areas = head_boxes_w * head_boxes_h
            self.focus_idx = np.argmax(areas)

        # Sanity check the focus index to ensure it is not above the detections
        if self.focus_idx >= self.num_detections:
            self.focus_idx = self.num_detections - 1

        # Get the head that is closest
        fx1, fy1, fx2, fy2 = head_boxes_xyxy[self.focus_idx]
        self.focus_frame = frame[int(fy1): int(fy2), int(fx1): int(fx2)]

        # Annotate all boxes onto the frame, set the closest to green border and
        # others to blue border
        for i, box in enumerate(head_boxes_xyxy):
            if i == self.focus_idx:
                annotate.box_label(
                    box, label=f'{confidence[i]:.2f}', color=self.focus_box_color)
            else:
                annotate.box_label(
                    box, label=f'{confidence[i]:.2f}', color=self.default_box_color)

        return annotate.im

    def detect_emotion(self):
        output = self.emotion_classifier(self.focus_frame)
        # Map to correct emotion
        emotion = output
        # Send to ABii
        self.abii_comm.send_fer_class(emotion)

    def detect(self, frame):
        # Perform inference
        outputs = self.default_detection(frame, verbose=False)
        # Get the dictionary of class ids (key) mapped to their string class
        # name (value)
        class_dict = outputs[0].names
        # Get the tensor containing the detected class ids
        classes: torch.Tensor = outputs[0].boxes.cls.int()
        classes = classes.numpy()
        # Get the confidence intervals
        confidence = outputs[0].boxes.conf
        # Convert types to numpy arrays
        boxes_xyxy = outputs[0].boxes.xyxy.numpy()

        # Annotate all boxes onto the frame with blue border
        annotate = Annotator(frame)
        for i, box in enumerate(boxes_xyxy):
            annotate.box_label(
                box,
                # This is the label to display on the bounding box
                label=f'{class_dict[classes[i]]} {confidence[i]:.2f}',
                color=self.default_box_color
            )

        return annotate.im

    def pose(self, frame):
        annotate = Annotator(frame)
        # Inference
        pose_results = self.pose_detection(frame, verbose=False)
        # Update the number of detections
        self.num_detections = len(pose_results[0].keypoints.data)
        # Return early if there are no detections
        if self.num_detections < 1:
            return annotate.im
        # Get the left and right ear points
        detected_poses = pose_results[0].keypoints.xy.numpy()
        people = detected_poses[:, 5:7]
        # print(people.shape)
        # TODO: Can probably do better than looping.
        distances = []
        for shoulders in people:
            l_shoulder, r_shoulder = shoulders
            distances.append(np.sqrt((l_shoulder[0] - r_shoulder[0]) **
                                     2 + (l_shoulder[1] - r_shoulder[1])**2))
        closest_pose_idx = np.argmax(distances)
        annotate.kpts(detected_poses[closest_pose_idx])
        return annotate.im

    def detect_hand(self, frame):
        hand_results = self.hand_detection(frame, verbose=False)

        # Update the number of detections
        self.num_detections = len(hand_results[0].keypoints.data)
        # print(len(hand_results[0].keypoints.data))
        annotate = Annotator(frame)
        # Return early if there are no detections
        if self.num_detections < 1:
            return annotate.im
        for i, hands in enumerate(hand_results[0].keypoints.data):
            annotate.kpts(hands)
            # Limit number of hands to plot to 2
            if i == 1:
                break
        return annotate.im

    def predict(self, frame):
        return self.predict_funcs[self.mode](frame)
