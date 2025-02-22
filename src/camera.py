import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from PySide6.QtCore import QSize, QTimer, Signal, Qt
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QImage, QPixmap
from inference import Inference
################################################################################


class Camera(QLabel):

    update_frame = Signal(tuple)
    update_stats = Signal(dict)

    def __init__(self, parent):
        QLabel.__init__(self)
        self.video_size = QSize(1920, 1080)
        self.setup_camera()
        self.model = Inference()

    def setup_camera(self):
        """Initialize camera.
        """
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,
                         self.video_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,
                         self.video_size.height())
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(
            'M', 'J', 'P', 'G'))  # depends on fourcc available camera
        self.capture.set(cv2.CAP_PROP_FPS, 30)

        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(1)

    def set_tab(self, new_tab: str):
        self.tab = new_tab

    def display_video_stream(self):
        """Read frame from camera and repaint QLabel widget.
        """
        # print(f'Reading Frame: {time.time()}')
        res, frame = self.capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        # Perform detection
        ann_frame = self.model.predict(frame)
        # Check that we have valid results
        if ann_frame is not None:
            display_frame = QImage(
                ann_frame, ann_frame.shape[1], ann_frame.shape[0],
                ann_frame.strides[0], QImage.Format.Format_RGB888
            )
            display_pm = QPixmap.fromImage(display_frame)
            self.update_frame.emit((display_pm))
