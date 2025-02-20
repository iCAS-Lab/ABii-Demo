"""_summary_
https://www.pythonguis.com/tutorials/pyside6-plotting-matplotlib/
"""

import time
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap, QKeyEvent
from PySide6.QtWidgets import QMainWindow, QLabel
from generated_files.MainWindow import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        # Init GUI
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.prev_press_time = time.time()
        self.press_threshold = 3.0

        # Connect signals to slots
        self.connect_signals()

        # Temp pixmap
        self.pixmap = QPixmap()

        # Scale contents to GUI, set False to preserve images quality
        self.ui.detectVideo.setScaledContents(True)

        self.showFullScreen()

    def connect_signals(self):
        self.ui.detectVideo.update_frame.connect(self.update_cameras)
        self.ui.detectVideo.update_stats.connect(self.update_tab)

    def keyPressEvent(self, event: QKeyEvent):
        now = time.time()
        if now - self.prev_press_time > self.press_threshold:
            self.prev_press_time = now
            if event.key() == Qt.Key.Key_Space:
                print('space pressed')
            elif event.key() == Qt.Key.Key_Left:
                print('left pressed')
            elif event.key() == Qt.Key.Key_Right:
                print('right pressed')
            elif event.key() == Qt.Key.Key_Up:
                print('up pressed')
            elif event.key() == Qt.Key.Key_Down:
                print('down pressed')
            elif event.key() == Qt.Key.Key_P:
                print('party activated')
            elif event.key() == Qt.Key.Key_I:
                print('introduction activated')
            else:
                print('invalid key pressed')
        else:
            print('key pressed too soon')

    def update_num_heads_detected(self, num_faces: int):
        pass

    def next_face(self):
        pass

    def previous_face(self):
        pass

    def reset_face(self):
        """This function should reset the focus to the closest face i.e.
        the face which has the bounding box with the highest area (w*h).
        """
        pass

    def camera_resolution_changed(self):
        pass

    def tab_changed(self):
        pass

    def update_tab(self, data: dict):
        pass

    def update_stats(self, stats: dict):
        pass

    def update_plots(self, plot_data: dict):
        pass

    def update_cameras(self, pixmaps: tuple):
        # print(f'Received Frame: {time.time()}')
        detect_pm = pixmaps
        self.pixmap = detect_pm
        detectVideoSize = self.ui.detectVideo.size()
        self.ui.detectVideo.setPixmap(detect_pm.scaled(
            detectVideoSize, Qt.AspectRatioMode.KeepAspectRatio))
