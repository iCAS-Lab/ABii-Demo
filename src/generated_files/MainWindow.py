# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainWindow.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QGridLayout, QLabel,
    QMainWindow, QSizePolicy, QTabWidget, QVBoxLayout,
    QWidget)

from camera import Camera

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.setWindowModality(Qt.WindowModality.NonModal)
        MainWindow.resize(1102, 785)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setStyleSheet(u"background-color: rgb(92, 92, 92);")
        MainWindow.setTabShape(QTabWidget.TabShape.Rounded)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, -1)
        self.detect = QFrame(self.centralwidget)
        self.detect.setObjectName(u"detect")
        sizePolicy.setHeightForWidth(self.detect.sizePolicy().hasHeightForWidth())
        self.detect.setSizePolicy(sizePolicy)
        self.verticalLayout_8 = QVBoxLayout(self.detect)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.detectLabel = QLabel(self.detect)
        self.detectLabel.setObjectName(u"detectLabel")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.detectLabel.sizePolicy().hasHeightForWidth())
        self.detectLabel.setSizePolicy(sizePolicy1)
        self.detectLabel.setMaximumSize(QSize(16777215, 64))
        self.detectLabel.setStyleSheet(u"background-color: rgb(115, 0, 10);")
        self.detectLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_8.addWidget(self.detectLabel)

        self.label = QLabel(self.detect)
        self.label.setObjectName(u"label")

        self.verticalLayout_8.addWidget(self.label)

        self.detectVideo = Camera(self.detect)
        self.detectVideo.setObjectName(u"detectVideo")
        sizePolicy.setHeightForWidth(self.detectVideo.sizePolicy().hasHeightForWidth())
        self.detectVideo.setSizePolicy(sizePolicy)
        self.detectVideo.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.verticalLayout_8.addWidget(self.detectVideo)


        self.gridLayout.addWidget(self.detect, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"ABii Demo", None))
        self.detectLabel.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\" font-size:24pt; font-weight:700; color:#ffffff;\">ABii the Smart Robot Tutor</span></p></body></html>", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p align=\"center\"><span style=\" font-size:18pt; font-weight:700;\">A Collaboration Between:</span></p><p align=\"center\"><span style=\" font-size:18pt; font-weight:700;\">Intelligent Circuits, Architectures, and Systems (iCAS) Lab &amp; VAN Robotics</span></p></body></html>", None))
        self.detectVideo.setText("")
    # retranslateUi

