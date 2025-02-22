import sys

from PySide6.QtWidgets import QApplication
import PySide6.QtGui
from gui import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    screen_size = app.primaryScreen().size().toTuple()
    print(screen_size)
    main_window = MainWindow()
    main_window.resize(screen_size[0], screen_size[1])
    sys.exit(app.exec())
