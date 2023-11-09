import os

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QLabel
from PyQt5.QtGui import QPixmap, QImage
from ultralytics import YOLO
from PIL import Image, ImageQt
import cv2

import gui



class MainWindow(QMainWindow, gui.Ui_MainWindow):
    def __init__(self): 
        super().__init__()
        self.setupUi(self)
        self.setFixedSize(self.size())

        self.pushButton_2.clicked.connect(self.choose_image)
        self.pushButton.clicked.connect(self.detect_and_show)

        self.model = YOLO("weights.pt")
        self.img = None
        self.HOME = os.getcwd()

    def choose_image(self):
        try:
            self.img = Image.open(QFileDialog.getOpenFileName(self, 'Open File', self.HOME, 'Images (*.png *.jpg *.bmp)')[0])
        finally:
            pass

    def detect_and_show(self):
        if self.img:
            results = self.model.predict(source=self.img)
            Image.fromarray(results[0].plot()).save('image.png')
            image = QPixmap('image.png')
            self.im.setPixmap(image)
            self.scrollArea.setWidget(self.im)



if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec()