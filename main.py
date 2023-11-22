import os

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QLabel
from PyQt5.QtGui import QPixmap, QImage
from ultralytics import YOLO
from PIL import Image, ImageQt
import cv2
import shutil

import gui



class MainWindow(QMainWindow, gui.Ui_MainWindow):
    def __init__(self): 
        super().__init__()
        self.setupUi(self)
        self.setFixedSize(self.size())

        self.pushButton_2.clicked.connect(self.choose_image)
        self.pushButton.clicked.connect(self.detect_and_show)
        self.textBox.textChanged.connect(self.img_list_change)

        self.model = YOLO("weights.pt")
        self.img = None
        self.HOME = os.getcwd()

    def choose_image(self):
        try:
            # self.img = Image.open(QFileDialog.getOpenFileNames(self, 'Open File', self.HOME, 'Images (*.png *.jpg *.bmp)')[0])
            self.img_list = QFileDialog.getOpenFileNames(self, 'Open File', self.HOME, 'Images (*.png *.jpg *.bmp)')[0]
            self.textBox.setText(';'.join(self.img_list))
        finally:
            pass

    def img_list_change(self):
        self.img_list = self.textBox.text().split(";")

    def detect_and_show(self):
        layout = QtWidgets.QGridLayout()
        img_labels = []
        try:
            for img in self.img_list:
                image = Image.open(img)
                results = self.model.predict(source=image)
                Image.fromarray(results[0].plot()).save('temp_image.png')
                if self.checkBox.isChecked():
                    shutil.copyfile('temp_image.png', "results/" + ''.join(img.split('/')[-1]) + "_DETECTED.png")
                image = QPixmap('temp_image.png')

                img_labels.append(QtWidgets.QLabel())
                img_labels[-1].setPixmap(image)
                layout.addWidget(img_labels[-1], img_labels.index(img_labels[-1]), 0)
            scroll_widget = QtWidgets.QWidget()
            scroll_widget.setLayout(layout)
            self.scrollArea.setWidget(scroll_widget)
            try:
                os.remove('temp_image.png')
            except OSError:
                pass
        except:
            try:
                os.remove('temp_image.png')
            except OSError:
                pass



if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec()