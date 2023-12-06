import os

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import QFileInfo, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QLabel
from PyQt5.QtGui import QPixmap, QImage
from ultralytics import YOLO
from PIL import Image, ImageQt
import cv2
import supervision as sv
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

        self.video = QVideoWidget()
        self.video.resize(640, 640)
        self.video.move(0, 0)
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.video)

        self.model = YOLO("weights.pt")
        self.img = None
        self.HOME = os.getcwd()

    def choose_image(self):
        try:
            self.img_list = QFileDialog.getOpenFileNames(self, 'Open File', self.HOME, 'Images and Videos (*.png *.jpg *.bmp *.mp4 *.avi)')[0]
            self.textBox.setText(';'.join(self.img_list))
        finally:
            pass

    def img_list_change(self):
        self.img_list = self.textBox.text().split(";")

    def process_frame(frame: np.ndarray) -> np.ndarray:
        print(20)
        model = YOLO("weights.pt")
        results = model(frame, imgsz=640)[0]
        print(2)
        detections = sv.Detections.from_yolov8(results)
        print(3)
        box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
        print(4)
        labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]
        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        print(1)

        return frame

    def detect_and_show(self):
        def process_frame(frame: np.ndarray) -> np.ndarray:
            print(20)
            model = YOLO("weights.pt")
            results = model(frame, imgsz=640)[0]
            print(2)
            detections = sv.Detections.from_yolov8(results)
            print(3)
            box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
            print(4)
            labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]
            frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
            print(1)
            return frame
        layout = QtWidgets.QGridLayout()
        img_labels = []
        try:
            for img in self.img_list:
                if img[len(img)-3:len(img)] in ["avi", "mp4"]:
                    path = os.path.dirname(os.path.abspath(__file__))
                    results = self.model(img, save=True, project=path, name='results', exist_ok=True)
                    if self.checkBox.isChecked():
                        shutil.copyfile('results/' + ''.join(img.split('/')[-1]), "results/" + ''.join(img.split('/')[-1]) + "_DETECTED.avi")
                    self.player.setMedia(QMediaContent(QUrl.fromLocalFile('results/' + ''.join(img.split('/')[-1]))))
                    self.player.setPosition(0)
                    self.video.show()
                    self.player.play()
                else:
                    image = Image.open(img)
                    results = self.model.predict(source=image)
                    Image.fromarray(results[0].plot()).save('temp_image.png')
                    if self.checkBox.isChecked():
                        shutil.copyfile('temp_image.png', "results/" + ''.join(img.split('/')[-1]) + "_DETECTED.png")
                    image = QPixmap('temp_image.png')
                    img_labels.append(QtWidgets.QLabel())
                    img_labels[-1].setPixmap(image)
                layout.addWidget(img_labels[-1], img_labels.index(img_labels[-1]), 0)
            print("bbbb")
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