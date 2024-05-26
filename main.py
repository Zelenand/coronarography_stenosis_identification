import os
import shutil
import sys

import cv2
import numpy as np
from scipy.spatial import distance
from PIL import Image
from PyQt5 import QtWidgets
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtCore import QObject
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtCore import pyqtSignal as Signal
from PyQt5.QtGui import QPixmap, QPalette
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QMainWindow, QWidget
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QLabel, \
    QSlider, QStyle, QSizePolicy, QFileDialog
from ultralytics import YOLO
import pathlib

import gui


class OutputLogger(QObject):
    emit_write = Signal(str, int)

    class Severity:
        DEBUG = 0
        ERROR = 1

    def __init__(self, io_stream, severity):
        super().__init__()

        self.io_stream = io_stream
        self.severity = severity

    def write(self, text):
        self.io_stream.write(text)
        self.emit_write.emit(text, self.severity)

    def flush(self):
        self.io_stream.flush()


OUTPUT_LOGGER_STDOUT = OutputLogger(sys.stdout, OutputLogger.Severity.DEBUG)
OUTPUT_LOGGER_STDERR = OutputLogger(sys.stderr, OutputLogger.Severity.ERROR)

sys.stdout = OUTPUT_LOGGER_STDOUT
sys.stderr = OUTPUT_LOGGER_STDERR


class Player(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PyQt5 Media Player")
        self.setGeometry(350, 100, 700, 500)

        p = self.palette()
        p.setColor(QPalette.Window, Qt.black)
        self.setPalette(p)

        self.init_ui()

        self.show()

    def init_ui(self):

        # create media player object
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        # create videowidget object

        videowidget = QVideoWidget()

        # create button for playing
        self.playBtn = QPushButton()
        self.playBtn.setEnabled(False)
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playBtn.clicked.connect(self.play_video)

        # create slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_position)

        # create label
        self.label = QLabel()
        self.label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # create hbox layout
        hboxLayout = QHBoxLayout()
        hboxLayout.setContentsMargins(0, 0, 0, 0)

        # set widgets to the hbox layout
        hboxLayout.addWidget(self.playBtn)
        hboxLayout.addWidget(self.slider)

        # create vbox layout
        vboxLayout = QVBoxLayout()
        vboxLayout.addWidget(videowidget)
        vboxLayout.addLayout(hboxLayout)
        vboxLayout.addWidget(self.label)

        self.setLayout(vboxLayout)

        self.mediaPlayer.setVideoOutput(videowidget)

        # media player signals

        self.mediaPlayer.stateChanged.connect(self.mediastate_changed)
        self.mediaPlayer.positionChanged.connect(self.position_changed)
        self.mediaPlayer.durationChanged.connect(self.duration_changed)

    def play_video(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()

        else:
            self.mediaPlayer.play()

    def mediastate_changed(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playBtn.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause)

            )

        else:
            self.playBtn.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay)

            )

    def position_changed(self, position):
        self.slider.setValue(position)

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)

    def set_position(self, position):
        self.mediaPlayer.setPosition(position)

    def handle_errors(self):
        self.playBtn.setEnabled(False)
        self.label.setText("Error: " + self.mediaPlayer.errorString())


class MainWindow(QMainWindow, gui.Ui_MainWindow):
    def __init__(self): 
        super().__init__()
        self.setupUi(self)
        self.setFixedSize(self.size())

        self.pushButton_2.clicked.connect(self.choose_image)
        self.pushButton_3.clicked.connect(self.choose_dir)
        self.pushButton.clicked.connect(self.detect_and_show)
        self.textBox.textChanged.connect(self.img_list_change)

        self.model = YOLO("weights.pt")
        self.img = None
        self.HOME = os.getcwd()

        OUTPUT_LOGGER_STDOUT.emit_write.connect(self.onUpdateText)
        OUTPUT_LOGGER_STDERR.emit_write.connect(self.onUpdateText)

        self.player = Player()
        self.player.hide()
        print("Start")

    def onUpdateText(self, text, severity):
        """Write console output to text widget."""
        self.textBrowser.append(text)

    def choose_image(self):
        try:
            self.img_list = QFileDialog.getOpenFileNames(self, 'Open File', self.HOME, 'Images and Videos (*.png *.jpg *.bmp *.mp4 *.avi)')[0]
            self.textBox.setText(';'.join(self.img_list))
        finally:
            pass

    def choose_dir(self):
        try:
            exts = [".png", ".jpg", ".bnp", ".mp4", ".avi"]
            #result = list(pathlib.Path(QFileDialog.getExistingDirectory(self, "Выбрать папку", ".")).rglob('*'))
            result = [str(p).replace("\\", "/") for p in pathlib.Path(QFileDialog.getExistingDirectory(self,"Выбрать папку",".")).rglob('*') if p.suffix in exts]
            res = result[0]
            self.img_list = result
            self.textBox.setText(';'.join(self.img_list))
        finally:
            pass

    def img_list_change(self):
        self.img_list = self.textBox.text().split(";")

    def calc_stenosis_degree(self, image):
        detected_box = cv2.medianBlur(image, 9)
        ret2, th2 = cv2.threshold(detected_box, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = 128
        th2 = cv2.threshold(th2, thresh, 255, cv2.THRESH_BINARY)[1]
        th3 = cv2.Canny(th2, 50, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (33, 33))
        closed = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        image_countoured = image.copy()
        cv2.drawContours(image_countoured, [contour], -1, (0, 255, 0), 2)
        closed_copy = closed.copy()
        skel = np.zeros(closed_copy.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        while True:
            open = cv2.morphologyEx(closed_copy, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(closed_copy, open)
            eroded = cv2.erode(closed_copy, element)
            skel = cv2.bitwise_or(skel, temp)
            closed_copy = eroded.copy()

            if cv2.countNonZero(closed_copy) == 0:
                break

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        skel = cv2.morphologyEx(skel, cv2.MORPH_CLOSE, kernel)
        skel_contours, _ = cv2.findContours(skel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        skel_contour = max(skel_contours, key=cv2.contourArea)

        x, y = image.shape
        skel_points = [i[0] for i in skel_contour]
        contour_points = [i[0] for i in contour]
        # Найдем ближайшие точки контура для каждой точки скелета
        closest_points = []
        for skel_point in skel_points:
            dist = [distance.euclidean(skel_point, p) for p in contour_points]
            closest_point = contour_points[np.argmin(dist)]
            closest_points.append(closest_point)

        # Вычислим расстояния между точками скелета и их ближайшими точками на контуре
        distances = [distance.euclidean(skel_points[i], closest_points[i]) for i in range(len(skel_points))]

        narrowest_part = min(distances)
        widest_part = max(distances)

        return (100 - (narrowest_part / widest_part * 100)) // 0.1 / 10

    def detect_and_show(self):
        layout = QtWidgets.QGridLayout()
        img_labels = []
        try:
            for img in self.img_list:
                self.textBrowser.append(img)
                QCoreApplication.processEvents()
                if img[len(img)-3:len(img)] in ["avi", "mp4"]:
                    path = os.path.dirname(os.path.abspath(__file__))
                    results = self.model(img, save=True, project=path, name='results', exist_ok=True)
                    print("Video processed")
                    if self.checkBox.isChecked():
                        shutil.copyfile('results/' + ''.join(img.split('/')[-1]), "results/" + ''.join(img.split('/')[-1]) + "_DETECTED.avi")
                        print("Video saved as", 'results/' + ''.join(img.split('/')[-1]), "results/" + ''.join(img.split('/')[-1]) + "_DETECTED.avi")

                    self.player.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile('results/' + ''.join(img.split('/')[-1]))))
                    self.player.mediaPlayer.setPosition(0)
                    self.player.playBtn.setEnabled(True)
                    self.player.show()
                    os.remove('results/' + ''.join(img.split('/')[-1]))

                else:
                    image = Image.open(img)
                    results = self.model.predict(source=image)
                    image = Image.open(img).convert('L')
                    if len(results):
                        detected_box = image.crop((int(results[0].boxes.xyxy[0][0]) - 15,
                                                   int(results[0].boxes.xyxy[0][1]) - 15,
                                                   int(results[0].boxes.xyxy[0][2]) + 15,
                                                   int(results[0].boxes.xyxy[0][3]) + 15))
                        detected_box = np.array(detected_box.convert('RGB'))[:, :, ::-1].copy()
                        detected_box = cv2.cvtColor(detected_box, cv2.COLOR_BGR2GRAY)
                        detected_box = cv2.resize(detected_box, (100, 100))
                        detected_box = cv2.medianBlur(detected_box, 9)
                        stenosis_degree = self.calc_stenosis_degree(detected_box)
                        print("Степень выраженности стеноза: " + str(stenosis_degree) + "%")
                    Image.fromarray(results[0].plot()).save('temp_image.png')
                    if self.checkBox.isChecked():
                        print("results/" + '/'.join(img.split('/')[1:]).replace(" ", "") + "_DETECTED.png")
                        dirname = "results/" + '/'.join(img.split('/')[1:][:-1])
                        print(dirname)
                        dirname = pathlib.Path(dirname)
                        dirname.mkdir(parents=True, exist_ok=True)
                        try:
                            shutil.copyfile('temp_image.png', "results/" + '/'.join(img.split('/')[1:]) + "_DETECTED.png")
                        except Exception as e:
                            print(e)
                        print("results/" + '/'.join(img.split('/')[1:]).replace(" ", "") + "_DETECTED.png")
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
    sys.stdout = sys.__stdout__