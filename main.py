import os
import shutil
import sys

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

    def img_list_change(self):
        self.img_list = self.textBox.text().split(";")


    def detect_and_show(self):
        layout = QtWidgets.QGridLayout()
        img_labels = []
        try:
            for img in self.img_list:
                self.textBrowser.append("* Processing " + img + " *")
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
    sys.stdout = sys.__stdout__