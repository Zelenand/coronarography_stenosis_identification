from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from PyQt5.QtGui import QPixmap
import gui



class MainWindow(QMainWindow, gui.Ui_MainWindow):
    def __init__(self): 
        super().__init__()
        self.setupUi(self)
        self.setFixedSize(self.size())

        self.pushButton_2.clicked.connect(self.choose_image)

    def choose_image(self):
        pixmap = QPixmap(QFileDialog.getOpenFileName(self, 'Open File', 'C:/', 'PNG File (*.png);;JPG File *.jpg)')[0])

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec()