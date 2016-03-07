import sys
from PyQt4 import QtGui, QtCore
from gui_ts_guess import Ui_MainWindow

class Window(QtGui.QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.ui =  Ui_MainWindow()
        self.ui.setupUi(self)


app = QtGui.QApplication(sys.argv)
gui = Window()
gui.show()
sys.exit(app.exec_())
