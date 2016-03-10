import sys
import horton as ht

from PyQt4 import QtGui, QtCore
from gui_ts_guess import Ui_MainWindow
from saddle import *


class Window(QtGui.QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.ui =  Ui_MainWindow()
        self.ui.setupUi(self)
        self.reactant_path = None
        self.product_path = None
        self.lable_ratio_value = 50
        self.ui.reactant.clicked.connect(self.reactant_open)
        self.ui.reactant_text.setReadOnly(True)
        self.ui.product.clicked.connect(self.product_open)
        self.ui.product_text.setReadOnly(True)
        self.ui.reactant_reset.clicked.connect(self.reactant_reset)
        self.ui.product_reset.clicked.connect(self.product_reset)
        self.ui.ratio_slide_bar.valueChanged.connect(self.change_ratio)
        self.ui.ratio_value.returnPressed.connect(self.return_ratio)
        self.ui.auto_ic_select.toggle()
        self.auto_ic_select = True
        self.auto_key_ic = False
        self.ui.auto_ic_select.stateChanged.connect(self.change_auto_ic)
        self.ui.auto_key_ic.stateChanged.connect(self.change_key_ic)


    def reactant_open(self):
        name = QtGui.QFileDialog.getOpenFileName(self,"Open file")
        if name:
            self.reactant_path = name
            with open(name, "r") as file:
                self.ui.reactant_text.setText(file.read())

    def product_open(self):
        name = QtGui.QFileDialog.getOpenFileName(self,"Open file")
        if name:
            self.product_path = name
            with open(name, "r") as file:
                self.ui.product_text.setText(file.read())

    def reactant_reset(self):
        self.ui.reactant_text.clear()
        self.reactant_path = None

    def product_reset(self):
        self.ui.product_text.clear()
        self.product_path = None

    def change_ratio(self):
        self.lable_ratio_value = self.ui.ratio_slide_bar.value()
        # print self.lable_ratio_value
        self.ui.ratio_value.setText("{}".format(self.lable_ratio_value / 100.)) 

    def return_ratio(self):
        text = self.ui.ratio_value.displayText()
        try:
            value = float(text)
        except ValueError:
            value = 0.50
            self.ui.ratio_value.setText("0.50")
        self.lable_ratio_value = value * 100
        value = min(max(self.lable_ratio_value, 0), 100.)
        self.ui.ratio_slide_bar.setValue(value)
        # print self.lable_ratio_value

    def change_auto_ic(self):
        self.auto_ic_select = not self.auto_ic_select
        # print self.auto_ic_select

    def change_key_ic(self):
        self.auto_key_ic = not self.auto_key_ic
        # print self.auto_key_ic

app = QtGui.QApplication(sys.argv)
gui = Window()
gui.show()
sys.exit(app.exec_())
