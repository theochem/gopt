import sys
import horton as ht
# import os

from saddle import TransitionSearch
from PyQt4 import QtGui, QtCore
from gui_ts_guess import Ui_MainWindow
from subprocess import Popen
from key_table_view import KeyIcTable


class Window(QtGui.QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.reactant_path = None
        self.product_path = None
        self.lable_ratio_value = 50
        self.ui.reactant.clicked.connect(self.reactant_open)
        self.ui.reactant_text.setReadOnly(True)
        self.ui.product.clicked.connect(self.product_open)
        self.ui.ts_guess.clicked.connect(self.get_ts_guess)
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
        self.ui.tc_progressBar.setValue(0)
        self.ui.save_xyz.clicked.connect(self.save_xyz)
        self.ui.actionSaddle.triggered.connect(self.about)
        self.ts_mol = None
        self.ui.view_vmd.clicked.connect(self.view_vmd)
        self.output = None
        self.ui.select_key_ic.clicked.connect(self.open_select_key_ci)

    def reactant_open(self):
        name = QtGui.QFileDialog.getOpenFileName(self,"Open file")
        if name:
            self.reactant_path = name
            with open(name, "r") as file:
                self.ui.reactant_text.setText(file.read())
        self.ui.tc_progressBar.setValue(0)


    def product_open(self):
        name = QtGui.QFileDialog.getOpenFileName(self,"Open file")
        if name:
            self.product_path = name
            with open(name, "r") as file:
                self.ui.product_text.setText(file.read())
        self.ui.tc_progressBar.setValue(0)


    def reactant_reset(self):
        self.ui.reactant_text.clear()
        self.reactant_path = None
        self.ui.tc_progressBar.setValue(0)
        self.output = None
        # choice = QtGui.QMessageBox.information(self, "Extrac!", "Really", QtGui.QMessageBox.Ok)

    def product_reset(self):
        self.ui.product_text.clear()
        self.product_path = None
        self.ui.tc_progressBar.setValue(0)
        self.output = None

    def change_ratio(self):
        self.lable_ratio_value = self.ui.ratio_slide_bar.value()
        # print self.lable_ratio_value
        self.ui.ratio_value.setText("{0}".format(self.lable_ratio_value / 100.)) 

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
        self.ui.key_ic_text.setEnabled(not self.auto_key_ic)
        # self.ui.select_key_ic.setEnabled(not self.auto_key_ic)

    def get_ts_guess(self):
        if self.reactant_path and self.product_path:
            react_mol = ht.IOData.from_file(str(self.reactant_path))
            produ_mol = ht.IOData.from_file(str(self.product_path))
            self.ui.tc_progressBar.setValue(20)
            ts_search = TransitionSearch.TransitionSearch(react_mol, produ_mol)
            self.return_ratio()
            ratio = self.lable_ratio_value / 100.
            ts_search.auto_ts_search()
            self.ui.tc_progressBar.setValue(50)
            if self.auto_ic_select == True:
                ts_search.auto_ic_select_combine()
                if self.auto_key_ic == True:
                    ts_search.auto_key_ic_select()
            self.ui.tc_progressBar.setValue(100)
            self.ts_mol = ts_search # change from ts_search.ts_state to just ts_search
            success = QtGui.QMessageBox.information(self, "Finished", "\n\nFinished!", QtGui.QMessageBox.Ok)
        else:
            fail = QtGui.QMessageBox.warning(self, "Can't do that", '''Can't do that
                \nPlease select reactant and product structure first''', QtGui.QMessageBox.Ok)

    def save_xyz(self):
        name = QtGui.QFileDialog.getSaveFileName(self, "Save .xyz")
        if name:
            name = name + ".xyz"
            with open(name, "w") as f:
                print >> f, len(self.ts_mol.ts_state.numbers)
                print >> f, getattr(self.ts_mol.ts_state, "title", name.split("/")[-1].split(".")[0])
                for i in range(len(self.ts_mol.ts_state.numbers)):
                    n = ht.periodic[self.ts_mol.ts_state.numbers[i]].symbol
                    x, y, z = self.ts_mol.ts_state.coordinates[i]/ht.angstrom
                    print >> f, '%2s %15.10f %15.10f %15.10f' % (n, x, y, z)
            self.output = name
        # print self.auto_key_ic

    def view_vmd(self):
        file_name = "{}".format(self.output)
        arg = ["vmd",file_name]
        a = Popen(arg)
        print a
        # os.system("vmd {0}".format(self.output))

    def open_select_key_ci(self):
        if self.ts_mol:
            molecule = self.ts_mol
            table_gui = KeyIcTable(molecule)
            table_gui.setWindowTitle("Key IC Selection --by Derrick")
            table_gui.exec_()
            # print self.ts_mol._ic_key_counter
        else:
            fail = QtGui.QMessageBox.warning(self, "Can't do that", '''Can't do that
                \nPlease generate transition guess structure first''', QtGui.QMessageBox.Ok)

    def about(self):
        popup = QtGui.QMessageBox.about(self, "About Saddle", '''<font size="6"><p align="center">Saddle</p></font>
            \n<font size="3"><p align="center">Copyright 2016 Horton Group</p></font>\n
            <font size="2"><p align="center">version 1.0 by Derrick</p></font>''')


app = QtGui.QApplication(sys.argv)
gui = Window()
gui.setWindowTitle("Saddle --by Derrick")
gui.show()
sys.exit(app.exec_())
