# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ts_guess.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(600, 800)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.product_reset = QtGui.QPushButton(self.centralwidget)
        self.product_reset.setObjectName(_fromUtf8("product_reset"))
        self.gridLayout.addWidget(self.product_reset, 3, 0, 1, 1)
        self.tc_progressBar = QtGui.QProgressBar(self.centralwidget)
        self.tc_progressBar.setProperty("value", 24)
        self.tc_progressBar.setObjectName(_fromUtf8("tc_progressBar"))
        self.gridLayout.addWidget(self.tc_progressBar, 7, 0, 1, 3)
        self.reactant = QtGui.QPushButton(self.centralwidget)
        self.reactant.setObjectName(_fromUtf8("reactant"))
        self.gridLayout.addWidget(self.reactant, 0, 0, 1, 1)
        self.ratio_value = QtGui.QLineEdit(self.centralwidget)
        self.ratio_value.setFrame(False)
        self.ratio_value.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.ratio_value.setObjectName(_fromUtf8("ratio_value"))
        self.gridLayout.addWidget(self.ratio_value, 4, 2, 1, 1)
        self.product_text = QtGui.QTextEdit(self.centralwidget)
        self.product_text.setObjectName(_fromUtf8("product_text"))
        self.gridLayout.addWidget(self.product_text, 2, 1, 2, 3)
        self.reactant_reset = QtGui.QPushButton(self.centralwidget)
        self.reactant_reset.setObjectName(_fromUtf8("reactant_reset"))
        self.gridLayout.addWidget(self.reactant_reset, 1, 0, 1, 1)
        self.product = QtGui.QPushButton(self.centralwidget)
        self.product.setObjectName(_fromUtf8("product"))
        self.gridLayout.addWidget(self.product, 2, 0, 1, 1)
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 4, 1, 1, 1)
        self.ratio_slide_bar = QtGui.QSlider(self.centralwidget)
        self.ratio_slide_bar.setProperty("value", 50)
        self.ratio_slide_bar.setOrientation(QtCore.Qt.Horizontal)
        self.ratio_slide_bar.setObjectName(_fromUtf8("ratio_slide_bar"))
        self.gridLayout.addWidget(self.ratio_slide_bar, 4, 3, 1, 1)
        self.auto_key_ic = QtGui.QCheckBox(self.centralwidget)
        self.auto_key_ic.setObjectName(_fromUtf8("auto_key_ic"))
        self.gridLayout.addWidget(self.auto_key_ic, 6, 0, 1, 1)
        self.reactant_text = QtGui.QTextEdit(self.centralwidget)
        self.reactant_text.setObjectName(_fromUtf8("reactant_text"))
        self.gridLayout.addWidget(self.reactant_text, 0, 1, 2, 3)
        self.ts_guess = QtGui.QPushButton(self.centralwidget)
        self.ts_guess.setObjectName(_fromUtf8("ts_guess"))
        self.gridLayout.addWidget(self.ts_guess, 7, 3, 1, 1)
        self.auto_ic_select = QtGui.QCheckBox(self.centralwidget)
        self.auto_ic_select.setObjectName(_fromUtf8("auto_ic_select"))
        self.gridLayout.addWidget(self.auto_ic_select, 5, 0, 1, 1)
        self.key_ic_text = QtGui.QTextEdit(self.centralwidget)
        self.key_ic_text.setObjectName(_fromUtf8("key_ic_text"))
        self.gridLayout.addWidget(self.key_ic_text, 5, 1, 2, 3)
        self.save_saddle = QtGui.QPushButton(self.centralwidget)
        self.save_saddle.setObjectName(_fromUtf8("save_saddle"))
        self.gridLayout.addWidget(self.save_saddle, 8, 3, 1, 1)
        self.save_xyz = QtGui.QPushButton(self.centralwidget)
        self.save_xyz.setObjectName(_fromUtf8("save_xyz"))
        self.gridLayout.addWidget(self.save_xyz, 8, 2, 1, 1)
        self.reactant.raise_()
        self.reactant_text.raise_()
        self.reactant_reset.raise_()
        self.product_text.raise_()
        self.product.raise_()
        self.product_reset.raise_()
        self.auto_ic_select.raise_()
        self.label.raise_()
        self.ts_guess.raise_()
        self.tc_progressBar.raise_()
        self.save_saddle.raise_()
        self.ratio_slide_bar.raise_()
        self.ratio_value.raise_()
        self.auto_key_ic.raise_()
        self.save_xyz.raise_()
        self.key_ic_text.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 600, 22))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.product_reset.setText(_translate("MainWindow", "Reset", None))
        self.reactant.setText(_translate("MainWindow", "Reactant", None))
        self.ratio_value.setText(_translate("MainWindow", "0.50", None))
        self.reactant_reset.setText(_translate("MainWindow", "Reset", None))
        self.product.setText(_translate("MainWindow", "Product", None))
        self.label.setText(_translate("MainWindow", "Initial Guess Ratio", None))
        self.auto_key_ic.setText(_translate("MainWindow", "Auto Key IC Select", None))
        self.ts_guess.setText(_translate("MainWindow", "Get TS Guess", None))
        self.auto_ic_select.setText(_translate("MainWindow", "Auto IC Select", None))
        self.save_saddle.setText(_translate("MainWindow", "Save as .saddle", None))
        self.save_xyz.setText(_translate("MainWindow", "Save as .xyz", None))

