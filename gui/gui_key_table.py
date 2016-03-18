# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui_key_table.ui'
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

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName(_fromUtf8("Dialog"))
        Dialog.resize(451, 536)
        self.verticalLayout = QtGui.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.keyic_table_widget = QtGui.QTableWidget(Dialog)
        self.keyic_table_widget.setEnabled(True)
        self.keyic_table_widget.setAlternatingRowColors(False)
        self.keyic_table_widget.setRowCount(0)
        self.keyic_table_widget.setObjectName(_fromUtf8("keyic_table_widget"))
        self.keyic_table_widget.setColumnCount(0)
        self.verticalLayout.addWidget(self.keyic_table_widget)
        self.return_button = QtGui.QDialogButtonBox(Dialog)
        self.return_button.setOrientation(QtCore.Qt.Horizontal)
        self.return_button.setStandardButtons(QtGui.QDialogButtonBox.Cancel|QtGui.QDialogButtonBox.Ok)
        self.return_button.setObjectName(_fromUtf8("return_button"))
        self.verticalLayout.addWidget(self.return_button)

        self.retranslateUi(Dialog)
        QtCore.QObject.connect(self.return_button, QtCore.SIGNAL(_fromUtf8("accepted()")), Dialog.accept)
        QtCore.QObject.connect(self.return_button, QtCore.SIGNAL(_fromUtf8("rejected()")), Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(_translate("Dialog", "Dialog", None))

