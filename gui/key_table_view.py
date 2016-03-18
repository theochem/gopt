import sys
# import os

from PyQt4 import QtGui, QtCore
from gui_key_table import Ui_Dialog
from horton import periodic

# from gui_design import Window

class KeyIcTable(QtGui.QDialog):

    def __init__(self, mol):
        super(KeyIcTable, self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.mol = mol
        self.ic_info = mol.ts_state.procedures
        self.atom_info = mol.ts_state.numbers
        self.ui.keyic_table_widget.setColumnCount(5)
        self.ui.keyic_table_widget.setRowCount(len(self.ic_info))
        # print self.ui.keyic_table_widget.rowCount()
        # print len(ic_info)+1
        self.ui.keyic_table_widget.setHorizontalHeaderLabels(['IC Type', 'Atom1', 'Atom2', 'Atom3', 'Atom4'])
        self.ui.keyic_table_widget.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
        for row in range(len(self.ic_info)):
            info = self.ic_info[row]
            if info[0] == 'add_bond_length':
                content = QtGui.QTableWidgetItem('Bond')
            elif info[0] == "add_bend_angle":
                content = QtGui.QTableWidgetItem('Angle')
            elif info[0] == 'add_dihed_angle':
                content = QtGui.QTableWidgetItem('Normal Dihedral')
            elif info[0] == 'add_dihed_angle_new_dot':
                content = QtGui.QTableWidgetItem('New Dot Dihedral')
            elif info[0] == 'add_dihed_angle_new_cross':
                content = QtGui.QTableWidgetItem('New Cross Dihedral')

            self.ui.keyic_table_widget.setItem(row, 0, content)
            content.setFlags(QtCore.Qt.ItemIsUserCheckable |  QtCore.Qt.ItemIsEnabled)
            content.setCheckState(QtCore.Qt.Unchecked)
            atoms = info[1]
            for i in range(len(atoms)):
                sym = periodic[self.atom_info[atoms[i]]].symbol
                index = atoms[i]
                item = QtGui.QTableWidgetItem('{}  ({})'.format(index, sym))
                item.setFlags(QtCore.Qt.ItemIsEnabled)
                self.ui.keyic_table_widget.setItem(row, i+1, item)
            for i in range(len(atoms), 5):
                item = QtGui.QTableWidgetItem("")
                item.setFlags(QtCore.Qt.ItemIsSelectable)
                self.ui.keyic_table_widget.setItem(row, i+1, item)

        self.ui.keyic_table_widget.itemClicked.connect(self.item_check)
        self.ui.return_button.accepted.connect(self.accept_key_ic)
        self.ui.return_button.rejected.connect(self.reject_key_ic)
        if self.mol._ic_key_counter > 0:
            for i in range(self.mol._ic_key_counter):
                self.ui.keyic_table_widget.item(i, 0).setCheckState(QtCore.Qt.Checked)
                for j in range(5):
                    if self.ui.keyic_table_widget.item(i, j).text():
                        self.ui.keyic_table_widget.item(i, j).setBackgroundColor(QtGui.QColor(200, 200, 200))

    def item_check(self, item):
        if item.checkState() == QtCore.Qt.Checked:
            row = item.row()
            for i in range(5):
                if self.ui.keyic_table_widget.item(row, i).text():
                    self.ui.keyic_table_widget.item(row, i).setBackgroundColor(QtGui.QColor(200, 200, 200))
        elif item.checkState() != QtCore.Qt.Checked:
            row = item.row()
            for i in range(5):
                if self.ui.keyic_table_widget.item(row, i).text():
                    self.ui.keyic_table_widget.item(row, i).setBackgroundColor(QtGui.QColor(255, 255, 255))

    def accept_key_ic(self):
        key_ic = []
        self.mol._ic_key_counter = 0
        rows = self.ui.keyic_table_widget.rowCount()
        for i in range(rows):
            if self.ui.keyic_table_widget.item(i, 0).checkState() == QtCore.Qt.Checked:
                key_ic.append(i)
        # print key_ic
        self.mol._arrange_key_ic(key_ic)
        # print self.mol._ic_key_counter

    def reject_key_ic(self):
        pass

# if __name__ == '__main__':
#     app = QtGui.QApplication(sys.argv)
#     # gui = KeyIcTable(data, atom_info)
#     gui.setWindowTitle("Key IC Selection --by Derrick")
#     gui.show()
#     sys.exit(app.exec_())
