import os
import sys
dir_path = os.path.dirname(sys.argv[0])
os.chdir(dir_path)
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QComboBox, QMainWindow,QHBoxLayout,
                             QTableWidget,QTableWidgetItem,QTableView,QGridLayout,QFrame,QScrollArea,QVBoxLayout,QCheckBox)
from PyQt5.QtCore import Qt,QAbstractTableModel,QPoint
from PyQt5.QtGui import QPixmap,QPainter
import pandas as pd
import numpy as np
from main import time_window_hist


class Label(QLabel):
    def __init__(self, img):
        super(Label, self).__init__()
        self.setFrameStyle(QFrame.StyledPanel)
        self.pixmap = QPixmap(img)

    def paintEvent(self, event):
        size = self.size()
        painter = QPainter(self)
        point = QPoint(0,0)
        scaledPix = self.pixmap.scaled(size, Qt.KeepAspectRatio, transformMode = Qt.SmoothTransformation)
        # start painting the label from left upper corner
        point.setX((size.width() - scaledPix.width())/2)
        point.setY((size.height() - scaledPix.height())/2)
        painter.drawPixmap(point, scaledPix)

class CheckableComboBox(QComboBox):
    def __init__(self):
        super().__init__()
        self._changed = False

        self.view().pressed.connect(self.handleItemPressed)

    def setItemChecked(self, index, checked=False):
        item = self.model().item(index, self.modelColumn())  # QStandardItem object

        if checked:
            item.setCheckState(Qt.Checked)
        else:
            item.setCheckState(Qt.Unchecked)

    def handleItemPressed(self, index):
        item = self.model().itemFromIndex(index)

        if item.checkState() == Qt.Checked:
            item.setCheckState(Qt.Unchecked)
        else:
            item.setCheckState(Qt.Checked)
        self._changed = True

    def hidePopup(self):
        if not self._changed:
            super().hidePopup()
        self._changed = False

    def itemChecked(self, index):
        item = self.model().item(index, self.modelColumn())
        return item.checkState() == Qt.Checked

class Heatmap(QWidget):
    def __init__(self):
        super().__init__()
        # create window
        grid = QGridLayout()
        grid.setAlignment(Qt.AlignTop)
        grid.setRowStretch(1,6)
        #grid.setRowStretch(3, 3)
        self.setLayout(grid)
        mainLayout = QHBoxLayout(self)
        self.resize(1280,720)
        self.setWindowTitle('Heat maps')

        #get data
        self.files = os.listdir("output")
        # self.cust_legal_code = dict(zip(self.plots_df["customer_legal_name"], self.plots_df["customer_name"]))
        # self.plot_legal_code = dict(zip(self.plots_df["plot_logical_name"], self.plots_df["plot_code"]))
        # self.cust_code_legal = dict(zip(self.plots_df["customer_name"], self.plots_df["customer_legal_name"]))
        # self.plot_code_legal = dict(zip(self.plots_df["plot_code"], self.plots_df["plot_logical_name"]))

        #what to show dropdown
        self.drop_files = QComboBox(self)
        self.drop_files.addItems(self.files)
        self.drop_files.show()
        self.drop_files.setEnabled(True)
        grid.addWidget(self.drop_files, *(0,0,1,1))
        self.drop_files.activated.connect(self.click_files)
        #col base buttom
        self.drop_col_base = QComboBox(self)
        self.drop_col_base.setEnabled(False)
        self.drop_col_base.addItems(["col base"])
        grid.addWidget(self.drop_col_base, *(0,1,1,1))
        self.drop_col_base.activated.connect(self.click_col_base)
        #action_base  buttom
        self.drop_action_base = QComboBox(self)
        self.drop_action_base.setEnabled(False)
        grid.addWidget(self.drop_action_base, *(0,2,1,1))
        self.drop_action_base.addItems(["action base"])
        self.drop_action_base.activated.connect(self.click_action_base)
        #col count buttom
        self.drop_col_count = QComboBox(self)
        self.drop_col_count.setEnabled(False)
        grid.addWidget(self.drop_col_count, *(0,3,1,1))
        self.drop_col_count.addItems(["col count"])
        self.drop_col_count.activated.connect(self.click_col_count)
        #action count buttom
        self.drop_action_count = CheckableComboBox()
        self.drop_action_count.setEnabled(False)
        grid.addWidget(self.drop_action_count, *(0,4,1,1))
        self.drop_action_count.addItems(["action count"])
        self.drop_action_count.activated.connect(self.click_action_count)
        #show buttom
        self.show_but = QPushButton(self)
        self.show_but.setText('show')
        self.show_but.setEnabled(False)
        grid.addWidget(self.show_but, *(0, 5,1,1))
        self.show_but.clicked.connect(self.click_show)

        self.click_files()

    def show_buttoms(self,arr):
        self.drop_files.setEnabled(arr[0])
        self.drop_col_base.setEnabled(arr[1])
        self.drop_action_base.setEnabled(arr[2])
        self.drop_col_count.setEnabled(arr[3])
        self.drop_action_count.setEnabled(arr[4])
        self.show_but.setEnabled(arr[5])
    def click_files(self):
        file = str(self.drop_files.currentText())
        self.show_buttoms([True, True, False, False, False,False])
        self.drop_col_base.clear()
        self.drop_action_base.clear()
        self.drop_col_count.clear()
        self.drop_action_count.clear()
        self.drop_action_base.addItems(["action base"])
        self.drop_col_count.addItems(["col count"])
        self.drop_action_count.addItems(["action count"])
        path = os.path.join("output",file,f"{file} windows.csv")
        df = pd.read_csv(path)
        self.df = df
        self.windows = df
        col_base = self.windows["col base"]
        self.drop_col_base.addItems(col_base.unique())
        self.click_col_base()

    def click_col_base(self):
        col_base = str(self.drop_col_base.currentText())
        self.show_buttoms([True, True, True, False, False, False])
        self.drop_action_base.clear()
        self.drop_col_count.clear()
        self.drop_action_count.clear()
        self.drop_col_count.addItems(["col count"])
        self.drop_action_count.addItems(["action count"])
        self.windows = self.df
        self.windows = self.windows[self.windows["col base"] == col_base]
        action_base = self.windows["action base"]
        self.drop_action_base.addItems(action_base.unique())
        self.click_action_base()

    def click_action_base(self):
        action_base = str(self.drop_action_base.currentText())
        col_base = str(self.drop_col_base.currentText())
        self.show_buttoms([True, True, True, True, False, False])
        self.drop_col_count.clear()
        self.drop_action_count.clear()
        self.drop_action_count.addItems(["action count"])
        self.windows = self.df
        self.windows = self.windows[self.windows["col base"] == col_base]
        self.windows = self.windows[self.windows["action base"] == action_base]
        col_count = self.windows["col count"]
        self.drop_col_count.addItems(col_count.unique())
        self.click_col_count()

    def click_col_count(self):
        col_count = str(self.drop_col_count.currentText())
        action_base = str(self.drop_action_base.currentText())
        col_base = str(self.drop_col_base.currentText())
        self.show_buttoms([True, True, True, True, True, False])
        self.drop_action_count.clear()
        self.windows = self.df
        self.windows = self.windows[self.windows["col base"] == col_base]
        self.windows = self.windows[self.windows["action base"] == action_base]
        self.windows = self.windows[self.windows["col count"] == col_count]
        self.action_count = self.windows["action count"]
        self.drop_action_count.addItems(["all"])
        self.drop_action_count.addItems(self.action_count.unique())
        self.click_action_count()

    def click_action_count(self):
        self.show_buttoms([True, True, True, True, True, True])
        self.click_show()

    def click_show(self):
        col_base, action_base, col_count = str(self.drop_col_base.currentText()),str(self.drop_action_base.currentText()),str(self.drop_col_count.currentText())
        chked_index = [ind for ind in range((self.drop_action_count.count())) if self.drop_action_count.itemChecked(ind)]
        if len(chked_index) == 0:
            return
        if 0 in chked_index:
            action_count = self.action_count.unique()
        else:
            action_count = np.array(["all"] + list(self.action_count.unique()))[chked_index]
        file = str(self.drop_files.currentText())
        save_path = os.path.join("output",file,file)
        img_path = f"{save_path} {col_base} window hist.png"
        window_df = time_window_hist(self.df, col_base, action_base, col_count, action_count, save_path)
        label = Label(img_path)
        self.layout().addWidget(label, *(1, 0, 1, 6))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Heatmap()
    window.show()
    print("app loaded")
    sys.exit(app.exec_())


#pyinstaller --onefile gui_script.spec  ##in terminal



