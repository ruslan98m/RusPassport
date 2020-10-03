from PyQt5 import QtCore, QtGui, QtWidgets
from interface import Ui_MainWindow, DisplayImageWidget
import sys

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    interface = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(interface)
    interface.show()
    sys.exit(app.exec_())

    #app = QtWidgets.QApplication(sys.argv)
    #display_image_widget = DisplayImageWidget()
    #display_image_widget.show()
    #sys.exit(app.exec_())