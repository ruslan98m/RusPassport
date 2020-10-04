from PyQt5 import QtCore, QtGui, QtWidgets
from pytesseract import pytesseract
import sys

from datavalidation.interface import Ui_MainWindow, DisplayImageWidget

from pkg_resources import resource_filename



pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
app = QtWidgets.QApplication(sys.argv)
interface = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
file = lambda fn: resource_filename('tests', 'data/%s' % fn)
ui.setupUi(interface,file("pas2.jpg")) #picture name
interface.show()
sys.exit(app.exec_())

