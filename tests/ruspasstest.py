'''
Test module for use with py.test.
Write each test as a function named test_<something>.
Read more here: http://pytest.org/

Author: Konstantin Tretyakov
License: MIT
'''
import cv2
from pkg_resources import resource_filename
from pytesseract import pytesseract

from pasportrecogniotion.image import read_mrz
from pasportrecogniotion.util.docdescription import DocDescription


pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Smoke test for Tesseract OCR
def testview():
    file = lambda fn : resource_filename('tests', 'data/%s' % fn)
    read_mrz(file('pas2.jpg'),file("RusPass.json"))

def testPassDescription():
    file = lambda fn: resource_filename('tests', 'data/%s' % fn)
    docdescr = DocDescription(file("RusPass.json"))
    docdescr.show(cv2.imread(file("pas3.jpg")))

cv2.namedWindow("test", cv2.WINDOW_NORMAL)
cv2.resizeWindow("test", 420, 600)

testview()
#testPassDescription()