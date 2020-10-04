import cv2
from pkg_resources import resource_filename
from pytesseract import pytesseract

from pasportrecogniotion.image import recognise_doc
from pasportrecogniotion.util.docdescription import DocDescription


pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Smoke test for Tesseract OCR
def testview():
    file = lambda fn : resource_filename('tests', 'data/%s' % fn)
    recognise_doc(cv2.imread(file("pas3.jpg")), file("RusPass.json"))

def testPassDescription():
    file = lambda fn: resource_filename('tests', 'data/%s' % fn)
    docdescr = DocDescription(file("RusPass.json"))
    docdescr.show(cv2.imread(file("pas1.jpg")))




#testview()
testPassDescription()