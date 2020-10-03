'''
Image processing for passport data extraction.

Author: Dziuba Alexandr
License: MIT
'''

import cv2
import numpy as np
from pasportrecogniotion.util.docdescription import DocDescription
from pasportrecogniotion.util.pipeline import Pipeline
from pasportrecogniotion.text import MRZ
from userInterface.passport_data import PassportData

def show(img):
    cv2.imshow("test", img)
    cv2.waitKey(0)


class OpenCVPreProc(object):
    """Preperocessing for OpenCV"""

    __depends__ = []
    __provides__ = ['rectKernel','sqKernel']

    def __init__(self):
        self.rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        self.sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))

    def __call__(self):
        return  self.rectKernel, self.sqKernel



class GrayConverter(object):
    """Convert img to GRAY"""

    __depends__ = ['rectKernel']
    __provides__ = ['img', 'img_real']

    def __init__(self, img):
        self.img = img

    def __call__(self, rectKernel):
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(img, (3, 3), 0)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

        return blackhat, img

class BooneTransform(object):
    """Processes `img_small` according to Hans Boone's method
    (http://www.pyimagesearch.com/2015/11/30/detecting-machine-readable-zones-in-passport-images/)
    Outputs a `img_binary` - a result of threshold_otsu(closing(sobel(black_tophat(img_small)))"""

    __depends__ = ['img', 'rectKernel', 'sqKernel']
    __provides__ = ['img_binary']

    def __init__(self, square_size=5):
        self.square_size = square_size

    def __call__(self, img, rectKernel, sqKernel):

        gradX = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)

        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
        res = cv2.erode(thresh, None, iterations=4)

        return res


class MRZBoxLocator(object):
    """Extracts putative passport's data as DataBlock-s instances from the contours of `img_binary`"""

    __depends__ = ['img_binary','img_real']
    __provides__ = ['boxes']

    def __init__(self, doc_description):
        self.doc_description = doc_description

    def __call__(self, img_binary, img_real):
        cs, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.doc_description.extract_data(img_real, cs)

        return self.doc_description.blocks



class BoxToMRZ(object):
    """Convert text to MRZ"""

    __depends__ = ['boxes']
    __provides__ = ['mrz']

    def __call__(self, boxes):
        text = boxes['MRZ']
        mrz = MRZ.from_ocr(text)
        mrz.aux['method'] = 'direct'
        # Now try improving the result via hacks
        if not mrz.valid:
            pass
        return mrz

class BoxToData(object):

    __depends__ = ['boxes']
    __provides__ = ['data']

    def __call__(self, boxes, mrz):

        data = PassportData()

        return data



class MRZPipeline(Pipeline):
    """This is the "currently best-performing" pipeline for parsing passport from a given image file."""

    def __init__(self, img, docfile, extra_cmdline_params=''):
        super(MRZPipeline, self).__init__()
        self.version = '1.0'
        self.add_component('opencv', OpenCVPreProc())
        self.add_component('loader', GrayConverter(img))
        self.add_component('boone', BooneTransform())
        self.add_component('box_locator', MRZBoxLocator(DocDescription(docfile)))
        self.add_component('box_to_mrz', BoxToMRZ())
        self.add_component('box_to_mrz', BoxToData())

    @property
    def result(self):
        return self['data']


def recognise_doc(img, doc_descr):
    """The main interface function to this module, encapsulating the recognition pipeline.
       Given an image filename, runs MRZPipeline on it, returning the parsed MRZ object.

    :param img: A img  to read the file data from.
    :param doc_descr: A file to read document description from.
    """
    p = MRZPipeline(img, doc_descr)
    mrz = p.result

    if mrz is not None:
        mrz.aux['text'] = p['text']
    return mrz


