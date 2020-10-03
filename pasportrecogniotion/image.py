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


class Loader(object):
    """Loads `file` to `img`."""

    __depends__ = ['rectKernel']
    __provides__ = ['img', 'img_real']

    def __init__(self, file, as_gray=True, pdf_aware=True):
        self.file = file
        self.as_gray = as_gray
        self.pdf_aware = pdf_aware

    def _imread(self, file, rectKernel):
        img = image = cv2.imread(file)
        if img is not None and len(img.shape) != 2:
            # The PIL plugin somewhy fails to load some images

            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(img, (3, 3), 0)
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
        return blackhat, img

    def __call__(self, rectKernel):
        if isinstance(self.file, str):
            return self._imread(self.file, rectKernel)
        return None

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

    __provides__ = ['mrz']
    __depends__ = ['boxes']

    def __call__(self, boxes):
        text = boxes['MRZ']
        mrz = MRZ.from_ocr(text)
        mrz.aux['method'] = 'direct'
        # Now try improving the result via hacks
        if not mrz.valid:
            pass
        return mrz




class MRZPipeline(Pipeline):
    """This is the "currently best-performing" pipeline for parsing MRZ from a given image file."""

    def __init__(self, file, docfile, extra_cmdline_params=''):
        super(MRZPipeline, self).__init__()
        self.version = '1.0'  # In principle we might have different pipelines in use, so possible backward compatibility is an issue
        self.file = file
        self.add_component('opencv', OpenCVPreProc())
        self.add_component('loader', Loader(file))
        self.add_component('scaler', Scaler())
        self.add_component('boone', BooneTransform())
        self.add_component('box_locator', MRZBoxLocator(DocDescription(docfile)))
        self.add_component('box_to_mrz', BoxToMRZ())

    @property
    def result(self):
        return self['mrz']


def read_mrz(file, doc_descr, save_roi=False, extra_cmdline_params=''):
    """The main interface function to this module, encapsulating the recognition pipeline.
       Given an image filename, runs MRZPipeline on it, returning the parsed MRZ object.

    :param file: A filename or a stream to read the file data from.
    :param save_roi: when this is True, the .aux['roi'] field will contain the Region of Interest where the MRZ was parsed from.
    :param extra_cmdline_params:extra parameters to the ocr.py
    """
    p = MRZPipeline(file, doc_descr, extra_cmdline_params)
    mrz = p.result

    if mrz is not None:
        mrz.aux['text'] = p['text']
        if save_roi:
            mrz.aux['roi'] = p['roi']
    return mrz


