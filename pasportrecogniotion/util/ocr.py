'''
PassportEye::Util: Interface between SKImage and the PyTesseract OCR
NB: You must have the "tesseract" tool present in your path for this to work.

Author: Dziuba Alexandr
License: MIT
'''

import tempfile
import numpy as np
from imageio import imwrite
from pytesseract import pytesseract


def ocr(img, lang='rus',whitelist = ""):

    if img is None or img.shape[-1] == 0:  # Issue #34
        return ''
    # Prevent annoying warning about lossy conversion to uint8
    if str(img.dtype).startswith('float') and np.nanmin(img) >= 0 and np.nanmax(img) <= 1:
        img = img.astype(np.float64) * (np.power(2.0, 8) - 1) + 0.499999999
        img = img.astype(np.uint8)

    config = ("--psm 6 ")

    res = pytesseract.image_to_string(img,
                              lang=lang,
                              config=config)

    return res

def ocreng(img, lang='eng'):

    if img is None or img.shape[-1] == 0:  # Issue #34
        return ''
    # Prevent annoying warning about lossy conversion to uint8
    if str(img.dtype).startswith('float') and np.nanmin(img) >= 0 and np.nanmax(img) <= 1:
        img = img.astype(np.float64) * (np.power(2.0, 8) - 1) + 0.499999999
        img = img.astype(np.uint8)

    config = ("--psm 6")

    res = pytesseract.image_to_string(img,
                              lang=lang,
                              config=config)

    return res

