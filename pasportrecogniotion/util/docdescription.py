import io

import cv2
import json
import numpy as np
from pasportrecogniotion.util.ocr import ocr, ocreng


class DataBlock():
    def __init__(self, block, name):
        self.height = block['height']
        self.width = block['width']
        self.posX = block['posX']
        self.posY = block['posY']
        self.direction = block['direction']
        self.name = name
        self.data = []
        self.mrz = []
        self.images = []
        self.whitelist = block['whitelist']

    def recognize(self, img):

        min_x, min_y, max_x, max_y = 10000, 10000, 0, 0
        self.images.sort(key=lambda box: cv2.boundingRect(box)[1])
        for box in self.images:
            x, y, w, h = cv2.boundingRect(box)
            max_y = max(max_y, h + y)
            max_x = max(max_x, w + x)
            min_x = min(min_x, x)
            min_y = min(min_y, y)

        ROI = img[min_y-10:max_y+10, min_x-10:max_x+10]
        #ROI = img[y - 10:y + 10 + h, x - 10:x + 10 + w]
        ROI = cv2.threshold(ROI, 210, 255, cv2.THRESH_TRUNC)[1]
        if self.direction != 'normal':
            if self.direction == 'right':
                ROI = cv2.rotate(ROI, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif self.direction == 'right':
                ROI = cv2.rotate(ROI, cv2.ROTATE_90_CLOCKWISE)
            else:
                ROI = cv2.rotate(ROI, cv2.ROTATE_180)

        if self.name != 'MRZ':
            res = ocr(ROI, 'rus')
            self.data.append("".join(filter(lambda x: x in self.whitelist, res[0:-1])))


        else:
            res = ocreng(ROI, 'eng')
            self.mrz.append("".join(filter(lambda x: x in self.whitelist, res[0:-1])))




class DocDescription(object):

    def __init__(self, file):
        data = {}
        with io.open(file, encoding='utf-8') as json_file:
            data = json.load(json_file)
        self.width = data['width']
        self.height = data['height']
        self.blocks = {}
        for i in data['blocks']:
            data_block = data['blocks'][i]
            self.blocks[i] = (DataBlock(data_block, i))
        self.name = data['name']

    def extract_data(self, img, contours):
        iws = img.shape[1] / self.width
        ihs = img.shape[0] / self.height

        rects = [cv2.minAreaRect(cont) for cont in (contours)]
        for key in self.blocks:
            block = self.blocks[key]
            for i, c in enumerate(rects):
                if c[1][1]*c[1][0] > 50 / iws / ihs:
                    if ((c[0][1] / ihs > block.posY) and (
                            c[0][1] / ihs < block.posY + block.height) and
                            (c[0][0] / iws > block.posX) and (
                                    c[0][0] / iws < block.posX + block.width)):
                        box = cv2.boxPoints(c)  # cv2.boxPoints(rect) for OpenCV 3.x
                        box = np.int0(box)
                        block.images.append(box)
            block.recognize(img.copy())



    def show(self, img=None):
        # Function for debuging

        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, self.width + 10, self.height + 10)
        blank_image = cv2.resize(img, (self.width + 10, self.height + 10))
        cv2.rectangle(blank_image, (5, 5), (self.width + 5, self.height + 5), (0, 255, 0), 2)
        for name in self.blocks:
            i = self.blocks[name]
            start_point = (i.posX + 5, i.posY + 5)
            end_point = (i.posX + i.width + 5, i.posY + i.height + 5)
            color = (255, 0, 0)
            cv2.rectangle(blank_image, start_point, end_point, color, 2)

        cv2.imshow(self.name, blank_image)
        cv2.waitKey(0)
