# from config import *

try:
    from crnn import CRNNHandle
    from angnet import AngleNetHandle
    from utils import draw_bbox, crop_rect, sorted_boxes, get_rotate_crop_image
    from dbnet.dbnet_infer import DBNET
except:
    from .crnn import CRNNHandle
    from .angnet import AngleNetHandle
    from .utils import draw_bbox, crop_rect, sorted_boxes, get_rotate_crop_image
    from .dbnet.dbnet_infer import DBNET

import sys, os

try:
    from ..rapid_ocr.ch_ppocr_server_v2_rec import TextRecognizer
except:
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
    from rapid_ocr.ch_ppocr_server_v2_rec import TextRecognizer
# from pyqtocr.rapid_ocr.ch_ppocr_server_v2_rec.text_recognize import TextRecognizer

from PIL import Image
import numpy as np
import cv2
import copy
import time
import traceback

from pathlib import Path
import os

root_path = os.path.dirname(__file__)


class OcrHandle(object):
    def __init__(self, args):
        self.args = args
        self.text_handle = DBNET(self.args.dbnet_model)
        self.crnn_handle = CRNNHandle(self.args.crnn_model)
        # from paddleocr.tools.infer.predict_rec import TextRecognizer
        # self.text_recognizer = TextRecognizer(self.args)
        # from ..rapid_ocr.ch_ppocr_server_v2_rec.text_recognize import TextRecognizer
        # self.text_recognizer = TextRecognizer(
        #     os.path.join(root_path, '../rapid_ocr/models/ch_ppocr_server_v2.0_rec_infer.onnx'))
        self.text_recognizer = TextRecognizer(self.args.rapidocr_rec_model_dir)
        if self.args.angle_detect:
            self.angle_handle = AngleNetHandle(self.args.angle_net_model)

    def crnnRecWithBox(self, im, boxes_list, score_list):
        """
        crnn模型，ocr识别
        @@model,
        @@converter,
        @@im:Array
        @@text_recs:text box
        @@ifIm:是否输出box对应的img

        """
        results = []
        boxes_list = sorted_boxes(np.array(boxes_list))

        line_imgs = []
        for index, (box, score) in enumerate(
                zip(boxes_list[:self.args.angle_detect_num], score_list[:self.args.angle_detect_num])):
            tmp_box = copy.deepcopy(box)
            partImg_array = get_rotate_crop_image(im, tmp_box.astype(np.float32))
            partImg = Image.fromarray(partImg_array).convert("RGB")
            line_imgs.append(partImg)

        angle_res = False
        if self.args.angle_detect:
            angle_res = self.angle_handle.predict_rbgs(line_imgs)

        count = 1
        for index, (box, score) in enumerate(zip(boxes_list, score_list)):

            tmp_box = copy.deepcopy(box)
            partImg_array = get_rotate_crop_image(im, tmp_box.astype(np.float32))

            partImg = Image.fromarray(partImg_array).convert("RGB")

            if self.args.angle_detect and angle_res:
                partImg = partImg.rotate(180)

            if not self.args.is_rgb:
                partImg = partImg.convert('L')

            try:
                if self.args.is_rgb:
                    simPred = self.crnn_handle.predict_rbg(partImg)  ##识别的文本
                else:
                    simPred = self.crnn_handle.predict(partImg)  ##识别的文本
            except Exception as e:
                print(traceback.format_exc())
                continue

            results.append([tmp_box, simPred, score])
            count += 1

        return results

    def text_predict(self, img, short_size=None):
        if short_size is None:
            short_size = self.args.short_size
        boxes_list, score_list = self.text_handle.process(np.asarray(img).astype(np.uint8), short_size=short_size)
        # result = self.crnnRecWithBox(np.array(img), boxes_list, score_list)
        img_crop_list = []
        for bno in range(len(boxes_list)):
            tmp_box = copy.deepcopy(boxes_list[bno])
            img_crop = get_rotate_crop_image(np.asarray(img).astype(np.uint8), tmp_box.astype(np.float32))
            img_crop_list.append(img_crop)
        rec_res, elapse = self.text_recognizer(img_crop_list)
        result = [[boxes_list[bno], rec_res[bno][0], rec_res[bno][1]] for bno in range(len(boxes_list))]
        return result
