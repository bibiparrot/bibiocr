import copy
import sys
from collections import OrderedDict

import math
from PyQt5.QtCore import QObject, pyqtSignal, QThread, Qt, QUrl, QProcess
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QSoundEffect

from mainwindow import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsPixmapItem, QMessageBox, QFileDialog, \
    QTableWidgetItem
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import json
import logging

log_format = (
    '[%(asctime)s] %(levelname)-8s %(name)-12s %(message)s')

# Define basic configuration
logging.basicConfig(
    # Define logging level
    level=logging.DEBUG,
    # Declare the object we created to format the log messages
    format=log_format,
    # Declare handlers
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

try:
    from onnx.cnocr_lite.crnn.CRNN import CRNNHandle
    from onnx.cnocr_lite.angnet.angle import AngleNetHandle
    from onnx.cnocr_lite.dbnet.dbnet_infer import DBNET
    from onnx.rapid_ocr.ch_ppocr_mobile_v2_rec.text_recognize import TextRecognizer
    from onnx.rapid_ocr.ch_ppocr_mobile_v2_det.text_detect import TextDetector
    from onnx.rapid_ocr.ch_ppocr_mobile_v2_cls.text_cls import TextClassifier
except:
    from .onnx.cnocr_lite.crnn.CRNN import CRNNHandle
    from .onnx.cnocr_lite.angnet.angle import AngleNetHandle
    from .onnx.cnocr_lite.dbnet.dbnet_infer import DBNET
    from .onnx.rapid_ocr.ch_ppocr_mobile_v2_rec.text_recognize import TextRecognizer
    from .onnx.rapid_ocr.ch_ppocr_mobile_v2_cls.text_cls import TextClassifier
    from .onnx.rapid_ocr.ch_ppocr_mobile_v2_det.text_detect import TextDetector

local_root = os.path.dirname(__file__)
os.chdir(os.path.abspath(os.path.dirname(__file__)))
local_ttsmp3_tempfile = os.path.abspath('~edge_tts_zh-CN-XiaomoNeural.mp3')


########################################################################################################################


########################################################################################################################

def cv2pil(image):
    ''' OpenCV -> PIL '''
    new_image = image.copy()
    if new_image.ndim == 2:  #
        pass
    elif new_image.shape[2] == 3:  #
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  #
        new_image = new_image[:, :, [2, 1, 0, 3]]
    new_image = Image.fromarray(new_image)
    return new_image


def pil2cv(image):
    ''' PIL -> OpenCV '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  #
        pass
    elif new_image.shape[2] == 3:  #
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:  #
        new_image = new_image[:, :, [2, 1, 0, 3]]
    return new_image


########################################################################################################################

def check_and_read_gif(img_path):
    if os.path.basename(img_path)[-3:].lower() in ['gif']:
        gif = cv2.VideoCapture(img_path)
        ret, frame = gif.read()
        if not ret:
            logging.info("Cannot read {}. This gif image maybe corrupted.")
            return None, False
        if len(frame.shape) == 2 or frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        imgvalue = frame[:, :, ::-1]
        return imgvalue, True
    return None, False


def read_cv(image_file):
    img, flag = check_and_read_gif(image_file)
    if not flag:
        img = cv2.imread(image_file)
    if img is None:
        img = Image.open(image_file)
        if img is not None:
            img = pil2cv(img)
    if img is None:
        logger.debug("error in loading image:{}".format(image_file))
    return img


def read_pil(image_file):
    img, flag = check_and_read_gif(image_file)
    if not flag:
        img = cv2.imread(image_file)
    if img is None:
        img = Image.open(image_file)
    else:
        img = cv2pil(img)
    if img is None:
        logger.debug("error in loading image:{}".format(image_file))
    return img


# def read_image(image_file):
#     img, flag = check_and_read_gif(image_file)
#     if not flag:
#         img = cv2.imread(image_file)
#     if img is None:
#         logger.debug("error in loading image:{}".format(image_file))
#     return img


def qImage2CvMat(qImage):
    if qImage is None:
        return None
    _qImage = qImage.convertToFormat(4)
    width = _qImage.width()
    height = _qImage.height()

    ptr = _qImage.bits()
    ptr.setsize(_qImage.byteCount())
    mat = np.array(ptr, dtype=np.uint8).reshape(height, width, 4)  # Copies the data
    return mat[..., :3]


def cvMat2QImage(cvMat):
    if cvMat is None:
        return None
    shape = cvMat.shape
    if cvMat.ndim == 4:
        cvMat = cv2.cvtColor(cvMat, cv2.COLOR_BGR2RGBA)
        qImage = QImage(cvMat.data, shape[1], shape[0], shape[1] * 4, QImage.Format_RGB32)
    elif cvMat.ndim == 3:
        cvMat = cv2.cvtColor(cvMat, cv2.COLOR_BGR2RGB)
        qImage = QImage(cvMat.data, shape[1], shape[0], shape[1] * 3, QImage.Format_RGB888)
    elif cvMat.ndim == 2:
        qImage = QImage(cvMat.data, shape[1], shape[0], shape[1] * 1, QImage.Format_Grayscale8)
    else:
        raise Exception("image channels must be 1,3,4")
    return qImage


########################################################################################################################
def get_config(name='config.json'):
    with open(os.path.join(local_root, name), "r", encoding='utf8') as jsonfile:
        data = json.load(jsonfile)
        logger.info(f"Read {name} successfully:\n{data}\n")
        jsonfile.close()
    return data


def get_config_path(path):
    return os.path.abspath(os.path.join(local_root, path))


def get_args(name='config.json'):
    config = get_config(name)
    args = OrderedDict()
    args.__dict__.update(config)
    logger.info(f"{name} args:\n{json.dumps(args.__dict__, indent=4)}\n")
    return args


########################################################################################################################


def draw_ocr_box_txt(image,
                     boxes,
                     txts,
                     scores=None,
                     drop_score=0.5,
                     font_path="msyh.ttc"):
    h, w = image.height, image.width
    img_left = image.copy()
    img_right = Image.new('RGB', (w, h), (255, 255, 255))

    import random

    random.seed(0)
    draw_left = ImageDraw.Draw(img_left)
    draw_right = ImageDraw.Draw(img_right)
    for idx, (box, txt) in enumerate(zip(boxes, txts)):
        if scores is not None and scores[idx] < drop_score:
            continue
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))
        draw_left.polygon(box, fill=color)
        draw_right.polygon(
            [
                box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                box[2][1], box[3][0], box[3][1]
            ],
            outline=color)
        box_height = math.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][
            1]) ** 2)
        box_width = math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][
            1]) ** 2)
        if box_height > 2 * box_width:
            font_size = max(int(box_width * 0.9), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            cur_y = box[0][1]
            for c in txt:
                char_size = font.getsize(c)
                draw_right.text(
                    (box[0][0] + 3, cur_y), c, fill=(0, 0, 0), font=font)
                cur_y += char_size[1]
        else:
            font_size = max(int(box_height * 0.8), 10)
            font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
            draw_right.text(
                [box[0][0], box[0][1]], txt, fill=(0, 0, 0), font=font)
    img_left = Image.blend(image, img_left, 0.5)
    img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
    img_show.paste(img_left, (0, 0, w, h))
    img_show.paste(img_right, (w, 0, w * 2, h))
    return np.array(img_show)


def vis_img(img, dt_boxes, rec_res, args):
    drop_score = args.drop_score
    font_path = args.vis_font_path
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    boxes = dt_boxes
    txts = [rec_res[i][0] for i in range(len(rec_res))]
    scores = [rec_res[i][1] for i in range(len(rec_res))]

    draw_img = draw_ocr_box_txt(
        image,
        boxes,
        txts,
        scores,
        drop_score=drop_score,
        font_path=font_path)
    return draw_img[:, :, ::-1]


def save_img(img, image_file, args):
    draw_img_save_dir = args.draw_img_save_dir
    os.makedirs(draw_img_save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(draw_img_save_dir, os.path.basename(image_file)), img)
    logger.debug("The visualized image saved in {}".format(
        os.path.join(draw_img_save_dir, os.path.basename(image_file))))


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def get_rotate_crop_image(img, points):
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    img_crop_width = int(np.linalg.norm(points[0] - points[1]))
    img_crop_height = int(np.linalg.norm(points[0] - points[3]))
    pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])

    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img_crop,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


########################################################################################################################


class Pyttsx3Thread(QThread):

    def __init__(self, txt):
        super(Pyttsx3Thread, self).__init__()
        self.txt = txt

    def run(self):
        import pyttsx3
        engine = pyttsx3.init(debug=True)
        volume = engine.getProperty('volume')
        engine.setProperty('volume', 1.0)
        engine.say(self.txt)
        engine.runAndWait()
        engine.stop()


# class Edge_ttsThread(QThread):
#     def __init__(self, txt=None):
#         super(Edge_ttsThread, self).__init__()
#         self.txt = txt
#
#     def run(self):
#         def text_to_mp3(txt, mp3file):
#             def find_edge_tts(dir=None):
#                 edge_tts_exe = 'edge-tts.exe'
#                 if dir is not None:
#                     edge_tts_path = os.path.join(dir, edge_tts_exe)
#                     if os.path.exists(edge_tts_path):
#                         return os.path.abspath(edge_tts_path)
#                 edge_tts_path = os.path.join('binary', edge_tts_exe)
#                 if os.path.exists(edge_tts_path):
#                     return os.path.abspath(edge_tts_path)
#                 if os.path.exists(edge_tts_exe):
#                     return os.path.abspath(edge_tts_exe)
#                 return edge_tts_exe
#
#             program = QProcess()
#             program.start(find_edge_tts(), ['--voice', 'zh-CN-XiaomoNeural', '--text', txt, '--write-media',
#                                             mp3file])
#             program.waitForFinished()
#
#         if os.path.exists(local_ttsmp3_tempfile):
#             os.remove(local_ttsmp3_tempfile)
#         text_to_mp3(self.txt, local_ttsmp3_tempfile)
#         logger.info(f"Generated {self.txt} into TTS file{local_ttsmp3_tempfile}")


class EdgeTtsThread(QThread):
    def __init__(self, txt=''):
        super(EdgeTtsThread, self).__init__()
        self.txt = txt.replace('\n', '')
        self.args = get_args('edge_tts.json')

    def run(self):
        from edge_tts import Communicate, SubMaker, list_voices
        import asyncio
        async def _tts(text, args):
            tts = Communicate()
            subs = SubMaker(args.overlapping)
            if args.write_media:
                media_file = open(get_config_path(args.write_media), "wb")  # pylint: disable=consider-using-with
            if args.custom_ssml:
                args.custom_ssml = get_config_path(args.custom_ssml)
            async for i in tts.run(
                    text,
                    args.boundary_type,
                    args.codec,
                    args.voice,
                    args.pitch,
                    args.rate,
                    args.volume,
                    customspeak=args.custom_ssml,
            ):
                if i[2] is not None:
                    if not args.write_media:
                        sys.stdout.buffer.write(i[2])
                    else:
                        media_file.write(i[2])
                elif i[0] is not None and i[1] is not None:
                    subs.create_sub(i[0], i[1])
            if args.write_media:
                media_file.close()
            if not args.write_subtitles:
                sys.stderr.write(subs.generate_subs())
            else:
                with open(get_config_path(args.write_subtitles), "w", encoding="utf-8") as file:
                    srt = subs.generate_subs()
                    file.write(srt)
                    logger.info(f"subtitles: {srt}")

        if os.path.exists(get_config_path(self.args.write_media)):
            os.remove(get_config_path(self.args.write_media))
        if os.path.exists(get_config_path(self.args.write_subtitles)):
            os.remove(get_config_path(self.args.write_subtitles))
        asyncio.new_event_loop().run_until_complete(_tts(self.txt, self.args))
        logger.info(f"Generated {self.txt} into TTS file{local_ttsmp3_tempfile}")


########################################################################################################################

class OcrWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, img, resDict):
        super().__init__()
        self.img = img
        self.args = get_args()
        self.text_handle = DBNET(self.args.cnocr_lite_dbnet_model)
        self.crnn_handle = CRNNHandle(self.args.cnocr_lite_crnn_model)
        self.angle_handle = AngleNetHandle(self.args.cnocr_lite_angle_net_model)

        self.text_detector = TextDetector(get_config_path(self.args.rapidocr_det_model_path))
        self.text_recognizer = TextRecognizer(get_config_path(self.args.rapidocr_rec_model_path),
                                              get_config_path(self.args.rapidocr_rec_char_dict_path))
        self.text_classifier = TextClassifier(get_config_path(self.args.rapidocr_cls_model_path))
        self.result = resDict

    @staticmethod
    def rec2table(dt_boxes, rec_res):
        s_table = ["(x1,\ty1)\t(x2,\ty2)\t(x3,\ty3)\t(x4,\ty4)\tprob.\ttext"]
        for i in range(len(dt_boxes)):
            box = dt_boxes[i]
            txt, prob = rec_res[i]
            # x1, y1, x2, y2, x3, y3, x4, y4 = box.reshape(-1)
            s_table.append(
                f"({box[0][0]:.0f},\t{box[0][1]:.0f})\t({box[1][0]:.0f},\t{box[1][1]:.0f})\t({box[2][0]:.0f},\t{box[2][1]:.0f})\t({box[3][0]:.0f},\t{box[3][1]:.0f})\t{prob:.3f}\t{txt}")
        return "\n".join(s_table)

    @staticmethod
    def rec2text(dt_boxes, rec_res):
        from io import StringIO
        f = StringIO()

        def box_same_line(box1, box2):
            x11, y11, x12, y12, x13, y13, x14, y14 = box1.reshape(-1)
            x21, y21, x22, y22, x23, y23, x24, y24 = box2.reshape(-1)
            if abs(y14 - y11) + abs(y24 - y21) > 1.1 * abs(y24 - y11):
                return True
            else:
                return False

        for i in range(len(dt_boxes)):
            box = dt_boxes[i]
            txt, prob = rec_res[i]
            if i == 0:
                f.write(txt)
            else:
                if box_same_line(dt_boxes[i - 1], dt_boxes[i]):
                    f.write('\t')
                else:
                    f.write('\n')
                f.write(txt)
        return f.getvalue()

    def run_rapid_ocr(self, img):
        img = img.copy()
        self.progress.emit(1)
        dt_boxes, elapse = self.text_detector(img)
        logger.debug("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None or len(dt_boxes) < 1:
            return [], []
        dt_boxes = sorted_boxes(dt_boxes)

        self.progress.emit(30)
        img_crop_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(img, tmp_box)
            img_crop_list.append(img_crop)

        self.progress.emit(50)
        if self.args.angle_detect:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            logger.debug("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        self.progress.emit(60)
        rec_res, elapse = self.text_recognizer(img_crop_list)
        logger.debug("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))

        self.progress.emit(90)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.args.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        logger.debug("filter_res num  : {}".format(
            len(filter_boxes)))
        return filter_boxes, filter_rec_res

    def run_ocr(self, img):
        ori_im = img.copy()
        self.progress.emit(1)
        boxes_list, score_list = self.text_handle.process(np.asarray(ori_im).astype(np.uint8),
                                                          short_size=self.args.short_size)

        boxes_list = sorted_boxes(boxes_list)

        self.progress.emit(20)

        logger.debug("dt_boxes num : {}".format(len(boxes_list)))
        if boxes_list is None:
            return None, None

        img_crop_list = []
        for bno in range(len(boxes_list)):
            tmp_box = copy.deepcopy(boxes_list[bno])
            img_crop = get_rotate_crop_image(np.asarray(ori_im).astype(np.uint8), tmp_box.astype(np.float32))
            img_crop_list.append(img_crop)
        rec_res, elapse = self.text_recognizer(img_crop_list)
        results = [[boxes_list[bno], rec_res[bno][0], rec_res[bno][1]] for bno in range(len(boxes_list))]
        logger.debug("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        self.progress.emit(80)

        boxes, rec_res = [], []
        for i, r in enumerate(results):
            rect, txt, confidence = r
            boxes.append(rect)
            rec_res.append([txt, confidence])
        # vimg = vis_img(pil2cv(ori_im), boxes, rec_res, self.args)

        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.args.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
            logger.debug("rec_res text  : {}, score : {:.3f}".format(
                text, score))
        self.progress.emit(80)
        return filter_boxes, filter_rec_res

    def run(self):
        """Long-running task."""
        logger.info(f"START ocr process ...")
        # self.result['ocr'] = self.run_ocr(self.img)
        self.result['ocr'] = self.run_rapid_ocr(self.img)
        logger.info(f"finish ocr function.")
        self.result['text'] = self.rec2table(*self.result['ocr'])
        # self.result['pure_text'] = "\n".join([item[0] for item in self.result['ocr'][1]])
        self.result['pure_text'] = self.rec2text(*self.result['ocr'])
        self.progress.emit(90)
        self.result['image'] = vis_img(self.img, *self.result['ocr'], self.args)
        logger.info(f"END ocr process !!!")
        self.progress.emit(100)
        self.finished.emit()


class MainWindow(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.show()
        self.clipboard = QApplication.clipboard()
        self.toolButton_5.setEnabled(False)
        self.toolButton_6.setEnabled(False)
        self.toolButton_7.setEnabled(False)
        self.toolButton_8.setEnabled(False)
        self.toolButton_9.setEnabled(False)

        self.toolButton.clicked.connect(
            self.on_toolButton_click
        )
        self.toolButton_2.clicked.connect(
            self.on_toolButton_2_click
        )

        self.toolButton_3.clicked.connect(
            self.on_toolButton_3_click
        )

        self.toolButton_4.clicked.connect(
            self.on_toolButton_4_click
        )

        self.toolButton_5.clicked.connect(
            self.on_toolButton_5_click
        )

        self.toolButton_6.clicked.connect(
            self.on_toolButton_6_click
        )

        self.toolButton_7.clicked.connect(
            self.on_toolButton_7_click
        )

        self.toolButton_8.clicked.connect(
            self.on_toolButton_8_click
        )

        self.toolButton_9.clicked.connect(
            self.on_toolButton_9_click
        )

        self.action.triggered.connect(
            self.on_toolButton_click
        )
        self.action_2.triggered.connect(
            self.on_toolButton_2_click
        )
        self.action_3.triggered.connect(
            self.on_toolButton_3_click
        )
        self.action_4.triggered.connect(
            self.on_toolButton_4_click
        )
        self.action_5.triggered.connect(
            self.on_toolButton_5_click
        )
        self.action_6.triggered.connect(
            self.on_toolButton_6_click
        )
        self.action_7.triggered.connect(
            self.on_toolButton_7_click
        )
        self.action_8.triggered.connect(
            self.on_toolButton_8_click
        )
        self.action_9.triggered.connect(
            self.on_toolButton_9_click
        )
        self.action_10.triggered.connect(
            self.close
        )
        self.action_11.triggered.connect(
            lambda: QMessageBox.about(self, '关于作者', 'mail:chunqishi@gmail.com, wechat:chunqishi')
        )
        self.result = {}
        self.needread = False

    def closeEvent(self, event):
        reply = QMessageBox.question(self, '关闭提示', "是否要关闭程序?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def reportProgress(self, n):
        self.progressBar.setProperty("value", n)

    def runLongTask(self, img):
        try:
            self.result.clear()
            self.toolButton.setEnabled(False)
            self.toolButton_2.setEnabled(False)
            self.toolButton_3.setEnabled(False)
            self.toolButton_4.setEnabled(False)
            self.toolButton_5.setEnabled(False)
            self.toolButton_6.setEnabled(False)
            self.toolButton_7.setEnabled(False)
            self.toolButton_8.setEnabled(False)
            self.toolButton_9.setEnabled(False)
            self.progressBar.setProperty("value", 0)
            qImage = cvMat2QImage(img)
            self.graphicsView.geometry()
            pmp = QPixmap(qImage)
            scene = QGraphicsScene()
            scene.addItem(QGraphicsPixmapItem(pmp))
            self.graphicsView.setScene(scene)
            # self.graphicsView.fitInView(scene.sceneRect(), mode=Qt.KeepAspectRatio)
            self.graphicsView.update()

            # Step 2: Create a QThread object
            self.thread = QThread()
            # Step 3: Create a worker object
            self.worker = OcrWorker(img, self.result)
            # Step 4: Move worker to the thread
            self.worker.moveToThread(self.thread)
            # Step 5: Connect signals and slots
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.progress.connect(self.reportProgress)
            # Step 6: Start the thread
            self.thread.start()
            # Step 7: init TTS Thread
            # self.ttsthread = Edge_ttsThread()
        except Exception as exp:
            import traceback, sys
            traceback.print_exc()
            exc_type, exc_value, exc_traceback = sys.exc_info()
            error = str(repr(traceback.format_exception(exc_type, exc_value, exc_traceback)))
            logger.debug(f"Error:{error}")

        def show_result():
            self.toolButton.setEnabled(True)
            self.toolButton_2.setEnabled(True)
            self.toolButton_3.setEnabled(True)
            self.toolButton_4.setEnabled(True)
            qImage = cvMat2QImage(self.result.get('image', None))
            if qImage is not None:
                pmp = QPixmap(qImage)
                scene = QGraphicsScene()
                scene.addItem(QGraphicsPixmapItem(pmp))
                self.graphicsView_2.setScene(scene)
                # self.graphicsView_2.fitInView(scene.sceneRect(), mode=Qt.KeepAspectRatio)
                self.graphicsView_2.update()
                self.textEdit.setPlainText(self.result.get('pure_text', ''))
                self.textEdit.show()
                # self.textBrowser.setText(self.result.get('pure_text', ''))
                # self.textBrowser.show()

            colnames = ['x1 (px)', 'y1 (px)', 'x2 (px)', 'y2 (px)', 'x3 (px)', 'y3 (px)', 'x4 (px)', 'y4 (px)', 'prob.',
                        'text']
            dt_boxes, rec_res = self.result['ocr']
            self.tableWidget.setWindowTitle(f"图片识别结果[{len(rec_res)}]")
            self.tableWidget.setColumnCount(len(colnames))
            self.tableWidget.setRowCount(len(dt_boxes))
            self.tableWidget.setHorizontalHeaderLabels(colnames)
            for i, box in enumerate(dt_boxes):
                txt, prob = rec_res[i]
                self.tableWidget.setItem(i, 0, QTableWidgetItem(f"{box[0][0]:.0f}"))
                self.tableWidget.setItem(i, 1, QTableWidgetItem(f"{box[0][1]:.0f}"))
                self.tableWidget.setItem(i, 2, QTableWidgetItem(f"{box[1][0]:.0f}"))
                self.tableWidget.setItem(i, 3, QTableWidgetItem(f"{box[1][1]:.0f}"))
                self.tableWidget.setItem(i, 4, QTableWidgetItem(f"{box[2][0]:.0f}"))
                self.tableWidget.setItem(i, 5, QTableWidgetItem(f"{box[2][1]:.0f}"))
                self.tableWidget.setItem(i, 6, QTableWidgetItem(f"{box[3][0]:.0f}"))
                self.tableWidget.setItem(i, 7, QTableWidgetItem(f"{box[3][1]:.0f}"))
                self.tableWidget.setItem(i, 8, QTableWidgetItem(f"{prob:.3f}"))
                self.tableWidget.setItem(i, 9, QTableWidgetItem(txt))
            self.tableWidget.show()

            self.toolButton_5.setEnabled(True)
            self.toolButton_6.setEnabled(True)
            self.toolButton_7.setEnabled(True)
            self.toolButton_8.setEnabled(True)

            if self.needread:
                pure_text = self.result.get('pure_text', '')
                # self.thread = Pyttsx3Thread(pure_text)
                self.ttsthread = EdgeTtsThread(pure_text)
                self.ttsthread.finished.connect(self.ttsthread.deleteLater)
                self.ttsthread.start()

                def play_tts():
                    self.toolButton_9.setEnabled(True)
                    self.player = QMediaPlayer()
                    self.player.setMedia(QMediaContent(QUrl.fromLocalFile(local_ttsmp3_tempfile)))
                    self.player.setVolume(100)
                    self.player.play()

                self.ttsthread.finished.connect(
                    lambda: play_tts()
                )

        self.thread.finished.connect(
            lambda: show_result()
        )

    def on_toolButton_click(self, needread=False):
        self.needread = needread
        mimedata = self.clipboard.mimeData()
        if mimedata.hasImage():
            qimg = self.clipboard.image()
            img = qImage2CvMat(qimg)
            logger.info(f"readed image[{img.shape}] from clipboard")
            self.runLongTask(img)
        else:
            QMessageBox.warning(self, '剪贴板无图片', '请复制图片到剪贴板！', QMessageBox.Ok, QMessageBox.Ok)

    def on_toolButton_2_click(self):
        self.needread = True
        self.on_toolButton_click(True)

    def on_toolButton_3_click(self, needread=False):
        self.needread = needread
        filename, filetype = QFileDialog.getOpenFileName(self, "选择图片文件", os.getcwd(),
                                                         "Image Files(*.bmp;*.png;*.jpg;*.jpeg;*.gif);;All Files(*)")

        if filename is not None and len(filename) > 3:
            img = read_cv(filename)
            if img is not None:
                logger.info(f"readed image[{img.shape}] from {filename}")
                self.runLongTask(img)
            else:
                QMessageBox.warning(self, '图片格式不支持', '请选择支持的图片格式！', QMessageBox.Ok, QMessageBox.Ok)
        else:
            QMessageBox.warning(self, '未打开图片', '请选择图片文件！', QMessageBox.Ok, QMessageBox.Ok)

    def on_toolButton_4_click(self):
        self.needread = True
        self.on_toolButton_3_click(True)

    def on_toolButton_5_click(self):
        self.clipboard.setText(self.result.get('text', ""))
        QMessageBox.information(self, '拷贝到剪贴板', '已将识别文本详细内容拷贝到剪贴板', QMessageBox.Ok, QMessageBox.Ok)

    def on_toolButton_6_click(self):
        self.clipboard.setText(self.result.get('pure_text', ""))
        QMessageBox.information(self, '拷贝到剪贴板', '已将识别结果文本部分拷贝到剪贴板', QMessageBox.Ok, QMessageBox.Ok)

    def on_toolButton_7_click(self):
        # text = self.textBrowser.document().toPlainText()
        text = self.textEdit.document().toPlainText()
        self.ttsthread = EdgeTtsThread(text)
        self.ttsthread.finished.connect(self.ttsthread.deleteLater)
        self.ttsthread.start()

        def play_tts():
            self.toolButton_9.setEnabled(True)
            self.player = QMediaPlayer()
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(local_ttsmp3_tempfile)))
            self.player.setVolume(100)
            self.player.play()

        self.ttsthread.finished.connect(
            lambda: play_tts()
        )


    def on_toolButton_8_click(self):
        csv_file, filetype = QFileDialog.getSaveFileName(self, '选择保存路径', os.path.join(os.getcwd(), '图片识别结果.csv'),
                                                         "CSV File(*.csv);;All Files(*)")
        if csv_file is None or len(csv_file) <= 3:
            QMessageBox.warning(self, '未选择保存路径', '请选择CSV文件保存路径！', QMessageBox.Ok, QMessageBox.Ok)
        else:
            dt_boxes, rec_res = self.result['ocr']
            logger.info(f'csv_file = {csv_file}')
            logger.info(f'dt_boxes = {dt_boxes}; rec_res = {rec_res}')
            with open(csv_file, "w", encoding='utf-8-sig', newline='') as csvFile:
                logger.info(f'csvFile = {csvFile}')
                import csv
                wr = csv.writer(csvFile, quotechar=',')
                titles = ['x1 (px)', 'y1 (px)', 'x2 (px)', 'y2 (px)', 'x3 (px)', 'y3 (px)', 'x4 (px)', 'y4 (px)',
                          'prob.',
                          'text']
                wr.writerow(titles)
                for i, box in enumerate(dt_boxes):
                    txt, prob = rec_res[i]
                    wr.writerow([f"{box[0][0]:.0f}",
                                 f"{box[0][1]:.0f}",
                                 f"{box[1][0]:.0f}",
                                 f"{box[1][1]:.0f}",
                                 f"{box[2][0]:.0f}",
                                 f"{box[2][1]:.0f}",
                                 f"{box[3][0]:.0f}",
                                 f"{box[3][1]:.0f}",
                                 f"{prob:.3f}",
                                 txt
                                 ])
            QMessageBox.information(self, '保存成功', f'已将识别信息保存到“{csv_file}“', QMessageBox.Ok, QMessageBox.Ok)

    def on_toolButton_9_click(self):
        mp3_file, filetype = QFileDialog.getSaveFileName(self, '选择保存路径', os.path.join(os.getcwd(), '图片文字朗读.mp3'),
                                                         "MP3 File(*.mp3);;All Files(*)")
        if mp3_file is None or len(mp3_file) <= 3:
            QMessageBox.warning(self, '未选择保存路径', '请选择MP3文件保存路径！', QMessageBox.Ok, QMessageBox.Ok)
        else:
            import shutil
            shutil.copy(local_ttsmp3_tempfile, mp3_file)
            logger.info(f'mp3File = {mp3_file}')
            QMessageBox.information(self, '保存成功', f'已将图片文字朗读保存到“{mp3_file}“', QMessageBox.Ok, QMessageBox.Ok)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    root = MainWindow()
    sys.exit(app.exec_())
