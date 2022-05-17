import argparse
import time
from model import OcrHandle
import tornado.web
import tornado.gen
import tornado.httpserver
import base64
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import datetime
import json
import cv2
import logging
import math
import numpy as np
import re
logger = logging.getLogger(__name__)

import os

_root = os.path.dirname(__file__)


def get_config():
    with open("config12.json", "r", encoding='utf8') as jsonfile:
        data = json.load(jsonfile)
        logger.info(f"Read config.json successfully:\n{data}\n")
        jsonfile.close()
    return data


def get_args():
    config = get_config()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vis_font_path", type=str, default="simfang.ttf")
    args = parser.parse_args()
    args.__dict__.update(config)
    logger.info(f"OCR args:\n{json.dumps(args.__dict__, indent=4)}\n")
    return args


def check_and_read_gif(img_path):
    if os.path.basename(img_path)[-3:] in ['gif', 'GIF']:
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
        from utils import pil2cv
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
        from utils import cv2pil
        img = cv2pil(img)
    if img is None:
        logger.debug("error in loading image:{}".format(image_file))
    return img


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


def ocr(imgfile):
    now_time = time.strftime("%Y-%m-%d", time.localtime(time.time()))
    # from config import max_post_time, dbnet_max_size, white_ips

    args = get_args()

    img = read_pil(imgfile)
    img_w, img_h = img.size
    print('image_size:', img_w, img_h)
    # short_size = args.dbnet_max_size // max(img_w, img_h) * min(img_w, img_h) // 32 * 32
    # short_size = min(img_w, img_h) // 32 * 32
    short_size = 256 * 8
    print('short_size', short_size)

    ocrhandle = OcrHandle(args)

    res = ocrhandle.text_predict(img, short_size)
    boxes, rec_res = [], []
    for i, r in enumerate(res):
        rect, txt, confidence = r
        boxes.append(rect)
        rec_res.append([txt, confidence])
    from utils import pil2cv
    vimg = vis_img(pil2cv(img), boxes, rec_res, args)
    save_img(vimg, imgfile, args)
    return boxes, rec_res

from LAC import LAC


class TextExtractor(object):

    @staticmethod
    def is_plate(text):
        regex = r'([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领A-Z]{1}[a-zA-Z](([DF]((?![IO])[a-zA-Z0-9](?![IO]))[0-9]{4})|([0-9]{5}[DF]))|[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领A-Z]{1}[A-Z]{1}[A-Z0-9]{4}[A-Z0-9挂学警港澳]{1})'
        m = re.search(regex, text)
        if not m:
            return None
        return m.group(1)

    @staticmethod
    def is_id_number(text):
        pattern = re.compile(r'[\u4e00-\u9fa5]')
        text = re.sub(pattern, "", text)
        if text is None:
            return None
        else:
            if len(text) != 18 and len(text) != 15:
                return None

            regularExpression = "(^[1-9]\\d{5}(18|19|20)\\d{2}((0[1-9])|(10|11|12))(([0-2][1-9])|10|20|30|31)\\d{3}[0-9Xx]$)|" \
                                "(^[1-9]\\d{5}\\d{2}((0[1-9])|(10|11|12))(([0-2][1-9])|10|20|30|31)\\d{3}$)"

            if re.match(regularExpression, text):
                if len(text) == 18:
                    n = text.upper()
                    # 前十七位加权因子
                    var = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
                    # 这是除以11后，可能产生的11位余数对应的验证码
                    var_id = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']

                    sum = 0
                    for i in range(0, 17):
                        sum += int(n[i]) * var[i]
                    sum %= 11
                    if (var_id[sum]) != str(n[17]):
                        # print("身份证号规则核验失败，校验码应为", var_id[sum], "，当前校验码是：", n[17])
                        return None
                return text
            else:
                return None

    @staticmethod
    def is_address(text=""):
        import re
        PATTERN1 = r'([\u4e00-\u9fa5]{2,5}?(?:省|自治区|市)){0,1}([\u4e00-\u9fa5]{2,7}?(?:区|县|州)){0,1}([\u4e00-\u9fa5]{2,7}?(?:镇)){0,1}([\u4e00-\u9fa5]{2,7}?(?:村|街|街道)){0,1}([\d]{1,3}?(号)){0,1}'
        pattern = re.compile(PATTERN1)
        m = pattern.search(text).group()
        if not m:
            return None
        else:
            text = text.strip()
            text = re.sub(r'^住址|^地址|^.址|^址', '', text)
            return text

    @staticmethod
    def match_name(text):
        # 先用正则判断
        pattern = r'^所有人|^所.人|^姓名|^.名|^名'
        str1 = re.findall(pattern, text)
        if str1:
            name = re.sub(pattern, '', text)
            return name
        else:
            user_name_list = []
            lac = LAC(mode="lac")
            lac_result = lac.run(text)
            my_re = re.compile(r'[A-Za-z]')
            for index, lac_label in enumerate(lac_result[1]):
                if lac_label == "PER":
                    name_tmp = lac_result[0][index]
                    res = re.findall(my_re, name_tmp)
                    if not res:
                        user_name_list.append(name_tmp)

            if len(user_name_list) == 0:
                return None
            else:
                return max(user_name_list)

    @staticmethod
    def find_best(texts):
        record = {
            'id': '',
            'name': '',
            'address': '',
            'plate': ''
        }
        ids, names, addresses, plates = [], [], [], []
        for text in texts:
            id = TextExtractor.is_id_number(text)
            ad = TextExtractor.is_address(text)
            nm = TextExtractor.match_name(text)
            pt = TextExtractor.is_plate(text)
            if id is not None:
                ids.append(id)
            if ad is not None:
                addresses.append(ad)
            if nm is not None:
                names.append(nm)
            if pt is not None:
                plates.append(pt)

        if len(ids) > 0:
            record['id'] = max(ids)
        if len(names) > 0:
            record['name'] = max(names)
        if len(addresses) > 0:
            record['address'] = max(addresses)
        if len(plates) > 0:
            record['plate'] = max(plates)
        return record



if __name__ == "__main__":
    root = '../TMP'

    records = []
    for file in os.listdir(root):
        if file.lower().endswith(('.bmp', '.png')):
            print('image file', file)
            record = {'image': file}

            records.append(record)
            boxes, rec_res = ocr(os.path.join(root, file))

            record.update(TextExtractor.find_best([txt for txt, prob in rec_res]))
    csv_file = '双证识别结果.csv'
    import csv
    with open(csv_file, "w", encoding='utf-8-sig', newline='') as csvFile:
        logger.info(f'csvFile = {csvFile}')
        wr = csv.writer(csvFile, quotechar=',')
        titles = ['图片', '姓名', '身份证', '车牌', '地址', '所属文件夹']
        wr.writerow(titles)
        for record in records:
            wr.writerow([os.path.basename(record['image']),
                         record.get('name', ''),
                         record.get('id', ''),
                         record.get('plate', ''),
                         record.get('address', ''),
                         os.path.dirname(record['image'])])