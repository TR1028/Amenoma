import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from PIL import Image
import logging
from tensorflow import get_logger
import tensorflow.keras.models

import charset_util

get_logger().setLevel(logging.ERROR)


# class OCR:
#     def __init__(self, model_path='mn_model.h5', scale_ratio=1):
#         pass

from ocr_m import find_beige_strip, is_empty_material_slot, crop_amount


class Config:
    name_coords = [33, 8, 619, 69]
    type_coords = [32, 89, 350, 134]
    star_coords = [30, 310, 350, 360]
    amount_coords = [10, 175, 150, 203]


class OCR:
    def __init__(self, model, scale_ratio=1):
        self.scale_ratio = scale_ratio
        self.model = tensorflow.keras.models.load_model(model)
        # Vocabulary pinned per model (rcc/models_EN/charset_en_mat.json or <model>.charset.json)
        # instead of being rebuilt from material_names_v*.json, see charset_util.
        self.charset = charset_util.load_charset_for_model(self.model, model, lang="EN", kind="material")
        self.characters = self.charset.characters

        self.width = self.charset.width
        self.height = self.charset.height
        self.max_length = self.charset.max_length
        # self.build_model(input_shape=(self.width, self.height))
        # self.model.load_weights(model_weight)

    def setScaleRatio(self, scaleRatio):
        self.scale_ratio = scaleRatio

    def detect_info(self, detail_img, item_img):
        info = self.extract_art_info(detail_img, item_img)
        keys = sorted(info.keys())
        preprocessed = [self.preprocess(info[key]) for key in keys]
        x = np.concatenate([p.T[None, :, :, None] for p in preprocessed], axis=0)
        raw_pred = np.array(self.model.predict(x, verbose=0), copy=True)
        blank_idx = self.charset.blank_index
        for i, p in enumerate(preprocessed):
            nz = np.flatnonzero(p.max(axis=0) > 0)
            nz_w = int(nz[-1]) + 1 if len(nz) else 1
            max_t = max(1, (nz_w + 3) // 4)
            if max_t < raw_pred.shape[1]:
                raw_pred[i, max_t:, :] = 0.0
                raw_pred[i, max_t:, blank_idx] = 1.0
        y = self.decode(raw_pred)
        res = {key: v for key, v in zip(keys, y)}
        if 'amount' in res and (not res['amount'] or not res['amount'].isdigit()):
            amt_idx = keys.index('amount')
            digit_decoded = self.decode_digits(raw_pred[amt_idx:amt_idx + 1])
            if digit_decoded and digit_decoded[0]:
                res['amount'] = digit_decoded[0]
        res['star'] = self.detect_star(detail_img)
        return res

    def decode_digits(self, pred):
        pred = np.asarray(pred)
        allowed = [i + 1 for i, c in enumerate(self.characters) if c in "0123456789"]
        if not allowed:
            return self.decode(pred)
        allowed.append(self.charset.blank_index)
        mask = np.zeros(pred.shape[-1], dtype=pred.dtype)
        mask[allowed] = 1
        return self.charset.greedy_decode(pred * mask)

    def extract_art_info(self, material_img, item_img):
        name = material_img.crop([i * self.scale_ratio for i in Config.name_coords])
        amount = crop_amount(item_img, self.scale_ratio)
        return {key: value for key, value in locals().items() if key not in ['material_img', 'item_img', 'self']}

    def detect_star(self, material_img):
        star = material_img.crop([i * self.scale_ratio for i in Config.star_coords])
        cropped_star = self.crop(self.normalize(self.to_gray(star), auto_inverse=False))
        coef = cropped_star.shape[1] / max(1, cropped_star.shape[0])
        coef = coef / 1.30882352 + 0.21568627
        return round(coef)

    def to_gray(self, text_img):
        text_img = np.array(text_img)
        if len(text_img.shape) > 2:
            text_img = (text_img[..., :3] @ [[[0.299], [0.587], [0.114]]])[:, :, 0]
        return np.array(text_img, np.float32)

    def normalize(self, img, auto_inverse=True):
        img -= img.min()
        denom = img.max()
        if denom > 0:
            img /= denom
        if auto_inverse and img[-1, -1] > 0.5:
            img = 1 - img
        return img

    def resplice(self, img):
        # Must stay identical to model_trainer's resplice (train/inference preprocessing match).
        line_height = 27
        if (img[30] == 0.).all():
            res1 = img[:line_height, :]
            res2 = img[-line_height:, :]
            space = np.zeros((res1.shape[0], 10))
            res = np.concatenate((res1, space, res2), axis=1)
            return self.crop(res, tol=0)
        # Two-line names whose first line has descenders: the empty gap row is not row 30, search
        # rows 20-40 instead.
        empty = np.flatnonzero((img[20:41] == 0.).all(axis=1))
        if len(empty) == 0:
            return img
        gap = 20 + int(empty[np.argmin(np.abs(empty - 10))])
        top = self.crop(img[:gap, :], tol=0)
        bottom = self.crop(img[gap:, :], tol=0)
        h = max(top.shape[0], bottom.shape[0])
        top = np.pad(top, [[0, h - top.shape[0]], [0, 0]], mode='constant')
        bottom = np.pad(bottom, [[0, h - bottom.shape[0]], [0, 0]], mode='constant')
        return self.crop(np.concatenate((top, np.zeros((h, 10)), bottom), axis=1), tol=0)

    def binarization(self, img, thresh=0.5):
        return np.where((img < thresh), 0, img)

    def crop(self, img, tol=0.7):
        # img is 2D image data
        # tol  is tolerance
        mask = img > tol
        m, n = img.shape
        mask0, mask1 = mask.any(0), mask.any(1)
        col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
        row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
        #     print(row_end-row_start, col_end-col_start)
        return img[row_start:row_end, col_start:col_end]

    def resize_to_height(self, img):
        height = self.height
        h0 = max(1, img.shape[0])
        w_out = max(1, int(img.shape[1] * height / h0))
        return (
                np.array(
                    Image.fromarray(np.uint8(img * 255)).resize(
                        (w_out, height),
                        Image.BILINEAR,
                    )
                ) / 255
        )

    def pad_to_width(self, img):
        width = self.width
        if img.shape[1] >= width:
            return img[:, :width]
        return np.pad(
            img, [[0, 0], [0, width - img.shape[1]]], mode="constant", constant_values=0
        )

    def zoom(self, img):
        height_ = 60
        h0 = max(1, img.shape[0])
        w_out = max(1, int(img.shape[1] * height_ / h0))
        return (np.array(
                    Image.fromarray(np.uint8(img * 255)).resize(
                        (w_out, height_),
                        Image.BILINEAR, )) / 255)

    def preprocess(self, text_img):
        result = self.to_gray(text_img)
        result = self.normalize(result, True)
        result = self.binarization(result)
        result = self.crop(result, tol=0)
        result = self.zoom(result)
        result = self.resplice(result)
        result = self.resize_to_height(result)
        result = self.pad_to_width(result)
        return result

    def decode(self, pred):
        # Greedy CTC decoding through the model's pinned charset (see ocr.OCR.decode).
        return self.charset.greedy_decode(np.asarray(pred))
