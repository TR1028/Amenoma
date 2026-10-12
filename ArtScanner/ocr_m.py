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

class Config:
    name_coords = [33, 8, 619, 69]
    type_coords = [32, 89, 350, 134]
    star_coords = [30, 310, 350, 360]
    amount_coords = [10, 175, 150, 203]


def find_beige_strip(item_img, scale_ratio=1.0):
    """Locate the (y_top, y_bot) rows of the beige label strip in a material card image.

    Returns (y_top, y_bot) inclusive in item_img pixel coordinates, or None if not found.
    """
    arr = np.asarray(item_img)
    if arr.ndim < 3 or arr.shape[2] < 3:
        return None
    arr = arr[..., :3].astype(np.int32)
    h, w, _ = arr.shape
    y_start = max(0, min(h - 1, int(round(130 * scale_ratio))))
    x0 = max(0, min(w - 1, int(round(6 * scale_ratio))))
    x1 = max(x0 + 1, min(w, int(round(18 * scale_ratio))))
    strip = arr[y_start:, x0:x1, :]
    is_beige = (
        (np.abs(strip[:, :, 0] - 233) <= 12) &
        (np.abs(strip[:, :, 1] - 229) <= 12) &
        (np.abs(strip[:, :, 2] - 220) <= 12)
    ).any(axis=1)
    best_start, best_end = None, None
    cur_start = None
    for i, val in enumerate(is_beige):
        if val and cur_start is None:
            cur_start = i
        elif not val and cur_start is not None:
            if best_start is None or (i - cur_start) >= (best_end - best_start + 1):
                best_start, best_end = cur_start, i - 1
            cur_start = None
    if cur_start is not None:
        i = len(is_beige)
        if best_start is None or (i - cur_start) >= (best_end - best_start + 1):
            best_start, best_end = cur_start, i - 1
    min_run = max(5, int(round(15 * scale_ratio)))
    if best_start is not None and (best_end - best_start + 1) >= min_run:
        return y_start + best_start, y_start + best_end
    return None


def is_empty_material_slot(item_img, scale_ratio=1.0):
    """Return True when item_img is an empty inventory slot, see layout.is_empty_material_slot."""
    import layout
    return layout.is_empty_material_slot(item_img, scale_ratio)


def crop_amount(item_img, scale_ratio=1.0):
    """Crop the material count region dynamically from the card's beige label strip."""
    arr = np.asarray(item_img)
    if arr.ndim >= 3 and arr.shape[2] >= 3:
        rgb = arr[..., :3]
        h, w, _ = rgb.shape
        bounds = find_beige_strip(item_img, scale_ratio)
        if bounds is not None:
            y_top, y_bot = bounds
            c_top = min(h - 1, y_top + int(round(6 * scale_ratio)))
            c_bot = min(h, y_bot + 1, y_top + int(round(32 * scale_ratio)))
            if c_bot - c_top >= max(4, int(round(12 * scale_ratio))):
                x0 = max(0, int(round(10 * scale_ratio)))
                x1 = min(w, int(round(150 * scale_ratio)))
                crop_arr = rgb[c_top:c_bot, x0:x1]
                # If the crop was clipped at the bottom edge of item_img and digit ink touches
                # the bottom row, extend the last row slightly so vertical stems (e.g. '4') stay intact.
                if c_bot >= h and (c_bot - c_top) < int(round(25 * scale_ratio)):
                    g_last = crop_arr[-1, :, :3].astype(np.float32) @ [0.299, 0.587, 0.114]
                    if float(g_last.min()) < 150.0:
                        pad_n = max(1, int(round(2 * scale_ratio)))
                        crop_arr = np.concatenate(
                            [crop_arr, np.repeat(crop_arr[-1:, :, :], pad_n, axis=0)], axis=0
                        )
                return Image.fromarray(crop_arr)
    return item_img.crop([i * scale_ratio for i in Config.amount_coords])


def mask_amount_padding(pred, preprocessed, blank_index, min_keep_prob=0.9):
    """Remove the trailing echo material_model_7.1.0 sometimes adds to amounts ('11' -> '111').

    The model emits the k-th digit around CTC timestep 2k, which on real amounts was always before
    ceil(ink_width / 4) (ink_width = inked columns of the 16 px high model input).  The echo is a
    low-confidence repeat of the last digit at/after that point (p 0.58-0.72, real digits >= 0.86).
    Late emissions are blanked only when they are also unsure, so a real last digit that lands
    late is kept.

    Must NOT be applied to the name row: the BiLSTM emits (CJK) name characters far right of the
    ink, blanking there drops the last characters of the name.

    :param pred: (timesteps, classes) prediction of one row, modified in place
    :param preprocessed: the (height, width) model input of that row
    """
    preprocessed = np.asarray(preprocessed)
    timesteps = pred.shape[0]
    if timesteps < 1 or preprocessed.ndim != 2:
        return pred
    cols = np.flatnonzero(preprocessed.max(axis=0) > 0)
    ink_w = int(cols[-1]) + 1 if len(cols) else 1
    for t in range(max(1, (ink_w + 3) // 4), timesteps):
        c = int(np.argmax(pred[t]))
        if c != blank_index and float(pred[t, c]) < min_keep_prob:
            pred[t, :] = 0.0
            pred[t, blank_index] = 1.0
    return pred


class OCR:
    def __init__(self, model, scale_ratio=1):
        self.scale_ratio = scale_ratio
        self.model = tensorflow.keras.models.load_model(model)
        # Vocabulary pinned per model (rcc/models_CHS/charset_chs_mat.json or <model>.charset.json)
        # instead of being rebuilt from material_names_v*.json, see charset_util.
        self.charset = charset_util.load_charset_for_model(self.model, model, lang="CHS", kind="material")
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
        if 'amount' in keys:
            amt_idx = keys.index('amount')
            mask_amount_padding(raw_pred[amt_idx], preprocessed[amt_idx], self.charset.blank_index)
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
                )
                / 255
        )

    def pad_to_width(self, img):
        width = self.width
        if img.shape[1] >= width:
            return img[:, :width]
        return np.pad(
            img, [[0, 0], [0, width - img.shape[1]]], mode="constant", constant_values=0
        )

    def preprocess(self, text_img):
        result = self.to_gray(text_img)
        result = self.normalize(result, True)
        result = self.binarization(result)
        result = self.crop(result)
        result = self.normalize(result, False)
        result = self.resize_to_height(result)
        result = self.pad_to_width(result)
        return result

    def decode(self, pred):
        # Greedy CTC decoding through the model's pinned charset (see ocr.OCR.decode).
        return self.charset.greedy_decode(np.asarray(pred))
