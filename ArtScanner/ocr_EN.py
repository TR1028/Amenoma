import os

import tensorflow.keras

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from PIL import Image
import logging
from tensorflow import get_logger
import tensorflow.keras.models

import charset_util
import panel_detect

get_logger().setLevel(logging.ERROR)


# class OCR:
#     def __init__(self, model_path='mn_model.h5', scale_ratio=1):
#         pass

class Config_EN:
    name_coords = [33, 4, 619, 72]
    type_coords = [32, 89, 350, 134]
    main_attr_name_coords = [35, 200, 350, 240]
    main_attr_value_coords = [35, 240, 350, 300]
    star_coords = [30, 310, 350, 360]
    level_coords = [43, 414, 112, 444]
    subattr_1_coords = [67, 480, 560, 520]
    subattr_2_coords = [67, 532, 560, 572]
    subattr_3_coords = [67, 584, 560, 624]
    subattr_4_coords = [67, 636, 560, 676]
    equipped_coords = [105, 1060, 500, 1100]
    # 5.2+: lock button moved left, Astral Mark button at the old lock position (see ocr.Config).
    lock_coords = [450, 385, 575, 475]
    astral_mark_coords = [525, 385, 650, 475]


class OCR:
    def __init__(self, model, scale_ratio=1):
        self.scale_ratio = scale_ratio
        self.model = tensorflow.keras.models.load_model(model)
        # Vocabulary pinned per model (rcc/models_EN/charset_en_art.json or <model>.charset.json),
        # see charset_util / ocr.OCR.
        self.charset = charset_util.load_charset_for_model(self.model, model, lang="EN", kind="artifact")
        self.characters = self.charset.characters
        # 6.0 grey "(unactivated)" substat lines are only read by models that know the marker
        self.reads_unactivated = charset_util.can_read_unactivated(self.characters, "EN")
        if not self.reads_unactivated:
            logging.getLogger().info("OCR model vocabulary has no '(unactivated)': grey (unactivated) "
                                     "substat lines are skipped")

        self.width = self.charset.width
        self.height = self.charset.height
        self.max_length = self.charset.max_length
        # self.build_model(input_shape=(self.width, self.height))
        # self.model.load_weights(model_weight)

    def setScaleRatio(self, scaleRatio):
        self.scale_ratio = scaleRatio

    def detect_info(self, art_img):
        info, unactivated = self.analyze_panel(art_img)
        keys = sorted(info.keys())
        x = np.concatenate([
            self.preprocess(info[key]).T[None, :, :, None]
            for key in keys], axis=0)
        y = self.model.predict(x)
        y = self.decode(y)
        astral_mark = self.detect_astral_mark(art_img)
        return {**{key: v for key, v in zip(keys, y)},
                **{'star': self.detect_star(art_img),
                   # marking an artifact with the Astral Mark always locks it
                   'locked': self.detect_lock(art_img) or astral_mark,
                   'astral_mark': astral_mark,
                   # substat keys whose line is the grey 6.0 "(unactivated)" substat
                   'unactivated': unactivated}}

    def extract_art_info_EN(self, art_img):
        return self.analyze_panel(art_img)[0]

    # same name as ocr.OCR, used by the debug dump
    extract_art_info = extract_art_info_EN

    def analyze_panel(self, art_img):
        """Crop the text fields of the detail panel.

        :return: (crops, unactivated) - crops: dict key -> PIL image of every field to OCR;
                 unactivated: list of 'subattr_N' keys shown as unactivated (6.0+, grey text).  Their
                 crop is only included if the model can read the "(unactivated)" marker.
        """
        s = self.scale_ratio
        crops = {
            'name': art_img.crop([i * s for i in Config_EN.name_coords]),
            'type': art_img.crop([i * s for i in Config_EN.type_coords]),
            'main_attr_name': art_img.crop([i * s for i in Config_EN.main_attr_name_coords]),
            'main_attr_value': art_img.crop([i * s for i in Config_EN.main_attr_value_coords]),
            'level': art_img.crop([i * s for i in Config_EN.level_coords]),
            'equipped': art_img.crop([i * s for i in Config_EN.equipped_coords]),
        }
        lines = [art_img.crop([i * s for i in coords]) for coords in (  # text colour [73, 83, 102]
            Config_EN.subattr_1_coords, Config_EN.subattr_2_coords,
            Config_EN.subattr_3_coords, Config_EN.subattr_4_coords)]
        present, unactivated = panel_detect.split_substat_lines(lines, s)
        for number in present:
            if number in unactivated and not self.reads_unactivated:
                continue  # dropped like in v3.4: the old model misreads the marker
            crops['subattr_%d' % number] = lines[number - 1]
        return crops, ['subattr_%d' % number for number in unactivated]

    def detect_star(self, art_img):
        star = art_img.crop([i * self.scale_ratio for i in Config_EN.star_coords])
        cropped_star = self.crop(self.normalize(self.to_gray(star), auto_inverse=False))
        coef = cropped_star.shape[1] / cropped_star.shape[0]
        coef = coef / 1.30882352 + 0.21568627
        return round(coef)

    def detect_lock(self, img) -> bool:
        return panel_detect.detect_lock(img, Config_EN.lock_coords, self.scale_ratio)

    def detect_astral_mark(self, img) -> bool:
        return panel_detect.detect_astral_mark(img, Config_EN.astral_mark_coords, self.scale_ratio)

    def to_gray(self, text_img):
        text_img = np.array(text_img)
        if len(text_img.shape) > 2:
            text_img = (text_img[..., :3] @ [[[0.299], [0.587], [0.114]]])[:, :, 0]
        return np.array(text_img, np.float32)

    def normalize(self, img, auto_inverse=True):
        img -= img.min()
        img /= img.max()
        if auto_inverse and img[-1, -1] > 0.5:
            img = 1 - img
        return img

    def binarization(self, img, thresh=0.5):
        return np.where((img < thresh), 0, img)

    def resplice(self, img):
        # Must stay identical to model_trainer's resplice (train/inference preprocessing match).
        line_height = 27
        if (img[30] == 0.).all():
            res1 = img[:line_height, :]
            res2 = img[-line_height:, :]
            space = np.zeros((res1.shape[0], 10))
            res = np.concatenate((res1, space, res2), axis=1)
            return self.crop(res, tol=0)
        # Two-line names whose first line has descenders (e.g. "Moment That Ceased Upon Waking" /
        # "From Grand Dreams"): the empty gap row is not row 30, search rows 20-40 instead.
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

    def zoom(self, img):
        height_ = 60
        return (
                np.array(
                    Image.fromarray(np.uint8(img * 255)).resize(
                        (int(img.shape[1] * height_ / img.shape[0]), height_),
                        Image.BILINEAR, )
                ) / 255)

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
        return (
                np.array(
                    Image.fromarray(np.uint8(img * 255)).resize(
                        (int(img.shape[1] * height / img.shape[0]), height),
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
        result = self.crop(result, tol=0)

        result = self.zoom(result)
        result = self.resplice(result)

        result = self.resize_to_height(result)
        result = self.pad_to_width(result)
        return result

    def decode(self, pred):
        # Greedy CTC decoding through the model's pinned charset (see ocr.OCR.decode).
        return self.charset.greedy_decode(np.asarray(pred))

    # def build_model(self, input_shape):
    #     input_img = Input(
    #         shape=(input_shape[0], input_shape[1], 1), name="image", dtype="float32"
    #     )
    #     mobilenet = MobileNetV3_Small(
    #         (input_shape[0], input_shape[1], 1), 0, alpha=1.0, include_top=False
    #     ).build()
    #     x = mobilenet(input_img)
    #     new_shape = ((input_shape[0] // 8), (input_shape[1] // 8) * 576)
    #     x = Reshape(target_shape=new_shape, name="reshape")(x)
    #     x = Dense(64, activation="relu", name="dense1")(x)
    #     x = Dropout(0.2)(x)
    #
    #     # RNNs
    #     x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.25))(x)
    #     x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.25))(x)
    #
    #     # Output layer
    #     output = Dense(len(self.characters) + 2, activation="softmax", name="dense2")(x)
    #
    #     # Define the model
    #     self.model = Model(inputs=[input_img], outputs=output, name="ocr_model_v1")
