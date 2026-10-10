# -*- coding: utf-8 -*-
"""Train the CHS artifact OCR model -> rcc/models_CHS/artifact_model_<version>.h5

7.x update (see README.md):
* data generation / preprocessing / CTC code moved to trainer_common.py and
  trainer_tf.py (shared by the four trainers, multiprocess data workers,
  graph-mode training instead of run_eagerly, no deprecated NumPy aliases,
  runs on TF 2.3 as well as on current TF with tf_keras)
* the training vocabulary is written to charset_chs_art.json next to the
  checkpoints and next to the exported .h5 (+ <model>.charset.json), so
  ArtScanner no longer rebuilds it from the data tables
* the vocabulary and the synthetic panels include the 6.0 unactivated 4th
  substat ("暴击率+3.9%（待激活）", drawn in grey)
* input width 240 -> 320 and max_length 15 -> 24: "元素充能效率+38.9%（待激活）"
  renders 240 px wide and has 17 characters

Run from model_trainer/ (or anywhere):
    python train_model_artifact.py --warm-start        # fine-tune the shipped model (recommended)
    python train_model_artifact.py                     # historical full schedule from scratch
    python train_model_artifact.py --schedule smoke    # 2 training steps + export (pipeline check)
    python train_model_artifact.py --charset-only      # vocabulary JSON + geometry report, no TF
    python train_model_artifact.py --evaluate ../ArtScanner/rcc/models_CHS/artifact_model_3.4.0.h5
    python train_model_artifact.py --help
"""
import numpy as np

import trainer_common as tc
import trainer_tf

SPEC = dict(
    lang='chs', kind='art',
    width=320, height=16, max_length=24,
    model_name='ocr_model_v1',
    checkpoint='artifact-weights-improvement-{epoch:02d}-{ctc_accu:.3f}.hdf5',
    min_lr=1e-6,
    warmup=((8192, 1), (4096, 2), (2048, 4), (1024, 8)),
    main=(512, 512),
    scale_ratio=1,
    script='train_model_artifact.py',
)


# test functions (--predict-image)
class Config:
    name_coords = [33, 8, 619, 69]
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
    lock_coords = [570, 405, 620, 455]


def extract_art_info(art_img, scale_ratio=1):
    """Crop the text fields of an artifact detail panel (historical test helper,
    ``np.float`` -> ``float``).  Grey 6.0 unactivated substat lines are kept (the
    model reads them with their "（待激活）" marker) when ArtScanner's
    panel_detect is available."""
    info = {}
    for key in ('name', 'type', 'main_attr_name', 'main_attr_value', 'level', 'equipped'):
        info[key] = art_img.crop([i * scale_ratio for i in getattr(Config, key + '_coords')])
    subs = [art_img.crop([i * scale_ratio for i in getattr(Config, 'subattr_{}_coords'.format(n))])  # [73, 83, 102]
            for n in range(1, 5)]
    try:
        from ArtScanner import panel_detect
        present, _unactivated = panel_detect.split_substat_lines(subs, scale_ratio)
    except ImportError:
        present = []
        for n, sub in enumerate(subs, 1):
            if np.all(np.abs(np.array(sub, float) - [[[73, 83, 102]]]).max(axis=-1) > 25):
                break
            present.append(n)
    for n in present:
        info['subattr_{}'.format(n)] = subs[n - 1]
    return info


def detect_star(art_img, scale_ratio=1):
    star = art_img.crop([i * scale_ratio for i in Config.star_coords])
    cropped_star = tc.crop(tc.normalize(tc.to_gray(star)))
    coef = cropped_star.shape[1] / cropped_star.shape[0]
    coef = coef / 1.30882352 + 0.21568627
    return round(coef)


def detect_lock(img, scale_ratio=1) -> bool:
    lock = img.crop([i * scale_ratio for i in Config.lock_coords])
    result = tc.to_gray(lock)
    result = tc.normalize(result, auto_inverse=False)
    result = np.where((result < 0.5), 0, 1)
    return bool(np.add.reduce(np.add.reduce(result)) < 500)


if __name__ == '__main__':
    raise SystemExit(trainer_tf.main(SPEC, extract_fields=extract_art_info))
