# -*- coding: utf-8 -*-
"""Train the CHS material OCR model -> rcc/models_CHS/material_model_<version>.h5

7.x update (see README.md):
* shared data generation / preprocessing / CTC code (trainer_common.py,
  trainer_tf.py): multiprocess data workers, graph-mode training, no
  deprecated NumPy aliases, TF 2.3 and current TF (tf_keras) supported
* the training vocabulary is written to charset_chs_mat.json next to the
  checkpoints and next to the exported .h5 (+ <model>.charset.json)
* input width 240 -> 320 and max_length 15 -> 20: 7.x names reach 12
  characters / 214 px ("切片辖域管控模块·零质点"), close to the old 240 px

Run from model_trainer/ (or anywhere):
    python train_model_materials.py --warm-start        # fine-tune the shipped model (recommended)
    python train_model_materials.py                     # historical full schedule from scratch
    python train_model_materials.py --schedule smoke    # 2 training steps + export (pipeline check)
    python train_model_materials.py --charset-only      # vocabulary JSON + geometry report, no TF
    python train_model_materials.py --help
"""
import trainer_tf

SPEC = dict(
    lang='chs', kind='mat',
    width=320, height=16, max_length=20,
    model_name='ocr_model_v1',
    checkpoint='materials-weights-improvement-{epoch:02d}-{ctc_accu:.3f}.hdf5',
    min_lr=1e-7,
    warmup=((8192, 1), (4096, 2), (2048, 4), (1024, 4)),
    main=(512, 512),
    script='train_model_materials.py',
)

if __name__ == '__main__':
    raise SystemExit(trainer_tf.main(SPEC))
