# -*- coding: utf-8 -*-
"""TensorFlow side of the Amenoma OCR trainers.

Model, CTC loss / accuracy, training schedules, warm start from an older model,
evaluation and export.  Used by the four ``train_model_*.py`` scripts (each one
only defines a ``SPEC``) and by ``converters/model_converter*.py``.

Supported environments
----------------------
* Python 3.8 + TF 2.3 (the ArtScanner runtime): models exported here load
  directly in ArtScanner.  Recommended for the final export.
* TF 2.4 - 2.15 (``tf.keras`` is Keras 2).
* TF >= 2.16 with the ``tf_keras`` package: ``TF_USE_LEGACY_KERAS=1`` is set
  before TensorFlow is imported.  Keras 3 is not supported (mobilenetv3.py
  uses Keras-2 backend ops, Keras 3 changed ``hard_sigmoid`` and removed
  ``ctc_batch_cost``).

A full ``.h5`` model written by TF >= 2.4 contains ``TFOpLambda`` layers that
TF 2.3 cannot deserialize.  The weights are interchangeable, so re-export such
models for ArtScanner with ``converters/model_converter.py`` under the
ArtScanner runtime (``--weights output/weights/<model>.weights.npz``).

Nothing in this module imports TensorFlow at import time: the data workers are
``spawn``-ed processes that re-import the training script.
"""
from __future__ import print_function

import argparse
import datetime
import glob
import os
import re
import sys
import time

import numpy as np

import trainer_common as tc

TRAINER_DIR = tc.TRAINER_DIR
DEFAULT_TRAIN_DIR = os.path.join(TRAINER_DIR, 'train')
DEFAULT_OUT_DIR = os.path.join(TRAINER_DIR, 'output')

RUNTIME_BASENAME = {('chs', 'art'): 'artifact_model', ('en', 'art'): 'artifact_model_EN',
                    ('chs', 'mat'): 'material_model', ('en', 'mat'): 'material_model_EN'}
MODEL_SUBDIR = {'chs': 'models_CHS', 'en': 'models_EN'}

SCHEDULES = ('full', 'finetune', 'smoke')


def log(*args):
    print('[trainer]', *args)
    sys.stdout.flush()


def target_version():
    import model_info
    return model_info.MODEL_TARGET_VERSION


def runtime_model_filename(spec, version):
    """File name ArtScanner loads: ``rcc/models_<LANG>/<this>``."""
    return '{}_{}.h5'.format(RUNTIME_BASENAME[(spec['lang'], spec['kind'])], version)


# --------------------------------------------------------------------------- TensorFlow
_TF = {}


def _keras_major(tf):
    keras = tf.keras
    name = getattr(keras, '__name__', '') or ''
    if 'tf_keras' in name:
        return 2
    version = getattr(keras, '__version__', None)
    if version:
        return int(str(version).split('.')[0])
    return 3 if name.startswith('keras') else 2


def import_tf():
    """Import TensorFlow with Keras 2 semantics; returns ``(tf, keras)``."""
    if _TF:
        return _TF['tf'], _TF['keras']
    os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')
    import tensorflow as tf
    try:
        keras = tf.keras
        major = _keras_major(tf)
    except Exception as e:  # TF >= 2.16 without tf_keras
        raise SystemExit('TensorFlow {} needs the tf_keras package (pip install "tf_keras~={}"): {}'.format(
            tf.__version__, '.'.join(tf.__version__.split('.')[:2]), e))
    if major >= 3:
        raise SystemExit('tf.keras is Keras 3 (TensorFlow {}). These trainers need Keras 2: pip install tf_keras '
                         'and set TF_USE_LEGACY_KERAS=1 (or use the Python 3.8 / TF 2.3 environment).'.format(
                             tf.__version__))
    _TF['tf'], _TF['keras'] = tf, keras
    return tf, keras


def tf_runtime_compatible():
    """True when full .h5 models written by this TF load in ArtScanner's TF 2.3."""
    tf, _ = import_tf()
    major, minor = [int(v) for v in tf.__version__.split('.')[:2]]
    return (major, minor) <= (2, 3)


def build_ocr_model(num_classes, width, height=16, name='ocr_model_v1'):
    """The historical CRNN (unchanged): MobileNetV3-Small without top ->
    Dense(96) -> 2x BiLSTM -> softmax over ``num_classes`` (= charset + 2).

    No weight depends on ``width`` (the time axis is ``width // 8``), so models
    of different input widths can share weights."""
    tf, keras = import_tf()
    from mobilenetv3 import MobileNetV3_Small  # imports tensorflow.keras: only after import_tf()

    if width % 8 or height % 8:
        raise ValueError('width and height must be multiples of 8, got {}x{}'.format(width, height))
    layers = keras.layers
    input_img = layers.Input(shape=(width, height, 1), name="image", dtype="float32")
    mobilenet = MobileNetV3_Small((width, height, 1), 0, alpha=1.0, include_top=False).build()
    x = mobilenet(input_img)
    new_shape = ((width // 8), (height // 8) * 576)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(96, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(192, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(96, return_sequences=True, dropout=0.25))(x)

    # Output layer
    output = layers.Dense(num_classes, activation="softmax", name="dense2")(x)
    return keras.models.Model(inputs=[input_img], outputs=output, name=name)


def make_ctc_loss():
    tf, keras = import_tf()

    @tf.autograph.experimental.do_not_convert
    def ctc_loss(y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True)

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        return keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

    return ctc_loss


def make_ctc_accuracy(codec, name='ctc_accu'):
    """Exact-match rate of the greedy CTC decoding (the historical ``ctc_accu``).

    The historical metric was NumPy code that forced ``run_eagerly=True``; the
    NumPy part now runs in ``tf.py_function`` so training runs in graph mode."""
    tf, keras = import_tf()

    def _count(y_true, y_pred):
        return np.float32(codec.count_correct(y_true.numpy(), y_pred.numpy()))

    class CTCAccuracy(keras.metrics.Metric):
        def __init__(self, name=name, **kwargs):
            super(CTCAccuracy, self).__init__(name=name, **kwargs)
            self.correct = self.add_weight(name='correct', initializer='zeros', dtype='float32')
            self.total = self.add_weight(name='total', initializer='zeros', dtype='float32')

        def update_state(self, y_true, y_pred, sample_weight=None):
            hits = tf.reshape(tf.py_function(_count, [y_true, y_pred], tf.float32), [])
            self.correct.assign_add(hits)
            self.total.assign_add(tf.cast(tf.shape(y_pred)[0], tf.float32))

        def result(self):
            return tf.math.divide_no_nan(self.correct, self.total)

    return CTCAccuracy(name)


# --------------------------------------------------------------------------- weights files
def weights_num_classes(path):
    """Output size (charset + 2) stored in a weights / model file, or None."""
    if path.endswith('.npz'):
        with np.load(path) as data:
            return int(data['arr_%d' % (len(data.files) - 1)].shape[0])
    import h5py
    found = []

    def visit(name, obj):
        parts = name.split('/')
        if isinstance(obj, h5py.Dataset) and 'dense2' in parts[:-1] and parts[-1].startswith('bias'):
            found.append(obj.shape)

    with h5py.File(path, 'r') as f:
        f.visititems(visit)
    return int(found[0][0]) if found else None


def save_weights_npz(model, path):
    """``model.get_weights()`` as ``arr_0..arr_N`` (portable across TF versions)."""
    folder = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(folder):
        os.makedirs(folder)
    np.savez(path, *model.get_weights())
    return path


def load_weights_any(model, path):
    """Load ``.npz`` (``get_weights`` order) or a Keras ``.h5/.hdf5`` file
    (weights-only checkpoint or full model) into ``model``."""
    if path.endswith('.npz'):
        with np.load(path) as data:
            arrays = [data['arr_%d' % i] for i in range(len(data.files))]
        expected = [tuple(w.shape) for w in model.get_weights()]
        got = [tuple(a.shape) for a in arrays]
        if expected != got:
            raise ValueError('{}: weight shapes do not match the model ({} vs {} arrays; first mismatch: {})'.format(
                path, len(got), len(expected), next(((a, b) for a, b in zip(got, expected) if a != b), None)))
        model.set_weights(arrays)
    else:
        model.load_weights(path)


def read_init_weights(path, num_classes, width, height):
    """Weights of ``path`` as NumPy arrays (built + loaded in a scratch model)."""
    tf, keras = import_tf()
    donor = build_ocr_model(num_classes, width, height)
    load_weights_any(donor, path)
    weights = donor.get_weights()
    del donor
    keras.backend.clear_session()
    return weights


def remap_output_layer(weights, old_charset, new_charset, seed=0):
    """Adapt ``dense2`` (last kernel/bias) from ``old_charset`` to ``new_charset``.

    Mask (column 0) and CTC blank (last column) are kept, characters in both
    charsets keep their trained column, new characters get a Glorot-uniform
    kernel column and the smallest old character bias (so they start unlikely
    and do not disturb the known characters).  All other layers are copied.
    """
    kernel, bias = weights[-2], weights[-1]
    if kernel.shape[1] != len(old_charset) + 2:
        raise ValueError('old charset has {} chars but the weights have {} classes'.format(
            len(old_charset), kernel.shape[1]))
    n_new = len(new_charset) + 2
    rng = np.random.RandomState(seed)
    limit = np.sqrt(6.0 / (kernel.shape[0] + n_new))
    new_kernel = rng.uniform(-limit, limit, size=(kernel.shape[0], n_new)).astype(kernel.dtype)
    char_bias = bias[1:-1]
    new_bias = np.full((n_new,), float(char_bias.min()) if len(char_bias) else 0.0, dtype=bias.dtype)
    new_kernel[:, 0], new_bias[0] = kernel[:, 0], bias[0]
    new_kernel[:, -1], new_bias[-1] = kernel[:, -1], bias[-1]
    old_index = dict((c, i + 1) for i, c in enumerate(old_charset))
    kept, added = 0, []
    for i, c in enumerate(new_charset):
        j = old_index.get(c)
        if j is None:
            added.append(c)
        else:
            new_kernel[:, i + 1], new_bias[i + 1] = kernel[:, j], bias[j]
            kept += 1
    dropped = sorted(set(old_charset) - set(new_charset))
    return list(weights[:-2]) + [new_kernel, new_bias], kept, added, dropped


def find_charset_json(weights_path, lang, kind, num_classes=None, explicit=None):
    """Charset JSON for a weights / model file (same lookup order as ArtScanner):
    ``explicit``, ``<stem>.charset.json``, ``charset_<lang>_<kind>.json`` next to it.
    Returns the loaded dict (with ``'path'``) or None."""
    stem = os.path.splitext(weights_path)[0]
    if stem.endswith('.weights'):
        stem = stem[:-len('.weights')]
    candidates = [explicit] if explicit else []
    candidates += [stem + '.charset.json', os.path.join(os.path.dirname(weights_path), tc.charset_filename(lang, kind))]
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        data = tc.load_charset_json(path)
        if num_classes is not None and len(data['charset']) + 2 != num_classes:
            if path == explicit:
                raise ValueError('{} has {} chars but {} has {} classes'.format(
                    path, len(data['charset']), weights_path, num_classes))
            continue
        data['path'] = path
        return data
    return None


def shipped_model(spec):
    """Newest shipped runtime model for this spec (ArtScanner/rcc/models_*/)."""
    folder = tc.SHIPPED_MODEL_DIRS[spec['lang']]
    base = RUNTIME_BASENAME[(spec['lang'], spec['kind'])]
    found = []
    for path in glob.glob(os.path.join(folder, base + '_*.h5')):
        version = os.path.basename(path)[len(base) + 1:-3]
        if re.match(r'^[0-9][0-9.]*$', version):
            found.append((tuple(int(v) for v in version.split('.') if v), path))
    if not found:
        raise SystemExit('no shipped model {}_<version>.h5 in {}'.format(base, folder))
    return sorted(found)[-1][1]


def list_checkpoints(spec, train_dir):
    """``[(accuracy, epoch, path)]`` of the checkpoints in ``train_dir`` written
    with the spec's pattern (the CHS pattern does not match EN checkpoints)."""
    prefix = spec['checkpoint'].split('{')[0]
    found = []
    for path in glob.glob(os.path.join(train_dir, prefix + '*.hdf5')):
        m = re.match(re.escape(prefix) + r'(\d+)-([0-9.]+)\.hdf5$', os.path.basename(path))
        if m:
            found.append((float(m.group(2)), int(m.group(1)), path))
    return found


def best_checkpoint(spec, train_dir, preferred=None):
    """``preferred`` if it exists in ``train_dir``, else the checkpoint with the
    highest accuracy (then epoch) matching the spec's checkpoint pattern."""
    if preferred and os.path.isfile(os.path.join(train_dir, preferred)):
        return os.path.join(train_dir, preferred)
    found = [(acc, epoch, os.path.getmtime(path), path) for acc, epoch, path in list_checkpoints(spec, train_dir)]
    return max(found)[-1] if found else None


# --------------------------------------------------------------------------- data / checks
def training_charset(spec):
    if spec['kind'] == 'art':
        strings = tc.artifact_charset_strings(spec['lang'])
    else:
        strings = tc.material_charset_strings(spec['lang'])
    return tc.build_charset(strings)


def extreme_labels(spec):
    if spec['kind'] == 'art':
        return tc.artifact_extreme_labels(spec['lang'])
    return tc.material_extreme_labels(spec['lang'])


def preflight(spec, charset, width, height, max_length, fonts, force=False):
    """Glyph coverage + geometry check of the current tables; aborts on problems."""
    primary_missing, fallback, uncovered = fonts.coverage(charset)
    log('font {}: {} chars, {} missing in the primary font{}'.format(
        os.path.basename(fonts.primary), len(charset), len(primary_missing),
        '' if not primary_missing else ' -> fallback ' + repr(sorted(set(fallback.values())))))
    if uncovered:
        raise SystemExit('no font has glyphs for {!r}: pass --fallback-font (e.g. a Noto Sans CJK / Microsoft '
                         'YaHei file) or set AMENOMA_FALLBACK_FONTS'.format(''.join(uncovered)))
    report = tc.geometry_report(extreme_labels(spec), spec['lang'], width, height, max_length, fonts,
                                two_line_threshold=35 if spec['lang'] == 'en' else None)
    log('geometry {}x{} max_length {}:\n{}'.format(width, height, max_length,
                                                    tc.format_geometry_report(report, width, max_length)))
    problems = report['too_long'] or report['ctc_overflow'] or report['too_wide']
    if problems and not force:
        raise SystemExit('labels do not fit the model geometry: increase --width / --max-length (or --force)')
    return report


def data_config(spec, charset, width, height, max_length, args):
    p_focus = float(getattr(args, 'focus', 0.0) or 0.0)
    known = None
    if p_focus > 0:
        path = getattr(args, 'focus_charset', None) or os.path.join(
            tc.SHIPPED_MODEL_DIRS[spec['lang']], tc.charset_filename(spec['lang'], spec['kind']))
        data = tc.load_charset_json(path)
        known = list(data['charset'] if isinstance(data, dict) else data)
        log('focus {}: strings with characters outside {} ({} chars), extra lines {}'.format(
            p_focus, path, len(known), getattr(args, 'extra_lines', 0)))
    return dict(lang=spec['lang'], kind=spec['kind'], charset=list(charset), max_length=max_length,
                width=width, height=height, font=args.font, fallback_fonts=args.fallback_font,
                samples_per_step=args.samples_per_step, p_unactivated=args.p_unactivated,
                p_focus=p_focus, known_chars=known, extra_lines=int(getattr(args, 'extra_lines', 0) or 0))


def charset_meta(spec, args, model_file=None, version=None, extra=None):
    meta = dict(model=model_file, target_version=version, source='model_trainer/' + spec['script'],
                font=os.path.basename(args.font or tc.DEFAULT_FONT),
                unactivated_markers=tc.UNACTIVATED_MARKERS[spec['lang']] if spec['kind'] == 'art' else None,
                created=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    meta.update(extra or {})
    return meta


# --------------------------------------------------------------------------- callbacks
def make_callbacks(keras, codec, val, args, monitor, throughput):
    """Logging/validation callback first: it adds ``val_ctc_accu`` to the logs
    seen by the checkpoint / LR / stopping callbacks that follow it."""

    class EpochLogger(keras.callbacks.Callback):
        def __init__(self):
            super(EpochLogger, self).__init__()
            self.t0 = None
            # Receive the original logs dict: TF 2.3 passes callbacks without this
            # flag a NumPy copy, and ModelCheckpoint (which has it) would not see
            # the val_ctc_accu added here.
            self._supports_tf_logs = True

        def on_epoch_begin(self, epoch, logs=None):
            self.t0 = time.time()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs if logs is not None else {}
            dt = max(time.time() - self.t0, 1e-9)
            steps = self.params.get('steps') or 0
            lines = 0.0
            for m in self.model.metrics:
                if getattr(m, 'name', '') == 'ctc_accu' and hasattr(m, 'total'):
                    lines = float(keras.backend.get_value(m.total))
            msg = 'epoch {}: loss {:.4f} ctc_accu {:.4f}'.format(epoch + 1, float(logs.get('loss', np.nan)),
                                                                  float(logs.get('ctc_accu', np.nan)))
            if val is not None:
                x, labels, keys = val
                pred = self.model.predict(x, batch_size=256, verbose=0)
                texts = codec.decode(pred)
                ok = [t == l for t, l in zip(texts, labels)]
                logs['val_ctc_accu'] = float(np.mean(ok))
                msg += ' val_ctc_accu {:.4f}'.format(logs['val_ctc_accu'])
                errors = [(l, t) for l, t, good in zip(labels, texts, ok) if not good]
                if errors:
                    msg += ' e.g. ' + '; '.join('{!r}->{!r}'.format(l, t) for l, t in errors[:3])
            opt = self.model.optimizer
            lr = float(keras.backend.get_value(getattr(opt, 'learning_rate', None) if hasattr(opt, 'learning_rate')
                                               else opt.lr))
            msg += ' | lr {:.2e} | {:.1f} steps/s, {:.0f} lines/s'.format(lr, steps / dt, lines / dt)
            throughput.append((steps, lines, dt))
            log(msg)

    class SolvedStop(keras.callbacks.Callback):
        """Stop once ``monitor`` >= threshold for ``patience`` consecutive epochs."""

        def __init__(self, threshold, patience):
            super(SolvedStop, self).__init__()
            self.threshold, self.patience, self.count = threshold, patience, 0

        def on_epoch_end(self, epoch, logs=None):
            value = (logs or {}).get(monitor)
            self.count = self.count + 1 if value is not None and value >= self.threshold else 0
            if self.count >= self.patience:
                log('{} >= {} for {} epochs: stopping'.format(monitor, self.threshold, self.patience))
                self.model.stop_training = True

    class BestWeights(keras.callbacks.Callback):
        def __init__(self):
            super(BestWeights, self).__init__()
            self.best, self.weights, self.epoch = -np.inf, None, None

        def on_epoch_end(self, epoch, logs=None):
            value = (logs or {}).get(monitor)
            if value is not None and value > self.best:
                self.best, self.weights, self.epoch = value, self.model.get_weights(), epoch + 1

    logger = EpochLogger()
    return logger, SolvedStop, BestWeights


# --------------------------------------------------------------------------- train
def schedule_phases(spec, args):
    """List of ``(steps_per_epoch, epochs, is_main)``.

    full: the historical schedule (warm-up fits without callbacks, then the
    long fit with checkpoint / LR reduction) + stop-when-solved.
    finetune: only the long fit, for a warm-started model.
    smoke: one short fit (end-to-end pipeline check)."""
    if args.schedule == 'smoke':
        return [(args.steps_per_epoch or 2, args.epochs or 1, True)]
    main_steps, main_epochs = spec['main']
    main = (args.steps_per_epoch or main_steps, args.epochs or (main_epochs if args.schedule == 'full' else 200), True)
    if args.schedule == 'full' and not args.no_warmup:
        return [(s, e, False) for s, e in spec['warmup']] + [main]
    return [main]


def keep_previous_charset(spec, path, charset, width, height, max_length):
    """If ``path`` holds a different vocabulary / geometry and checkpoints of
    that earlier run exist, rename it to ``<name>.<timestamp>.json`` instead of
    silently overwriting it: the converters would pair it with the wrong
    checkpoints."""
    if not os.path.isfile(path):
        return None
    old_ckpts = list_checkpoints(spec, os.path.dirname(path))
    if not old_ckpts:
        return None
    try:
        old = tc.load_charset_json(path)
        same = (list(old['charset']) == list(charset) and int(old.get('width', width)) == width
                and int(old.get('height', height)) == height and int(old.get('max_length', max_length)) == max_length)
    except Exception:  # unreadable: keep it as well
        same = False
    if same:
        return None
    stamp = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y%m%d-%H%M%S')
    backup, n = '{}.{}.json'.format(os.path.splitext(path)[0], stamp), 1
    while os.path.exists(backup):
        backup, n = '{}.{}-{}.json'.format(os.path.splitext(path)[0], stamp, n), n + 1
    os.rename(path, backup)
    log('WARNING: {} belonged to an earlier run with another vocabulary/geometry: renamed to {}. Its {} '
        'checkpoint(s) in {} need --charset {} to be converted, and new checkpoints may overwrite them: '
        'consider a fresh --train-dir.'.format(os.path.basename(path), os.path.basename(backup), len(old_ckpts),
                                               os.path.dirname(path), backup))
    return backup


def train(spec, args):
    tf, keras = import_tf()
    lang, kind = spec['lang'], spec['kind']
    width, height = args.width or spec['width'], spec['height']
    max_length = args.max_length or spec['max_length']
    fonts = tc.FontBank(args.font, args.fallback_font)
    charset = training_charset(spec)
    preflight(spec, charset, width, height, max_length, fonts, force=args.force)
    codec = tc.LabelCodec(charset, max_length)
    version = args.version or target_version()
    seed = args.seed if args.seed is not None else int(time.time()) % (2 ** 31)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    train_dir = os.path.abspath(args.train_dir)
    if not os.path.isdir(train_dir):
        os.makedirs(train_dir)
    # Training vocabulary next to the checkpoints (used by the converters).
    charset_path = os.path.join(train_dir, tc.charset_filename(lang, kind))
    keep_previous_charset(spec, charset_path, charset, width, height, max_length)
    tc.save_charset_json(charset_path, charset, width, height, max_length,
                         lang, kind, **charset_meta(spec, args, version=version))

    init = args.init_from
    if args.warm_start and not init:
        init = shipped_model(spec)
    init_weights = None
    if init:
        num_classes = weights_num_classes(init)
        found = find_charset_json(init, lang, kind, num_classes, explicit=args.init_charset)
        if found is None:
            if num_classes not in (None, len(charset) + 2):
                raise SystemExit('no charset JSON with {} classes found for {}: pass --init-charset'.format(
                    num_classes, init))
            old_charset, source = charset, 'training charset'
        else:
            old_charset, source = found['charset'], found['path']
        weights = read_init_weights(init, len(old_charset) + 2, width, height)
        init_weights, kept, added, dropped = remap_output_layer(weights, old_charset, charset, seed)
        log('warm start from {} (charset {}: {} chars): {} chars kept, {} new{}, {} dropped{}'.format(
            init, source, len(old_charset), kept, len(added), ' ' + repr(''.join(added[:60])) if added else '',
            len(dropped), ' ' + repr(''.join(dropped[:60])) if dropped else ''))

    model = build_ocr_model(len(charset) + 2, width, height, spec['model_name'])
    if init_weights is not None:
        model.set_weights(init_weights)
    lr = args.lr or (1e-3 if args.schedule == 'full' and not init else 5e-4)
    model.compile(loss=make_ctc_loss(), optimizer=keras.optimizers.Adam(learning_rate=lr),
                  metrics=[make_ctc_accuracy(codec)])
    if args.verbose >= 2:
        model.summary()
    log('{} {}: {} classes, input {}x{}, max_length {}, lr {}, schedule {}, {} worker(s), {} panel(s)/step, '
        'TF {}'.format(lang, kind, codec.num_classes, width, height, max_length, lr, args.schedule, args.workers,
                       args.samples_per_step, tf.__version__))

    cfg = data_config(spec, charset, width, height, max_length, args)
    val = tc.validation_set(cfg, args.val_lines, seed=12345) if args.val_lines > 0 else None
    monitor = 'val_ctc_accu' if val is not None else 'ctc_accu'
    throughput = []
    logger, SolvedStop, BestWeights = make_callbacks(keras, codec, val, args, monitor, throughput)
    best = BestWeights()
    main_callbacks = [
        best,
        keras.callbacks.ModelCheckpoint(os.path.join(train_dir, spec['checkpoint']), monitor=monitor, verbose=1,
                                        save_best_only=True, save_weights_only=True, mode='max'),
        keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, min_lr=spec['min_lr'], verbose=1,
                                          patience=args.lr_patience, mode='max'),
        SolvedStop(args.stop_at, args.stop_patience),
    ]
    if args.patience > 0:
        main_callbacks.append(keras.callbacks.EarlyStopping(monitor=monitor, mode='max', patience=args.patience))

    stream = tc.BatchStream(cfg, workers=args.workers, seed=seed)
    t_start = time.time()
    try:
        for steps, epochs, is_main in schedule_phases(spec, args):
            callbacks = [logger] + (main_callbacks if is_main else []) + [keras.callbacks.TerminateOnNaN()]
            log('fit: {} epoch(s) x {} steps{}'.format(epochs, steps, '' if is_main else ' (warm-up)'))
            history = model.fit(stream, steps_per_epoch=steps, epochs=epochs, callbacks=callbacks,
                                verbose=1 if args.verbose >= 2 else (2 if args.verbose else 0))
            if np.isnan(history.history.get('loss', [0])[-1]):
                raise SystemExit('loss is NaN')
    except KeyboardInterrupt:
        log('interrupted: exporting the best weights so far')
    finally:
        stream.close()
    log('training time {:.1f} min'.format((time.time() - t_start) / 60))
    if throughput:
        steps = sum(t[0] for t in throughput)
        lines = sum(t[1] for t in throughput)
        dt = sum(t[2] for t in throughput)
        log('throughput: {:.2f} steps/s, {:.0f} lines/s over {} steps'.format(steps / dt, lines / dt, steps))

    weights = best.weights if best.weights is not None else model.get_weights()
    if best.weights is not None:
        log('best {} {:.4f} at epoch {}'.format(monitor, best.best, best.epoch))
    return export_model(spec, weights, charset, width, height, max_length, args, version,
                        extra_meta=dict(best=dict(monitor=monitor, value=float(best.best), epoch=best.epoch)
                                        if best.weights is not None else None,
                                        init_from=os.path.basename(init) if init else None))


# --------------------------------------------------------------------------- export
def export_model(spec, weights, charset, width, height, max_length, args, version, extra_meta=None, out_dir=None):
    """Write ``<out>/models_<LANG>/<runtime name>.h5`` + ``<stem>.charset.json`` +
    ``charset_<lang>_<kind>.json`` and ``<out>/weights/<stem>.weights.npz`` +
    ``<stem>.charset.json`` (so that the .npz can be converted on its own)."""
    tf, keras = import_tf()
    lang, kind = spec['lang'], spec['kind']
    keras.backend.clear_session()
    model = build_ocr_model(len(charset) + 2, width, height, spec['model_name'])
    model.set_weights(weights)
    out_dir = os.path.abspath(out_dir or args.out_dir)
    folder = os.path.join(out_dir, MODEL_SUBDIR[lang])
    if not os.path.isdir(folder):
        os.makedirs(folder)
    fname = runtime_model_filename(spec, version)
    stem = os.path.splitext(fname)[0]
    h5_path = os.path.join(folder, fname)
    model.save(h5_path, include_optimizer=False)
    npz_path = save_weights_npz(model, os.path.join(out_dir, 'weights', stem + '.weights.npz'))
    meta = charset_meta(spec, args, model_file=fname, version=version,
                        extra=dict(tensorflow=tf.__version__, weights=os.path.basename(npz_path), **(extra_meta or {})))
    json_paths = [tc.save_charset_json(os.path.join(folder, stem + '.charset.json'), charset, width, height,
                                       max_length, lang, kind, **meta),
                  tc.save_charset_json(os.path.join(folder, tc.charset_filename(lang, kind)), charset, width, height,
                                       max_length, lang, kind, **meta)]
    tc.save_charset_json(os.path.join(os.path.dirname(npz_path), stem + '.charset.json'), charset, width, height,
                         max_length, lang, kind, **meta)
    # Round trip: the saved .h5 must reload and give the same predictions.
    probe = np.random.RandomState(0).rand(2, width, height, 1).astype(np.float32)
    reloaded = keras.models.load_model(h5_path, compile=False)
    diff = float(np.abs(reloaded.predict(probe, verbose=0) - model.predict(probe, verbose=0)).max())
    if diff > 1e-4:
        raise SystemExit('reloaded model differs from the trained one (max diff {})'.format(diff))
    log('exported {} (+ {}; weights {} + {}.charset.json)'.format(
        h5_path, ', '.join(os.path.basename(p) for p in json_paths), npz_path, stem))
    if not tf_runtime_compatible():
        log('NOTE: {} was written by TF {}; ArtScanner\'s TF 2.3 runtime cannot load it. Re-export it there with: '
            'python converters/model_converter.py --only {} --weights {}'.format(
                fname, tf.__version__, spec_key(spec), npz_path))
    return h5_path


def spec_key(spec):
    return spec['kind'] + ('_en' if spec['lang'] == 'en' else '')


def convert(spec, weights_path, out_dir=None, version=None, charset_path=None, args=None):
    """Re-export a checkpoint (``.hdf5`` / ``.h5`` / ``.npz``) as a runtime model."""
    import_tf()
    args = args or parse_args(spec, [])
    num_classes = weights_num_classes(weights_path)
    found = find_charset_json(weights_path, spec['lang'], spec['kind'], num_classes, explicit=charset_path)
    if found is None:
        charset = training_charset(spec)
        if num_classes not in (None, len(charset) + 2):
            raise SystemExit('{}: {} classes, but no matching charset JSON was found (the current tables give {}); '
                             'pass --charset'.format(weights_path, num_classes, len(charset) + 2))
        log('WARNING: no charset JSON next to {}, using the vocabulary of the current tables'.format(weights_path))
        width, height, max_length = spec['width'], spec['height'], spec['max_length']
    else:
        charset = found['charset']
        width = int(found.get('width', spec['width']))
        height = int(found.get('height', spec['height']))
        max_length = int(found.get('max_length', spec['max_length']))
    weights = read_init_weights(weights_path, len(charset) + 2, width, height)
    return export_model(spec, weights, charset, width, height, max_length, args, version or target_version(),
                        extra_meta=dict(converted_from=os.path.basename(weights_path)), out_dir=out_dir)


def convert_checkpoint(spec, preferred_name=None, weights=None, out_dir=None, version=None, charset=None,
                       weights_dir=None):
    """converters/ entry point: ``weights``, else ``preferred_name`` (model_info) or
    the best checkpoint in ``weights_dir`` (default ``train/``)."""
    weights_dir = weights_dir or DEFAULT_TRAIN_DIR
    if not weights:
        weights = best_checkpoint(spec, weights_dir, preferred_name)
    if not weights:
        raise SystemExit('{}: no {}*.hdf5 checkpoint in {} (pass --weights)'.format(
            spec_key(spec), spec['checkpoint'].split('{')[0], weights_dir))
    log('{}: converting {}'.format(spec_key(spec), weights))
    return convert(spec, weights, out_dir=out_dir or DEFAULT_OUT_DIR, version=version, charset_path=charset)


# --------------------------------------------------------------------------- evaluate / predict
def load_for_inference(spec, path, charset_path=None):
    """``(model, charset, max_length)`` for a runtime model or a checkpoint."""
    tf, keras = import_tf()
    num_classes = weights_num_classes(path)
    found = find_charset_json(path, spec['lang'], spec['kind'], num_classes, explicit=charset_path)
    if found is None:
        charset = training_charset(spec)
        if num_classes not in (None, len(charset) + 2):
            raise SystemExit('no charset JSON with {} classes for {}: pass --charset'.format(num_classes, path))
        found = {}
    else:
        charset = found['charset']
    model = None
    if path.endswith('.h5'):
        try:
            model = keras.models.load_model(path, compile=False)
        except Exception as e:  # e.g. TF 2.3 models in newer Keras: rebuild + load weights
            log('load_model failed ({}: {}), rebuilding the architecture'.format(type(e).__name__, str(e)[:120]))
    if model is None:
        model = build_ocr_model(len(charset) + 2, int(found.get('width', spec['width'])),
                                int(found.get('height', spec['height'])), spec['model_name'])
        load_weights_any(model, path)
    if int(model.output_shape[-1]) != len(charset) + 2:
        raise SystemExit('model has {} classes, charset {}'.format(model.output_shape[-1], len(charset) + 2))
    max_length = int(found.get('max_length', spec['max_length']))
    return model, charset, max_length, found.get('path', 'training charset')


def evaluate(spec, args):
    model, charset, max_length, source = load_for_inference(spec, args.evaluate, args.charset)
    width, height = int(model.input_shape[1]), int(model.input_shape[2])
    codec = tc.LabelCodec(charset, max_length)
    p_unactivated = args.p_unactivated
    if spec['kind'] == 'art' and codec.unknown_chars(''.join(tc.UNACTIVATED_MARKERS[spec['lang']])):
        p_unactivated = 0.0  # older vocabulary: cannot output the 6.0 marker
    cfg = data_config(spec, charset, width, height, max_length, args)
    cfg['p_unactivated'] = p_unactivated
    t0 = time.time()
    x, labels, keys = tc.validation_set(cfg, args.eval_lines, seed=args.seed if args.seed is not None else 4242,
                                        skip_unknown=True)
    pred = model.predict(x, batch_size=256, verbose=0)
    texts = codec.decode(pred)
    ok = np.array([t == l for t, l in zip(texts, labels)])
    log('evaluate {} (charset {}, {} chars, input {}x{}): {:.4f} exact on {} synthetic lines '
        '(p_unactivated {}, {:.1f}s)'.format(args.evaluate, source, len(charset), width, height, ok.mean(), len(ok),
                                             p_unactivated, time.time() - t0))
    groups = {}
    for key, good in zip(keys, ok):
        groups.setdefault(re.sub(r'\d+', '', key), []).append(good)
    for key in sorted(groups):
        log('  {:<24} {:.4f} ({} lines)'.format(key, np.mean(groups[key]), len(groups[key])))
    errors = [(l, t) for l, t, good in zip(labels, texts, ok) if not good]
    for l, t in errors[:20]:
        log('  error: {!r} -> {!r}'.format(l, t))
    return float(ok.mean())


def predict_image(spec, args, extract_fields):
    """Historical test helper: OCR the fields of a detail-panel screenshot."""
    from PIL import Image
    model, charset, max_length, _ = load_for_inference(spec, args.model, args.charset)
    width, height = int(model.input_shape[1]), int(model.input_shape[2])
    codec = tc.LabelCodec(charset, max_length)
    img = Image.open(args.predict_image).convert('RGB')
    fields = extract_fields(img, args.scale_ratio)
    keys = sorted(fields.keys())
    x = np.concatenate([tc.preprocess(fields[k], spec['lang'], width, height).T[None, :, :, None] for k in keys])
    for key, text in zip(keys, codec.decode(model.predict(x, verbose=0))):
        print('{:<16} {}'.format(key, text))


# --------------------------------------------------------------------------- CLI
def collect_strings(obj, out):
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            collect_strings(k, out)
            collect_strings(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_strings(v, out)
    return out


def check_font(spec, args):
    import json
    fonts = tc.FontBank(args.font, args.fallback_font)
    chars = set(training_charset(spec))
    for path in args.check_font:
        with open(path, encoding='utf-8') as f:
            chars |= set(''.join(collect_strings(json.load(f), [])))
    missing, fallback, uncovered = fonts.coverage(chars)
    log('{}: {} distinct chars checked, {} missing in {}'.format(
        spec['script'], len(chars), len(missing), os.path.basename(fonts.primary)))
    for c in missing:
        log('  U+{:04X} {!r} -> {}'.format(ord(c), c, fallback.get(c, 'NOT COVERED')))
    return 1 if uncovered else 0


def charset_only(spec, args):
    width, height = args.width or spec['width'], spec['height']
    max_length = args.max_length or spec['max_length']
    fonts = tc.FontBank(args.font, args.fallback_font)
    charset = training_charset(spec)
    preflight(spec, charset, width, height, max_length, fonts, force=True)
    folder = os.path.join(os.path.abspath(args.out_dir), MODEL_SUBDIR[spec['lang']])
    path = tc.save_charset_json(os.path.join(folder, tc.charset_filename(spec['lang'], spec['kind'])), charset, width,
                                height, max_length, spec['lang'], spec['kind'],
                                **charset_meta(spec, args, version=args.version or target_version()))
    log('wrote {} ({} chars)'.format(path, len(charset)))
    return 0


def default_workers():
    return min(4, max(1, (os.cpu_count() or 2) // 4))


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show defaults, except the uninformative None / False / []."""

    def _get_help_string(self, action):
        if action.default is None or action.default is False or action.default == []:
            return action.help
        return super(_HelpFormatter, self)._get_help_string(action)


def parse_args(spec, argv=None):
    p = argparse.ArgumentParser(
        prog=spec['script'], formatter_class=_HelpFormatter,
        description='Train the {} {} OCR model (default input {}x{}, max_length {}). See README.md.'.format(
            spec['lang'].upper(), 'artifact' if spec['kind'] == 'art' else 'material', spec['width'], spec['height'],
            spec['max_length']))
    g = p.add_argument_group('mode (default: train)')
    g.add_argument('--charset-only', action='store_true', help='write the charset JSON + geometry report (no TF)')
    g.add_argument('--check-font', nargs='*', metavar='JSON', default=None,
                   help='check glyph coverage of the vocabulary (+ all strings of the given JSON files)')
    g.add_argument('--evaluate', metavar='MODEL', help='accuracy of a model / checkpoint on synthetic data')
    g.add_argument('--convert', metavar='WEIGHTS', help='re-export a checkpoint (.hdf5/.h5/.npz) as runtime model')
    g.add_argument('--predict-image', metavar='PNG', help='OCR a detail-panel screenshot with --model')
    t = p.add_argument_group('training')
    t.add_argument('--schedule', choices=SCHEDULES, default=None,
                   help='full: historical schedule (default from scratch); finetune: long phase only (default '
                        'with --warm-start/--init-from); smoke: a couple of steps end-to-end')
    t.add_argument('--warm-start', action='store_true',
                   help='initialise from the newest shipped model in ArtScanner/rcc/models_*/ (output layer '
                        'remapped to the new charset); implies --schedule finetune unless given')
    t.add_argument('--init-from', metavar='WEIGHTS', help='initialise from this model / checkpoint / .npz')
    t.add_argument('--init-charset', metavar='JSON', help='charset of --init-from (default: found next to it)')
    t.add_argument('--steps-per-epoch', type=int, default=None)
    t.add_argument('--epochs', type=int, default=None)
    t.add_argument('--no-warmup', action='store_true', help='full schedule without the warm-up fits')
    t.add_argument('--lr', type=float, default=None, help='Adam learning rate (default 1e-3 from scratch, 5e-4 '
                                                          'when warm-starting)')
    t.add_argument('--lr-patience', type=int, default=5)
    t.add_argument('--patience', type=int, default=0, help='early stopping patience in epochs (0: off)')
    t.add_argument('--stop-at', type=float, default=0.999, help='stop when the monitored accuracy reaches this...')
    t.add_argument('--stop-patience', type=int, default=5, help='...for this many epochs in a row')
    t.add_argument('--samples-per-step', type=int, default=1, help='panels per batch (1 = historical)')
    t.add_argument('--workers', type=int, default=default_workers(), help='data generation processes (0: inline)')
    t.add_argument('--val-lines', type=int, default=None,
                   help='fixed synthetic validation lines (0: off; default 1024, 64 for smoke)')
    t.add_argument('--p-unactivated', type=float, default=0.35,
                   help='probability that an artifact sample has a grey 6.0 "unactivated" substat')
    t.add_argument('--focus', type=float, default=0.0,
                   help='training data only: probability of drawing artifact names / characters / material '
                        'names from the strings with characters outside --focus-charset')
    t.add_argument('--focus-charset', metavar='JSON', default=None,
                   help='vocabulary treated as known by --focus (default: pinned charset of the shipped model)')
    t.add_argument('--extra-lines', type=int, default=0,
                   help='training data only: extra name + equipped lines per artifact panel')
    t.add_argument('--seed', type=int, default=None)
    t.add_argument('--force', action='store_true', help='train even if labels do not fit the geometry')
    o = p.add_argument_group('model / files')
    o.add_argument('--width', type=int, default=None, help='input width (multiple of 8)')
    o.add_argument('--max-length', type=int, default=None)
    o.add_argument('--version', default=None, help='version in the exported file name (default: '
                                                   'ArtScanner/config.py TARGET_VERSION)')
    o.add_argument('--train-dir', default=DEFAULT_TRAIN_DIR, help='checkpoints')
    o.add_argument('--out-dir', default=DEFAULT_OUT_DIR, help='exported runtime models')
    o.add_argument('--font', default=tc.DEFAULT_FONT)
    o.add_argument('--fallback-font', action='append', default=[], help='fallback font(s) for missing glyphs')
    o.add_argument('--model', help='model for --predict-image')
    o.add_argument('--charset', help='charset JSON for --evaluate/--convert/--predict-image')
    o.add_argument('--scale-ratio', type=float, default=spec.get('scale_ratio', 1.0), help='--predict-image')
    o.add_argument('--eval-lines', type=int, default=2048)
    o.add_argument('--verbose', type=int, default=1, help='0: quiet, 1: one line per epoch, 2: Keras progress bar')
    args = p.parse_args(argv)
    if args.schedule is None:
        args.schedule = 'finetune' if (args.warm_start or args.init_from) else 'full'
    if args.val_lines is None:
        args.val_lines = 64 if args.schedule == 'smoke' else 1024
    return args


def main(spec, argv=None, extract_fields=None):
    args = parse_args(spec, argv)
    if args.check_font is not None:
        return check_font(spec, args)
    if args.charset_only:
        return charset_only(spec, args)
    if args.evaluate:
        evaluate(spec, args)
        return 0
    if args.convert:
        convert(spec, args.convert, out_dir=args.out_dir, version=args.version, charset_path=args.charset, args=args)
        return 0
    if args.predict_image:
        if extract_fields is None or not args.model:
            raise SystemExit('--predict-image needs --model (and is only available for artifact models)')
        predict_image(spec, args, extract_fields)
        return 0
    train(spec, args)
    return 0
