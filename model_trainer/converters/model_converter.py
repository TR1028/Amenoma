"""Export trained weights as ArtScanner runtime models (+ charset JSON files).

Run it with the ArtScanner runtime (Python 3.8 + TF 2.3) so that the .h5 files
load there (models saved by TF >= 2.4 contain layers TF 2.3 cannot read; the
weights files are interchangeable):

    python converters/model_converter.py                    # all four models
    python converters/model_converter.py --only art_en --weights output/weights/artifact_model_EN_X.weights.npz

Weights per model, in this order: --weights (only together with --only <one
model>), model_info.<X>_MODEL_NAME in --weights-dir, the best checkpoint in
--weights-dir (highest accuracy in the file name, then epoch).  The vocabulary
comes from <stem>.charset.json / charset_<lang>_<kind>.json next to the weights
(the trainers write it into train/); without one it is rebuilt from the current
data tables, as before.  Output: <out-dir>/models_<LANG>/<model>_<version>.h5,
<model>_<version>.charset.json and charset_<lang>_<kind>.json.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model_converter_artifact as mca  # noqa: E402
import model_converter_artifact_EN as mcaEN  # noqa: E402
import model_converter_material as mcm  # noqa: E402
import model_converter_material_EN as mcmEN  # noqa: E402
import model_info as mi  # noqa: E402

CONVERTERS = {'art': mca, 'art_en': mcaEN, 'mat': mcm, 'mat_en': mcmEN}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--only', default=','.join(CONVERTERS), help='comma separated subset of ' + ', '.join(CONVERTERS))
    p.add_argument('--weights', help='weights file (.hdf5 / .h5 / .npz) for the single model given by --only')
    p.add_argument('--charset', help='charset JSON for --weights (default: found next to it)')
    p.add_argument('--weights-dir', default=os.path.join(os.path.dirname(_HERE), mi.MODEL_DIR))
    p.add_argument('--out-dir', default=os.path.join(os.path.dirname(_HERE), mi.OUTPUT_DIR))
    p.add_argument('--version', default=mi.MODEL_TARGET_VERSION)
    args = p.parse_args(argv)
    keys = [k.strip() for k in args.only.split(',') if k.strip()]
    unknown = [k for k in keys if k not in CONVERTERS]
    if unknown:
        p.error('unknown model(s) {}'.format(unknown))
    if (args.weights or args.charset) and len(keys) != 1:
        p.error('--weights / --charset need exactly one model in --only')
    os.makedirs(args.out_dir, exist_ok=True)
    for key in keys:
        CONVERTERS[key].convert(weights=args.weights, out_dir=args.out_dir, version=args.version,
                                charset=args.charset, weights_dir=args.weights_dir)


if __name__ == '__main__':
    main()
