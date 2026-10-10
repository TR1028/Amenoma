import json
import os

try:  # imported as ArtScanner.MaterialInfo (model_trainer) ...
    from . import config as _config
except ImportError:  # ... or as a top-level module (the app, PyInstaller builds, tests)
    import config as _config

current_path = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(current_path, 'rcc/material_names_%s.json' % _config.DATA_VERSION),
                      encoding='utf-8'))

MaterialsNameCHS = data['CHS']
MaterialsNameEN = data['EN']

if __name__ == '__main__':
    print(f"{MaterialsNameCHS=}")
    print(f"{MaterialsNameEN=}")
