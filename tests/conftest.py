"""Shared pytest setup for the ArtScanner unit tests.

The scanner modules are top-level modules inside ArtScanner/ (``import ArtsInfo``, ``import utils``),
so ArtScanner/ is put on sys.path.  Windows-only / input-hook packages (pywin32, mouse, keyboard,
mss) are replaced by inert stub modules when they are not installed, and ``utils``' log file
handler is not created, so that the pure logic can be tested on any OS without a game window:

    python -m pytest tests          (Python 3.8 runtime env, or any newer Python with numpy/Pillow/
                                     Levenshtein; ZODB is needed for test_art_saver, TF / h5py for
                                     the model checks, otherwise those tests are skipped)
"""
import importlib
import logging
import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTSCANNER_DIR = os.path.join(REPO_ROOT, 'ArtScanner')
if ARTSCANNER_DIR not in sys.path:
    sys.path.insert(0, ARTSCANNER_DIR)


def _unavailable(module, name):
    def call(*args, **kwargs):
        raise RuntimeError('%s.%s is not available in the test environment' % (module, name))
    call.__name__ = name
    return call


def _stub_module(name):
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:
        pass
    module = types.ModuleType(name)

    def module_getattr(attr, _name=name):
        if attr.startswith('__'):
            raise AttributeError(attr)
        return _unavailable(_name, attr)

    module.__dict__['__getattr__'] = module_getattr
    module.__dict__['__amenoma_test_stub__'] = True
    sys.modules[name] = module


for _name in ('win32api', 'win32con', 'win32gui', 'win32process', 'mouse', 'keyboard', 'mss'):
    _stub_module(_name)


class _NoFileHandler(logging.NullHandler):
    """utils.py creates ./Amenoma.log at import time; keep the tests from writing it."""

    def __init__(self, *args, **kwargs):
        logging.NullHandler.__init__(self)


_FileHandler = logging.FileHandler
logging.FileHandler = _NoFileHandler
try:
    import utils  # noqa: E402,F401
finally:
    logging.FileHandler = _FileHandler
