import sys
import keyboard
import unicodedata
import win32api
import win32con
import win32gui
import win32process
from PIL import Image
from mss import mss
import Levenshtein
import ArtsInfo
import logging
import MaterialInfo


class _NullStream:
    encoding = "utf-8"
    closed = False

    def write(self, s):
        return len(s) if s is not None else 0

    def writelines(self, lines):
        pass

    def flush(self):
        pass

    def isatty(self):
        return False

    def close(self):
        pass


def ensure_standard_streams():
    if sys.stdout is None:
        sys.stdout = _NullStream()
    if sys.stderr is None:
        sys.stderr = _NullStream()
    if getattr(sys, "__stdout__", None) is None:
        sys.__stdout__ = sys.stdout
    if getattr(sys, "__stderr__", None) is None:
        sys.__stderr__ = sys.stderr

    loggers = [logging.getLogger()]
    for lg in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(lg, logging.Logger):
            loggers.append(lg)
    for lg in loggers:
        for handler in getattr(lg, "handlers", ()):
            if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) is None:
                handler.stream = sys.stderr


ensure_standard_streams()

logger = logging.getLogger()
logHandler = logging.FileHandler("./Amenoma.log", encoding='utf-8')
logHandler.setFormatter(logging.Formatter("[%(levelname)7s] %(asctime)s /%(module)10s[%(lineno)3d]\t"
                                          "/%(message)s"))
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

def calcFormatWidth(text, target):
    return target - sum(unicodedata.east_asian_width(c) in 'WF' for c in text)


def setWindowToForeground(hwnd):
    keyboard.press('alt')
    win32gui.ShowWindow(hwnd, 5)
    win32gui.SetForegroundWindow(hwnd)
    keyboard.release('alt')


def getProcessName(hwnd):
    try:
        pid = win32process.GetWindowThreadProcessId(hwnd)
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, pid[1])
        return win32process.GetModuleFileNameEx(handle, 0)
    except:
        return "Unknown"


def findWindowsByName(name):
    windows = []

    def winEnumHandler(hwnd, ctx):
        win_name = win32gui.GetWindowText(hwnd)
        win_size = win32gui.GetClientRect(hwnd)[2:]
        if win32gui.IsWindowVisible(hwnd) and name in win_name and ((win_size[0] * win_size[1]) != 0 or len(name) > 0):
            windows.append((hwnd, win_name, getProcessName(hwnd), win_size))

    win32gui.EnumWindows(winEnumHandler, None)
    return windows


def captureRect(rect):
    with mss() as sct:
        monitor = {"top": rect[0], "left": rect[1], "width": rect[2], "height": rect[3]}
        sct_img = sct.grab(monitor)
        # Convert to PIL/Pillow Image
        return Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')


def captureWindow(hwnd, local_rect=None):
    if local_rect is None:
        local_rect = win32gui.GetClientRect(hwnd)
    else:
        local_rect = tuple([round(i) for i in local_rect])
    screen_rect = [*local_rect]
    screen_rect[:2] = win32gui.ClientToScreen(hwnd, local_rect[:2])
    screen_rect[2:] = win32gui.ClientToScreen(hwnd, local_rect[2:])
    return captureRect(
        [screen_rect[1], screen_rect[0], screen_rect[2] - screen_rect[0], screen_rect[3] - screen_rect[1]])


_DIGIT_TRANS = str.maketrans({
    'O': '0', 'o': '0', 'D': '0',
    'l': '1', 'I': '1', 'i': '1', '|': '1',
    'Z': '2', 'z': '2',
    'S': '5', 's': '5',
    'B': '8',
})


def decodeValue(v):
    if type(v) != str:
        return v
    s = v.strip().translate(_DIGIT_TRANS)
    if "%" in s:
        return float(s.replace('%', '').replace(',', '').replace('+', '').strip()) / 100
    else:
        cleaned = s.replace(',', '').replace('+', '').strip()
        if not cleaned.lstrip('-').isdigit():
            digits = ''.join(ch for ch in cleaned if ch.isdigit())
            if digits:
                cleaned = ('-' if cleaned.startswith('-') else '') + digits
        return int(cleaned)


def attr_auto_correct(attr: str) -> str:
    corr_name = ''
    dis = 10000000
    if attr in ArtsInfo.MainAttrNames.values():
        return attr
    for n in ArtsInfo.MainAttrNames.values():
        ndis = Levenshtein.distance(attr, n)
        if ndis < dis:
            dis = ndis
            corr_name = n
    if dis == 0:
        pass
    elif dis <= (len(attr) // 3):
        logger.info(f"Corrected attribute from [{attr}] to [{corr_name}] with distance {dis}")
    else:
        logger.warning(f"Corrected attribute from [{attr}] to [{corr_name}] with distance {dis}")
    return corr_name


def name_auto_correct(name: str) -> str:
    corr_name = ""
    dis = 10000000
    if name in ArtsInfo.ArtNames:
        return name
    for arts in ArtsInfo.ArtNames:
        for rname in arts:
            ndis = Levenshtein.distance(name, rname)
            if ndis < dis:
                corr_name = rname
                dis = ndis
    if dis == 0:
        pass
    elif dis <= (len(name) // 3):
        logger.info(f"Corrected name from [{name}] to [{corr_name}] with distance {dis}")
    else:
        logger.warning(f"Corrected name from [{name}] to [{corr_name}] with distance {dis}")
    return corr_name


def attr_auto_correct_EN(attr: str) -> str:
    corr_name = ''
    dis = 10000000
    if attr in ArtsInfo.MainAttrNames_EN.values():
        return attr
    for n in ArtsInfo.MainAttrNames_EN.values():
        ndis = Levenshtein.distance(attr, n)
        if ndis < dis:
            dis = ndis
            corr_name = n
    if dis == 0:
        pass
    elif dis <= (len(attr) // 3):
        logger.info(f"Corrected attribute from [{attr}] to [{corr_name}] with distance {dis}")
    else:
        logger.warning(f"Corrected attribute from [{attr}] to [{corr_name}] with distance {dis}")
    return corr_name


def name_auto_correct_EN(name: str) -> str:
    corr_name = ""
    dis = 10000000
    for arts in ArtsInfo.ArtNames_EN:
        for rname in arts:
            ndis = Levenshtein.distance(name, rname)
            if ndis < dis:
                corr_name = rname
                dis = ndis
    if dis == 0:
        pass
    elif dis <= (len(name) // 3):
        logger.info(f"Corrected name from [{name}] to [{corr_name}] with distance {dis}")
    else:
        logger.warning(f"Corrected name from [{name}] to [{corr_name}] with distance {dis}")
    return corr_name


def type_auto_correct(name: str) -> str:
    corr_name = ""
    dis = 10000000
    if name in ArtsInfo.TypeNames:
        return name
    for tname in ArtsInfo.TypeNames:
        ndis = Levenshtein.distance(name, tname)
        if ndis < dis:
            corr_name = tname
            dis = ndis
    if dis == 0:
        pass
    elif dis <= (len(name) // 3):
        logger.info(f"Corrected type from [{name}] to [{corr_name}] with distance {dis}")
    else:
        logger.warning(f"Corrected type from [{name}] to [{corr_name}] with distance {dis}")
    return corr_name


def type_auto_correct_EN(name: str) -> str:
    corr_name = ""
    dis = 10000000
    if name in ArtsInfo.TypeNames_EN:
        return name
    for tname in ArtsInfo.TypeNames_EN:
        ndis = Levenshtein.distance(name, tname)
        if ndis < dis:
            corr_name = tname
            dis = ndis
    if dis == 0:
        pass
    elif dis <= (len(name) // 3):
        logger.info(f"Corrected type from [{name}] to [{corr_name}] with distance {dis}")
    else:
        logger.warning(f"Corrected type from [{name}] to [{corr_name}] with distance {dis}")
    return corr_name


def equipped_auto_correct(name: str) -> str:
    if Levenshtein.distance(name[-3:], "已装备") > 1:
        return ""
    corr_name = ""
    dis = 10000000
    if name[:-3] in ArtsInfo.UsersCHS:
        return name[:-3]
    for tname in ArtsInfo.UsersCHS:
        ndis = Levenshtein.distance(name[:-3], tname)
        if ndis < dis:
            corr_name = tname
            dis = ndis
    if dis == 0:
        pass
    elif dis <= (len(name) // 3):
        logger.info(f"Corrected character from [{name}] to [{corr_name}] with distance {dis}")
    else:
        corr_name = "Traveler"
        logger.warning(f"Failed to recognize [{name}]. Used [Traveler]")
    return corr_name


def equipped_auto_correct_EN(name: str) -> str:
    if len(name) < 10:
        return ""
    if Levenshtein.distance(name[:10], "Equipped: ") > 4:
        return ""
    corr_name = ""
    dis = 10000000
    if name[10:] in ArtsInfo.UsersEN:
        return name[10:]
    for tname in ArtsInfo.UsersEN:
        ndis = Levenshtein.distance(name[10:], tname)
        if ndis < dis:
            corr_name = tname
            dis = ndis
    if dis == 0:
        pass
    elif dis <= (len(name) // 3):
        logger.info(f"Corrected character from [{name}] to [{corr_name}] with distance {dis}")
    else:
        corr_name = "Traveler"
        logger.warning(f"Failed to recognize [{name}]. Used [Traveler]")
    return corr_name


def material_name_auto_correct(name: str) -> str:
    corr_name = ""
    dis = 10000000
    norm_name = name.replace("原坯", "原胚")
    if norm_name in MaterialInfo.MaterialsNameCHS:
        return norm_name
    for item in MaterialInfo.MaterialsNameCHS:
        ndis = Levenshtein.distance(norm_name, item)
        if ndis < dis:
            corr_name = item
            dis = ndis
    if dis == 0:
        pass
    elif dis <= (len(norm_name) // 3):
        logger.info(f"Corrected material name from [{name}] to [{corr_name}] with distance {dis}")
    else:
        logger.warning(f"Corrected material  name from [{name}] to [{corr_name}] with distance {dis}")
    return corr_name


def material_name_auto_correct_EN(name: str) -> str:
    corr_name = ""
    dis = 10000000
    if name in MaterialInfo.MaterialsNameEN:
        return name
    for item in MaterialInfo.MaterialsNameEN:
        ndis = Levenshtein.distance(name, item)
        if ndis < dis:
            corr_name = item
            dis = ndis
    if dis == 0:
        pass
    elif dis <= (len(name) // 3):
        logger.info(f"Corrected material name from [{name}] to [{corr_name}] with distance {dis}")
    else:
        logger.warning(f"Corrected material  name from [{name}] to [{corr_name}] with distance {dis}")
    return corr_name
