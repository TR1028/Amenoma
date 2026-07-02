"""Crop material amount images from saved debug inputs.

Examples:
    python tests/debug_crop_from_saved_image.py materials/debug/000001_item.png out.png
    python tests/debug_crop_from_saved_image.py materials/debug/000001_item.png out.png --scale-ratio 0.75
    python tests/debug_crop_from_saved_image.py screenshot.png out.png --mode screenshot --row 2 --col 3
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTSCANNER_DIR = REPO_ROOT / "ArtScanner"


AMOUNT_COORDS = (10, 175, 150, 203)
BASE_ITEM_WIDTH = 164.39
BASE_ITEM_HEIGHT = 203.21


def build_offline_game_info(width, height):
    scale_ratio = min(width / 2560, height / 1440)
    art_width = 164.39 * scale_ratio
    art_height = 203.21 * scale_ratio
    art_gap_x = 30.89 * scale_ratio
    art_gap_y = 30.35 * scale_ratio
    left_margin = (155 if width < 2 * height else 295.33) * scale_ratio
    right_margin = (871.33 if width < 2 * height else 967.33) * scale_ratio
    art_cols = int((width - left_margin - right_margin + art_gap_x) // (art_width + art_gap_x))
    art_shift = (
        (width - left_margin - right_margin + art_gap_x)
        - art_cols * (art_width + art_gap_x)
    ) / 2

    return SimpleNamespace(
        scale_ratio=scale_ratio,
        art_width=art_width,
        art_height=art_height,
        art_gap_x=art_gap_x,
        art_gap_y=art_gap_y,
        first_art_x=left_margin + art_shift,
        first_art_y=161 * scale_ratio,
        lastrow_offset=125.5 * scale_ratio,
    )


def scaled_box(coords, scale_ratio):
    return tuple(round(i * scale_ratio) for i in coords)


def infer_item_scale_ratio(item_img):
    width, height = item_img.size
    return min(width / BASE_ITEM_WIDTH, height / BASE_ITEM_HEIGHT)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop the material amount region from a saved item image or inventory screenshot."
    )
    parser.add_argument("input", help="Saved item_img or full inventory screenshot.")
    parser.add_argument("output", help="Output path for the amount crop image.")
    parser.add_argument(
        "--mode",
        choices=("item", "screenshot"),
        default="item",
        help="Use item for saved item_img, screenshot for a full inventory screenshot.",
    )
    parser.add_argument("--row", type=int, default=0, help="Grid row for screenshot mode.")
    parser.add_argument("--col", type=int, default=0, help="Grid column for screenshot mode.")
    parser.add_argument(
        "--lastrow",
        action="store_true",
        help="Apply the same lastrow_offset used by MaterialScannerLogic.",
    )
    parser.add_argument(
        "--scale-ratio",
        type=float,
        default=None,
        help=(
            "OCR crop scale ratio. In item mode, defaults to inferring from the saved item image size; "
            "in screenshot mode, defaults to inferring from screenshot size."
        ),
    )
    parser.add_argument(
        "--item-output",
        default=None,
        help="Optional path to save the intermediate item crop in screenshot mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required to crop images. Install the project's runtime dependencies.") from exc

    image = Image.open(args.input)

    if args.mode == "screenshot":
        sys.path.insert(0, str(ARTSCANNER_DIR))
        from material_scanner_logic import getMaterialItemCoord

        game_info = build_offline_game_info(*image.size)
        item_coord = getMaterialItemCoord(game_info, args.row, args.col, args.lastrow)
        item_img = image.crop(item_coord)
        scale_ratio = args.scale_ratio if args.scale_ratio is not None else game_info.scale_ratio
        if args.item_output:
            Path(args.item_output).parent.mkdir(parents=True, exist_ok=True)
            item_img.save(args.item_output)
    else:
        item_img = image
        scale_ratio = args.scale_ratio if args.scale_ratio is not None else infer_item_scale_ratio(item_img)

    amount_crop = item_img.crop(scaled_box(AMOUNT_COORDS, scale_ratio))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    amount_crop.save(output)


if __name__ == "__main__":
    main()
