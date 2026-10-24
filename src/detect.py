#!/usr/bin/env python3
"""
detect.py  —  YOLOv8 Arabic line detector
==========================================
Importable function and CLI for detecting text lines on a page image.

The detector returns one bounding box per line, sorted top-to-bottom.
Class labels (printed / handwritten) are unreliable on real pages — use
boxes only, not class predictions.

CLI usage
---------
    python src/detect.py page.jpg
    python src/detect.py page.jpg --draw --out annotated.jpg
    python src/detect.py page.jpg --weights path/to/best.pt

Python usage
------------
    from src.detect import detect_lines

    boxes = detect_lines("page.jpg")           # list of (x0,y0,x1,y1)
    boxes = detect_lines(pil_image)            # PIL Image also accepted
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

from PIL import Image

# ── defaults ──────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = _HERE.parent / "best.pt"

IMGSZ   = 1280
CONF    = 0.4
IOU     = 0.5
PAD     = 4
MIN_H   = 15     # minimum box height in pixels


# ── core function ─────────────────────────────────────────────────────────────
def detect_lines(
    image: Union[str, Path, Image.Image],
    weights: Optional[Union[str, Path]] = None,
    imgsz: int = IMGSZ,
    conf: float = CONF,
    iou: float = IOU,
    pad: int = PAD,
    min_h: int = MIN_H,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect text-line bounding boxes on a page image.

    Parameters
    ----------
    image   : path to an image file, or a PIL Image.
    weights : path to best.pt. Defaults to best.pt next to this repo root.
    imgsz   : YOLO inference image size (longest side, pixels).
    conf    : confidence threshold.
    iou     : IoU threshold for NMS.
    pad     : pixels to add around each raw box.
    min_h   : minimum box height (boxes shorter than this are expanded).

    Returns
    -------
    List of (x0, y0, x1, y1) tuples (int, pixel coordinates), sorted
    top-to-bottom by vertical centre. Class labels are NOT returned because
    they are unreliable on real pages.

    Notes
    -----
    - The top of each box is trimmed to the bottom of the box above to
      remove vertical overlaps before returning.
    - Boxes are clamped to lie inside the image.
    """
    from ultralytics import YOLO

    wpath = weights or DEFAULT_WEIGHTS
    model = YOLO(str(wpath))

    if isinstance(image, (str, Path)):
        page = Image.open(image).convert("RGB")
    else:
        page = image.convert("RGB")

    W, H = page.size
    res = model.predict(page, imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
    raw = [list(map(int, b)) for b in res.boxes.xyxy.cpu().numpy()]
    raw.sort(key=lambda b: (b[1] + b[3]) / 2)

    out: List[Tuple[int, int, int, int]] = []
    prev_bottom = 0

    for x0, y0, x1, y1 in raw:
        x0, x1 = max(0, x0 - pad), min(W, x1 + pad)
        y0, y1 = max(0, y0 - pad), min(H, y1 + pad)

        if prev_bottom and y0 < prev_bottom:
            y0 = prev_bottom                         # trim overlap

        if y1 - y0 < min_h:                          # enforce minimum height
            y0 = max(0, min(y0, H - min_h))
            y1 = min(H, y0 + min_h)

        out.append((x0, y0, x1, y1))
        prev_bottom = max(prev_bottom, y1)

    return out


def draw_boxes(
    image: Union[str, Path, Image.Image],
    boxes: List[Tuple[int, int, int, int]],
    output_path: Optional[Union[str, Path]] = None,
) -> Image.Image:
    """
    Draw numbered boxes on *image* and optionally save to *output_path*.

    Returns the annotated PIL Image.
    """
    from PIL import ImageDraw, ImageFont

    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")

    draw = ImageDraw.Draw(img)
    try:
        fnt = ImageFont.truetype(str(_HERE.parent / "assets" / "font.ttf"), 18)
    except Exception:
        fnt = ImageFont.load_default()

    for i, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        draw.rectangle([x0, y0, x1, y1], outline=(220, 50, 50), width=2)
        draw.text((x0 + 3, y0 + 1), str(i), font=fnt, fill=(30, 100, 200))

    if output_path:
        img.save(str(output_path))

    return img


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Detect text lines on an Arabic page image."
    )
    parser.add_argument("image", help="Path to input image.")
    parser.add_argument("--weights", default=None,
                        help="Path to best.pt (default: ../best.pt).")
    parser.add_argument("--draw", action="store_true",
                        help="Draw numbered boxes on the image.")
    parser.add_argument("--out", default=None,
                        help="Save annotated image to this path (implies --draw).")
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--conf",  type=float, default=CONF)
    parser.add_argument("--iou",   type=float, default=IOU)
    args = parser.parse_args()

    boxes = detect_lines(
        args.image,
        weights=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
    )
    print(f"{len(boxes)} line(s) detected:")
    for i, (x0, y0, x1, y1) in enumerate(boxes, start=1):
        print(f"  {i:3d}: ({x0}, {y0}, {x1}, {y1})")

    if args.draw or args.out:
        out_path = args.out or Path(args.image).stem + "_detected.jpg"
        annotated = draw_boxes(args.image, boxes, output_path=out_path)
        print(f"Annotated image saved to: {out_path}")


if __name__ == "__main__":
    main()
