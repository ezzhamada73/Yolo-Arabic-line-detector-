#!/usr/bin/env python3
"""
make_synthetic_pages.py
=======================
Generate synthetic 2500x2500 page images for training the YOLOv8 line detector.

Based on: notebooks_kaggle/generate_arabic_ocr_images (1).ipynb (newest version).
This is the generator used to produce the training data for best.pt.

Each page contains a random mix of:
  - Printed lines: Arabic text from Arabic Wikipedia, rendered with Amiri or
    another Arabic font at a random size (28–40 pt).
  - Handwritten lines: real handwritten line images from the KHATT dataset.

Output layout (YOLO format):
  <output_dir>/
    images/train/  page_0000.jpg ...
    images/val/    page_0000.jpg ...
    labels/train/  page_0000.txt ...
    labels/val/    page_0000.txt ...
    arabic_lines.yaml

Label format:  class_id x_center y_center width height  (normalized 0–1)
  class 0 = printed_line
  class 1 = handwritten_line

Requirements
------------
  pip install pillow datasets arabic-reshaper python-bidi

Usage
-----
    # Minimal: download KHATT from Kaggle first, then:
    python scripts/make_synthetic_pages.py \\
        --khatt-train /path/to/KHATT/Train_deskewed/Train_deskewed \\
        --khatt-val   /path/to/KHATT/Validate_deskewed/Validate_deskewed \\
        --khatt-csv-train /path/to/KHATT/Train.csv \\
        --khatt-csv-val   /path/to/KHATT/Validation.csv \\
        --fonts Amiri-Regular.ttf \\
        --out ./generated_pages \\
        --n-train 2000 --n-val 200

Notes
-----
- The printed-line generator loads ~500 articles from Arabic Wikipedia (streaming).
- The script does NOT require a GPU.
- On Kaggle/Colab with the KHATT dataset already mounted, point --khatt-train
  to the mounted path.
- The 'arabic-reshaper' and 'python-bidi' packages are needed only when NOT
  running on Google Colab (Colab handles RTL automatically).
  On Kaggle, arabic-reshaper is also needed.
"""

import argparse
import os
import random
import re
import sys

from PIL import Image, ImageDraw, ImageFont


# ── text helpers ──────────────────────────────────────────────────────────────
def _render_text(text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    """Render *text* as a white-background RGB image using *font*."""
    tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    bbox = tmp.textbbox((0, 0), text, font=font)
    l, t, r, b = bbox
    tw, th = r - l, b - t
    img = Image.new("RGB", (tw + 20, th + 20), (255, 255, 255))
    ImageDraw.Draw(img).text((10, 10), text, font=font, fill=(0, 0, 0))
    return img


def _degrade_page(img: Image.Image) -> Image.Image:
    """Apply a slight yellowish tint to mimic aged paper."""
    import numpy as np
    arr = img.convert("RGB")
    a = __import__("numpy").asarray(arr).astype("float32")
    tint = __import__("numpy").array([
        random.randint(235, 255),
        random.randint(225, 250),
        random.randint(200, 245),
    ]) / 255.0
    a = (a * tint).clip(0, 255).astype("uint8")
    return Image.fromarray(a)


# ── generators ────────────────────────────────────────────────────────────────
def _hw_generator(df, img_dir: str):
    """Infinite generator of handwritten line images from KHATT."""
    while True:
        shuffled = df.sample(frac=1).reset_index(drop=True)
        for filename in shuffled.iloc[:, 0]:
            filename = str(filename).replace(".tif", ".jpg")
            p = os.path.join(img_dir, filename)
            try:
                yield Image.open(p).convert("RGB")
            except Exception:
                pass


def _printed_generator(sentences: list, fonts: list):
    """Infinite generator of printed line images."""
    while True:
        random.shuffle(sentences)
        for text in sentences:
            font = random.choice(fonts)
            yield _render_text(text, font)


# ── page builder ───────────────────────────────────────────────────────────────
def _make_page(hw_gen, pr_gen) -> tuple[Image.Image, list]:
    """
    Build one 2500x2500 canvas with alternating printed/handwritten lines.

    Returns (canvas, boxes) where boxes is a list of
    (class_id, x0, y0, x1, y1) in pixel coordinates.
    """
    canvas = Image.new("RGB", (2500, 2500), (255, 255, 255))
    target_h = random.randint(60, 110)
    y = 100
    x = random.randint(60, 300)
    boxes = []

    while True:
        use_printed = random.random() < 0.5
        src = next(pr_gen) if use_printed else next(hw_gen)
        label = 0 if use_printed else 1

        w, h = src.size
        if h == 0:
            continue
        new_h = target_h
        new_w = int(w * new_h / h)
        if x + new_w > 2500:
            continue

        scaled = src.resize((new_w, new_h))
        canvas.paste(scaled, (x, y))
        boxes.append((label, x, y, x + new_w, y + new_h))

        y += scaled.size[1] + 20
        if y + target_h >= 2400:
            break

    return canvas, boxes


def _yolo_label(boxes: list, W: int = 2500, H: int = 2500) -> list[str]:
    lines = []
    for cls, x1, y1, x2, y2 in boxes:
        xc = (x1 + x2) / 2 / W
        yc = (y1 + y2) / 2 / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def _save(canvas: Image.Image, labels: list[str], out_dir: str, name: str, split: str):
    img_dir = os.path.join(out_dir, "images", split)
    lbl_dir = os.path.join(out_dir, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    canvas.save(os.path.join(img_dir, f"{name}.jpg"))
    with open(os.path.join(lbl_dir, f"{name}.txt"), "w") as f:
        f.write("\n".join(labels) + "\n")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Generate synthetic YOLO detector pages.")
    p.add_argument("--khatt-train",     required=True, help="KHATT Train_deskewed dir.")
    p.add_argument("--khatt-val",       required=True, help="KHATT Validate_deskewed dir.")
    p.add_argument("--khatt-csv-train", required=True, help="KHATT Train.csv.")
    p.add_argument("--khatt-csv-val",   required=True, help="KHATT Validation.csv.")
    p.add_argument("--fonts", nargs="+", required=True, help="One or more .ttf font files.")
    p.add_argument("--out",      default="./generated_pages", help="Output directory.")
    p.add_argument("--n-train",  type=int, default=2000, help="Number of training pages.")
    p.add_argument("--n-val",    type=int, default=200,  help="Number of validation pages.")
    p.add_argument("--font-sizes", nargs="+", type=int, default=[28, 32, 36, 40])
    p.add_argument("--wiki-articles", type=int, default=500,
                   help="Number of Arabic Wikipedia articles to sample from.")
    args = p.parse_args()

    # Load fonts
    fonts = [ImageFont.truetype(fp, sz)
             for fp in args.fonts
             for sz in args.font_sizes]
    print(f"Loaded {len(fonts)} font variants from {len(args.fonts)} file(s).")

    # Load Arabic Wikipedia sentences
    print("Streaming Arabic Wikipedia …")
    try:
        from datasets import load_dataset
        ds = load_dataset("wikimedia/wikipedia", "20231101.ar",
                          split="train", streaming=True)
        texts, sentences = [], []
        for i, row in enumerate(ds):
            texts.append(row["text"])
            if i >= args.wiki_articles:
                break
        for t in texts:
            for piece in re.split(r"[!؟.،؛\n]", t):
                piece = piece.strip()
                if 10 <= len(piece) <= 200:
                    sentences.append(piece)
        print(f"  {len(sentences)} printed sentences ready.")
    except Exception as e:
        sys.exit(f"Failed to load Wikipedia sentences: {e}")

    # Load KHATT CSVs
    import pandas as pd
    train_df = pd.read_csv(args.khatt_csv_train)
    val_df   = pd.read_csv(args.khatt_csv_val)
    print(f"KHATT: {len(train_df)} train / {len(val_df)} val handwritten lines.")

    hw_train = _hw_generator(train_df, args.khatt_train)
    hw_val   = _hw_generator(val_df,   args.khatt_val)
    pr_gen   = _printed_generator(sentences, fonts)

    # Generate pages
    print(f"Generating {args.n_train} train pages …")
    for i in range(args.n_train):
        canvas, boxes = _make_page(hw_train, pr_gen)
        canvas = _degrade_page(canvas)
        _save(canvas, _yolo_label(boxes), args.out, f"page_{i:04d}", "train")
        if (i + 1) % 100 == 0:
            print(f"  train: {i + 1}/{args.n_train}")

    print(f"Generating {args.n_val} val pages …")
    for i in range(args.n_val):
        canvas, boxes = _make_page(hw_val, pr_gen)
        canvas = _degrade_page(canvas)
        _save(canvas, _yolo_label(boxes), args.out, f"page_{i:04d}", "val")

    # Write YAML
    yaml_path = os.path.join(args.out, "arabic_lines.yaml")
    with open(yaml_path, "w") as f:
        f.write(f'path: "{os.path.abspath(args.out)}"\n')
        f.write("train: images/train\nval: images/val\n\nnames:\n")
        f.write("  0: printed_line\n  1: handwritten_line\n")
    print(f"Done. YAML written to {yaml_path}")


if __name__ == "__main__":
    main()
