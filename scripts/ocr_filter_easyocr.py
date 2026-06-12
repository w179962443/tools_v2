#!/usr/bin/env python3
"""
OCR Filter Script (EasyOCR version)

Recursively scans a directory for image files, runs EasyOCR on each,
and moves images whose extracted text length exceeds --min-chars to
the output directory (preserving relative subdirectory structure).

Usage examples:
    python scripts/ocr_filter_easyocr.py --input-dir ./images --output-dir ./text_images
    python scripts/ocr_filter_easyocr.py --input-dir ./images --output-dir ./out --min-chars 50 --dry-run
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import easyocr
from PIL import Image


IMAGE_EXTENSIONS: set = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
}


MAX_DIMENSION = 4000
OCR_TIMEOUT = 3.0


def _exceeds_text_threshold(reader, img_array, threshold, timeout=OCR_TIMEOUT):
    """Check if OCR text length exceeds threshold, stopping recognition early.

    Uses EasyOCR's detect+recognize split: detection runs once on the full
    image, then each detected text region is recognized individually. As soon
    as accumulated character count exceeds the threshold, remaining regions
    are skipped. Also returns True if total OCR time exceeds timeout seconds.
    """
    t0 = time.monotonic()

    h_lists, f_lists = reader.detect(img_array, reformat=False)
    img_grey = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    if time.monotonic() - t0 > timeout:
        return True

    total = 0
    boxes = h_lists[0] if h_lists else []
    free_boxes = f_lists[0] if f_lists else []

    for bbox in boxes:
        result = reader.recognize(img_grey, [bbox], [], reformat=False, detail=0)
        for text in result:
            total += len(text)
        if total > threshold:
            return True
        if time.monotonic() - t0 > timeout:
            return True

    for bbox in free_boxes:
        result = reader.recognize(img_grey, [], [bbox], reformat=False, detail=0)
        for text in result:
            total += len(text)
        if total > threshold:
            return True
        if time.monotonic() - t0 > timeout:
            return True

    return False


def _safe_dest(dest_dir: Path, src: Path) -> Path:
    """Return a collision-free destination path inside dest_dir."""
    dest = dest_dir / src.name
    if not dest.exists():
        return dest
    counter = 1
    while True:
        dest = dest_dir / f"{src.stem}_{counter}{src.suffix}"
        if not dest.exists():
            return dest
        counter += 1


def process_directory(
    input_dir: str,
    output_dir: str,
    min_chars: int = 50,
    lang: list = None,
    use_gpu: bool = False,
    dry_run: bool = False,
) -> None:
    """Recursively process images in input_dir, move text-rich ones to output_dir.

    Args:
        input_dir: Directory containing source images (searched recursively).
        output_dir: Destination directory for matched images.
        min_chars: Text-length threshold (exclusive). Default 50.
        lang: List of EasyOCR language codes. Default ['ch_sim', 'en'].
        use_gpu: Use GPU for inference.
        dry_run: Report actions without moving files.
    """
    if lang is None:
        lang = ["ch_sim", "en"]

    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()

    if not input_path.is_dir():
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    image_files = sorted(
        f
        for f in input_path.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        print(f"No image files found in: {input_dir}")
        return

    print(f"Found {len(image_files)} image(s)  |  threshold: >{min_chars} chars")

    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    print("Initializing EasyOCR…")
    reader = easyocr.Reader(lang, gpu=use_gpu)

    moved = skipped = errors = 0

    for idx, img_file in enumerate(image_files, 1):
        prefix = f"[{idx:>{len(str(len(image_files)))}}/{len(image_files)}]"
        rel = img_file.relative_to(input_path)
        print(f"{prefix} {rel}", end="", flush=True)

        try:
            with Image.open(str(img_file)) as pil_img:
                w, h = pil_img.size
                oversized = w > MAX_DIMENSION or h > MAX_DIMENSION
                if not oversized:
                    img_array = np.array(pil_img.convert("RGB"))

            if oversized:
                print(f"  →  oversized ({w}x{h})  →  MOVE")
                if not dry_run:
                    dest_subdir = output_path / img_file.parent.relative_to(input_path)
                    dest_subdir.mkdir(parents=True, exist_ok=True)
                    dest = _safe_dest(dest_subdir, img_file)
                    shutil.move(str(img_file), str(dest))
                moved += 1
                continue

            t0 = time.monotonic()
            exceeds = _exceeds_text_threshold(reader, img_array, min_chars)
            elapsed = time.monotonic() - t0

            if exceeds:
                reason = "timeout" if elapsed > OCR_TIMEOUT else f">{min_chars} chars"
                print(f"  →  {reason}  →  MOVE")
                if not dry_run:
                    dest_subdir = output_path / img_file.parent.relative_to(input_path)
                    dest_subdir.mkdir(parents=True, exist_ok=True)
                    dest = _safe_dest(dest_subdir, img_file)
                    shutil.move(str(img_file), str(dest))
                moved += 1
            else:
                print(f"  →  ≤{min_chars} chars  →  skip")
                skipped += 1

        except Exception as exc:
            exc_msg = str(exc)
            if "UnidentifiedImageError" in type(exc).__name__ or "cannot identify" in exc_msg.lower():
                print(f"  →  SKIP (corrupt/unreadable)")
                skipped += 1
            else:
                print(f"  →  ERROR: {exc}")
                errors += 1

    print()
    print(f"Done.  Moved: {moved}  |  Skipped: {skipped}  |  Errors: {errors}")
    if dry_run:
        print("(Dry-run mode — no files were moved)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move images whose OCR text exceeds a character threshold (EasyOCR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ocr_filter_easyocr.py --input-dir ./images --output-dir ./text_images
  python scripts/ocr_filter_easyocr.py --input-dir ./images --output-dir ./out --min-chars 50
  python scripts/ocr_filter_easyocr.py --input-dir ./images --output-dir ./out --lang ch_sim en
  python scripts/ocr_filter_easyocr.py --input-dir ./images --output-dir ./out --gpu
  python scripts/ocr_filter_easyocr.py --input-dir ./images --output-dir ./out --dry-run
        """,
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        metavar="DIR",
        help="Source directory containing image files (searched recursively)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Destination directory for images that pass the threshold",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=50,
        metavar="N",
        help="Move images with more than N extracted characters (default: 50)",
    )
    parser.add_argument(
        "--lang",
        nargs="+",
        default=["ch_sim", "en"],
        help="EasyOCR language codes: ch_sim, ch_tra, en, ja, ko, … (default: ch_sim en)",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without moving any files",
    )

    args = parser.parse_args()

    process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_chars=args.min_chars,
        lang=args.lang,
        use_gpu=args.gpu,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()