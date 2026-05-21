#!/usr/bin/env python3
"""
OCR Filter Script

Reads every image file in an input directory, runs PaddleOCR on it,
and moves images whose extracted text exceeds --min-chars characters to
the output directory.

Usage examples:
    python scripts/ocr_filter.py --input-dir ./images --output-dir ./text_images
    python scripts/ocr_filter.py --input-dir ./images --output-dir ./out --dry-run
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None  # type: ignore

# Allow running from project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.ocr_tool import OCRProcessor  # noqa: E402

IMAGE_EXTENSIONS: set = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
}


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
    min_chars: int = 30,
    lang: str = "ch",
    use_gpu: bool = False,
    dry_run: bool = False,
) -> None:
    """Process images in input_dir and move those with ≥ min_chars text.

    Args:
        input_dir: Directory containing source images.
        output_dir: Destination directory for matched images.
        min_chars: Text-length threshold (exclusive). Default 30.
        lang: PaddleOCR language code. Default 'ch'.
        use_gpu: Use GPU for inference.
        dry_run: Report actions without moving files.
    """
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()

    if not input_path.is_dir():
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    image_files = sorted(
        f
        for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        print(f"No image files found in: {input_dir}")
        return

    print(f"Found {len(image_files)} image(s)  |  threshold: >{min_chars} chars")

    if not dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    print("Initializing OCR processor…")
    processor = OCRProcessor(lang=lang, use_gpu=use_gpu)

    moved = skipped = errors = 0

    for idx, img_file in enumerate(image_files, 1):
        prefix = f"[{idx:>{len(str(len(image_files)))}}/{len(image_files)}]"
        print(f"{prefix} {img_file.name}", end="", flush=True)

        # --- dimension guard: skip OCR entirely, move directly to output ---
        if _PILImage is not None:
            with _PILImage.open(str(img_file)) as _im:
                _w, _h = _im.size
            if _w > 4000 or _h > 4000:
                print(f"  →  oversized ({_w}x{_h})  →  MOVE")
                if not dry_run:
                    output_path.mkdir(parents=True, exist_ok=True)
                    dest = _safe_dest(output_path, img_file)
                    shutil.move(str(img_file), str(dest))
                moved += 1
                continue

        try:
            over = processor.exceeds_text_length(str(img_file), min_chars)

            if over:
                print("  →  >%d chars  →  MOVE" % min_chars)
                if not dry_run:
                    dest = _safe_dest(output_path, img_file)
                    shutil.move(str(img_file), str(dest))
                moved += 1
            else:
                print("  →  ≤%d chars  →  skip" % min_chars)
                skipped += 1

        except Exception as exc:
            print(f"  →  ERROR: {exc}")
            errors += 1

    print()
    print(f"Done.  Moved: {moved}  |  Skipped: {skipped}  |  Errors: {errors}")
    if dry_run:
        print("(Dry-run mode — no files were moved)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move images whose OCR text exceeds a character threshold",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ocr_filter.py --input-dir ./images --output-dir ./text_images
  python scripts/ocr_filter.py --input-dir ./images --output-dir ./out --min-chars 50
  python scripts/ocr_filter.py --input-dir ./images --output-dir ./out --lang en
  python scripts/ocr_filter.py --input-dir ./images --output-dir ./out --gpu
  python scripts/ocr_filter.py --input-dir ./images --output-dir ./out --dry-run
        """,
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        metavar="DIR",
        help="Source directory containing image files",
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
        default=30,
        metavar="N",
        help="Move images with more than N extracted characters (default: 30)",
    )
    parser.add_argument(
        "--lang",
        default="ch",
        help="PaddleOCR language code: ch, en, japan, … (default: ch)",
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
