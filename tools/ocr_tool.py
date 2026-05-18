#!/usr/bin/env python3
"""
OCR workflow tool based on PaddleOCR.

Provides both a programmatic Python API and a command-line interface.

CLI Usage:
    python -m tools.ocr_tool --image photo.jpg
    python tools/ocr_tool.py --image photo.jpg --lang en --output-json result.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# --- PaddlePaddle compatibility fixes (must be set before paddle is imported) ---
# Hide CUDA devices so the GPU version of Paddle doesn't try to initialise CUDA
# DLLs that exceed the Windows paging-file budget.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
# Disable the new PIR (Program IR) executor, which conflicts with MKL-DNN on
# PaddlePaddle 3.x and causes native crashes. OCR inference models are also
# added to paddlex's NEWIR_BLOCKLIST (see new_ir_blocklist.py) so that
# config.enable_new_ir(False) is called at predictor-creation time.
os.environ.setdefault("FLAGS_enable_pir_api", "0")
# Skip network connectivity checks at startup.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
# ---------------------------------------------------------------------------

try:
    from PIL import Image as _PILImage
except ImportError:
    _PILImage = None  # type: ignore

try:
    from paddleocr import PaddleOCR
except ImportError:
    print(
        "Error: paddleocr is not installed.\n" "Install with: pip install paddleocr",
        file=sys.stderr,
    )
    sys.exit(1)


class OCRProcessor:
    """OCR processor using PaddleOCR.

    Supports Chinese (ch), English (en), and many other languages.
    """

    def __init__(
        self,
        lang: str = "ch",
        use_gpu: bool = False,
        show_log: bool = False,
    ) -> None:
        """Initialize OCR processor.

        Args:
            lang: Language code for OCR. 'ch' supports Chinese + English.
                  Other options: 'en', 'japan', 'korean', 'french', etc.
            use_gpu: Whether to use GPU acceleration.
            show_log: Whether to show PaddleOCR internal logs.
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self._ocr: Optional[PaddleOCR] = None
        self._show_log = show_log

    @property
    def ocr(self) -> PaddleOCR:
        """Lazily initialize PaddleOCR (first call downloads models)."""
        if self._ocr is None:
            self._ocr = PaddleOCR(
                use_textline_orientation=True,
                lang=self.lang,
                device="gpu" if self.use_gpu else "cpu",
                # Disable the document-unwarping step (UVDoc model): it tries
                # to process the full-resolution image in one shot which can
                # exhaust RAM on machines with a large/tall scan image.
                use_doc_unwarping=False,
                # Limit the long side of the image fed to the text-detector so
                # that the preprocessed tensor stays within available memory.
                text_det_limit_side_len=960,
            )
        return self._ocr

    @staticmethod
    def _check_image_size(image_path: str, limit: int = 4000) -> None:
        """Raise ValueError if either dimension of the image exceeds limit."""
        if _PILImage is None:
            return
        with _PILImage.open(image_path) as im:
            w, h = im.size
        if w > limit or h > limit:
            raise ValueError(
                f"Image dimension {w}x{h} exceeds the {limit}px limit "
                f"(file: {image_path})"
            )

    def extract_text(self, image_path: str) -> str:
        """Extract all text from an image as a single string.

        Args:
            image_path: Path to the image file.

        Returns:
            Concatenated text extracted from the image.

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If either image dimension exceeds 4000 px.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        self._check_image_size(image_path)

        try:
            result = self.ocr.predict(image_path)
        except Exception as exc:
            raise RuntimeError(f"OCR prediction failed: {exc}") from exc

        text_parts = []
        if result and result[0]:
            for text in result[0]["rec_texts"]:
                text_parts.append(text)

        return "".join(text_parts)

    def get_text_length(self, image_path: str) -> int:
        """Return the number of characters extracted from an image.

        Args:
            image_path: Path to the image file.

        Returns:
            Character count of extracted text.
        """
        return len(self.extract_text(image_path))

    def exceeds_text_length(self, image_path: str, threshold: int) -> bool:
        """Return True as soon as accumulated OCR text exceeds threshold.

        Iterates over the per-block results returned by PaddleOCR and stops
        counting the moment the running total surpasses *threshold*, avoiding
        unnecessary work when the image clearly has enough text.

        Args:
            image_path: Path to the image file.
            threshold: Character count to beat (exclusive: > threshold).

        Returns:
            True if extracted text length > threshold, False otherwise.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        self._check_image_size(image_path)

        try:
            result = self.ocr.predict(image_path)
        except Exception as exc:
            raise RuntimeError(f"OCR prediction failed: {exc}") from exc

        total = 0
        if result and result[0]:
            for text in result[0]["rec_texts"]:
                total += len(text)
                if total > threshold:
                    return True
        return False

    def process_image(self, image_path: str) -> dict:
        """Process an image and return detailed OCR results.

        Args:
            image_path: Path to the image file.

        Returns:
            Dict with keys:
                - 'text' (str): full concatenated text
                - 'length' (int): character count
                - 'lines' (list): per-line dicts with 'text', 'confidence', 'bbox'

        Raises:
            FileNotFoundError: If the image file does not exist.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        self._check_image_size(image_path)

        try:
            result = self.ocr.predict(image_path)
        except Exception as exc:
            raise RuntimeError(f"OCR prediction failed: {exc}") from exc

        lines = []
        if result and result[0]:
            res = result[0]
            for text, score, poly in zip(
                res["rec_texts"], res["rec_scores"], res["rec_polys"]
            ):
                lines.append(
                    {
                        "text": text,
                        "confidence": float(score),
                        "bbox": poly,
                    }
                )

        full_text = "".join(ln["text"] for ln in lines)
        return {
            "text": full_text,
            "length": len(full_text),
            "lines": lines,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCR tool based on PaddleOCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m tools.ocr_tool --image photo.jpg
  python -m tools.ocr_tool --image photo.jpg --lang en
  python -m tools.ocr_tool --image photo.jpg --output-json result.json
  python -m tools.ocr_tool --image photo.jpg --text-only
  python -m tools.ocr_tool --image photo.jpg --length-only
        """,
    )

    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--lang",
        default="ch",
        help="OCR language code (default: ch). Options: ch, en, japan, korean, …",
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument(
        "--output-json", metavar="FILE", help="Save full results to a JSON file"
    )
    parser.add_argument(
        "--text-only", action="store_true", help="Print only the extracted text"
    )
    parser.add_argument(
        "--length-only",
        action="store_true",
        help="Print only the character count (useful for scripting)",
    )

    args = parser.parse_args()

    processor = OCRProcessor(lang=args.lang, use_gpu=args.gpu)

    if args.length_only:
        print(processor.get_text_length(args.image), flush=True)
        return

    try:
        result = processor.process_image(args.image)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

    if args.text_only:
        print(result["text"], flush=True)
        return

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output_json}", flush=True)
        return

    print(f"File  : {args.image}", flush=True)
    print(f"Chars : {result['length']}", flush=True)
    print(f"Lines : {len(result['lines'])}", flush=True)
    if result["length"] == 0:
        print("Text  : (no text detected)", flush=True)
    else:
        print("Text  :", flush=True)
        print(result["text"], flush=True)


if __name__ == "__main__":
    main()
