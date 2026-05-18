#!/usr/bin/env python3
"""OCR and translate a screen region once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.subtitle_translation_tool import (  # noqa: E402
    SubtitleTranslatorConfig,
    capture_translate_once,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture one screen region, OCR it, and translate it.")
    parser.add_argument("x1", type=int)
    parser.add_argument("y1", type=int)
    parser.add_argument("x2", type=int)
    parser.add_argument("y2", type=int)
    parser.add_argument("--source", default=None, help="Source language (default: env/config auto)")
    parser.add_argument("--target", default=None, help="Target language (default: env/config zh)")
    parser.add_argument("--ocr-lang", default=None, help="Tesseract OCR language (default: chi_sim+eng)")
    parser.add_argument("--tesseract-cmd", default=None, help="Path to tesseract.exe")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable OCR preprocessing")
    parser.add_argument("--output-json", help="Write result JSON to a file")
    args = parser.parse_args()

    config = SubtitleTranslatorConfig.from_env()
    if args.source:
        config.source_lang = args.source
    if args.target:
        config.target_lang = args.target
    if args.ocr_lang:
        config.ocr_language = args.ocr_lang
    if args.tesseract_cmd:
        config.tesseract_cmd = args.tesseract_cmd

    try:
        result = capture_translate_once(
            bbox=(args.x1, args.y1, args.x2, args.y2),
            config=config,
            preprocess=not args.no_preprocess,
        )
    except Exception as exception:
        print(f"Error: {exception}", file=sys.stderr)
        return 1

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved result to: {args.output_json}")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())