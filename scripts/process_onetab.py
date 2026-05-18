#!/usr/bin/env python3
"""Merge, deduplicate, classify, and export OneTab text files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.text_processing_tool import ONETAB_CATEGORIES, process_onetab_files  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process OneTab export .txt files into category files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/process_onetab.py --input d:\\demo-project-onetab
  python scripts/process_onetab.py --input d:\\demo-project-onetab --output d:\\output
        """,
    )
    parser.add_argument("--input", required=True, help="Directory containing OneTab .txt exports")
    parser.add_argument("--output", help="Output directory (default: <input>/output)")
    parser.add_argument(
        "--separator-interval",
        type=int,
        default=30,
        help="Insert a blank line every N records (default: 30)",
    )
    args = parser.parse_args()

    try:
        result = process_onetab_files(
            input_dir=args.input,
            output_dir=args.output,
            separator_interval=args.separator_interval,
        )
    except Exception as exception:
        print(f"Error: {exception}", file=sys.stderr)
        return 1

    print(f"Input files: {len(result['input_files'])}")
    print(f"Merged records: {result['merged_count']}")
    print(f"Deduplicated records: {result['deduplicated_count']}")
    print(f"Duplicates removed: {result['duplicate_count']}")
    print("Category counts:")
    for category in ONETAB_CATEGORIES:
        print(f"  {category}: {result['category_counts'][category]}")
    print(f"Output directory: {result['output_dir']}")
    for category, output_path in result["output_files"].items():
        print(f"  {category}: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())