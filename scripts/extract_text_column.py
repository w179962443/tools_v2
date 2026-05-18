#!/usr/bin/env python3
"""Extract one CSV column to a plain text file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.text_processing_tool import extract_text_column, read_csv_column  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract a CSV column into a UTF-8 text file.")
    parser.add_argument("input_file", help="Source CSV or CSV-formatted TXT file")
    parser.add_argument("-o", "--output-file", help="Output TXT path")
    parser.add_argument("-c", "--column", default="text", help="Column name (default: text)")
    args = parser.parse_args()

    try:
        output_path = extract_text_column(
            input_file=args.input_file,
            output_file=args.output_file,
            column_name=args.column,
        )
        row_count = len(read_csv_column(args.input_file, args.column))
    except Exception as exception:
        print(f"Error: {exception}", file=sys.stderr)
        return 1

    print(f"Extracted {row_count} rows from '{args.column}' to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())