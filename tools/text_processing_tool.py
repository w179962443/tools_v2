"""
Small text-processing utilities migrated from the older daily tools folder.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ONETAB_CATEGORIES: tuple[str, ...] = ("github", "zhihu", "ai", "other")


def detect_csv_dialect(sample: str) -> csv.Dialect:
    """Detect a CSV dialect, falling back to the Excel dialect."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        if dialect.delimiter in {"\r", "\n"}:
            raise csv.Error("Line breaks are not valid delimiters.")
        return dialect
    except csv.Error:
        return csv.get_dialect("excel")


def read_csv_column(input_file: str | Path, column_name: str = "text") -> list[str]:
    """Read non-empty values from a named column in a CSV-like file."""
    input_path = Path(input_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    encodings = ["utf-8-sig", "utf-8", "gb18030"]
    last_error: UnicodeDecodeError | None = None

    for encoding in encodings:
        try:
            with input_path.open("r", encoding=encoding, newline="") as handle:
                sample = handle.read(4096)
                handle.seek(0)
                dialect = detect_csv_dialect(sample)
                reader = csv.DictReader(handle, dialect=dialect)

                if reader.fieldnames is None:
                    raise ValueError("Input file does not contain a header row.")

                if column_name not in reader.fieldnames:
                    available_columns = ", ".join(reader.fieldnames)
                    raise ValueError(
                        f"Column '{column_name}' was not found. "
                        f"Available columns: {available_columns}"
                    )

                return [
                    row[column_name].strip()
                    for row in reader
                    if row.get(column_name) and row[column_name].strip()
                ]
        except UnicodeDecodeError as decode_error:
            last_error = decode_error
            continue

    if last_error is not None:
        raise ValueError(
            "Unable to decode the input file. Tried: " + ", ".join(encodings)
        ) from last_error

    raise ValueError("Failed to read the input file.")


def extract_text_column(
    input_file: str | Path,
    output_file: str | Path | None = None,
    column_name: str = "text",
) -> Path:
    """Extract a CSV column to a UTF-8 text file and return the output path."""
    input_path = Path(input_file)
    output_path = (
        Path(output_file)
        if output_file is not None
        else input_path.with_name(f"{input_path.stem}_{column_name}.txt")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = read_csv_column(input_path, column_name=column_name)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def read_onetab_file(file_path: str | Path) -> list[str]:
    """Read a OneTab export text file, dropping blank lines."""
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle if line.strip()]


def extract_onetab_url(record: str) -> str:
    """Extract the URL part from a OneTab record."""
    if "|" in record:
        return record.split("|", 1)[0].strip()
    return record.strip()


def merge_onetab_files(file_paths: list[str | Path]) -> list[str]:
    """Interleave records from multiple OneTab export files."""
    all_records = [read_onetab_file(file_path) for file_path in file_paths]
    max_length = max((len(records) for records in all_records), default=0)
    merged: list[str] = []

    for record_index in range(max_length):
        for records in all_records:
            if record_index < len(records):
                merged.append(records[record_index])
    return merged


def deduplicate_onetab_records(records: list[str]) -> list[str]:
    """Deduplicate OneTab records by URL, preserving the first occurrence."""
    seen_urls: set[str] = set()
    deduplicated: list[str] = []

    for record in records:
        url = extract_onetab_url(record)
        if url not in seen_urls:
            seen_urls.add(url)
            deduplicated.append(record)
    return deduplicated


def classify_onetab_record(record: str) -> str:
    """Classify a OneTab record into github, zhihu, ai, or other."""
    url = extract_onetab_url(record).lower()
    if "github" in url:
        return "github"
    if "zhihu" in url:
        return "zhihu"
    if any(keyword in url for keyword in ("chatgpt", "claude", "gemini")):
        return "ai"
    return "other"


def add_visual_separators(records: list[str], interval: int = 30) -> list[str]:
    """Insert blank lines every ``interval`` records for easier reading."""
    if interval <= 0:
        return records.copy()

    separated: list[str] = []
    for record_index, record in enumerate(records):
        separated.append(record)
        if (record_index + 1) % interval == 0 and record_index + 1 < len(records):
            separated.append("")
    return separated


def process_onetab_files(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    separator_interval: int = 30,
    timestamp: str | None = None,
) -> dict[str, object]:
    """Merge, deduplicate, classify, and write OneTab export files."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_path}")

    output_path = Path(output_dir) if output_dir is not None else input_path / "output"
    output_path.mkdir(parents=True, exist_ok=True)

    input_files = sorted(path for path in input_path.glob("*.txt") if path.is_file())
    if not input_files:
        raise FileNotFoundError(f"No .txt files found in: {input_path}")

    merged_records = merge_onetab_files(input_files)
    deduplicated_records = deduplicate_onetab_records(merged_records)

    classified: dict[str, list[str]] = defaultdict(list)
    for record in deduplicated_records:
        category = classify_onetab_record(record)
        classified[category].append(record)

    time_prefix = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_files: dict[str, Path] = {}
    for category in ONETAB_CATEGORIES:
        records = classified.get(category, [])
        if not records:
            continue
        category_path = output_path / f"{time_prefix}_{category}.txt"
        category_path.write_text(
            "\n".join(add_visual_separators(records, separator_interval)) + "\n",
            encoding="utf-8",
        )
        output_files[category] = category_path

    return {
        "input_files": input_files,
        "output_dir": output_path,
        "output_files": output_files,
        "merged_count": len(merged_records),
        "deduplicated_count": len(deduplicated_records),
        "duplicate_count": len(merged_records) - len(deduplicated_records),
        "category_counts": {category: len(classified.get(category, [])) for category in ONETAB_CATEGORIES},
    }