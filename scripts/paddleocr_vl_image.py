#!/usr/bin/env python3
"""Run PaddleOCR-VL-1.5 on one image and write the result to one file.

Usage examples:
    python scripts/paddleocr_vl_image.py --image ./page.png --output ./result.md
    python scripts/paddleocr_vl_image.py -i ./page.png -o ./result.json
    python scripts/paddleocr_vl_image.py -i ./page.png -o ./result.md --device gpu
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List, Literal, Optional, Union

OutputFormat = Literal["markdown", "json"]


def _load_paddleocr_vl():
    try:
        from paddleocr import PaddleOCRVL
    except ImportError as import_error:
        raise RuntimeError(
            "PaddleOCRVL is unavailable. Install PaddlePaddle 3.2.1+ and "
            'PaddleOCR with: python -m pip install -U "paddleocr[doc-parser]"'
        ) from import_error

    return PaddleOCRVL


def _infer_output_format(
    output_path: Path,
    output_format: Optional[str],
) -> OutputFormat:
    if output_format is not None:
        return output_format  # type: ignore[return-value]
    if output_path.suffix.lower() == ".json":
        return "json"
    return "markdown"


def _result_files(directory: Path, suffixes: Iterable[str]) -> List[Path]:
    suffix_set = {suffix.lower() for suffix in suffixes}
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in suffix_set
    )


def _write_markdown(markdown_files: List[Path], output_path: Path) -> None:
    if not markdown_files:
        raise RuntimeError("PaddleOCR-VL did not save any Markdown result files.")

    parts: List[str] = []
    for index, markdown_file in enumerate(markdown_files, start=1):
        text = markdown_file.read_text(encoding="utf-8").strip()
        if len(markdown_files) > 1:
            parts.append(
                f"<!-- PaddleOCR-VL result {index}: {markdown_file.name} -->\n\n"
            )
        parts.append(text)
        parts.append("\n\n")

    output_path.write_text("".join(parts).rstrip() + "\n", encoding="utf-8")


def _write_json(json_files: List[Path], output_path: Path) -> None:
    if not json_files:
        raise RuntimeError("PaddleOCR-VL did not save any JSON result files.")

    payloads = []
    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as file_obj:
            payloads.append(json.load(file_obj))

    payload = payloads[0] if len(payloads) == 1 else payloads
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")


def run_paddleocr_vl_image(
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    output_format: Optional[OutputFormat] = None,
    device: str = "cpu",
    engine: str = "paddlepaddle",
    use_doc_orientation_classify: bool = False,
    use_doc_unwarping: bool = False,
    use_layout_detection: bool = True,
    verbose: bool = False,
) -> Path:
    """Run PaddleOCR-VL on a single image and save Markdown or JSON output."""
    image = Path(image_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    selected_format = _infer_output_format(output, output_format)

    if not image.is_file():
        raise FileNotFoundError(f"Image file not found: {image}")

    output.parent.mkdir(parents=True, exist_ok=True)

    if device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    PaddleOCRVL = _load_paddleocr_vl()

    pipeline_kwargs: Dict[str, object] = {
        "device": device,
        "use_doc_orientation_classify": use_doc_orientation_classify,
        "use_doc_unwarping": use_doc_unwarping,
        "use_layout_detection": use_layout_detection,
    }
    if engine == "transformers":
        pipeline_kwargs["engine"] = "transformers"

    print("Initializing PaddleOCR-VL-1.5 pipeline...", file=sys.stderr, flush=True)
    pipeline = PaddleOCRVL(**pipeline_kwargs)

    print(f"Running OCR: {image}", file=sys.stderr, flush=True)
    results = list(pipeline.predict(str(image)))
    if not results:
        raise RuntimeError("PaddleOCR-VL returned no results.")

    with TemporaryDirectory(prefix="paddleocr_vl_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        for result in results:
            if verbose:
                result.print()
            if selected_format == "json":
                result.save_to_json(save_path=temp_dir)
            else:
                result.save_to_markdown(save_path=temp_dir)

        if selected_format == "json":
            _write_json(_result_files(temp_dir, [".json"]), output)
        else:
            _write_markdown(_result_files(temp_dir, [".md", ".markdown"]), output)

    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR-VL-1.5 on one image and write the result to a file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/paddleocr_vl_image.py --image ./page.png --output ./result.md
  python scripts/paddleocr_vl_image.py -i ./page.png -o ./result.json
  python scripts/paddleocr_vl_image.py -i ./page.png -o ./result.md --device gpu
        """,
    )
    parser.add_argument(
        "-i",
        "--image",
        required=True,
        help="Path to the input image file",
    )
    parser.add_argument("-o", "--output", required=True, help="Path to the output file")
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        help="Output format. Defaults to JSON for .json outputs, otherwise Markdown.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "PaddleOCR device, for example cpu, gpu, xpu, dcu, or metax_gpu "
            "(default: cpu)"
        ),
    )
    parser.add_argument(
        "--engine",
        choices=["paddlepaddle", "transformers"],
        default="paddlepaddle",
        help="Inference engine (default: paddlepaddle)",
    )
    parser.add_argument(
        "--use-doc-orientation-classify",
        action="store_true",
        help="Enable document orientation classification before parsing",
    )
    parser.add_argument(
        "--use-doc-unwarping",
        action="store_true",
        help="Enable document unwarping before parsing",
    )
    parser.add_argument(
        "--no-layout-detection",
        action="store_true",
        help="Disable layout detection and ordering",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print PaddleOCR-VL structured results while processing",
    )

    args = parser.parse_args()

    try:
        saved_path = run_paddleocr_vl_image(
            image_path=args.image,
            output_path=args.output,
            output_format=args.format,
            device=args.device,
            engine=args.engine,
            use_doc_orientation_classify=args.use_doc_orientation_classify,
            use_doc_unwarping=args.use_doc_unwarping,
            use_layout_detection=not args.no_layout_detection,
            verbose=args.verbose,
        )
    except Exception as exception:
        print(f"Error: {exception}", file=sys.stderr)
        return 1

    print(f"Saved PaddleOCR-VL result to: {saved_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
