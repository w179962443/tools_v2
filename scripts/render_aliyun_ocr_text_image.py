#!/usr/bin/env python3
"""Render text extracted from Aliyun OCR JSON into an image.

Usage examples:
  python scripts/render_aliyun_ocr_text_image.py -i result.json -o result.png
  python scripts/render_aliyun_ocr_text_image.py -i result.json -o text.png \
      --source block
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageColor, ImageDraw, ImageFont
except ImportError as import_error:  # pragma: no cover - runtime dependency check
    Image = None  # type: ignore[assignment]
    ImageColor = None  # type: ignore[assignment]
    ImageDraw = None  # type: ignore[assignment]
    ImageFont = None  # type: ignore[assignment]
    PIL_IMPORT_ERROR = import_error
else:
    PIL_IMPORT_ERROR = None


DEFAULT_WIDTH = 1080
DEFAULT_MARGIN = 48
DEFAULT_FONT_SIZE = 30
DEFAULT_LINE_SPACING = 10
DEFAULT_PARAGRAPH_SPACING = 24
DEFAULT_BACKGROUND = "#ffffff"
DEFAULT_FOREGROUND = "#111827"


WINDOWS_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\Deng.ttf",
]

OTHER_FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]


def ensure_pillow_available() -> None:
    if PIL_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "Pillow is required. Install with: pip install Pillow"
    ) from PIL_IMPORT_ERROR


def load_json(path: str) -> Any:
    input_path = Path(path).expanduser().resolve()
    with input_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _decode_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return _decode_json_values(json.loads(stripped))
    except json.JSONDecodeError:
        return value


def _decode_json_values(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _decode_json_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_json_values(item) for item in value]
    return _decode_json_string(value)


def _mapping_get(mapping: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]

    lowered = {str(key).lower(): value for key, value in mapping.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None:
            return value
    return None


def _response_body(payload: Any) -> Any:
    payload = _decode_json_values(payload)
    if not isinstance(payload, Mapping):
        return payload
    body = _mapping_get(payload, ["body", "Body"])
    return body if body is not None else payload


def _response_data(payload: Any) -> Mapping[str, Any]:
    body = _response_body(payload)
    if not isinstance(body, Mapping):
        return {}

    data = _mapping_get(body, ["Data", "data"])
    data = _decode_json_values(data)
    return data if isinstance(data, Mapping) else {}


def _subimages(data: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    subimages = _mapping_get(data, ["SubImages", "subImages"])
    if not isinstance(subimages, list):
        return []
    return [item for item in subimages if isinstance(item, Mapping)]


def _extract_detail_texts(
    data: Mapping[str, Any],
    info_names: Sequence[str],
    detail_names: Sequence[str],
    content_names: Sequence[str],
) -> List[str]:
    texts = []
    for subimage in _subimages(data):
        info = _mapping_get(subimage, info_names)
        if not isinstance(info, Mapping):
            continue
        details = _mapping_get(info, detail_names)
        if not isinstance(details, list):
            continue
        for detail in details:
            if not isinstance(detail, Mapping):
                continue
            content = _mapping_get(detail, content_names)
            if isinstance(content, str) and content.strip():
                texts.append(content.strip())
    return texts


def extract_sections(payload: Any, source: str = "auto") -> Tuple[List[str], str]:
    data = _response_data(payload)
    if not data:
        return [], "none"

    extractors = {
        "paragraph": lambda: _extract_detail_texts(
            data,
            ["ParagraphInfo", "paragraphInfo"],
            ["ParagraphDetails", "paragraphDetails"],
            ["ParagraphContent", "paragraphContent"],
        ),
        "row": lambda: _extract_detail_texts(
            data,
            ["RowInfo", "rowInfo"],
            ["RowDetails", "rowDetails"],
            ["RowContent", "rowContent"],
        ),
        "block": lambda: _extract_detail_texts(
            data,
            ["BlockInfo", "blockInfo"],
            ["BlockDetails", "blockDetails"],
            ["BlockContent", "blockContent"],
        ),
        "content": lambda: [
            text.strip()
            for text in [str(_mapping_get(data, ["Content", "content"]) or "")]
            if text.strip()
        ],
    }

    if source == "auto":
        for candidate in ("paragraph", "row", "block", "content"):
            texts = extractors[candidate]()
            if texts:
                if candidate in ("row", "block"):
                    return ["\n".join(texts)], candidate
                return texts, candidate
        return [], "none"

    texts = extractors[source]()
    if source in ("row", "block") and texts:
        return ["\n".join(texts)], source
    return texts, source


def _font_candidates() -> List[Path]:
    return [Path(item) for item in WINDOWS_FONT_CANDIDATES + OTHER_FONT_CANDIDATES]


def load_font(font_path: Optional[str], font_size: int) -> Any:
    ensure_pillow_available()
    candidates = [Path(font_path).expanduser()] if font_path else _font_candidates()
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), font_size)

    if font_path:
        raise FileNotFoundError(f"Font file not found: {font_path}")

    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except OSError as os_error:
        raise RuntimeError(
            "No usable font was found. Pass --font with a Chinese-capable font path."
        ) from os_error


def text_width(draw: Any, text: str, font: Any) -> float:
    if not text:
        return 0.0
    return float(draw.textlength(text, font=font))


def wrap_visual_line(draw: Any, text: str, font: Any, max_width: int) -> List[str]:
    if not text:
        return [""]

    lines = []
    current = ""
    for char in text:
        candidate = current + char
        if current and text_width(draw, candidate, font) > max_width:
            lines.append(current.rstrip())
            current = char.lstrip()
        else:
            current = candidate

    if current:
        lines.append(current.rstrip())
    return lines or [""]


def wrap_section(draw: Any, section: str, font: Any, max_width: int) -> List[str]:
    wrapped = []
    visual_lines = section.splitlines() or [section]
    for visual_line in visual_lines:
        wrapped.extend(wrap_visual_line(draw, visual_line, font, max_width))
    return wrapped


def line_height(draw: Any, font: Any, line_spacing: int) -> int:
    bbox = draw.textbbox((0, 0), "Ag\u56fd", font=font)
    return bbox[3] - bbox[1] + line_spacing


def parse_color(value: str) -> Tuple[int, int, int]:
    ensure_pillow_available()
    return ImageColor.getrgb(value)


def render_text_image(
    sections: Sequence[str],
    output_path: str,
    width: int = DEFAULT_WIDTH,
    margin: int = DEFAULT_MARGIN,
    font_size: int = DEFAULT_FONT_SIZE,
    line_spacing: int = DEFAULT_LINE_SPACING,
    paragraph_spacing: int = DEFAULT_PARAGRAPH_SPACING,
    font_path: Optional[str] = None,
    background: str = DEFAULT_BACKGROUND,
    foreground: str = DEFAULT_FOREGROUND,
) -> Path:
    ensure_pillow_available()
    if width <= margin * 2:
        raise ValueError("--width must be greater than twice --margin.")

    font = load_font(font_path, font_size)
    text_color = parse_color(foreground)
    background_color = parse_color(background)
    measure_image = Image.new("RGB", (width, 10), background_color)
    measure_draw = ImageDraw.Draw(measure_image)
    max_text_width = width - margin * 2
    computed_line_height = line_height(measure_draw, font, line_spacing)

    section_lines = [
        wrap_section(measure_draw, section, font, max_text_width)
        for section in sections
    ]
    content_height = 0
    for index, lines in enumerate(section_lines):
        content_height += max(1, len(lines)) * computed_line_height
        if index < len(section_lines) - 1:
            content_height += paragraph_spacing

    height = max(margin * 2 + content_height, margin * 2 + computed_line_height)
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)

    y = margin
    for section_index, lines in enumerate(section_lines):
        for line in lines:
            draw.text((margin, y), line, fill=text_color, font=font)
            y += computed_line_height
        if section_index < len(section_lines) - 1:
            y += paragraph_spacing

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    return output


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Aliyun OCR JSON text/paragraphs into a PNG image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/render_aliyun_ocr_text_image.py -i result.json -o result.png
  python scripts/render_aliyun_ocr_text_image.py -i result.json -o text.png \
      --source block --width 1200 --font-size 32
        """,
    )
    parser.add_argument("-i", "--input", required=True, help="Aliyun OCR JSON file")
    parser.add_argument("-o", "--output", required=True, help="Output image file")
    parser.add_argument(
        "--source",
        choices=["auto", "paragraph", "row", "block", "content"],
        default="auto",
        help="Which OCR text field to render (default: auto)",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--margin", type=int, default=DEFAULT_MARGIN)
    parser.add_argument("--font-size", type=int, default=DEFAULT_FONT_SIZE)
    parser.add_argument("--line-spacing", type=int, default=DEFAULT_LINE_SPACING)
    parser.add_argument(
        "--paragraph-spacing",
        type=int,
        default=DEFAULT_PARAGRAPH_SPACING,
    )
    parser.add_argument("--font", help="Path to a TTF/TTC font file")
    parser.add_argument("--background", default=DEFAULT_BACKGROUND)
    parser.add_argument("--foreground", default=DEFAULT_FOREGROUND)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        payload = load_json(args.input)
        sections, selected_source = extract_sections(payload, args.source)
        if not sections:
            raise RuntimeError(f"No OCR text found for source: {args.source}")
        output = render_text_image(
            sections=sections,
            output_path=args.output,
            width=args.width,
            margin=args.margin,
            font_size=args.font_size,
            line_spacing=args.line_spacing,
            paragraph_spacing=args.paragraph_spacing,
            font_path=args.font,
            background=args.background,
            foreground=args.foreground,
        )
    except Exception as exception:
        print(f"Error: {exception}", file=sys.stderr)
        return 1

    print(f"Rendered {selected_source} text to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
