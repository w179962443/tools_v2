#!/usr/bin/env python3
"""Call Aliyun OCR RecognizeAllText for one local image/PDF or image URL.

Usage examples:
    python scripts/aliyun_ocr_image.py -i data/v1.jpg -o result.json
    python scripts/aliyun_ocr_image.py -i data/v1.jpg --text-only
    python scripts/aliyun_ocr_image.py -i table.png --type Advanced --table
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Set, Tuple

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
except ImportError:  # pragma: no cover - runtime message is clearer here
    Image = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]
    UnidentifiedImageError = OSError  # type: ignore[assignment]

try:
    from alibabacloud_ocr_api20210707.client import Client as AliyunOcrClient
    from alibabacloud_ocr_api20210707 import models as ocr_models
    from alibabacloud_tea_openapi import models as open_api_models
    from alibabacloud_tea_util import models as util_models
except ImportError as import_error:  # pragma: no cover - shown at runtime only
    AliyunOcrClient = None  # type: ignore[assignment]
    ocr_models = None  # type: ignore[assignment]
    open_api_models = None  # type: ignore[assignment]
    util_models = None  # type: ignore[assignment]
    SDK_IMPORT_ERROR = import_error
else:
    SDK_IMPORT_ERROR = None


DEFAULT_ENDPOINT = "ocr-api.cn-hangzhou.aliyuncs.com"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "aliyun_ocr_config.json"
ALIYUN_MIN_DIMENSION = 5
ALIYUN_MAX_DIMENSION = 8192
DEFAULT_JPEG_QUALITY = 90
MAX_BODY_BYTES = 10 * 1024 * 1024


def ensure_sdk_available() -> None:
    if SDK_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "Aliyun OCR SDK is not installed in the current Python environment. "
        "Install it with: pip install alibabacloud_ocr_api20210707==3.1.3"
    ) from SDK_IMPORT_ERROR


def resolve_config_path(config_path: Optional[str] = None) -> Path:
    selected_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    selected_path = selected_path.expanduser()
    if selected_path.is_absolute():
        return selected_path
    return (Path.cwd() / selected_path).resolve()


def load_aliyun_config(
    config_path: Optional[str] = None,
) -> Tuple[Mapping[str, Any], Path]:
    resolved_path = resolve_config_path(config_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Aliyun OCR config file not found: {resolved_path}")

    try:
        with resolved_path.open("r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    except json.JSONDecodeError as json_error:
        raise ValueError(
            f"Aliyun OCR config file is not valid JSON: {resolved_path}"
        ) from json_error

    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Aliyun OCR config file must contain a JSON object: {resolved_path}"
        )
    return payload, resolved_path


def _config_string(config: Mapping[str, Any], names: Iterable[str]) -> Optional[str]:
    lowered = {str(key).lower(): value for key, value in config.items()}
    value = None
    for name in names:
        if name in config:
            value = config[name]
            break
        lowered_value = lowered.get(name.lower())
        if lowered_value is not None:
            value = lowered_value
            break

    if value is None:
        return None
    value_text = str(value).strip()
    return value_text or None


def create_client(
    config_path: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> Any:
    """Create an Aliyun OCR client using access keys from the JSON config."""
    ensure_sdk_available()
    credential_config, resolved_config_path = load_aliyun_config(config_path)
    access_key_id = _config_string(
        credential_config,
        ["access_key_id", "accessKeyId", "ALIBABA_CLOUD_ACCESS_KEY_ID"],
    )
    access_key_secret = _config_string(
        credential_config,
        [
            "access_key_secret",
            "accessKeySecret",
            "ALIBABA_CLOUD_ACCESS_KEY_SECRET",
        ],
    )
    security_token = _config_string(
        credential_config,
        ["security_token", "securityToken", "ALIBABA_CLOUD_SECURITY_TOKEN"],
    )
    config_endpoint = _config_string(credential_config, ["endpoint"])

    missing_fields = []
    if access_key_id is None:
        missing_fields.append("access_key_id")
    if access_key_secret is None:
        missing_fields.append("access_key_secret")
    if missing_fields:
        raise RuntimeError(
            f"Missing {', '.join(missing_fields)} in Aliyun OCR config file: "
            f"{resolved_config_path}"
        )

    config = _build_model(
        open_api_models.Config,
        [
            (("access_key_id",), access_key_id),
            (("access_key_secret",), access_key_secret),
            (("security_token",), security_token),
        ],
    )
    config.access_key_id = access_key_id
    config.access_key_secret = access_key_secret
    if security_token is not None:
        config.security_token = security_token
    config.endpoint = endpoint or config_endpoint or DEFAULT_ENDPOINT
    return AliyunOcrClient(config)


def _constructor_parameters(model_class: type) -> Optional[Set[str]]:
    try:
        signature = inspect.signature(model_class.__init__)
    except (TypeError, ValueError):
        return None

    parameters = set()
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return None
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            parameters.add(name)
    return parameters


def _select_supported_name(
    model_class: type,
    candidates: Sequence[str],
) -> Optional[str]:
    parameters = _constructor_parameters(model_class)
    if parameters is None:
        return candidates[0]
    for candidate in candidates:
        if candidate in parameters:
            return candidate
    return None


def _build_model(
    model_class: type,
    values: Iterable[Tuple[Sequence[str], Any]],
) -> Any:
    kwargs = {}
    for candidates, value in values:
        if value is None:
            continue
        selected_name = _select_supported_name(model_class, candidates)
        if selected_name is not None:
            kwargs[selected_name] = value
    return model_class(**kwargs)


def _find_model_class(class_names: Iterable[str]) -> Optional[type]:
    if ocr_models is None:
        return None
    for class_name in class_names:
        model_class = getattr(ocr_models, class_name, None)
        if model_class is not None:
            return model_class
    return None


def _build_advanced_config(args: argparse.Namespace) -> Optional[Any]:
    config_values = [
        (("output_row",), args.output_row),
        (("output_paragraph",), args.output_paragraph),
        (("output_table",), args.output_table),
        (("output_char_info",), args.output_char_info),
        (("is_line_less_table",), args.line_less_table),
        (("is_hand_writing_table",), args.handwriting_table),
        (("output_table_excel", "output_tableexcel"), args.output_table_excel),
        (("output_table_html", "output_tablehtml"), args.output_table_html),
    ]
    if not any(value is not None for _, value in config_values):
        return None

    config_class = _find_model_class(["RecognizeAllTextRequestAdvancedConfig"])
    if config_class is None:
        return {
            candidates[0]: value
            for candidates, value in config_values
            if value is not None
        }
    return _build_model(config_class, config_values)


def _build_table_config(args: argparse.Namespace) -> Optional[Any]:
    config_values = [
        (("is_hand_writing_table",), args.handwriting_table),
        (("is_line_less_table",), args.line_less_table),
        (("output_table_excel", "output_tableexcel"), args.output_table_excel),
        (("output_table_html", "output_tablehtml"), args.output_table_html),
    ]
    if not any(value is not None for _, value in config_values):
        return None

    config_class = _find_model_class(["RecognizeAllTextRequestTableConfig"])
    if config_class is None:
        return {
            candidates[0]: value
            for candidates, value in config_values
            if value is not None
        }
    return _build_model(config_class, config_values)


def _build_multi_lan_config(args: argparse.Namespace) -> Optional[Any]:
    if args.languages is None:
        return None

    config_class = _find_model_class(["RecognizeAllTextRequestMultiLanConfig"])
    config_values = [(("languages",), args.languages)]
    if config_class is None:
        return {"languages": args.languages}
    return _build_model(config_class, config_values)


def _build_id_card_config(args: argparse.Namespace) -> Optional[Any]:
    if args.output_id_card_quality is None:
        return None

    config_class = _find_model_class(["RecognizeAllTextRequestIdCardConfig"])
    config_values = [(("output_id_card_quality",), args.output_id_card_quality)]
    if config_class is None:
        return {"output_id_card_quality": args.output_id_card_quality}
    return _build_model(config_class, config_values)


def _build_international_id_card_config(args: argparse.Namespace) -> Optional[Any]:
    if args.country is None:
        return None

    config_class = _find_model_class(
        ["RecognizeAllTextRequestInternationalIdCardConfig"]
    )
    config_values = [(("country",), args.country)]
    if config_class is None:
        return {"country": args.country}
    return _build_model(config_class, config_values)


def _build_international_business_license_config(
    args: argparse.Namespace,
) -> Optional[Any]:
    if args.country is None:
        return None

    config_class = _find_model_class(
        ["RecognizeAllTextRequestInternationalBusinessLicenseConfig"]
    )
    config_values = [(("country",), args.country)]
    if config_class is None:
        return {"country": args.country}
    return _build_model(config_class, config_values)


def _validate_preprocess_options(max_dimension: int, jpeg_quality: int) -> None:
    if max_dimension < ALIYUN_MIN_DIMENSION or max_dimension > ALIYUN_MAX_DIMENSION:
        raise ValueError(
            "--max-image-dimension must be between "
            f"{ALIYUN_MIN_DIMENSION} and {ALIYUN_MAX_DIMENSION}."
        )
    if jpeg_quality < 1 or jpeg_quality > 95:
        raise ValueError("--jpeg-quality must be between 1 and 95.")


def _ensure_body_size(body: bytes, source_description: str) -> bytes:
    size = len(body)
    if size > MAX_BODY_BYTES:
        raise ValueError(
            f"{source_description} is {size / 1024 / 1024:.2f} MB, which exceeds "
            "Aliyun OCR's 10 MB body upload limit. Use --url with an accessible "
            "image URL instead, or lower --jpeg-quality."
        )
    return body


def _image_to_jpeg_bytes(image: Any, jpeg_quality: int) -> bytes:
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba_image = image.convert("RGBA")
        background = Image.new("RGBA", rgba_image.size, (255, 255, 255, 255))
        background.alpha_composite(rgba_image)
        image = background.convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    output = BytesIO()
    image.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
    return output.getvalue()


def _preprocess_oversized_image(
    path: Path,
    max_dimension: int,
    oversize_policy: str,
    jpeg_quality: int,
) -> Optional[bytes]:
    if Image is None or ImageOps is None:
        raise RuntimeError(
            "Pillow is required to truncate oversized local images. "
            "Install with: pip install Pillow"
        )

    try:
        with Image.open(path) as image:
            image.load()
            image = ImageOps.exif_transpose(image)
            width, height = image.size

            if width < ALIYUN_MIN_DIMENSION or height < ALIYUN_MIN_DIMENSION:
                raise ValueError(
                    f"Image dimension {width}x{height} is below Aliyun OCR's "
                    f"minimum {ALIYUN_MIN_DIMENSION}px limit."
                )
            if width <= max_dimension and height <= max_dimension:
                return None
            if oversize_policy == "fail":
                raise ValueError(
                    f"Image dimension {width}x{height} exceeds Aliyun OCR's "
                    f"{ALIYUN_MAX_DIMENSION}px limit."
                )

            if oversize_policy == "resize":
                processed = image.copy()
                processed.thumbnail(
                    (max_dimension, max_dimension),
                    Image.Resampling.LANCZOS,
                )
            else:
                crop_box = (0, 0, min(width, max_dimension), min(height, max_dimension))
                processed = image.crop(crop_box)

            print(
                "Preprocessed oversized image for Aliyun OCR: "
                f"{path.name} {width}x{height} -> "
                f"{processed.size[0]}x{processed.size[1]} ({oversize_policy})",
                file=sys.stderr,
            )
            return _image_to_jpeg_bytes(processed, jpeg_quality)
    except UnidentifiedImageError:
        return None


def _read_image_body(
    image_path: str,
    max_dimension: int,
    oversize_policy: str,
    jpeg_quality: int,
) -> bytes:
    _validate_preprocess_options(max_dimension, jpeg_quality)
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    if oversize_policy != "none":
        processed_body = _preprocess_oversized_image(
            path,
            max_dimension,
            oversize_policy,
            jpeg_quality,
        )
        if processed_body is not None:
            return _ensure_body_size(processed_body, "Preprocessed image")

    return _ensure_body_size(path.read_bytes(), "Input file")


def build_recognize_all_text_request(args: argparse.Namespace) -> Any:
    ensure_sdk_available()
    request_class = getattr(ocr_models, "RecognizeAllTextRequest", None)
    if request_class is None:
        raise RuntimeError(
            "The installed alibabacloud_ocr_api20210707 package does not expose "
            "RecognizeAllTextRequest. Please upgrade the Aliyun OCR SDK."
        )

    coordinate = None if args.output_coordinate == "none" else args.output_coordinate
    body = (
        _read_image_body(
            args.image,
            args.max_image_dimension,
            args.oversize_policy,
            args.jpeg_quality,
        )
        if args.image
        else None
    )
    normalized_type = args.type.replace("_", "").lower()
    advanced_config = (
        _build_advanced_config(args) if normalized_type == "advanced" else None
    )
    multi_lan_config = (
        _build_multi_lan_config(args) if normalized_type == "multilang" else None
    )
    table_config = _build_table_config(args) if normalized_type == "table" else None
    id_card_config = (
        _build_id_card_config(args) if normalized_type == "idcard" else None
    )
    international_id_card_config = (
        _build_international_id_card_config(args)
        if normalized_type == "internationalidcard"
        else None
    )
    international_business_license_config = (
        _build_international_business_license_config(args)
        if normalized_type == "internationalbusinesslicense"
        else None
    )

    request_values = [
        (("url",), args.url),
        (("body",), body),
        (("type",), args.type),
        (("output_figure",), args.output_figure),
        (("output_qrcode", "output_qr_code"), args.output_qrcode),
        (("output_bar_code", "output_barcode"), args.output_bar_code),
        (("output_stamp",), args.output_stamp),
        (("output_coordinate",), coordinate),
        (("output_oricoord", "output_ori_coord"), args.output_oricoord),
        (("output_kvexcel", "output_kv_excel"), args.output_kv_excel),
        (("page_no",), args.page_no),
        (("advanced_config",), advanced_config),
        (("multi_lan_config", "multi_lang_config"), multi_lan_config),
        (("table_config",), table_config),
        (("id_card_config",), id_card_config),
        (("international_id_card_config",), international_id_card_config),
        (
            ("international_business_license_config",),
            international_business_license_config,
        ),
    ]
    return _build_model(request_class, request_values)


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
    if hasattr(value, "to_map"):
        value = value.to_map()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {str(key): _decode_json_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_json_values(item) for item in value]
    if isinstance(value, tuple):
        return [_decode_json_values(item) for item in value]
    return _decode_json_string(value)


def normalize_response(response: Any, include_transport: bool = False) -> Any:
    raw_response = _decode_json_values(response)
    if include_transport or not isinstance(raw_response, Mapping):
        return raw_response

    body = raw_response.get("body") or raw_response.get("Body")
    return body if body is not None else raw_response


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


def extract_content(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        return ""

    data = _mapping_get(payload, ["Data", "data"])
    if isinstance(data, str):
        data = _decode_json_string(data)
    if not isinstance(data, Mapping):
        return ""

    content = _mapping_get(data, ["Content", "content"])
    return content if isinstance(content, str) else ""


def write_json(payload: Any, output_path: str) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)
        file_obj.write("\n")
    return path


def write_text(text: str, output_path: str) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def print_summary(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    request_id = _mapping_get(payload, ["RequestId", "requestId", "request_id"])
    data = _mapping_get(payload, ["Data", "data"])
    if isinstance(data, str):
        data = _decode_json_string(data)

    if request_id:
        print(f"RequestId : {request_id}")
    if isinstance(data, Mapping):
        width = _mapping_get(data, ["Width", "width"])
        height = _mapping_get(data, ["Height", "height"])
        sub_image_count = _mapping_get(data, ["SubImageCount", "subImageCount"])
        if width and height:
            print(f"Image     : {width}x{height}")
        if sub_image_count is not None:
            print(f"SubImages : {sub_image_count}")

    content = extract_content(payload)
    print("Content   :")
    print(content if content else "(no text content returned)")


def recognize(args: argparse.Namespace) -> Any:
    client = create_client(config_path=args.config, endpoint=args.endpoint)
    request = build_recognize_all_text_request(args)
    method = getattr(client, "recognize_all_text_with_options", None)
    if method is None:
        raise RuntimeError(
            "The installed Aliyun OCR client does not expose "
            "recognize_all_text_with_options. Please upgrade the SDK."
        )
    response = method(request, util_models.RuntimeOptions())
    return normalize_response(response, include_transport=args.raw_response)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use Aliyun OCR RecognizeAllText to recognize one image/PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/aliyun_ocr_image.py -i data/v1.jpg -o result.json
  python scripts/aliyun_ocr_image.py -i data/v1.jpg --text-only
  python scripts/aliyun_ocr_image.py --url https://example.com/page.png --type Advanced
    python scripts/aliyun_ocr_image.py -i table.png --type Advanced --table \
            --coordinate points
    python scripts/aliyun_ocr_image.py -i long.png --oversize-policy resize

Credentials:
    The script reads config/aliyun_ocr_config.json by default.
    Copy/fill config/aliyun_ocr_config.template.json if needed.
        """,
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("-i", "--image", help="Path to a local image/PDF file")
    source_group.add_argument("--url", help="Publicly accessible image/PDF URL")

    parser.add_argument(
        "-o",
        "--output",
        help="Save normalized OCR response JSON to this file",
    )
    parser.add_argument(
        "--text-output",
        help="Save only Data.Content text to this file",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Print only Data.Content",
    )
    parser.add_argument(
        "--type",
        default="Advanced",
        help="RecognizeAllText Type value, such as Advanced, General, Table, IdCard",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help=f"Override config endpoint (default: {DEFAULT_ENDPOINT})",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to Aliyun OCR JSON config with access_key_id/access_key_secret",
    )
    parser.add_argument(
        "--coordinate",
        dest="output_coordinate",
        choices=["none", "points", "rectangle"],
        default="none",
        help="Coordinate output format (default: none)",
    )
    parser.add_argument(
        "--original-coordinate",
        dest="output_oricoord",
        action="store_true",
        default=None,
        help="Return coordinates in the original image coordinate system",
    )
    parser.add_argument(
        "--figure", dest="output_figure", action="store_true", default=None
    )
    parser.add_argument(
        "--qrcode", dest="output_qrcode", action="store_true", default=None
    )
    parser.add_argument(
        "--barcode", dest="output_bar_code", action="store_true", default=None
    )
    parser.add_argument(
        "--stamp", dest="output_stamp", action="store_true", default=None
    )
    parser.add_argument(
        "--kv-excel", dest="output_kv_excel", action="store_true", default=None
    )
    parser.add_argument(
        "--page-no",
        type=int,
        help="PDF/OFD page number for supported types",
    )
    parser.add_argument(
        "--oversize-policy",
        choices=["crop", "resize", "fail", "none"],
        default="crop",
        help=(
            "How to handle local images over 8192px on either side: crop truncates "
            "from the top-left, resize keeps the whole image, fail raises an error, "
            "none uploads the original file (default: crop)."
        ),
    )
    parser.add_argument(
        "--max-image-dimension",
        type=int,
        default=ALIYUN_MAX_DIMENSION,
        help="Maximum local image width/height before upload (default: 8192)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help="JPEG quality for preprocessed oversized images, 1-95 (default: 90)",
    )

    advanced_group = parser.add_argument_group("Advanced/Table options")
    advanced_group.add_argument(
        "--row", dest="output_row", action="store_true", default=None
    )
    advanced_group.add_argument(
        "--paragraph", dest="output_paragraph", action="store_true", default=None
    )
    advanced_group.add_argument(
        "--table", dest="output_table", action="store_true", default=None
    )
    advanced_group.add_argument(
        "--char-info", dest="output_char_info", action="store_true", default=None
    )
    advanced_group.add_argument(
        "--line-less-table", dest="line_less_table", action="store_true", default=None
    )
    advanced_group.add_argument(
        "--handwriting-table",
        dest="handwriting_table",
        action="store_true",
        default=None,
    )
    advanced_group.add_argument(
        "--table-excel", dest="output_table_excel", action="store_true", default=None
    )
    advanced_group.add_argument(
        "--table-html", dest="output_table_html", action="store_true", default=None
    )

    type_specific_group = parser.add_argument_group("Type-specific options")
    type_specific_group.add_argument(
        "--languages",
        help='Language list for Type=MultiLang, for example "eng,chn,lading"',
    )
    type_specific_group.add_argument(
        "--country",
        help="Country for InternationalIdCard or InternationalBusinessLicense",
    )
    type_specific_group.add_argument(
        "--id-card-quality",
        dest="output_id_card_quality",
        action="store_true",
        default=None,
        help="Enable ID-card quality detection for Type=IdCard",
    )
    parser.add_argument(
        "--raw-response",
        action="store_true",
        help="Keep SDK transport fields such as headers and statusCode in JSON output",
    )
    return parser.parse_args(argv)


def format_aliyun_error(exception: Exception) -> str:
    message = getattr(exception, "message", None) or str(exception)
    data = getattr(exception, "data", None)
    recommend = None
    if isinstance(data, Mapping):
        recommend = data.get("Recommend") or data.get("recommend")

    parts = [f"Error: {message}"]
    if recommend:
        parts.append(f"Recommend: {recommend}")
    return "\n".join(parts)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        payload = recognize(args)
    except Exception as exception:
        print(format_aliyun_error(exception), file=sys.stderr)
        return 1

    content = extract_content(payload)

    if args.output:
        saved_path = write_json(payload, args.output)
        print(f"Saved OCR JSON result to: {saved_path}")
    if args.text_output:
        saved_text_path = write_text(content, args.text_output)
        print(f"Saved OCR text result to: {saved_text_path}")

    if args.text_only:
        print(content)
    elif not args.output:
        print_summary(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
