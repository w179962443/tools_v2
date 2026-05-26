#!/usr/bin/env python3
"""Generate Chinese speech and optionally convert it with a Manbo RVC model.

Usage examples:
    python scripts/text_to_manbo_voice.py "今天也要元气满满。" --tts-only
    python scripts/text_to_manbo_voice.py "今天也要元气满满。" \
        --model data/rvc_models/manbo/manbo.pth \
        --rvc-command "python infer_cli.py --input {input} --output {output}"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.tts_rvc_tool import (  # noqa: E402
    DEFAULT_INDEX_FILE,
    DEFAULT_MODEL_FILE,
    DEFAULT_TTS_VOICE,
    RVC_COMMAND_ENV_VAR,
    synthesize_chinese_with_rvc,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Chinese speech, then convert it with a local Manbo "
            "RVC model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python scripts/text_to_manbo_voice.py "今天也要元气满满。" --tts-only
    python scripts/text_to_manbo_voice.py "今天也要元气满满。" -o out.wav --tts-only
    python scripts/text_to_manbo_voice.py --input-text-file input.txt --tts-only

For RVC conversion, pass --model and --rvc-command, or set
{RVC_COMMAND_ENV_VAR}. See docs/tts_rvc_usage.md for full command templates.

RVC command placeholders:
  {{input}}        Base TTS WAV file
  {{output}}       Final output WAV file
  {{model}}        RVC .pth model file
  {{index}}        RVC .index file, or empty when unavailable
  {{index_option}} Expands to "--index <file>" when an index exists
  {{pitch}}        Semitone pitch shift
  {{f0_method}}    F0 extraction method, usually rmvpe or harvest

You can also set {RVC_COMMAND_ENV_VAR} instead of passing --rvc-command.
        """,
    )
    parser.add_argument("text", nargs="?", help="Chinese text to synthesize")
    parser.add_argument(
        "--input-text-file",
        help="Read Chinese text from a UTF-8 text file instead of the positional text",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output WAV path (default: data/voice_outputs/manbo_<timestamp>.wav)",
    )
    parser.add_argument(
        "--model",
        help=f"RVC .pth model path (default: {DEFAULT_MODEL_FILE})",
    )
    parser.add_argument(
        "--index",
        help=f"RVC .index path (default: {DEFAULT_INDEX_FILE} when it exists)",
    )
    parser.add_argument(
        "--voice",
        default=DEFAULT_TTS_VOICE,
        help=f"Edge TTS Chinese voice (default: {DEFAULT_TTS_VOICE})",
    )
    parser.add_argument(
        "--tts-rate",
        default="+0%",
        help="Edge TTS speaking rate, for example +10%% or -10%% (default: +0%%)",
    )
    parser.add_argument(
        "--rvc-command",
        help="External RVC inference command template with placeholders",
    )
    parser.add_argument(
        "--rvc-working-dir",
        help="Directory where the external RVC command should run",
    )
    parser.add_argument(
        "--pitch",
        type=int,
        default=0,
        help="RVC pitch shift in semitones (default: 0)",
    )
    parser.add_argument(
        "--f0-method",
        default="rmvpe",
        help="RVC F0 extraction method (default: rmvpe)",
    )
    parser.add_argument(
        "--tts-only",
        action="store_true",
        help="Only generate the base Chinese TTS WAV; skip RVC conversion",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep base TTS MP3/WAV files beside the output",
    )

    args = parser.parse_args()
    try:
        text = _read_text(args.text, args.input_text_file)
        result = synthesize_chinese_with_rvc(
            text=text,
            output_file=args.output,
            model_file=args.model,
            index_file=args.index,
            tts_voice=args.voice,
            tts_rate=args.tts_rate,
            rvc_command=args.rvc_command,
            rvc_working_dir=args.rvc_working_dir,
            pitch=args.pitch,
            f0_method=args.f0_method,
            keep_intermediate=args.keep_intermediate,
            tts_only=args.tts_only,
        )
        print(f"Output: {result['output_file']}")
        if args.tts_only:
            print("TTS-only mode: output is the base voice, not the Manbo RVC voice.")
        print("Done.")
        return 0
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as exception:
        print(f"Error: {exception}", file=sys.stderr)
        return 1


def _read_text(text: str | None, input_text_file: str | None) -> str:
    if text and input_text_file:
        raise ValueError("Pass either positional text or --input-text-file, not both.")
    if input_text_file:
        text_path = Path(input_text_file)
        if not text_path.is_file():
            raise FileNotFoundError(f"Text file not found: {text_path}")
        return text_path.read_text(encoding="utf-8").strip()
    if text:
        return text.strip()
    raise ValueError("Provide Chinese text or --input-text-file.")


if __name__ == "__main__":
    raise SystemExit(main())
