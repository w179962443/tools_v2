#!/usr/bin/env python3
"""Transcribe one audio/video file to a timestamped CSV.

Usage examples:
    python scripts/audio_to_text.py audio.mp3
    python scripts/audio_to_text.py video.mp4 -m large -l en -o transcript.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.audio_transcription_tool import (
    WHISPER_MODELS,
    transcribe_audio,
)  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Use Whisper to transcribe an audio/video file to a timestamped CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/audio_to_text.py audio.mp3
  python scripts/audio_to_text.py audio.wav -m turbo -l zh
  python scripts/audio_to_text.py video.mp4 -m large -l en -o transcript.csv
  python scripts/audio_to_text.py audio.mp3 -m turbo -l zh -d D:\\whisper_models

Resume mode:
  If the output CSV already exists, only segments after the last end_time are appended.
        """,
    )
    parser.add_argument("audio_file", help="Input audio/video file")
    parser.add_argument(
        "-m",
        "--model",
        default="base",
        choices=WHISPER_MODELS,
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "-l",
        "--language",
        default="auto",
        help="Language code: zh, en, ja, auto, etc. (default: auto)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV path (default: <input>_transcript.csv)",
    )
    parser.add_argument("-d", "--model-dir", help="Whisper model cache directory")
    parser.add_argument(
        "--no-force-simplified",
        action="store_true",
        help="Keep original Chinese text instead of converting to simplified Chinese",
    )

    args = parser.parse_args()
    try:
        transcribe_audio(
            audio_file=args.audio_file,
            model_name=args.model,
            language=args.language,
            output_file=args.output,
            model_dir=args.model_dir,
            force_simplified=not args.no_force_simplified,
        )
        print("Done.")
        return 0
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as exception:
        print(f"Error: {exception}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
