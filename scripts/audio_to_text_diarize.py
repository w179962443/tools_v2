#!/usr/bin/env python3
"""Transcribe one audio/video file with optional speaker diarization."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.audio_transcription_tool import (  # noqa: E402
    WHISPER_MODELS,
    transcribe_with_diarization,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Use WhisperX to transcribe audio/video to CSV with speaker labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/audio_to_text_diarize.py audio.mp3 -m turbo -l zh
  python scripts/audio_to_text_diarize.py audio.mp3 -m turbo -l zh -t hf_xxxx
  python scripts/audio_to_text_diarize.py audio.mp3 -t hf_xxxx --min-speakers 2 --max-speakers 2

Set HF_TOKEN in the environment instead of passing -t to enable diarization.
        """,
    )
    parser.add_argument("audio_file", help="Input audio/video file")
    parser.add_argument(
        "-m",
        "--model",
        default="turbo",
        choices=WHISPER_MODELS,
        help="WhisperX model size (default: turbo)",
    )
    parser.add_argument(
        "-l",
        "--language",
        default="zh",
        help="Language code: zh, en, auto, etc. (default: zh)",
    )
    parser.add_argument(
        "-t",
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace token for pyannote speaker diarization",
    )
    parser.add_argument("-o", "--output", help="Output CSV path (default: <input>_diarize.csv)")
    parser.add_argument("-d", "--model-dir", help="Whisper model cache directory")
    parser.add_argument("--min-speakers", type=int, default=None, help="Minimum speaker count")
    parser.add_argument("--max-speakers", type=int, default=None, help="Maximum speaker count")
    parser.add_argument(
        "--no-force-simplified",
        action="store_true",
        help="Keep original Chinese text instead of converting to simplified Chinese",
    )

    args = parser.parse_args()
    try:
        transcribe_with_diarization(
            audio_file=args.audio_file,
            model_name=args.model,
            language=args.language,
            output_file=args.output,
            model_dir=args.model_dir,
            hf_token=args.token,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
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