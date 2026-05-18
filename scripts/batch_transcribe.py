#!/usr/bin/env python3
"""Batch-transcribe all supported audio/video files in a directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.audio_transcription_tool import (  # noqa: E402
    AUDIO_VIDEO_EXTENSIONS,
    WHISPER_MODELS,
    batch_transcribe_directory,
    find_media_files,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Transcribe every supported audio/video file in a directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Supported extensions: {' '.join(sorted(AUDIO_VIDEO_EXTENSIONS))}",
    )
    parser.add_argument("directory", help="Directory to scan")
    parser.add_argument("-m", "--model", default="turbo", choices=WHISPER_MODELS)
    parser.add_argument("-l", "--language", default="zh", help="Language code (default: zh)")
    parser.add_argument("-d", "--model-dir", help="Whisper model cache directory")
    parser.add_argument("-r", "--recursive", action="store_true", help="Scan subdirectories")
    parser.add_argument("--skip-existing", action="store_true", help="Skip media files with sibling .csv files")
    parser.add_argument("--dry-run", action="store_true", help="List work without transcribing")
    parser.add_argument(
        "--no-force-simplified",
        action="store_true",
        help="Keep original Chinese text instead of converting to simplified Chinese",
    )
    args = parser.parse_args()

    try:
        media_files = find_media_files(args.directory, recursive=args.recursive)
        print(f"Found {len(media_files)} media file(s).")
        result = batch_transcribe_directory(
            directory=args.directory,
            model_name=args.model,
            language=args.language,
            model_dir=args.model_dir,
            recursive=args.recursive,
            skip_existing=args.skip_existing,
            force_simplified=not args.no_force_simplified,
            dry_run=args.dry_run,
        )
    except Exception as exception:
        print(f"Error: {exception}", file=sys.stderr)
        return 1

    print("\nBatch transcription complete.")
    print(f"Success: {len(result['success'])}")
    print(f"Skipped: {len(result['skipped'])}")
    print(f"Failed : {len(result['failed'])}")
    if result["failed"]:
        for path in result["failed"]:
            print(f"  - {path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())