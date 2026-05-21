#!/usr/bin/env python3
"""Create subtitles, transcript text, polished text, and notes from video.

Usage examples:
    python scripts/video_to_subtitle_notes.py lecture.mp4 -m turbo -l zh
    python scripts/video_to_subtitle_notes.py lecture.mp4 -o ./lecture_outputs
    python scripts/video_to_subtitle_notes.py lecture.mp4 --skip-llm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.audio_transcription_tool import WHISPER_MODELS  # noqa: E402
from tools.video_learning_notes_tool import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_CLEAN_PROMPT_PATH,
    DEFAULT_LLM_MODEL,
    DEFAULT_NOTES_PROMPT_PATH,
    DEFAULT_API_KEY,
    generate_video_learning_package,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate SRT subtitles, raw text, polished transcript, and Markdown notes from one video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/video_to_subtitle_notes.py lecture.mp4 -m turbo -l zh
  python scripts/video_to_subtitle_notes.py lecture.mp4 -o ./lecture_outputs --llm-model qwen-plus
  python scripts/video_to_subtitle_notes.py lecture.mp4 --skip-llm

The LLM call reads DASHSCOPE_API_KEY or OPENAI_API_KEY unless --api-key is passed.
        """,
    )
    parser.add_argument("video_file", help="Input video/audio file")
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory (default: <video>, with random suffix on conflict)",
    )
    parser.add_argument(
        "-m",
        "--whisper-model",
        default="turbo",
        choices=WHISPER_MODELS,
        help="Whisper model size (default: turbo)",
    )
    parser.add_argument(
        "-l",
        "--language",
        default="zh",
        help="Language code: zh, en, auto, etc. (default: zh)",
    )
    parser.add_argument("-d", "--model-dir", help="Whisper model cache directory")
    parser.add_argument(
        "--no-force-simplified",
        action="store_true",
        help="Keep original Chinese text instead of converting to simplified Chinese",
    )
    parser.add_argument(
        "--subtitle-line-chars",
        type=int,
        default=42,
        help="Maximum characters per subtitle line before wrapping (default: 42)",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Only generate transcript CSV, SRT, and raw txt; do not call the LLM",
    )
    parser.add_argument(
        "--clean-prompt-file",
        default=str(DEFAULT_CLEAN_PROMPT_PATH),
        help="Prompt template file for transcript cleanup",
    )
    parser.add_argument(
        "--notes-prompt-file",
        default=str(DEFAULT_NOTES_PROMPT_PATH),
        help="Prompt template file for study notes generation",
    )
    parser.add_argument(
        "--llm-model", default=DEFAULT_LLM_MODEL, help="Chat model name"
    )
    parser.add_argument(
        "--api-key", default=DEFAULT_API_KEY, help="DashScope/OpenAI-compatible API key"
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="LLM temperature (default: 0.2)"
    )
    parser.add_argument(
        "--quiet-llm",
        action="store_true",
        help="Do not stream the LLM answer to stdout while generating files",
    )

    args = parser.parse_args()
    try:
        result = generate_video_learning_package(
            video_file=args.video_file,
            output_dir=args.output_dir,
            whisper_model=args.whisper_model,
            language=args.language,
            model_dir=args.model_dir,
            force_simplified=not args.no_force_simplified,
            clean_prompt_file=args.clean_prompt_file,
            notes_prompt_file=args.notes_prompt_file,
            llm_model=args.llm_model,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            subtitle_line_chars=args.subtitle_line_chars,
            skip_llm=args.skip_llm,
            stream_llm=not args.quiet_llm,
        )
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as exception:
        print(f"Error: {exception}", file=sys.stderr)
        return 1

    print("\nVideo learning package complete.")
    print(f"Detected language: {result.detected_language}")
    print(f"Segments: {result.segment_count}")
    for name, path in result.outputs.as_dict().items():
        print(f"{name}: {path}")
    if result.llm_skipped:
        print("LLM step skipped; polished transcript and notes were not generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
