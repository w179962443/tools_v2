#!/usr/bin/env python3
"""Run the realtime Whisper transcription web app."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.realtime_transcription_tool import DEFAULT_REALTIME_CONFIG, run_realtime_server  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the realtime transcription web server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    parser.add_argument("--output-dir", default="recordings", help="Transcript output directory")
    parser.add_argument("--model", default=DEFAULT_REALTIME_CONFIG["model_name"], help="Default Whisper model")
    parser.add_argument("--language", default=DEFAULT_REALTIME_CONFIG["language"], help="Default language")
    parser.add_argument("--interval", type=float, default=DEFAULT_REALTIME_CONFIG["transcribe_interval"], help="Transcription interval seconds")
    parser.add_argument("--gpu", action="store_true", help="Load Whisper on CUDA")
    args = parser.parse_args()

    config = {
        "model_name": args.model,
        "language": args.language,
        "transcribe_interval": args.interval,
        "use_gpu": args.gpu,
    }
    print(f"Starting realtime transcription server: http://127.0.0.1:{args.port}")
    run_realtime_server(
        host=args.host,
        port=args.port,
        debug=args.debug,
        output_dir=args.output_dir,
        config=config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())