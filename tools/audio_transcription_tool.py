"""
Audio and video transcription utilities built around Whisper and WhisperX.

This module contains the reusable package API migrated from the older daily
data-processing scripts. Command-line wrappers live in ``scripts/``.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Iterable


AUDIO_VIDEO_EXTENSIONS: set[str] = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".webm",
    ".aac",
    ".wma",
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".wmv",
    ".flv",
    ".ts",
    ".m4v",
}

WHISPER_MODELS: tuple[str, ...] = (
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "turbo",
)

CSV_FIELDNAMES: list[str] = [
    "start_time",
    "end_time",
    "start_timestamp",
    "end_timestamp",
    "duration",
    "text",
]

DIARIZED_CSV_FIELDNAMES: list[str] = [
    "start_time",
    "end_time",
    "start_timestamp",
    "end_timestamp",
    "duration",
    "speaker",
    "text",
]


try:
    from opencc import OpenCC

    _OPENCC_CONVERTER = OpenCC("t2s")
except ImportError:
    _OPENCC_CONVERTER = None

try:
    import zhconv as _zhconv
except ImportError:
    _zhconv = None


def convert_to_simplified(text: str) -> str:
    """Convert Chinese text to simplified Chinese when a converter exists."""
    if _OPENCC_CONVERTER is not None:
        return _OPENCC_CONVERTER.convert(text)
    if _zhconv is not None:
        return _zhconv.convert(text, "zh-cn")
    return text


def has_simplified_converter() -> bool:
    """Return whether a traditional-to-simplified converter is available."""
    return _OPENCC_CONVERTER is not None or _zhconv is not None


def format_timestamp(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS.mmm``."""
    total_milliseconds = int(round(float(seconds) * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    remaining_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}.{milliseconds:03d}"


def read_last_timestamp(csv_file: str | Path) -> float:
    """Read the final ``end_time`` from an existing transcript CSV."""
    csv_path = Path(csv_file)
    if not csv_path.exists():
        return 0.0

    last_timestamp = 0.0
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with csv_path.open("r", encoding=encoding, newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    try:
                        last_timestamp = float(row.get("end_time") or last_timestamp)
                    except (TypeError, ValueError):
                        continue
            return last_timestamp
        except UnicodeDecodeError:
            continue
    return last_timestamp


def check_gpu(verbose: bool = True) -> bool:
    """Return whether CUDA is available through PyTorch."""
    try:
        import torch
    except ImportError:
        if verbose:
            print("PyTorch is not installed; transcription will use CPU if possible.")
        return False

    if torch.cuda.is_available():
        if verbose:
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        return True

    if verbose:
        print("No GPU detected; using CPU.")
    return False


def _default_transcript_path(audio_file: str | Path, suffix: str) -> Path:
    audio_path = Path(audio_file)
    return audio_path.parent / f"{audio_path.stem}{suffix}.csv"


def _write_csv_rows(
    output_file: str | Path,
    fieldnames: Iterable[str],
    rows: list[dict[str, Any]],
) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    mode = "a" if file_exists else "w"

    with output_path.open(mode, encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def _preview_rows(rows: list[dict[str, Any]], speaker: bool = False) -> None:
    if not rows:
        return

    print("\nRecent rows:")
    print("-" * 80)
    for row in rows[-3:]:
        text = str(row["text"])
        preview = text[:50] + ("..." if len(text) > 50 else "")
        speaker_prefix = f"[{row['speaker']}] " if speaker else ""
        print(
            f"{speaker_prefix}[{row['start_timestamp']} --> "
            f"{row['end_timestamp']}] {preview}"
        )
    print("-" * 80)


def transcribe_audio(
    audio_file: str | Path,
    model_name: str = "base",
    language: str = "auto",
    output_file: str | Path | None = None,
    model_dir: str | Path | None = None,
    force_simplified: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Transcribe an audio/video file to a timestamped CSV with resume support."""
    try:
        import whisper
    except ImportError as import_error:
        raise ImportError(
            "openai-whisper is required. Install it with: pip install openai-whisper"
        ) from import_error

    audio_path = Path(audio_file)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if model_name not in WHISPER_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Use one of {WHISPER_MODELS}.")

    output_path = Path(output_file) if output_file else _default_transcript_path(audio_path, "_transcript")
    last_timestamp = read_last_timestamp(output_path)
    is_resume = last_timestamp > 0

    if verbose:
        print("Resuming transcript." if is_resume else "Starting new transcript.")
        if is_resume:
            print(f"Last timestamp: {format_timestamp(last_timestamp)}")

    has_gpu = check_gpu(verbose=verbose)
    device = "cuda" if has_gpu else "cpu"

    if verbose:
        print(f"Loading Whisper model: {model_name}")
        if model_dir:
            print(f"Model directory: {model_dir}")

    model = whisper.load_model(
        model_name,
        device=device,
        download_root=str(model_dir) if model_dir else None,
    )

    options: dict[str, Any] = {"fp16": has_gpu, "verbose": False}
    if language != "auto":
        options["language"] = language
    if language == "zh":
        options["initial_prompt"] = "以下是简体中文的转录内容："

    if verbose:
        print(f"Transcribing: {audio_path}")
    result = model.transcribe(str(audio_path), **options)
    detected_language = result.get("language", "unknown")
    segments = result.get("segments", [])
    new_segments = [segment for segment in segments if segment.get("end", 0) > last_timestamp]

    if verbose:
        print(f"Detected language: {detected_language}")
        print(f"Total segments: {len(segments)}")
        if is_resume:
            print(f"Skipped segments: {len(segments) - len(new_segments)}")
        print(f"New segments: {len(new_segments)}")

    rows: list[dict[str, Any]] = []
    for segment in new_segments:
        text = str(segment.get("text", "")).strip()
        if force_simplified and detected_language == "zh":
            text = convert_to_simplified(text)
        start_time = float(segment["start"])
        end_time = float(segment["end"])
        rows.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "start_timestamp": format_timestamp(start_time),
                "end_timestamp": format_timestamp(end_time),
                "duration": round(end_time - start_time, 3),
                "text": text,
            }
        )

    if not rows:
        if verbose:
            print("No new content to write.")
        result["output_file"] = str(output_path)
        result["new_rows"] = 0
        return result

    if force_simplified and detected_language == "zh" and verbose:
        if has_simplified_converter():
            print("Converted Chinese text to simplified Chinese.")
        else:
            print("No Chinese converter installed. Consider: pip install opencc-python-reimplemented")

    _write_csv_rows(output_path, CSV_FIELDNAMES, rows)
    if verbose:
        print(f"Transcript saved to: {output_path}")
        print(f"Rows added: {len(rows)}")
        _preview_rows(rows)

    result["output_file"] = str(output_path)
    result["new_rows"] = len(rows)
    return result


def transcribe_with_diarization(
    audio_file: str | Path,
    model_name: str = "turbo",
    language: str = "zh",
    output_file: str | Path | None = None,
    model_dir: str | Path | None = None,
    hf_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    force_simplified: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Transcribe with WhisperX and optionally assign speaker labels."""
    try:
        import whisperx
    except ImportError as import_error:
        raise ImportError("whisperx is required. Install it with: pip install whisperx") from import_error

    audio_path = Path(audio_file)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if model_name not in WHISPER_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Use one of {WHISPER_MODELS}.")

    output_path = Path(output_file) if output_file else _default_transcript_path(audio_path, "_diarize")
    token = hf_token or os.environ.get("HF_TOKEN")
    last_timestamp = read_last_timestamp(output_path)
    is_resume = last_timestamp > 0

    if verbose:
        print("Resuming diarized transcript." if is_resume else "Starting new diarized transcript.")
        if is_resume:
            print(f"Last timestamp: {format_timestamp(last_timestamp)}")

    has_gpu = check_gpu(verbose=verbose)
    device = "cuda" if has_gpu else "cpu"
    compute_type = "float16" if has_gpu else "int8"
    whisper_language = None if language == "auto" else language

    if verbose:
        print(f"Loading WhisperX model: {model_name}")
    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=whisper_language,
        download_root=str(model_dir) if model_dir else None,
    )

    audio = whisperx.load_audio(str(audio_path))
    transcribe_options: dict[str, Any] = {"batch_size": 16}
    if language == "zh":
        transcribe_options["initial_prompt"] = "以下是简体中文的转录内容："

    if verbose:
        print(f"Transcribing: {audio_path}")
    result = model.transcribe(audio, **transcribe_options)
    detected_language = result.get("language", language or "unknown")

    if verbose:
        print(f"Transcription complete. Detected language: {detected_language}")

    try:
        align_language = detected_language if detected_language != "unknown" else "en"
        align_model, metadata = whisperx.load_align_model(
            language_code=align_language,
            device=device,
        )
        result = whisperx.align(
            result.get("segments", []),
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        if verbose:
            print("Timestamp alignment complete.")
    except Exception as exception:
        if verbose:
            print(f"Timestamp alignment failed; using original timestamps: {exception}")

    if token:
        try:
            diarize_kwargs: dict[str, Any] = {"audio": audio}
            if min_speakers is not None:
                diarize_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarize_kwargs["max_speakers"] = max_speakers

            diarize_model = whisperx.DiarizationPipeline(use_auth_token=token, device=device)
            diarize_segments = diarize_model(**diarize_kwargs)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            if verbose:
                print("Speaker diarization complete.")
        except Exception as exception:
            if verbose:
                print(f"Speaker diarization failed; continuing without speakers: {exception}")
    elif verbose:
        print("No HuggingFace token provided; skipping speaker diarization.")

    segments = result.get("segments", [])
    new_segments = [segment for segment in segments if segment.get("end", 0) > last_timestamp]
    if verbose:
        if is_resume:
            print(f"Skipped segments: {len(segments) - len(new_segments)}")
        print(f"New segments: {len(new_segments)}")

    rows: list[dict[str, Any]] = []
    for segment in new_segments:
        text = str(segment.get("text", "")).strip()
        if force_simplified and detected_language == "zh":
            text = convert_to_simplified(text)
        start_time = float(segment["start"])
        end_time = float(segment["end"])
        rows.append(
            {
                "start_time": start_time,
                "end_time": end_time,
                "start_timestamp": format_timestamp(start_time),
                "end_timestamp": format_timestamp(end_time),
                "duration": round(end_time - start_time, 3),
                "speaker": segment.get("speaker", "UNKNOWN"),
                "text": text,
            }
        )

    if not rows:
        if verbose:
            print("No new content to write.")
        result["output_file"] = str(output_path)
        result["new_rows"] = 0
        return result

    _write_csv_rows(output_path, DIARIZED_CSV_FIELDNAMES, rows)
    if verbose:
        print(f"Diarized transcript saved to: {output_path}")
        print(f"Rows added: {len(rows)}")
        _preview_rows(rows, speaker=True)

    result["output_file"] = str(output_path)
    result["new_rows"] = len(rows)
    return result


def find_media_files(directory: str | Path, recursive: bool = False) -> list[Path]:
    """Find supported audio/video files under a directory."""
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    if not directory_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {directory_path}")

    pattern = "**/*" if recursive else "*"
    return sorted(
        media_path
        for media_path in directory_path.glob(pattern)
        if media_path.is_file() and media_path.suffix.lower() in AUDIO_VIDEO_EXTENSIONS
    )


def batch_transcribe_directory(
    directory: str | Path,
    model_name: str = "turbo",
    language: str = "zh",
    model_dir: str | Path | None = None,
    recursive: bool = False,
    skip_existing: bool = False,
    force_simplified: bool = True,
    dry_run: bool = False,
) -> dict[str, list[Path]]:
    """Transcribe every supported media file in a directory to sibling CSV files."""
    media_files = find_media_files(directory, recursive=recursive)
    skipped: list[Path] = []
    succeeded: list[Path] = []
    failed: list[Path] = []

    if skip_existing:
        pending_files: list[Path] = []
        for media_path in media_files:
            if media_path.with_suffix(".csv").exists():
                skipped.append(media_path)
            else:
                pending_files.append(media_path)
        media_files = pending_files

    for index, media_path in enumerate(media_files, 1):
        output_path = media_path.with_suffix(".csv")
        print(f"\n[{index}/{len(media_files)}] {media_path}")
        print(f"Output: {output_path}")
        if dry_run:
            skipped.append(media_path)
            continue

        try:
            transcribe_audio(
                audio_file=media_path,
                model_name=model_name,
                language=language,
                output_file=output_path,
                model_dir=model_dir,
                force_simplified=force_simplified,
                verbose=True,
            )
            succeeded.append(media_path)
        except Exception as exception:
            print(f"Failed: {exception}")
            failed.append(media_path)

    return {"success": succeeded, "failed": failed, "skipped": skipped}