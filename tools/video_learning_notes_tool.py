"""Generate subtitles, raw transcripts, polished text, and study notes from video.

The workflow reuses the Whisper transcription utility, then calls an OpenAI-
compatible chat model for transcript cleanup and note generation.
"""

from __future__ import annotations

import asyncio
import csv
import os
import platform
import random
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio_transcription_tool import transcribe_audio


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_LLM_MODEL = "qwen3.6-max-preview"
DEFAULT_API_KEY = None
DEFAULT_CLEAN_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "video_clean_transcript_prompt.md"
)
DEFAULT_NOTES_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "prompts"
    / "video_study_notes_prompt.md"
)


@dataclass(frozen=True)
class VideoLearningOutputs:
    output_dir: Path
    transcript_csv: Path
    subtitle_srt: Path
    raw_text: Path
    polished_text: Path
    learning_notes: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "output_dir": str(self.output_dir),
            "transcript_csv": str(self.transcript_csv),
            "subtitle_srt": str(self.subtitle_srt),
            "raw_text": str(self.raw_text),
            "polished_text": str(self.polished_text),
            "learning_notes": str(self.learning_notes),
        }


@dataclass(frozen=True)
class VideoLearningResult:
    outputs: VideoLearningOutputs
    detected_language: str
    segment_count: int
    llm_skipped: bool


def create_output_dir(video_file: str | Path, output_dir: str | Path | None = None) -> Path:
    video_path = Path(video_file)
    base_dir = Path(output_dir) if output_dir else video_path.parent / video_path.stem

    if output_dir:
        try:
            base_dir.mkdir(parents=True, exist_ok=True)
            if base_dir.is_dir():
                return base_dir
        except OSError:
            pass
    else:
        try:
            base_dir.mkdir(parents=True, exist_ok=False)
            return base_dir
        except OSError:
            pass

    parent_dir = base_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    base_name = base_dir.name or video_path.stem
    for _ in range(100):
        candidate = parent_dir / f"{base_name}_{random.randint(1000, 999999)}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except OSError:
            continue

    raise OSError(f"Cannot create output directory near: {base_dir}")


def build_output_paths(video_file: str | Path, output_dir: str | Path) -> VideoLearningOutputs:
    video_path = Path(video_file)
    target_dir = Path(output_dir)
    stem = video_path.stem
    return VideoLearningOutputs(
        output_dir=target_dir,
        transcript_csv=target_dir / f"{stem}_transcript.csv",
        subtitle_srt=target_dir / f"{stem}.srt",
        raw_text=target_dir / f"{stem}_raw_text.txt",
        polished_text=target_dir / f"{stem}_polished_text.txt",
        learning_notes=target_dir / f"{stem}_learning_notes.md",
    )


def _read_csv_rows(csv_file: str | Path) -> list[dict[str, str]]:
    csv_path = Path(csv_file)
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with csv_path.open("r", encoding=encoding, newline="") as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot decode transcript CSV: {csv_path}")


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_srt_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    remaining_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d},{milliseconds:03d}"


def _wrap_subtitle_text(text: str, max_chars: int) -> str:
    clean_text = " ".join(text.split())
    if max_chars <= 0 or len(clean_text) <= max_chars:
        return clean_text
    if " " in clean_text:
        return textwrap.fill(
            clean_text,
            width=max_chars,
            break_long_words=False,
            break_on_hyphens=False,
        )
    return "\n".join(clean_text[index : index + max_chars] for index in range(0, len(clean_text), max_chars))


def write_srt(rows: list[dict[str, str]], output_file: str | Path, max_line_chars: int = 42) -> int:
    subtitle_blocks: list[str] = []
    subtitle_index = 1
    for row in rows:
        text = str(row.get("text") or "").strip()
        if not text:
            continue

        start_time = _to_float(row.get("start_time"))
        end_time = _to_float(row.get("end_time"), start_time + 0.5)
        if end_time <= start_time:
            end_time = start_time + 0.5

        subtitle_blocks.append(
            "\n".join(
                [
                    str(subtitle_index),
                    f"{_format_srt_timestamp(start_time)} --> {_format_srt_timestamp(end_time)}",
                    _wrap_subtitle_text(text, max_line_chars),
                ]
            )
        )
        subtitle_index += 1

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n\n".join(subtitle_blocks) + "\n", encoding="utf-8")
    return len(subtitle_blocks)


def build_raw_text(rows: list[dict[str, str]]) -> str:
    parts = [str(row.get("text") or "").strip() for row in rows]
    raw_text = "\n".join(part for part in parts if part)
    if not raw_text:
        raise ValueError("Transcript has no text content.")
    return raw_text + "\n"


def render_prompt(
    prompt_template: str,
    text: str,
    source_file: str | Path,
    detected_language: str,
) -> str:
    replacements = {
        "{{SOURCE_FILE}}": str(source_file),
        "{{DETECTED_LANGUAGE}}": detected_language or "unknown",
        "{{RAW_TEXT}}": text.strip(),
        "{{TEXT}}": text.strip(),
        "{{text}}": text.strip(),
    }
    prompt = prompt_template
    for marker, value in replacements.items():
        prompt = prompt.replace(marker, value)
    return prompt


def read_prompt_template(prompt_file: str | Path) -> str:
    prompt_path = Path(prompt_file)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def normalize_output_text(text: str) -> str:
    return text.strip() + "\n" if text.strip() else ""


def normalize_delta_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            text = getattr(item, "text", None)
            if text:
                chunks.append(text)
        return "".join(chunks)
    return str(content)


async def get_chat_completion(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float = 0.2,
    stream_to_stdout: bool = True,
) -> str:
    try:
        from openai import AsyncOpenAI
    except ImportError as import_error:
        raise RuntimeError("Missing dependency 'openai'. Install it with: pip install openai") from import_error

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        stream=True,
    )

    chunks: list[str] = []
    async for event in stream:
        if not event.choices:
            continue
        text = normalize_delta_content(event.choices[0].delta.content)
        if not text:
            continue
        if stream_to_stdout:
            print(text, end="", flush=True)
        chunks.append(text)

    if stream_to_stdout:
        print()
    return "".join(chunks)


def resolve_api_key(api_key: str | None = None) -> str:
    resolved = api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    if not resolved:
        raise EnvironmentError("Set DASHSCOPE_API_KEY/OPENAI_API_KEY or pass --api-key to call the LLM.")
    return resolved


async def generate_video_learning_package_async(
    video_file: str | Path,
    output_dir: str | Path | None = None,
    whisper_model: str = "turbo",
    language: str = "zh",
    model_dir: str | Path | None = None,
    force_simplified: bool = True,
    clean_prompt_file: str | Path = DEFAULT_CLEAN_PROMPT_PATH,
    notes_prompt_file: str | Path = DEFAULT_NOTES_PROMPT_PATH,
    llm_model: str = DEFAULT_LLM_MODEL,
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.2,
    subtitle_line_chars: int = 42,
    skip_llm: bool = False,
    stream_llm: bool = True,
) -> VideoLearningResult:
    video_path = Path(video_file)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    target_dir = create_output_dir(video_path, output_dir)
    outputs = build_output_paths(video_path, target_dir)

    transcription_result = transcribe_audio(
        audio_file=video_path,
        model_name=whisper_model,
        language=language,
        output_file=outputs.transcript_csv,
        model_dir=model_dir,
        force_simplified=force_simplified,
    )
    detected_language = str(transcription_result.get("language") or language or "unknown")

    rows = _read_csv_rows(outputs.transcript_csv)
    write_srt(rows, outputs.subtitle_srt, max_line_chars=subtitle_line_chars)
    raw_text = build_raw_text(rows)
    outputs.raw_text.write_text(raw_text, encoding="utf-8")

    if skip_llm:
        return VideoLearningResult(
            outputs=outputs,
            detected_language=detected_language,
            segment_count=len(rows),
            llm_skipped=True,
        )

    resolved_api_key = resolve_api_key(api_key)
    clean_prompt = render_prompt(
        prompt_template=read_prompt_template(clean_prompt_file),
        text=raw_text,
        source_file=video_path,
        detected_language=detected_language,
    )
    if stream_llm:
        print("\nCleaning transcript with LLM...")
    polished_text = await get_chat_completion(
        prompt=clean_prompt,
        model=llm_model,
        api_key=resolved_api_key,
        base_url=base_url,
        temperature=temperature,
        stream_to_stdout=stream_llm,
    )
    polished_text = normalize_output_text(polished_text)
    outputs.polished_text.write_text(polished_text, encoding="utf-8")

    notes_prompt = render_prompt(
        prompt_template=read_prompt_template(notes_prompt_file),
        text=polished_text,
        source_file=video_path,
        detected_language=detected_language,
    )
    if stream_llm:
        print("\nGenerating study notes with LLM...")
    learning_notes = await get_chat_completion(
        prompt=notes_prompt,
        model=llm_model,
        api_key=resolved_api_key,
        base_url=base_url,
        temperature=temperature,
        stream_to_stdout=stream_llm,
    )
    learning_notes = normalize_output_text(learning_notes)
    outputs.learning_notes.write_text(learning_notes, encoding="utf-8")

    return VideoLearningResult(
        outputs=outputs,
        detected_language=detected_language,
        segment_count=len(rows),
        llm_skipped=False,
    )


def generate_video_learning_package(**kwargs: Any) -> VideoLearningResult:
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(generate_video_learning_package_async(**kwargs))