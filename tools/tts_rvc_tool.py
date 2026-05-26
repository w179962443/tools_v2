"""Chinese text-to-speech plus RVC voice conversion orchestration."""

from __future__ import annotations

import asyncio
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TTS_VOICE = "zh-CN-XiaoxiaoNeural"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "voice_outputs"
DEFAULT_MODEL_FILE = PROJECT_ROOT / "data" / "rvc_models" / "manbo" / "manbo.pth"
DEFAULT_INDEX_FILE = PROJECT_ROOT / "data" / "rvc_models" / "manbo" / "manbo.index"
RVC_COMMAND_ENV_VAR = "MANBO_RVC_COMMAND"
FALLBACK_RVC_COMMAND_ENV_VAR = "RVC_COMMAND"


def default_voice_output_path(prefix: str = "manbo") -> Path:
    """Build a timestamped default WAV output path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_OUTPUT_DIR / f"{prefix}_{timestamp}.wav"


def synthesize_chinese_with_rvc(
    text: str,
    output_file: str | Path | None = None,
    model_file: str | Path | None = None,
    index_file: str | Path | None = None,
    tts_voice: str = DEFAULT_TTS_VOICE,
    tts_rate: str = "+0%",
    rvc_command: str | None = None,
    rvc_working_dir: str | Path | None = None,
    pitch: int = 0,
    f0_method: str = "rmvpe",
    keep_intermediate: bool = False,
    tts_only: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Generate Chinese speech, optionally converting it with a local RVC model.

    The workflow is text -> Edge TTS base voice -> WAV -> external RVC command.
    The RVC command template must include input and output placeholders.
    """
    normalized_text = _normalize_text(text)
    output_path = _resolve_output_file(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_model_file: Path | None = None
    resolved_index_file: Path | None = None
    command_template: str | None = None
    working_dir: Path | None = None

    if not tts_only:
        resolved_model_file = _resolve_model_file(model_file)
        resolved_index_file = _resolve_index_file(index_file)
        command_template = _resolve_rvc_command(rvc_command)
        if not command_template:
            raise RuntimeError(
                "No RVC command configured. Pass --rvc-command or set "
                f"{RVC_COMMAND_ENV_VAR}. The command must include "
                "{input} and {output} placeholders."
            )
        working_dir = _resolve_working_dir(rvc_working_dir)

    intermediate_dir, temp_context = _prepare_intermediate_dir(
        output_path,
        keep_intermediate,
    )
    base_tts_mp3 = intermediate_dir / "base_tts.mp3"
    base_tts_wav = intermediate_dir / "base_tts.wav"

    try:
        if verbose:
            print(f"Generating base Chinese speech with {tts_voice}...")
        _save_edge_tts_audio(
            text=normalized_text,
            voice=tts_voice,
            rate=tts_rate,
            output_file=base_tts_mp3,
        )
        _convert_audio_to_wav(base_tts_mp3, base_tts_wav)

        if tts_only:
            shutil.copyfile(base_tts_wav, output_path)
            if verbose:
                print(f"TTS-only WAV written to: {output_path}")
            return _build_result(
                output_path=output_path,
                text=normalized_text,
                tts_voice=tts_voice,
                tts_rate=tts_rate,
                tts_only=True,
                model_file=None,
                index_file=None,
                rvc_command=None,
                rvc_working_dir=None,
                intermediate_dir=intermediate_dir if keep_intermediate else None,
                base_tts_mp3=base_tts_mp3 if keep_intermediate else None,
                base_tts_wav=base_tts_wav if keep_intermediate else None,
            )

        if output_path.exists():
            output_path.unlink()

        assert resolved_model_file is not None
        assert command_template is not None
        rendered_command = _render_rvc_command(
            command_template=command_template,
            input_file=base_tts_wav,
            output_file=output_path,
            model_file=resolved_model_file,
            index_file=resolved_index_file,
            pitch=pitch,
            f0_method=f0_method,
        )
        if verbose:
            print("Running RVC voice conversion...")
        _run_rvc_command(rendered_command, working_dir=working_dir, verbose=verbose)
        if not output_path.is_file():
            raise FileNotFoundError(
                "RVC command finished but did not create the expected output: "
                f"{output_path}"
            )
        if verbose:
            print(f"Converted WAV written to: {output_path}")

        return _build_result(
            output_path=output_path,
            text=normalized_text,
            tts_voice=tts_voice,
            tts_rate=tts_rate,
            tts_only=False,
            model_file=resolved_model_file,
            index_file=resolved_index_file,
            rvc_command=rendered_command,
            rvc_working_dir=working_dir,
            intermediate_dir=intermediate_dir if keep_intermediate else None,
            base_tts_mp3=base_tts_mp3 if keep_intermediate else None,
            base_tts_wav=base_tts_wav if keep_intermediate else None,
        )
    finally:
        if temp_context is not None:
            temp_context.cleanup()


def _normalize_text(text: str) -> str:
    normalized_text = str(text or "").strip()
    if not normalized_text:
        raise ValueError("Text is empty. Provide Chinese text to synthesize.")
    return normalized_text


def _resolve_output_file(output_file: str | Path | None) -> Path:
    output_path = Path(output_file) if output_file else default_voice_output_path()
    if not output_path.suffix:
        output_path = output_path.with_suffix(".wav")
    if output_path.suffix.lower() != ".wav":
        raise ValueError(f"Output file must use a .wav suffix: {output_path}")
    return output_path


def _resolve_model_file(model_file: str | Path | None) -> Path:
    model_path = Path(model_file) if model_file else DEFAULT_MODEL_FILE
    if not model_path.is_file():
        raise FileNotFoundError(
            "RVC model file not found. Put the Manbo model at "
            f"{DEFAULT_MODEL_FILE} or pass --model: {model_path}"
        )
    return model_path


def _resolve_index_file(index_file: str | Path | None) -> Path | None:
    if index_file is None:
        return DEFAULT_INDEX_FILE if DEFAULT_INDEX_FILE.is_file() else None

    index_path = Path(index_file)
    if not index_path.is_file():
        raise FileNotFoundError(f"RVC index file not found: {index_path}")
    return index_path


def _resolve_rvc_command(rvc_command: str | None) -> str | None:
    command = (
        rvc_command
        or os.environ.get(RVC_COMMAND_ENV_VAR)
        or os.environ.get(FALLBACK_RVC_COMMAND_ENV_VAR)
    )
    command = str(command or "").strip()
    return command or None


def _resolve_working_dir(rvc_working_dir: str | Path | None) -> Path | None:
    if rvc_working_dir is None:
        return None
    working_dir = Path(rvc_working_dir)
    if not working_dir.is_dir():
        raise NotADirectoryError(f"RVC working directory not found: {working_dir}")
    return working_dir


def _prepare_intermediate_dir(
    output_path: Path,
    keep_intermediate: bool,
) -> tuple[Path, Any | None]:
    if keep_intermediate:
        intermediate_dir = output_path.parent / f"{output_path.stem}_intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        return intermediate_dir, None

    temp_context = tempfile.TemporaryDirectory(prefix="tts_rvc_")
    return Path(temp_context.name), temp_context


def _save_edge_tts_audio(
    text: str,
    voice: str,
    rate: str,
    output_file: str | Path,
) -> None:
    try:
        import edge_tts
    except ImportError as exception:
        raise RuntimeError(
            "edge-tts is not installed. Install it with: pip install edge-tts"
        ) from exception

    async def save_audio() -> None:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
        await communicate.save(str(output_file))

    asyncio.run(save_audio())


def _convert_audio_to_wav(input_file: str | Path, output_file: str | Path) -> None:
    try:
        import ffmpeg
    except ImportError as exception:
        raise RuntimeError(
            "ffmpeg-python is not installed. Install it with: pip install ffmpeg-python"
        ) from exception

    try:
        (
            ffmpeg.input(str(input_file))
            .output(str(output_file), acodec="pcm_s16le", ac=1, ar="44100")
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exception:
        stderr = _decode_process_output(getattr(exception, "stderr", b""))
        raise RuntimeError(
            "FFmpeg failed to convert the base TTS audio to WAV. "
            "Install system FFmpeg and make sure it is on PATH."
            + (f"\n{stderr}" if stderr else "")
        ) from exception


def _render_rvc_command(
    command_template: str,
    input_file: Path,
    output_file: Path,
    model_file: Path,
    index_file: Path | None,
    pitch: int,
    f0_method: str,
) -> str:
    if not _has_any_placeholder(command_template, ("input", "input_file")):
        raise ValueError("RVC command must include {input} or {input_file}.")
    if not _has_any_placeholder(command_template, ("output", "output_file")):
        raise ValueError("RVC command must include {output} or {output_file}.")

    index_value = _quote_for_shell(index_file) if index_file else ""
    index_option = f"--index {index_value}" if index_file else ""
    replacements = {
        "input": _quote_for_shell(input_file),
        "input_file": _quote_for_shell(input_file),
        "output": _quote_for_shell(output_file),
        "output_file": _quote_for_shell(output_file),
        "model": _quote_for_shell(model_file),
        "model_file": _quote_for_shell(model_file),
        "index": index_value,
        "index_file": index_value,
        "index_option": index_option,
        "pitch": str(int(pitch)),
        "f0_method": str(f0_method),
        "f0": str(f0_method),
    }

    rendered_command = command_template
    for key, value in replacements.items():
        rendered_command = rendered_command.replace("{" + key + "}", value)
    return rendered_command


def _has_any_placeholder(command_template: str, names: tuple[str, ...]) -> bool:
    return any("{" + name + "}" in command_template for name in names)


def _quote_for_shell(value: str | Path | None) -> str:
    if value is None:
        return ""
    text = str(value)
    if platform.system().lower().startswith("win"):
        return subprocess.list2cmdline([text])
    return shlex.quote(text)


def _run_rvc_command(
    command: str,
    working_dir: Path | None,
    verbose: bool,
) -> None:
    completed_process = subprocess.run(
        command,
        shell=True,
        cwd=str(working_dir) if working_dir else None,
        text=True,
        capture_output=True,
    )
    if verbose and completed_process.stdout.strip():
        print(completed_process.stdout.strip())
    if completed_process.returncode != 0:
        detail = completed_process.stderr.strip() or completed_process.stdout.strip()
        raise RuntimeError(
            f"RVC command failed with exit code {completed_process.returncode}."
            + (f"\n{detail}" if detail else "")
        )


def _build_result(
    output_path: Path,
    text: str,
    tts_voice: str,
    tts_rate: str,
    tts_only: bool,
    model_file: Path | None,
    index_file: Path | None,
    rvc_command: str | None,
    rvc_working_dir: Path | None,
    intermediate_dir: Path | None,
    base_tts_mp3: Path | None,
    base_tts_wav: Path | None,
) -> dict[str, Any]:
    return {
        "output_file": str(output_path),
        "text_length": len(text),
        "tts_voice": tts_voice,
        "tts_rate": tts_rate,
        "tts_only": tts_only,
        "model_file": str(model_file) if model_file else None,
        "index_file": str(index_file) if index_file else None,
        "rvc_command": rvc_command,
        "rvc_working_dir": str(rvc_working_dir) if rvc_working_dir else None,
        "intermediate_dir": str(intermediate_dir) if intermediate_dir else None,
        "base_tts_mp3": str(base_tts_mp3) if base_tts_mp3 else None,
        "base_tts_wav": str(base_tts_wav) if base_tts_wav else None,
    }


def _decode_process_output(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()
