"""tools_v2 package public API.

Heavy third-party dependencies are imported lazily so one workflow can be used
without installing every optional backend used by the others.
"""

from __future__ import annotations

from typing import Any


__all__ = [
	"OCRProcessor",
	"NSFWDetector",
	"CLIPDetector",
	"FalconsaiDetector",
	"transcribe_audio",
	"transcribe_with_diarization",
	"batch_transcribe_directory",
	"find_media_files",
	"generate_video_learning_package",
	"AudioRecorder",
	"WhisperTranscriber",
	"TranscriptionLogger",
	"create_realtime_app",
	"run_realtime_server",
	"ScreenOCR",
	"StabilityDetector",
	"TencentTranslator",
	"SubtitleTranslatorConfig",
	"extract_text_column",
	"process_onetab_files",
]


def __getattr__(name: str) -> Any:
	if name == "OCRProcessor":
		from .ocr_tool import OCRProcessor

		return OCRProcessor
	if name in {"NSFWDetector", "CLIPDetector", "FalconsaiDetector"}:
		from .nsfw_tool import CLIPDetector, FalconsaiDetector, NSFWDetector

		return {
			"NSFWDetector": NSFWDetector,
			"CLIPDetector": CLIPDetector,
			"FalconsaiDetector": FalconsaiDetector,
		}[name]
	if name in {
		"transcribe_audio",
		"transcribe_with_diarization",
		"batch_transcribe_directory",
		"find_media_files",
	}:
		from . import audio_transcription_tool as module

		return getattr(module, name)
	if name in {"generate_video_learning_package"}:
		from . import video_learning_notes_tool as module

		return getattr(module, name)
	if name in {
		"AudioRecorder",
		"WhisperTranscriber",
		"TranscriptionLogger",
		"create_realtime_app",
		"run_realtime_server",
	}:
		from . import realtime_transcription_tool as module

		return getattr(module, name)
	if name in {
		"ScreenOCR",
		"StabilityDetector",
		"TencentTranslator",
		"SubtitleTranslatorConfig",
	}:
		from . import subtitle_translation_tool as module

		return getattr(module, name)
	if name in {"extract_text_column", "process_onetab_files"}:
		from . import text_processing_tool as module

		return getattr(module, name)
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
