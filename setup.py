from setuptools import setup, find_packages

audio_requires = [
    "openai-whisper>=20231117",
    "torch>=2.0.0",
    "torchaudio>=2.0.0",
    "ffmpeg-python>=0.2.0",
    "opencc-python-reimplemented>=0.1.7",
]

llm_requires = [
    "openai>=1.0.0",
]

realtime_requires = [
    "Flask>=2.3.0",
    "Flask-CORS>=4.0.0",
    "sounddevice>=0.4.6",
    "librosa>=0.10.0",
    "scipy>=1.11.0",
    "python-dotenv>=1.0.0",
    "requests>=2.31.0",
]

diarize_requires = [
    "whisperx",
    "pyannote.audio>=3.1.0",
]

subtitle_requires = [
    "PyQt5>=5.15.0",
    "pytesseract>=0.3.10",
    "tencentcloud-sdk-python>=3.0.0",
]

setup(
    name="tools_v2",
    version="0.2.0",
    description="OCR, NSFW, transcription, subtitle translation, and data workflow tools",
    packages=find_packages(),
    include_package_data=True,
    package_data={"prompts": ["*.md"]},
    python_requires=">=3.8",
    install_requires=[
        "paddlepaddle>=2.5.0",
        "paddleocr>=2.7.0",
        "nudenet>=3.4.0",
        "opencv-python>=4.8.0",
        "Pillow>=9.0.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "audio": audio_requires,
        "video-notes": audio_requires + llm_requires,
        "realtime": audio_requires + realtime_requires,
        "diarize": audio_requires + diarize_requires,
        "subtitle": subtitle_requires,
        "all": audio_requires + llm_requires + realtime_requires + diarize_requires + subtitle_requires,
    },
    entry_points={
        "console_scripts": [
            "ocr-tool=tools.ocr_tool:main",
            "nsfw-tool=tools.nsfw_tool:main",
            "ocr-filter=scripts.ocr_filter:main",
            "nsfw-filter=scripts.nsfw_filter:main",
            "audio-to-text=scripts.audio_to_text:main",
            "audio-to-text-diarize=scripts.audio_to_text_diarize:main",
            "batch-transcribe=scripts.batch_transcribe:main",
            "video-to-subtitle-notes=scripts.video_to_subtitle_notes:main",
            "extract-text-column=scripts.extract_text_column:main",
            "process-onetab=scripts.process_onetab:main",
            "realtime-transcriber=scripts.realtime_transcriber_web:main",
            "subtitle-translator=scripts.subtitle_translator_gui:main",
            "screen-ocr-translate=scripts.screen_ocr_translate_once:main",
        ],
    },
)
