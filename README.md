# tools_v2

一套可作为 Python 包导入、也可通过 `scripts/` 直接运行的数据与媒体处理工具。现已整合旧版 `daliy_dataprocess_tools` 的日常能力：

- **OCR 工作流**：提取图片文字，并按文字量过滤图片。
- **NSFW 检测工作流**：检测图片/视频 NSFW 内容并分类移动。
- **音视频转文字**：Whisper 转录到带时间轴 CSV，支持断点续传和简体中文转换。
- **视频字幕与学习笔记**：为单个视频生成 SRT 字幕、raw text、整理稿和学习笔记 Markdown。
- **说话人识别转录**：WhisperX + pyannote 输出带 speaker 标签的 CSV。
- **批量转录**：遍历目录内音视频并逐个转写。
- **实时转录**：麦克风实时录音、Whisper 转写、Web 页面查看和下载结果。
- **字幕翻译**：选择屏幕区域，Tesseract OCR 稳定检测后调用腾讯云翻译。
- **中文文本转 RVC 音色语音**：先生成中文底音，再调用本地 RVC 后端转换为目标音色。
- **文本辅助处理**：提取 CSV `text` 列、合并/去重/分类 OneTab 导出文件。

## 目录结构

```
tools_v2/
├── tools/               # 核心工具库（可作为 Python 包导入）
│   ├── __init__.py
│   ├── ocr_tool.py      # OCR 工作流 + CLI
│   ├── nsfw_tool.py     # NSFW 检测工作流 + CLI
│   └── tts_rvc_tool.py  # 中文 TTS + RVC 编排
├── scripts/             # 批量处理脚本（从项目根目录运行）
│   ├── ocr_filter.py    # 按文字量过滤并移动图片
│   ├── nsfw_filter.py   # 检测并移动 NSFW 图片/视频
│   ├── audio_to_text.py
│   ├── video_to_subtitle_notes.py
│   ├── audio_to_text_diarize.py
│   ├── batch_transcribe.py
│   ├── realtime_transcriber_web.py
│   ├── subtitle_translator_gui.py
│   ├── screen_ocr_translate_once.py
│   ├── text_to_manbo_voice.py
│   ├── extract_text_column.py
│   └── process_onetab.py
├── docs/                # 文档
│   ├── installation.md
│   ├── ocr_usage.md
│   ├── nsfw_usage.md
│   ├── audio_transcription_usage.md
│   ├── video_learning_notes_usage.md
│   ├── realtime_transcription_usage.md
│   ├── subtitle_translation_usage.md
│   ├── tts_rvc_usage.md
│   ├── manbo_voice_plan.md
│   └── text_processing_usage.md
├── prompts/             # 大模型 prompt 模板
│   ├── video_clean_transcript_prompt.md
│   └── video_study_notes_prompt.md
├── .vscode/
│   └── settings.json    # VSCode 使用 conda whisper2 环境
├── requirements.txt
└── setup.py
```

## 快速开始

详见 [docs/installation.md](docs/installation.md)。

### OCR 过滤

```bash
python scripts/ocr_filter.py --input-dir ./images --output-dir ./text_images
```

### NSFW 过滤

```bash
python scripts/nsfw_filter.py \
    --input-dir ./media \
    --output-images ./nsfw_images \
    --output-videos ./nsfw_videos
```

### 音视频转文字

```bash
python scripts/audio_to_text.py audio.mp3 -m turbo -l zh
```

### 视频字幕与学习笔记

```bash
python scripts/video_to_subtitle_notes.py lecture.mp4 -m turbo -l zh
```

需要设置 `DASHSCOPE_API_KEY` 或传入 `--api-key`。默认会输出到按视频文件名命名的目录；如果目录无法创建，会追加随机数字后缀。整理正文和生成学习笔记分别使用 `prompts/video_clean_transcript_prompt.md` 与 `prompts/video_study_notes_prompt.md`。

### 批量转录

```bash
python scripts/batch_transcribe.py ./recordings -r -m turbo -l zh
```

### 实时转录 Web

```bash
python scripts/realtime_transcriber_web.py --port 5000 --model base --language auto
```

打开 `http://127.0.0.1:5000` 使用。

### 字幕翻译 GUI

```bash
python scripts/subtitle_translator_gui.py
```

需要先安装 Tesseract OCR，并设置 `TENCENT_SECRET_ID` / `TENCENT_SECRET_KEY`。

### 中文文本转 RVC 音色语音

```bash
python scripts/text_to_manbo_voice.py "今天也要元气满满。" --tts-only
```

有合法可用的本地 RVC 模型后，可传入模型路径和外部 RVC 推理命令：

```bash
python scripts/text_to_manbo_voice.py "今天也要元气满满。" \
    --model data/rvc_models/manbo/manbo.pth \
    --index data/rvc_models/manbo/manbo.index \
    --rvc-command "python infer_cli.py --input {input} --output {output} --model {model} {index_option} --pitch {pitch} --f0-method {f0_method}"
```

### CSV 文本列提取 / OneTab 处理

```bash
python scripts/extract_text_column.py transcript.csv -c text
python scripts/process_onetab.py --input ./onetab_exports
```
