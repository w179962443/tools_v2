# 音视频转文字使用说明

## Python API

```python
from tools.audio_transcription_tool import transcribe_audio

transcribe_audio(
    "audio.mp3",
    model_name="turbo",
    language="zh",
    output_file="audio.csv",
)
```

输出 CSV 字段：`start_time`、`end_time`、`start_timestamp`、`end_timestamp`、`duration`、`text`。

如果输出 CSV 已存在，会读取最后一行 `end_time` 并只追加后续分段。

## 命令行

```bash
python scripts/audio_to_text.py audio.mp3 -m turbo -l zh
python scripts/audio_to_text.py video.mp4 -m large -l en -o transcript.csv
python scripts/audio_to_text.py audio.mp3 -m turbo -l zh -d D:\whisper_models
```

## 说话人识别

```bash
python scripts/audio_to_text_diarize.py meeting.mp3 -m turbo -l zh -t hf_xxxx
python scripts/audio_to_text_diarize.py meeting.mp3 --min-speakers 2 --max-speakers 2
```

也可以设置环境变量 `HF_TOKEN`。输出 CSV 额外包含 `speaker` 字段。

## 批量转录

```bash
python scripts/batch_transcribe.py ./recordings -r -m turbo -l zh
python scripts/batch_transcribe.py ./recordings --skip-existing
```

支持格式：`.mp3` `.wav` `.m4a` `.flac` `.ogg` `.webm` `.aac` `.wma` `.mp4` `.mkv` `.avi` `.mov` `.wmv` `.flv` `.ts` `.m4v`。
