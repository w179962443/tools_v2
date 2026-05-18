# 视频字幕与学习笔记生成

这个工作流会对一个音视频文件完成以下输出：

- Whisper 时间轴 CSV：`*_transcript.csv`
- SRT 字幕：`*.srt`
- 原始全文：`*_raw_text.txt`
- 大模型整理后的完整文字：`*_polished_text.txt`
- 学习笔记：`*_learning_notes.md`

默认输出目录使用视频文件名，例如 `lecture.mp4` 会输出到 `lecture/`。如果同名目录无法创建，会自动追加随机数字后缀，例如 `lecture_48291/`。

## 准备

安装音视频转写和大模型调用依赖：

```bash
pip install -r requirements.txt
```

设置 DashScope API Key：

```bash
set DASHSCOPE_API_KEY=你的Key
```

PowerShell：

```powershell
$env:DASHSCOPE_API_KEY="你的Key"
```

## 命令行

```bash
python scripts/video_to_subtitle_notes.py lecture.mp4 -m turbo -l zh
```

指定输出目录和大模型：

```bash
python scripts/video_to_subtitle_notes.py lecture.mp4 -o ./lecture_outputs --llm-model qwen-plus
```

只生成字幕和 raw text，不调用大模型：

```bash
python scripts/video_to_subtitle_notes.py lecture.mp4 --skip-llm
```

指定 prompt 文件：

```bash
python scripts/video_to_subtitle_notes.py lecture.mp4 \
    --clean-prompt-file prompts/video_clean_transcript_prompt.md \
    --notes-prompt-file prompts/video_study_notes_prompt.md
```

默认整理正文 prompt 位于 `prompts/video_clean_transcript_prompt.md`，默认学习笔记 prompt 位于 `prompts/video_study_notes_prompt.md`。两个任务会分别调用大模型。

## Python API

```python
from tools.video_learning_notes_tool import generate_video_learning_package

result = generate_video_learning_package(
    video_file="lecture.mp4",
    whisper_model="turbo",
    language="zh",
    clean_prompt_file="prompts/video_clean_transcript_prompt.md",
    notes_prompt_file="prompts/video_study_notes_prompt.md",
)

print(result.outputs.learning_notes)
```
