# 中文文本转 RVC 音色语音

这个工具用于把指定中文生成语音，并在有本地 RVC 模型时转换为目标音色。完整流程是：中文文本 -> 普通中文 TTS 底音 -> RVC 音色转换 -> WAV 文件。

RVC 不是文本转语音模型，它需要一段已有语音作为输入。因此没有 RVC 模型时，只能使用 `--tts-only` 生成普通中文语音，不能得到目标音色。

## 准备

安装语音生成依赖：

```bash
pip install -e .[voice]
```

还需要系统可用的 FFmpeg，并确保 `ffmpeg` 在 `PATH` 中。

如果要转换为曼波音色，请将合法可用的 RVC 模型放到：

```text
data/rvc_models/manbo/manbo.pth
data/rvc_models/manbo/manbo.index
```

`.index` 文件可选但推荐保留，通常能改善相似度。RVC 推理后端需要单独安装，例如 Applio、RVC WebUI 或其他提供命令行推理能力的实现。

## 命令行

先只生成普通中文底音，用来确认 `edge-tts`、网络和 FFmpeg 正常：

```bash
python scripts/text_to_manbo_voice.py "今天也要元气满满。" --tts-only
```

指定输出文件：

```bash
python scripts/text_to_manbo_voice.py "今天也要元气满满。" \
    -o data/voice_outputs/test.wav \
    --tts-only
```

从文本文件读取：

```bash
python scripts/text_to_manbo_voice.py --input-text-file input.txt --tts-only
```

配置 RVC 转换时，需要传入外部 RVC 推理命令模板。模板里的占位符会在运行时替换为实际路径：

```bash
python scripts/text_to_manbo_voice.py "今天也要元气满满。" \
    --model data/rvc_models/manbo/manbo.pth \
    --index data/rvc_models/manbo/manbo.index \
    --rvc-working-dir D:\RVC \
    --rvc-command "python infer_cli.py --input {input} --output {output} --model {model} {index_option} --pitch {pitch} --f0-method {f0_method}"
```

也可以设置环境变量，避免每次输入完整命令：

```powershell
$env:MANBO_RVC_COMMAND='python infer_cli.py --input {input} --output {output} --model {model} {index_option} --pitch {pitch} --f0-method {f0_method}'
python scripts/text_to_manbo_voice.py "今天也要元气满满。" --model data/rvc_models/manbo/manbo.pth
```

可用占位符：

| 占位符 | 含义 |
| --- | --- |
| `{input}` / `{input_file}` | 底音 TTS WAV 文件 |
| `{output}` / `{output_file}` | 最终输出 WAV 文件 |
| `{model}` / `{model_file}` | RVC `.pth` 模型文件 |
| `{index}` / `{index_file}` | RVC `.index` 文件；没有时为空 |
| `{index_option}` | 有 index 时展开为 `--index <file>`，否则为空 |
| `{pitch}` | 半音偏移，默认 `0` |
| `{f0_method}` / `{f0}` | F0 提取方法，默认 `rmvpe` |

## Python API

```python
from tools.tts_rvc_tool import synthesize_chinese_with_rvc

result = synthesize_chinese_with_rvc(
    text="今天也要元气满满。",
    output_file="data/voice_outputs/manbo.wav",
    model_file="data/rvc_models/manbo/manbo.pth",
    index_file="data/rvc_models/manbo/manbo.index",
    rvc_command=(
        "python infer_cli.py --input {input} --output {output} "
        "--model {model} {index_option} --pitch {pitch} --f0-method {f0_method}"
    ),
    rvc_working_dir="D:/RVC",
)

print(result["output_file"])
```

无模型时只试 TTS：

```python
from tools.tts_rvc_tool import synthesize_chinese_with_rvc

synthesize_chinese_with_rvc(
    text="今天也要元气满满。",
    output_file="data/voice_outputs/base.wav",
    tts_only=True,
)
```

## 常见问题

**没有模型能不能生成曼波音色？** 不能。没有 RVC 模型时只能生成普通中文 TTS 底音。

**底音 TTS 是什么？** 它是给 RVC 的输入语音。RVC 负责改变音色，但节奏、停顿和部分发音习惯仍会受底音影响。

**`.index` 必须有吗？** 不是必须，但推荐使用。很多 RVC 模型搭配 index 时相似度会更稳定。

**为什么这里不直接内置 RVC？** RVC 生态里的 CLI、依赖和 GPU 兼容性差异很大。当前工具先提供稳定的编排层，通过外部命令连接你已配置好的 RVC 后端。

**输出为什么固定 WAV？** RVC 后端通常以 WAV 作为输入输出最稳定；需要其他格式时，可以在生成后再用 FFmpeg 转换。