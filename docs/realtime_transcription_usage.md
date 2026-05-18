# 实时转录使用说明

实时转录模块提供录音、Whisper 转写、会话日志和 Flask Web 页面。

## 启动 Web 应用

```bash
python scripts/realtime_transcriber_web.py --port 5000 --model base --language auto
```

打开 `http://127.0.0.1:5000`，选择音源、语言和模型后开始转录。转录文件会保存到 `recordings/`。

## Python API

```python
from tools.realtime_transcription_tool import AudioRecorder, WhisperTranscriber

recorder = AudioRecorder(sample_rate=16000, chunk_duration=0.5)
transcriber = WhisperTranscriber(model_name="base", language="auto")
```

系统声音录制仍依赖操作系统层面的 loopback 或虚拟声卡配置；没有配置时会使用默认输入设备。
