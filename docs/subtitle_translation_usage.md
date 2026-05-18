# 字幕 OCR 翻译使用说明

该工具监控屏幕选区，内容稳定后用 Tesseract OCR 识别文本，再调用腾讯云机器翻译。

## 前置要求

1. 安装 Tesseract OCR，并安装中文简体和英文语言包。
2. 设置腾讯云密钥：

```powershell
$env:TENCENT_SECRET_ID = "你的SecretId"
$env:TENCENT_SECRET_KEY = "你的SecretKey"
```

## GUI

```bash
python scripts/subtitle_translator_gui.py
```

点击“选择区域”后拖拽屏幕字幕区域，再点击“开始监控”。

## 一次性 OCR + 翻译

```bash
python scripts/screen_ocr_translate_once.py 100 100 700 260 --target zh
```

## Python API

```python
from tools.subtitle_translation_tool import capture_translate_once

result = capture_translate_once((100, 100, 700, 260))
print(result["source_text"])
print(result["translation"]["target_text"])
```
