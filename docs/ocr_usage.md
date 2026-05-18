# OCR 工作流使用说明

基于 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)，支持中文、英文及多种语言。

---

## Python API

```python
from tools.ocr_tool import OCRProcessor

# 初始化（首次调用自动下载模型）
processor = OCRProcessor(lang="ch", use_gpu=False)

# 提取全文
text = processor.extract_text("photo.jpg")
print(text)

# 获取字符数
n = processor.get_text_length("photo.jpg")
print(f"文字数量：{n}")

# 获取详细结果（含每行文字、置信度、坐标）
result = processor.process_image("photo.jpg")
print(result["text"])        # 全文
print(result["length"])      # 字符数
for line in result["lines"]:
    print(line["text"], line["confidence"])
```

### `OCRProcessor` 构造参数

| 参数       | 类型 | 默认值  | 说明                                                                |
| ---------- | ---- | ------- | ------------------------------------------------------------------- |
| `lang`     | str  | `"ch"`  | 语言代码。`ch`=中英混合，`en`=英文，`japan`=日文，`korean`=韩文，等 |
| `use_gpu`  | bool | `False` | 是否使用 GPU                                                        |
| `show_log` | bool | `False` | 是否显示 PaddleOCR 内部日志                                         |

---

## 命令行接口

```bash
# 基本用法
python -m tools.ocr_tool --image photo.jpg

# 只输出文字
python -m tools.ocr_tool --image photo.jpg --text-only

# 只输出字符数（适合脚本）
python -m tools.ocr_tool --image photo.jpg --length-only

# 指定语言
python -m tools.ocr_tool --image photo.jpg --lang en

# GPU 加速
python -m tools.ocr_tool --image photo.jpg --gpu

# 保存 JSON 结果
python -m tools.ocr_tool --image photo.jpg --output-json result.json
```

### 参数说明

| 参数                 | 说明                  |
| -------------------- | --------------------- |
| `--image`            | 输入图片路径（必填）  |
| `--lang`             | 语言代码，默认 `ch`   |
| `--gpu`              | 启用 GPU              |
| `--text-only`        | 只打印提取的文字      |
| `--length-only`      | 只打印字符数          |
| `--output-json FILE` | 将完整结果保存为 JSON |

---

## OCR Filter 脚本

批量处理目录下所有图片，将文字量超过阈值的图片移动到输出目录。

```bash
python scripts/ocr_filter.py \
    --input-dir ./images \
    --output-dir ./text_images

# 调整阈值（默认 30）
python scripts/ocr_filter.py \
    --input-dir ./images \
    --output-dir ./text_images \
    --min-chars 50

# 预览（不实际移动）
python scripts/ocr_filter.py \
    --input-dir ./images \
    --output-dir ./text_images \
    --dry-run
```

### 参数说明

| 参数           | 默认值  | 说明                           |
| -------------- | ------- | ------------------------------ |
| `--input-dir`  | —       | 源图片目录（必填）             |
| `--output-dir` | —       | 输出目录（必填）               |
| `--min-chars`  | `30`    | 文字字符数阈值，超过此值才移动 |
| `--lang`       | `ch`    | OCR 语言                       |
| `--gpu`        | `False` | 启用 GPU                       |
| `--dry-run`    | `False` | 预览模式，不移动文件           |

### 支持的图片格式

`.jpg` `.jpeg` `.png` `.bmp` `.webp` `.tiff` `.tif`
