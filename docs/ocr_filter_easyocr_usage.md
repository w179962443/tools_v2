# EasyOCR 图片文字量过滤使用说明

基于 [EasyOCR](https://github.com/JaidedAI/EasyOCR) 的图片文字量过滤脚本。递归扫描指定目录下的所有图片，将含文字量超过阈值的图片移动到输出目录，保留子目录结构。

与 PaddleOCR 版本（`scripts/ocr_filter.py`）的区别：
- 使用 EasyOCR 替代 PaddleOCR，安装更轻量，不依赖 PaddlePaddle 框架
- 递归扫描子目录，而非仅扫描一层
- 保留移动图片的相对子目录结构
- 自动跳过宽/高超过 4000px 的大图（直接移动，不做 OCR）
- 超过 3 秒仍未完成 OCR 的图片自动判定为文字过多并移动
- 使用 detect+recognize 分步识别，达到字符阈值后立即停止，加快处理速度

---

## 环境准备

使用 conda 环境 `iv`：

```bash
conda activate iv
pip install easyocr
```

首次运行时会自动下载模型文件（简体中文 `craft_mlt_25k.zip` + `zh_sim_g2.pth` + 英文 `english_g2.pth`）。

---

## 命令行用法

```bash
# 基本用法：递归扫描目录，文字量 >50 字符的图片移动到输出目录
python scripts/ocr_filter_easyocr.py \
    --input-dir "D:\pictures\待筛选" \
    --output-dir "D:\pictures\带文字"

# 调整阈值（默认 50）
python scripts/ocr_filter_easyocr.py \
    --input-dir ./images \
    --output-dir ./text_images \
    --min-chars 100

# 启用 GPU 加速
python scripts/ocr_filter_easyocr.py \
    --input-dir ./images \
    --output-dir ./text_images \
    --gpu

# 指定语言（默认简体中文+英文）
python scripts/ocr_filter_easyocr.py \
    --input-dir ./images \
    --output-dir ./text_images \
    --lang ch_sim en ja

# 预览模式（不实际移动文件）
python scripts/ocr_filter_easyocr.py \
    --input-dir ./images \
    --output-dir ./text_images \
    --dry-run
```

---

## 参数说明

| 参数           | 默认值            | 说明                                                       |
| -------------- | ----------------- | ---------------------------------------------------------- |
| `--input-dir`  | —                 | 源图片目录（必填），递归扫描所有子目录                     |
| `--output-dir` | —                 | 输出目录（必填），保留子目录结构                           |
| `--min-chars`  | `50`              | 文字字符数阈值，超过此值才移动                             |
| `--lang`       | `ch_sim en`       | EasyOCR 语言代码，可指定多个。常用：`ch_sim`(简体)、`ch_tra`(繁体)、`en`(英文)、`ja`(日文)、`ko`(韩文) |
| `--gpu`        | `False`           | 启用 GPU 加速                                              |
| `--dry-run`    | `False`           | 预览模式，只显示处理结果，不实际移动文件                   |

---

## 处理逻辑

对每张图片按以下顺序判断：

1. **尺寸检查**：宽或高 > 4000px → 直接移动（跳过 OCR）
2. **OCR 超时**：单张图片 OCR 耗时 > 3 秒 → 判定为文字过多，移动
3. **文字量阈值**：检测文字区域后逐区域识别，累计字符数 > `--min-chars` → 移动
4. **文字量不足**：所有区域识别完毕，字符数 ≤ `--min-chars` → 跳过

---

## 输出示例

```
Found 1523 image(s)  |  threshold: >50 chars
Initializing EasyOCR…
[   1/1523] photo1.jpg  →  >50 chars  →  MOVE
[   2/1523] photo2.png  →  ≤50 chars  →  skip
[   3/1523] scan.jpg     →  oversized (5120x3840)  →  MOVE
[   4/1523] big.png      →  timeout  →  MOVE
[   5/1523] corrupt.gif  →  SKIP (corrupt/unreadable)

Done.  Moved: 423  |  Skipped: 1098  |  Errors: 2
```

---

## 支持的图片格式

`.jpg` `.jpeg` `.png` `.bmp` `.webp` `.tiff` `.tif`

---

## 注意事项

- Windows 下路径含中文或空格时，请用引号包裹路径参数
- 同名文件移动到输出目录时会自动添加数字后缀避免覆盖
- 中文路径完全支持（使用 PIL 读取图片后转 numpy 数组，绕过 OpenCV 的 Unicode 路径限制）