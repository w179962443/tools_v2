# NSFW 检测工作流使用说明

基于 [NudeNet](https://github.com/notAI-tech/NudeNet)，支持图片和视频（通过采样帧检测）。

---

## NSFW 标签定义

以下 NudeNet 标签被判定为 NSFW：

| 标签                       | 说明           |
| -------------------------- | -------------- |
| `FEMALE_GENITALIA_EXPOSED` | 女性生殖器裸露 |
| `FEMALE_BREAST_EXPOSED`    | 女性胸部裸露   |
| `MALE_GENITALIA_EXPOSED`   | 男性生殖器裸露 |
| `ANUS_EXPOSED`             | 肛门裸露       |
| `BUTTOCKS_EXPOSED`         | 臀部裸露       |

---

## Python API

### 检测图片

```python
from tools.nsfw_tool import NSFWDetector

detector = NSFWDetector(threshold=0.5)

result = detector.detect_image("photo.jpg")
print(result["is_nsfw"])        # True / False
print(result["nsfw_labels"])    # ['FEMALE_BREAST_EXPOSED', …]
```

### 检测视频

```python
result = detector.detect_video(
    "video.mp4",
    sample_interval=2.0,   # 每 2 秒采样一帧
    max_frames=100,        # 最多检查 100 帧
)
print(result["is_nsfw"])
print(result["nsfw_frames"])    # [{"frame_idx": 42, "timestamp": 1.68, "nsfw_labels": […]}, …]
print(result["checked_frames"]) # 实际检查的帧数
```

### 统一接口（自动识别图片/视频）

```python
result = detector.detect("file.mp4", sample_interval=3.0)
print(result["file_type"])  # "image" 或 "video"
print(result["is_nsfw"])
```

### `NSFWDetector` 构造参数

| 参数        | 类型  | 默认值 | 说明                                        |
| ----------- | ----- | ------ | ------------------------------------------- |
| `threshold` | float | `0.5`  | 置信度阈值（0–1），低于此值的检测结果被忽略 |

---

## 命令行接口

```bash
# 检测图片
python -m tools.nsfw_tool --file photo.jpg

# 检测视频
python -m tools.nsfw_tool --file video.mp4

# 调整阈值
python -m tools.nsfw_tool --file photo.jpg --threshold 0.6

# 调整视频采样参数
python -m tools.nsfw_tool --file video.mp4 --sample-interval 5 --max-frames 50

# 保存 JSON 结果
python -m tools.nsfw_tool --file photo.jpg --output-json result.json

# 仅返回退出码（0=安全，1=NSFW），适合 shell 脚本
python -m tools.nsfw_tool --file photo.jpg --status-only
```

### 参数说明

| 参数                 | 默认值  | 说明                   |
| -------------------- | ------- | ---------------------- |
| `--file`             | —       | 输入文件路径（必填）   |
| `--threshold`        | `0.5`   | 置信度阈值             |
| `--sample-interval`  | `2.0`   | 视频采样间隔（秒）     |
| `--max-frames`       | `100`   | 每个视频最多检查的帧数 |
| `--output-json FILE` | —       | 保存完整结果为 JSON    |
| `--status-only`      | `False` | 仅返回退出码           |

---

## NSFW Filter 脚本

批量检测目录下所有图片和视频，将 NSFW 文件分别移动到两个输出目录。

```bash
python scripts/nsfw_filter.py \
    --input-dir ./media \
    --output-images ./nsfw_images \
    --output-videos ./nsfw_videos

# 调高阈值减少误判
python scripts/nsfw_filter.py \
    --input-dir ./media \
    --output-images ./nsfw_images \
    --output-videos ./nsfw_videos \
    --threshold 0.65

# 对大视频稀疏采样（每 10 秒一帧）
python scripts/nsfw_filter.py \
    --input-dir ./media \
    --output-images ./nsfw_images \
    --output-videos ./nsfw_videos \
    --sample-interval 10 --max-frames 30

# 预览模式
python scripts/nsfw_filter.py \
    --input-dir ./media \
    --output-images ./nsfw_images \
    --output-videos ./nsfw_videos \
    --dry-run
```

### 参数说明

| 参数                | 默认值  | 说明                      |
| ------------------- | ------- | ------------------------- |
| `--input-dir`       | —       | 源媒体目录（必填）        |
| `--output-images`   | —       | NSFW 图片输出目录（必填） |
| `--output-videos`   | —       | NSFW 视频输出目录（必填） |
| `--threshold`       | `0.5`   | 置信度阈值                |
| `--sample-interval` | `2.0`   | 视频采样间隔（秒）        |
| `--max-frames`      | `100`   | 每个视频最多检查帧数      |
| `--dry-run`         | `False` | 预览模式，不移动文件      |

### 支持的格式

**图片**：`.jpg` `.jpeg` `.png` `.bmp` `.webp` `.gif` `.tiff` `.tif`

**视频**：`.mp4` `.avi` `.mkv` `.mov` `.wmv` `.flv` `.webm` `.m4v` `.mpeg` `.mpg` `.ts`

---

## 性能建议

- **阈值**：默认 `0.5` 平衡精度与召回。若误判多，可提高至 `0.6`–`0.7`。
- **视频采样**：`sample_interval` 越大，速度越快，但可能漏检短暂 NSFW 片段。
- **提前退出**：视频检测在找到第一帧 NSFW 内容后立即停止，避免扫描整个视频。
