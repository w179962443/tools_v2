# PaddleOCR v3 兼容性修复记录

本文档记录了将 `tools/ocr_tool.py` 从 PaddleOCR v2 API 迁移到 v3 API，并解决多个运行时崩溃问题的完整过程。

**环境**

- OS: Windows
- paddlepaddle-gpu: 3.3.1
- paddleocr: 3.4.1
- conda env: `iv`
- 推理设备: CPU（无可用 GPU）

---

## 问题一：`TypeError: PaddleOCR.predict() got an unexpected keyword argument 'cls'`

### 原因

PaddleOCR v3 将推理 API 从 `ocr()` 改为 `predict()`，同时移除了 `cls` 参数。

### 修复

| 旧 API（v2）             | 新 API（v3）       |
| ------------------------ | ------------------ |
| `ocr.ocr(img, cls=True)` | `ocr.predict(img)` |

**结果结构也发生了变化：**

```python
# v2 返回：[[[bbox, [text, score]], ...]]
result[0][0][1][0]  # text
result[0][0][1][1]  # score

# v3 返回：[OCRResult({"rec_texts": [...], "rec_scores": [...], "rec_polys": [...]})]
result[0]["rec_texts"]   # 所有行的文字列表
result[0]["rec_scores"]  # 所有行的置信度列表
result[0]["rec_polys"]   # 所有行的多边形坐标列表
```

`extract_text()` 和 `process_image()` 均已按新结构更新。

---

## 问题二：运行时无任何输出（进程 exit code 1）

### 原因

原始代码没有捕获异常，PaddlePaddle 内部的 C++ 级别崩溃直接终止进程，Python 层面不输出任何错误信息。

### 修复

在所有 `predict()` 调用处加了 `try/except`，所有 `print()` 加了 `flush=True`，崩溃信息才得以显示：

```
(Unimplemented) ConvertPirAttribute2RuntimeAttribute not support
[pir::ArrayAttribute<pir::DoubleAttribute>]  (at onednn_instruction.cc:118)
```

---

## 问题三：PIR + OneDNN 原生崩溃（核心问题）

### 根本原因

PaddlePaddle 3.3.1 的 **PIR（新 IR 执行器）** 与 **OneDNN（MKL-DNN）** 在 CPU 上存在不兼容 bug：

1. `static_infer.py` 的 CPU 路径会调用 `config.enable_new_ir(True)`
2. OneDNN 的 `onednn_placement_pass` 将算子分配给 OneDNN 内核
3. 执行时 `onednn_instruction.cc:118` 在尝试转换 `pir::ArrayAttribute<pir::DoubleAttribute>` 时崩溃

当显式传入 `enable_mkldnn=False` 时，会切换到 `run_mode="paddle"` 路径，该路径同样调用 `config.enable_new_ir(True)`，触发同样的崩溃。

### 尝试过但失败的方案

| 方案                                                         | 失败原因                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `os.environ["FLAGS_use_mkldnn"] = "0"`                       | flag 名称错误                                                |
| `os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "False"` | paddle 已在 import 时读取该值，设置太晚                      |
| `ocr_version="PP-OCRv4"`                                     | 使用不同模型，但同样的代码路径，同样崩溃                     |
| `enable_mkldnn=False` 构造参数                               | 切换到 `run_mode="paddle"`，仍调用 `enable_new_ir(True)`     |
| 运行时 patch `NEWIR_BLOCKLIST`                               | OCR 构造时 paddleocr/paddlex 模块已缓存旧值                  |
| Patch `BasePredictor.__init__`                               | 在 `PaddleOCR()` 调用后 patch，predictor 已建好，太晚        |
| 向构造函数传 `pp_option` 参数                                | PaddleOCR v3 不接受 `pp_option` 参数                         |
| `device_type` setter patch                                   | setter 内部重新 import paddle，强制 `FLAGS_enable_pir_api=1` |

### 最终修复方案（双重保险）

#### 方案 A：在 paddle import 之前设置环境变量

```python
# 必须在任何 paddle/paddleocr import 之前设置
os.environ.setdefault("FLAGS_enable_pir_api", "0")
```

在 `ocr_tool.py` 顶部、所有 import 之前设置。`os.environ.setdefault` 保证不覆盖用户已设置的值。

#### 方案 B：将 OCR 模型加入 `NEWIR_BLOCKLIST`（磁盘修改）

编辑 `site-packages/paddlex/inference/utils/new_ir_blocklist.py`，将所有 OCR 相关模型加入黑名单，使 `static_infer.py` 在创建 predictor 时调用 `config.enable_new_ir(False)`：

```python
# 新增条目（在原有列表末尾追加）
"PP-OCRv5_server_det",
"PP-OCRv5_server_rec",
"PP-OCRv5_mobile_det",
"PP-OCRv5_mobile_rec",
"en_PP-OCRv5_mobile_rec",
"PP-OCRv4_mobile_det",
"PP-OCRv4_mobile_rec",
"en_PP-OCRv4_mobile_rec",
"PP-OCRv3_mobile_det",
"PP-OCRv3_mobile_rec",
"PP-LCNet_x1_0_doc_ori",
"UVDoc",
"PP-LCNet_x1_0_textline_ori",
```

文件路径：

```
C:\ProgramData\Anaconda3\envs\iv\lib\site-packages\paddlex\inference\utils\new_ir_blocklist.py
```

---

## 问题四：CUDA DLL 加载失败（paging file 不足）

### 错误信息

```
OSError: [WinError 1455] 页面文件太小，无法完成操作。
Error loading "...\nvidia\cublas\bin\cublas64_12.dll"
```

### 原因

安装的是 `paddlepaddle-gpu` 3.3.1，包含 cuBLAS、cuFFT 等大型 CUDA DLL。即使指定 `device='cpu'`，paddle 在初始化时仍会加载部分 CUDA 库，Windows 默认的虚拟内存（页面文件）不够用。

### 修复方案

**方式一（推荐）：增大 Windows 页面文件**

1. Win+R → 输入 `sysdm.cpl` → 回车
2. 高级 → 性能 → 设置 → 高级 → 虚拟内存 → 更改
3. 取消勾选"自动管理所有驱动器的分页文件大小"
4. 选择 C: → 自定义大小 → 初始值：**8192 MB**，最大值：**16384 MB**
5. 点击"设置" → 确定 → 重启 Windows

**方式二：设置 `CUDA_VISIBLE_DEVICES=-1`**

在任何 paddle import 之前设置，阻止 paddle 尝试使用 GPU 设备（不能完全阻止 DLL 加载，但可减少部分问题）：

```python
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
```

---

## 问题五：大图像内存不足

### 错误信息

```
numpy._core._exceptions._ArrayMemoryError: Unable to allocate 40.5 MiB
for an array with shape (8304, 1280) and data type float32
```

### 原因

测试图像 `v1.jpg` 尺寸为 1280×8304（长图）。默认启用的 UVDoc 文档矫正模型试图对全分辨率图像做处理，分配的张量超出可用内存。

文本检测模型也有类似问题，默认 `text_det_limit_side_len=4000` 时会生成形状 `(1, 3, 4000, 608)` 的张量（约 28 MiB）。

### 修复

在 `PaddleOCR` 构造时传入：

```python
use_doc_unwarping=False,      # 禁用 UVDoc 文档矫正（对普通截图无需此步骤）
text_det_limit_side_len=960,  # 将检测器输入图像的长边限制为 960px
```

---

## 最终代码状态

### `tools/ocr_tool.py` 关键改动

```python
# 文件顶部，所有 import 之前
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# OCRProcessor.ocr property
self._ocr = PaddleOCR(
    use_textline_orientation=True,
    lang=self.lang,
    device="gpu" if self.use_gpu else "cpu",
    use_doc_unwarping=False,
    text_det_limit_side_len=960,
)

# process_image() 结果解析
result = self.ocr.predict(image_path)
if result and result[0]:
    res = result[0]
    for text, score, poly in zip(res["rec_texts"], res["rec_scores"], res["rec_polys"]):
        lines.append({"text": text, "confidence": float(score), "bbox": poly})
```

### `new_ir_blocklist.py` 改动

在 `NEWIR_BLOCKLIST` 列表末尾追加了所有 PP-OCR 系列模型名称（见上文"方案 B"）。

---

## 验证命令

页面文件扩大并重启后，运行：

```powershell
conda activate iv
cd D:\001-vibecoding\tools_v2
python -m tools.ocr_tool --image v1.jpg
```

预期输出示例：

```
File  : v1.jpg
Chars : 342
Lines : 28
Text  :
（识别出的文字内容）
```
