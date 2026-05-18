# 安装说明

## 环境要求

- Python 3.8+（推荐使用 conda `whisper2` 环境）
- Windows / Linux / macOS

## 1. 激活 conda 环境

```bash
conda activate whisper2
```

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

> **提示**：PaddleOCR、NudeNet、Whisper 首次运行时会自动下载模型文件，请确保网络连接。
> Whisper 需要系统可用的 FFmpeg。

### 可选：GPU 加速

若需 GPU 支持，请将 `paddlepaddle` 替换为 `paddlepaddle-gpu`：

```bash
pip uninstall paddlepaddle
pip install paddlepaddle-gpu
```

## 3. 安装为可编辑包（可选）

在项目根目录执行，可在任意位置 `import tools`：

```bash
pip install -e .
```

若不安装，脚本已在运行时自动将项目根目录加入 `sys.path`，无需此步骤即可正常使用 `scripts/` 下的脚本。

### 可选能力依赖

如果只想安装某类迁移能力，也可以用 extras：

```bash
pip install -e .[audio]
pip install -e .[realtime]
pip install -e .[subtitle]
pip install -e .[diarize]
```

说话人识别需要 HuggingFace Token，并需要在 HuggingFace 上接受 pyannote 模型协议。

## 4. VSCode 集成

`.vscode/settings.json` 已配置默认使用 conda `whisper2` 环境。在 VSCode 中打开项目根目录，Python 解释器将自动指向 `C:\ProgramData\Anaconda3\envs\whisper2\python.exe`。
