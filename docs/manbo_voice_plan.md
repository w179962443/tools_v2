# 曼波中文语音生成计划

新增一个命令行能力：输入中文文本，先用中文 TTS 生成“底音”音频，再把这段音频交给用户本地提供的曼波 RVC 模型做音色转换，最终输出 WAV。当前没有 RVC 模型，因此计划只覆盖推理集成、模型放置约定和使用说明；不包含模型训练、模型下载或分发。

## Steps

1. 前置约定：RVC 不是直接读文字的模型，它需要一段已有语音作为输入；底音 TTS 就是把中文先念出来的普通语音。等获得合法可用的曼波 RVC 模型后，将模型放在 `data/rvc_models/manbo/`，建议文件名为 `manbo.pth` 和 `manbo.index`。该目录位于现有 `/data` 下，已被 `.gitignore` 忽略，适合存放大模型权重。
2. 后端选择：MVP 使用 `edge-tts` 生成中文底音；RVC 转换使用外部命令适配器，通过 `--rvc-command` 或 `MANBO_RVC_COMMAND` 传入已安装 RVC 推理命令模板，避免把不稳定且依赖重的 RVC WebUI/Applio 代码直接塞进本仓库。
3. 新建核心工具模块 `tools/tts_rvc_tool.py`，提供可复用函数 `synthesize_chinese_with_rvc(...) -> dict[str, Any]`。参数包含 `text`、`output_file`、`model_file`、`index_file`、`tts_voice`、`rvc_command`、`pitch`、`f0_method`、`keep_intermediate`、`verbose`。
4. 在核心模块里拆出三个内部步骤：文本校验与输出路径准备；`edge-tts` 生成临时底音音频；调用 RVC 后端生成最终 WAV。每一步都做清晰错误提示，例如模型文件不存在、没有传 RVC 命令、RVC 命令退出失败、输出文件未生成。
5. 增加 `--tts-only` 模式，用来在还没有 RVC 模型时先验证“中文文本转普通语音”链路。这不满足最终曼波音色，但能帮助先确认安装、网络和 FFmpeg 都正常。
6. 新建 CLI 包装脚本 `scripts/text_to_manbo_voice.py`，沿用现有脚本的 `argparse.RawDescriptionHelpFormatter`、示例 epilog、`0/1/130` 返回码模式。CLI 支持直接传一句中文，也支持 `--input-text-file`；默认输出到 `data/voice_outputs/manbo_YYYYMMDD_HHMMSS.wav`。
7. 在 `setup.py` 添加 `voice_requires`，包含 `edge-tts>=6.1.0` 和 `ffmpeg-python>=0.2.0`；新增 extras `voice`，并把它加入 `all`。新增 console script：`text-to-manbo-voice=scripts.text_to_manbo_voice:main`。
8. 更新 `requirements.txt`，新增“Text-to-speech + RVC orchestration”小节，加入 `edge-tts`；RVC 推理依赖不强行写入主 requirements，而是在文档里说明由用户单独安装/配置外部 RVC 后端。
9. 新建使用文档 `docs/tts_rvc_usage.md`，内容包括：底音 TTS 是什么；为什么需要 RVC 模型；推荐模型目录；`--tts-only` 示例；完整 RVC 示例；`--rvc-command` 模板占位符；常见问题。
10. 更新 `docs/installation.md`，在可选能力依赖中加入 `pip install -e .[voice]`，并提示需要系统 FFmpeg；若使用外部 RVC 后端，按该后端文档单独配置。
11. 更新 `README.md`，在能力列表和快速开始中加入“中文文本转 RVC 音色语音”，示例命令只展示本地模型路径，不提供模型下载链接。
12. 可选后续阶段：如果之后明确选择某个 RVC 实现，比如 Applio、RVC WebUI CLI 或稳定 Python 包，再增加一个明确的 backend adapter，并把 `--backend command` 扩展为 `--backend command/native`。

## Relevant Files

- `tools/tts_rvc_tool.py`：核心工具模块。
- `scripts/text_to_manbo_voice.py`：命令行入口。
- `setup.py`：新增 `voice` extra 和 console script。
- `requirements.txt`：新增 TTS 编排依赖。
- `docs/tts_rvc_usage.md`：使用说明。
- `docs/installation.md`：安装说明。
- `README.md`：功能列表和快速开始。
- `.gitignore`：当前已有 `/data`，模型放在 `data/rvc_models/` 时无需改动。

## Verification

1. 静态检查：实现后用编辑器诊断检查新增/修改的 Python 文件；不主动运行安装、构建或终端测试，除非明确要求。
2. 无模型验证：用 `--tts-only` 生成一段中文普通语音，确认文本输入、输出路径、`edge-tts` 和 FFmpeg 链路能跑通。
3. 有模型验证：将合法可用的 `manbo.pth` 和可选 `manbo.index` 放到 `data/rvc_models/manbo/`，运行短句命令，确认最终 WAV 存在、可播放，且中间文件按 `--keep-intermediate` 配置保留或清理。
4. 错误路径验证：分别测试缺少模型、缺少 RVC 命令、RVC 输出不存在时的报错，确保提示能指向下一步。

## Decisions

- 包含：中文文本输入、普通中文 TTS 底音、外部 RVC 推理编排、命令行入口、Python API、文档与依赖声明。
- 不包含：训练曼波 RVC 模型、下载/分发角色或真人声音模型、绕过模型授权限制、GUI/Web 界面。
- 推荐默认：先用 `edge-tts` 做底音，先用外部 RVC 命令模板做转换；这比直接绑定某个 RVC 包更适合当前工具仓库，也更容易在 Windows conda 环境里排错。

## Further Considerations

1. 真正的“曼波音色”前提是拥有合法可用的 RVC 模型；没有模型时只能做到普通中文 TTS 或把代码能力搭好。
2. RVC 模型通常会映射到具体声线或表演素材，使用前需要确认来源、授权和使用场景。
3. 如果后续目标是“一条命令从零安装并运行 RVC”，需要另开计划选择具体 RVC 实现，因为那会显著增加依赖、环境和 GPU 兼容性范围。