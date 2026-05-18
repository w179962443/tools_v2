# 文本辅助处理使用说明

## 提取 CSV 文本列

```bash
python scripts/extract_text_column.py transcript.csv -c text
python scripts/extract_text_column.py transcript.csv -c text -o transcript.txt
```

支持逗号、Tab、分号、竖线等常见分隔符，并尝试 `utf-8-sig`、`utf-8`、`gb18030` 编码。

## OneTab 导出处理

```bash
python scripts/process_onetab.py --input ./onetab_exports
python scripts/process_onetab.py --input ./onetab_exports --output ./onetab_output
```

处理流程：交错合并多个 `.txt` 文件，按 URL 去重，分类为 `github`、`zhihu`、`ai`、`other`，并每 30 条插入空行方便阅读。
