#!/usr/bin/env python3
"""Call an LLM for each question row in a CSV and judge against the answer.

Usage examples:
    python scripts/judge_learning_question_csv.py input.csv -o judged.csv
    python scripts/judge_learning_question_csv.py input.csv --llm-model qwen-plus
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.video_learning_notes_tool import (  # noqa: E402
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_LLM_MODEL,
    get_chat_completion,
    resolve_api_key,
)

PROMPT_TEMPLATE = """我希望你你能根据我提供的json格式的题目，给出你的答案，注意，只需要输出答案

例1:
题干是：
{"question": "大模型可以自动访问用户的内部系统，无需额外软件集成。"}

你的答案：
false

例2：
题干是：
{"question": "AI的发展与战争需求密切相关，弹道计算的需求直接催生了ENIAC的诞生。"}

你的答案：
true

例3：
题干是：
{"question":"用户上传一篇长篇报告后，要求模型\"帮我提炼出三条核心观点\"。这一操作最主要体现了大语言模型的哪项功能？","options":[{"key":"A","value":"总结提炼"},{"key":"B","value":"深度研究"},{"key":"C","value":"联网搜索"},{"key":"D","value":"翻译与润色"}]}

你的答案：
A

现在按照我的要求，回答我的问题：

题干是：
{{content}}

你的答案：
"""

REQUIRED_COLUMNS = ("id", "content", "answer")
OUTPUT_COLUMNS = ("id", "content", "answer", "raw_llm_answer", "judge")
LETTER_ORDER = tuple(string.ascii_uppercase)
LETTER_ANSWERS = set(LETTER_ORDER)
BOOLEAN_ANSWERS = {"true", "false"}
TOKEN_REGEX = re.compile(r"(?i)(?<![A-Za-z])(true|false|[A-Z])(?![A-Za-z])")
PREFIX_REGEX = re.compile(r"(?i)(?:答案|answer)\s*[：:\-]?\s*(true|false|[A-Z])")
OPTION_TOKEN_REGEX = re.compile(r"(?i)(?<![A-Za-z])([A-Z])(?![A-Za-z])")


def build_prompt(content: str) -> str:
    return PROMPT_TEMPLATE.replace("{{content}}", content)


def _normalize_token(token: str) -> str | None:
    stripped = token.strip()
    stripped = stripped.strip("`")
    stripped = stripped.strip().strip('"').strip("'").strip("“”‘’")
    stripped = stripped.strip("。.,，:：;；!?！？()[]{}<>")

    lowered = stripped.lower()
    if lowered in BOOLEAN_ANSWERS:
        return lowered

    uppered = stripped.upper()
    if uppered in LETTER_ANSWERS and len(uppered) == 1:
        return uppered

    return None


def _normalize_multi_answer(tokens: list[str]) -> tuple[str, ...]:
    unique_tokens = {token for token in tokens if token in LETTER_ANSWERS}
    return tuple(letter for letter in LETTER_ORDER if letter in unique_tokens)


def _strip_answer_prefix(text: str) -> str:
    return re.sub(r"(?i)^\s*(?:答案|answer)\s*[：:\-]?\s*", "", text).strip()


def _extract_multi_choice_answer(text: str) -> tuple[str, ...] | None:
    candidates = [_strip_answer_prefix(text)]
    candidates.extend(_strip_answer_prefix(line) for line in text.splitlines())

    for candidate in candidates:
        if not candidate:
            continue

        compact_candidate = re.sub(r"[\s,，、/;；|]+", "", candidate)
        compact_candidate = (
            compact_candidate.replace("和", "").replace("与", "").replace("及", "")
        )
        if re.fullmatch(r"(?i)[A-Z]{2,26}", compact_candidate):
            return _normalize_multi_answer([char.upper() for char in compact_candidate])

        option_tokens = [
            match.group(1).upper() for match in OPTION_TOKEN_REGEX.finditer(candidate)
        ]
        normalized_multi = _normalize_multi_answer(option_tokens)
        if len(normalized_multi) > 1:
            return normalized_multi

    return None


def _extract_single_choice_answer(text: str) -> str | None:
    direct = _normalize_token(text)
    if direct is not None:
        return direct

    for line in text.splitlines():
        normalized = _normalize_token(line)
        if normalized is not None:
            return normalized

    prefixed_match = PREFIX_REGEX.search(text)
    if prefixed_match:
        normalized = _normalize_token(prefixed_match.group(1))
        if normalized is not None:
            return normalized

    token_match = TOKEN_REGEX.search(text)
    if token_match:
        normalized = _normalize_token(token_match.group(1))
        if normalized is not None:
            return normalized

    return None


def normalize_expected_answer(answer_text: str) -> str | tuple[str, ...]:
    try:
        payload = json.loads(answer_text)
    except json.JSONDecodeError as exception:
        raise ValueError(f"Invalid answer JSON: {answer_text}") from exception

    if "correct" not in payload:
        raise ValueError(f"Answer JSON missing 'correct': {answer_text}")

    correct = payload["correct"]
    if isinstance(correct, bool):
        return "true" if correct else "false"

    if isinstance(correct, str):
        normalized = _extract_single_choice_answer(correct)
        if normalized is not None:
            return normalized

    if isinstance(correct, list):
        normalized_tokens: list[str] = []
        for item in correct:
            if not isinstance(item, str):
                raise ValueError(f"Unsupported answer format: {answer_text}")
            normalized = _normalize_token(item)
            if normalized is None or normalized not in LETTER_ANSWERS:
                raise ValueError(f"Unsupported answer format: {answer_text}")
            normalized_tokens.append(normalized)

        if not normalized_tokens:
            raise ValueError(f"Unsupported answer format: {answer_text}")
        return _normalize_multi_answer(normalized_tokens)

    raise ValueError(f"Unsupported answer format: {answer_text}")


def normalize_model_answer(raw_answer: str) -> str | tuple[str, ...] | None:
    direct = _normalize_token(raw_answer)
    if direct is not None:
        return direct

    for line in raw_answer.splitlines():
        normalized = _normalize_token(line)
        if normalized is not None:
            return normalized

    prefixed_match = PREFIX_REGEX.search(raw_answer)
    if prefixed_match:
        normalized = _normalize_token(prefixed_match.group(1))
        if normalized is not None:
            return normalized

    if re.search(r"(?i)true|false", raw_answer):
        token_match = TOKEN_REGEX.search(raw_answer)
        if token_match:
            normalized = _normalize_token(token_match.group(1))
            if normalized is not None:
                return normalized

    normalized_multi = _extract_multi_choice_answer(raw_answer)
    if normalized_multi is not None:
        return normalized_multi

    token_match = TOKEN_REGEX.search(raw_answer)
    if token_match:
        normalized = _normalize_token(token_match.group(1))
        if normalized is not None:
            return normalized

    return None


def read_csv_rows(
    csv_path: Path, required_columns: tuple[str, ...]
) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in required_columns if column not in fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        return [dict(row) for row in reader]


def read_rows(input_csv: Path) -> list[dict[str, str]]:
    return read_csv_rows(input_csv, REQUIRED_COLUMNS)


def build_output_row(input_row: dict[str, str]) -> dict[str, str]:
    return {
        "id": str(input_row.get("id", "")).strip(),
        "content": str(input_row.get("content", "")),
        "answer": str(input_row.get("answer", "")),
        "raw_llm_answer": "",
        "judge": "",
    }


def load_existing_output_rows(output_csv: Path) -> list[dict[str, str]]:
    rows = read_csv_rows(output_csv, REQUIRED_COLUMNS)
    normalized_rows: list[dict[str, str]] = []
    for row in rows:
        normalized_row = build_output_row(row)
        normalized_row["raw_llm_answer"] = str(row.get("raw_llm_answer", ""))
        normalized_row["judge"] = str(row.get("judge", ""))
        normalized_rows.append(normalized_row)
    return normalized_rows


def merge_output_rows(
    input_rows: list[dict[str, str]],
    existing_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    existing_by_id = {
        str(row.get("id", "")).strip(): row
        for row in existing_rows
        if str(row.get("id", "")).strip()
    }
    merged_rows: list[dict[str, str]] = []

    for index, input_row in enumerate(input_rows):
        merged_row = build_output_row(input_row)
        row_id = merged_row["id"]
        existing_row = existing_by_id.get(row_id)

        if existing_row is None and index < len(existing_rows):
            candidate_row = existing_rows[index]
            if str(candidate_row.get("id", "")).strip() == row_id:
                existing_row = candidate_row

        if existing_row is None:
            merged_rows.append(merged_row)
            continue

        existing_content = str(existing_row.get("content", ""))
        existing_answer = str(existing_row.get("answer", ""))
        existing_raw_answer = str(existing_row.get("raw_llm_answer", ""))
        existing_judge = str(existing_row.get("judge", ""))

        if existing_content == merged_row["content"]:
            merged_row["raw_llm_answer"] = existing_raw_answer
            if existing_answer == merged_row["answer"]:
                merged_row["judge"] = existing_judge

        merged_rows.append(merged_row)

    return merged_rows


def load_or_initialize_output_rows(
    input_rows: list[dict[str, str]],
    output_csv: Path,
) -> list[dict[str, str]]:
    if output_csv.is_file():
        existing_rows = load_existing_output_rows(output_csv)
        return merge_output_rows(input_rows, existing_rows)
    return [build_output_row(row) for row in input_rows]


def is_multi_select_question(row: dict[str, str]) -> bool:
    try:
        payload = json.loads(str(row.get("answer", "")))
        correct = payload.get("correct")
        return isinstance(correct, list) and len(correct) > 1
    except (json.JSONDecodeError, AttributeError):
        return False


def is_row_completed(row: dict[str, str]) -> bool:
    raw_llm_answer = str(row.get("raw_llm_answer", "")).strip()
    judge = str(row.get("judge", "")).strip()
    return bool(raw_llm_answer and judge)


def judge_row_from_raw_answer(row: dict[str, str]) -> str:
    expected_answer = normalize_expected_answer(str(row.get("answer", "")))
    normalized_llm_answer = normalize_model_answer(str(row.get("raw_llm_answer", "")))
    return "符合" if normalized_llm_answer == expected_answer else "不符"


def needs_llm_call(row: dict[str, str]) -> bool:
    return not is_row_completed(row) and not str(row.get("raw_llm_answer", "")).strip()


def write_rows(output_csv: Path, rows: list[dict[str, str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


async def judge_rows(
    rows: list[dict[str, str]],
    output_csv: Path,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    stream_llm: bool,
) -> list[dict[str, str]]:
    total = len(rows)

    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("id", "")).strip()
        if is_multi_select_question(row):
            print(
                f"[{index}/{total}] Skipping id={row_id or '<empty>'} (multi-select, skipped)",
                flush=True,
            )
            continue

        if is_row_completed(row):
            print(
                f"[{index}/{total}] Skipping id={row_id or '<empty>'} (already completed)",
                flush=True,
            )
            continue

        if str(row.get("raw_llm_answer", "")).strip():
            print(
                f"[{index}/{total}] Finalizing id={row_id or '<empty>'} from existing raw_llm_answer",
                flush=True,
            )
            row["judge"] = judge_row_from_raw_answer(row)
            write_rows(output_csv, rows)
            continue

        print(f"[{index}/{total}] Processing id={row_id or '<empty>'}", flush=True)
        content = str(row.get("content", ""))
        raw_llm_answer = await get_chat_completion(
            prompt=build_prompt(content),
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            stream_to_stdout=stream_llm,
        )
        row["raw_llm_answer"] = raw_llm_answer
        row["judge"] = judge_row_from_raw_answer(row)
        write_rows(output_csv, rows)

    return rows


def resolve_output_path(input_csv: Path, output: str | None) -> Path:
    if output:
        return Path(output)
    return input_csv.with_name(f"{input_csv.stem}_judged.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call an LLM for each CSV row and judge the answer against the answer JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/judge_learning_question_csv.py data/wy_learning_question.csv
  python scripts/judge_learning_question_csv.py data/wy_learning_question.csv -o data/wy_learning_question_judged.csv
  python scripts/judge_learning_question_csv.py data/wy_learning_question.csv --llm-model qwen-plus --stream-llm

The LLM call reads DASHSCOPE_API_KEY or OPENAI_API_KEY unless --api-key is passed.
        """,
    )
    parser.add_argument(
        "input_csv", help="Input CSV path containing id/content/answer columns"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV path (default: <input>_judged.csv)",
    )
    parser.add_argument(
        "--llm-model", default=DEFAULT_LLM_MODEL, help="Chat model name"
    )
    parser.add_argument(
        "--api-key", default=DEFAULT_API_KEY, help="DashScope/OpenAI-compatible API key"
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="LLM temperature (default: 0.0)"
    )
    parser.add_argument(
        "--stream-llm",
        action="store_true",
        help="Stream the model answer to stdout while each row is processed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = resolve_output_path(input_csv, args.output)

    if not input_csv.is_file():
        print(f"Error: input CSV not found: {input_csv}", file=sys.stderr)
        return 1

    try:
        input_rows = read_rows(input_csv)
        judged_rows = load_or_initialize_output_rows(input_rows, output_csv)
        write_rows(output_csv, judged_rows)

        api_key = ""
        if any(needs_llm_call(row) for row in judged_rows):
            api_key = resolve_api_key(args.api_key)

        judged_rows = asyncio.run(
            judge_rows(
                rows=judged_rows,
                output_csv=output_csv,
                model=args.llm_model,
                api_key=api_key,
                base_url=args.base_url,
                temperature=args.temperature,
                stream_llm=args.stream_llm,
            )
        )
        write_rows(output_csv, judged_rows)
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    except Exception as exception:
        print(f"Error: {exception}", file=sys.stderr)
        return 1

    print(f"Done. Wrote {len(judged_rows)} rows to: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
