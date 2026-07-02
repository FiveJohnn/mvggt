from __future__ import annotations

import argparse
import os
from pathlib import Path

from agentic_grounding.io_utils import read_json, write_json
from agentic_grounding.query.compiler import QueryCompiler
from agentic_grounding.vlm.openai_compatible import OpenAICompatibleVLMClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile referring expressions into cached JSON graphs")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query-key", default="description")
    parser.add_argument("--id-key", default="query_id")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="VLM_API_KEY")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    records = read_json(args.input)
    if not isinstance(records, list):
        raise TypeError("Query compiler input must be a JSON list of annotation records")
    output_path = Path(args.output)
    completed: dict[str, dict] = {}
    if output_path.exists() and not args.force:
        previous = read_json(output_path)
        if isinstance(previous, list):
            for item in previous:
                key = str(item.get(args.id_key, item.get("ann_id", "")))
                if key and "compiled_query" in item:
                    completed[key] = item
    client = OpenAICompatibleVLMClient(
        base_url=args.base_url,
        model=args.model,
        api_key=os.getenv(args.api_key_env, ""),
    )
    compiler = QueryCompiler(client)
    output = []
    for index, record in enumerate(records):
        record_id = str(record.get(args.id_key, record.get("ann_id", "")))
        if record_id in completed:
            output.append(completed[record_id])
            continue
        if "compiled_query" in record and not args.force:
            output.append(record)
            continue
        try:
            compiled = compiler.compile_record(record, args.query_key, args.id_key)
        except Exception as exc:
            if args.fail_fast:
                raise
            compiled = dict(record)
            compiled["compiler_error"] = f"{type(exc).__name__}: {exc}"
        output.append(compiled)
        if (index + 1) % 20 == 0:
            write_json(args.output, output)
            print(f"compiled {index + 1}/{len(records)}", flush=True)
    write_json(args.output, output)


if __name__ == "__main__":
    main()
