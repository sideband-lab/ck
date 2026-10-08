#!/usr/bin/env python3
"""Report embedding-related Rust line coverage from cargo llvm-cov JSON."""

import argparse
import json
import sys
from pathlib import Path


DEDICATED_FILES = (
    "ck-embed/src/lib.rs",
    "ck-embed/src/mixedbread.rs",
    "ck-embed/src/reranker.rs",
    "ck-embed/src/tokenizer.rs",
    "ck-models/src/lib.rs",
    "ck-engine/src/semantic_v3.rs",
)

SHARED_FILES = (
    "ck-index/src/lib.rs",
    "ck-engine/src/lib.rs",
    "ck-chunk/src/lib.rs",
    "ck-chunk/src/query_chunker.rs",
    "ck-core/src/lib.rs",
    "ck-cli/src/main.rs",
    "ck-cli/src/mcp_server.rs",
    "ck-cli/src/mcp/context.rs",
    "ck-tui/src/app.rs",
    "ck-tui/src/chunks.rs",
    "ck-tui/src/config.rs",
)


def read_coverage(report_path):
    with report_path.open(encoding="utf-8") as report_file:
        report = json.load(report_file)
    return [file for data in report.get("data", []) for file in data.get("files", [])]


def find_file(files, suffix):
    matches = [
        file
        for file in files
        if file.get("filename", "").replace("\\", "/").endswith("/" + suffix)
        or file.get("filename", "").replace("\\", "/") == suffix
    ]
    if len(matches) > 1:
        raise ValueError(f"multiple coverage entries match {suffix}")
    return matches[0] if matches else None


def line_percent(file):
    return file["summary"]["lines"]["percent"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--report", action="store_true", help="print coverage without enforcing thresholds")
    mode.add_argument("--check", action="store_true", help="require at least 80%% in every dedicated file")
    parser.add_argument("coverage_json", type=Path, help="JSON output from cargo llvm-cov")
    args = parser.parse_args()

    try:
        files = read_coverage(args.coverage_json)
        print("Dedicated embedding files (80% minimum):")
        failures = []
        for suffix in DEDICATED_FILES:
            file = find_file(files, suffix)
            if file is None:
                print(f"  {suffix}: missing")
                failures.append(suffix)
                continue
            percent = line_percent(file)
            print(f"  {suffix}: {percent:.1f}%")
            if percent < 80.0:
                failures.append(suffix)

        print("Shared embedding paths (reported separately):")
        for suffix in SHARED_FILES:
            file = find_file(files, suffix)
            if file is None:
                print(f"  {suffix}: missing")
            else:
                print(f"  {suffix}: {line_percent(file):.1f}%")

        if args.check and failures:
            print("Dedicated coverage below 80% or missing: " + ", ".join(failures), file=sys.stderr)
            return 1
        return 0
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"coverage report error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
