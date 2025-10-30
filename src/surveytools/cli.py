# file: src/surveytools/cli.py
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import List

import pandas as pd

from .cleaning import fix_df_text, fill_client_tokens
from .summarize import TableSummarizer
from .types import SummaryConfig


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="surveytools",
        description="Headless survey summarizer: Excel/CSV → per-question sheets (+optional charts)"
    )
    p.add_argument("--input", type=str, required=True, help="Input Excel/CSV. Supports dbfs:/ and local paths.")
    p.add_argument("--sheet", type=str, help="Excel sheet name/index if input is Excel.")
    p.add_argument("--output", type=str, required=True, help="Output .xlsx path (dbfs:/ or local).")
    p.add_argument("--groups", nargs="*", default=[], help="Group/demographic columns to break out by.")
    p.add_argument("--mode", choices=["percent", "fraction", "count"], default="percent", help="Numeric mode.")
    p.add_argument("--decimals", type=int, default=1, help="Decimal places.")
    p.add_argument("--question-prefix", default="Q", help="Prefix for question columns (default: Q).")
    p.add_argument("--delimiter", default="|", help="Multi-select delimiter (informational).")
    p.add_argument("--select", default="", help="Filter questions: comma list OR single regex.")
    p.add_argument("--charts-dir", type=str, help="Optional directory for overall charts (dbfs:/ or local).")
    p.add_argument("--charts-likert-only", action="store_true", help="Chart Likert/scale questions only.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def _dbfs_to_local(p: str) -> Path:
    return Path("/dbfs") / p[6:] if p.startswith("dbfs:/") else Path(p)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# Excel sheet rules: max 31 chars; no : \ / ? * [ ]
_INVALID_SHEET_CHARS = re.compile(r"[:\\/*?\[\]]")


def _sheet_name(name: str, seen: set[str]) -> str:
    base = _INVALID_SHEET_CHARS.sub("", str(name)) or "Sheet"
    base = base[:31]
    if base not in seen:
        seen.add(base); return base
    i = 1
    while True:
        suf = f"_{i}"
        cand = (base[: 31 - len(suf)]) + suf
        if cand not in seen:
            seen.add(cand); return cand
        i += 1


def _is_likert_like(summarizer: TableSummarizer, question: str) -> bool:
    return summarizer.detect_question_type(question) == "scale"


def main(argv: List[str] | None = None) -> int:
    a = parse_args(argv)
    lvl = logging.WARNING if a.quiet else (logging.DEBUG if a.verbose else logging.INFO)
    logging.basicConfig(level=lvl, format="%(message)s")
    log = logging.getLogger("surveytools")

    in_local = _dbfs_to_local(a.input)
    out_local = _dbfs_to_local(a.output)
    charts_local = _dbfs_to_local(a.charts_dir) if a.charts_dir else None

    if not in_local.exists():
        print(f"Input not found: {a.input}", file=sys.stderr); return 2
    if out_local.suffix.lower() != ".xlsx":
        print("Only .xlsx outputs are supported for --output.", file=sys.stderr); return 2
    if charts_local:
        charts_local.mkdir(parents=True, exist_ok=True)

    # Read input
    try:
        if in_local.suffix.lower() in {".xlsx", ".xls"}:
            sheet = a.sheet if a.sheet is not None else 0
            df = pd.read_excel(in_local, sheet_name=sheet)
        else:
            df = pd.read_csv(in_local)
    except Exception as e:
        print(f"Failed to read input: {e}", file=sys.stderr); return 2

    # Clean/normalize
    df = fill_client_tokens(fix_df_text(df))

    # Config + summarizer
    cfg = SummaryConfig(output_mode=a.mode, decimals=a.decimals, as_percent=(a.mode == "percent"))
    summ = TableSummarizer(df, cfg)

    # Question selection
    questions = [c for c in df.columns if str(c).startswith(a.question_prefix) and c not in summ.group_options]
    if a.select:
        if "," in a.select:
            parts = [s.strip() for s in a.select.split(",") if s.strip()]
            rx = re.compile("|".join(map(re.escape, parts)))
        else:
            rx = re.compile(a.select)
        questions = [q for q in questions if rx.search(str(q))]
    if not questions:
        print("No question columns found after filtering.", file=sys.stderr); return 3

    # Validate groups
    present_groups, missing_groups = [], []
    for g in a.groups:
        (present_groups if g in df.columns else missing_groups).append(g)
    if missing_groups:
        log.warning("Missing group columns (skipped): %s", ", ".join(missing_groups))

    # Write workbook
    _ensure_parent(out_local)
    seen: set[str] = set()
    wrote_any = False

    with pd.ExcelWriter(out_local, engine="openpyxl") as writer:
        for q in questions:
            try:
                res = summ.summarize_overall_and_demo(q, present_groups)
                sheet = _sheet_name(q, seen)
                res.to_excel(writer, sheet_name=sheet, index=False)
                wrote_any = True

                if charts_local:
                    try:
                        from .report_assistant import save_overall_chart
                        if (not a.charts_likert_only) or _is_likert_like(summ, q):
                            png_name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(q))[:80] + ".png"
                            save_overall_chart(df, q, path=str(charts_local / png_name), config=cfg)
                    except Exception as ce:
                        log.debug("Chart skip for %s: %s", q, ce)
            except Exception as e:
                log.warning("Failed to summarize %s: %s", q, e)

    if not wrote_any:
        print("No sheets were written due to errors.", file=sys.stderr); return 4

    print(f"Wrote Excel: {out_local}")
    if charts_local: print(f"Charts dir: {charts_local}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
