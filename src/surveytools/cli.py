cli_code = r'''
from __future__ import annotations
import argparse, re, sys, logging
from pathlib import Path
import pandas as pd
from .cleaning import fix_df_text, fill_client_tokens
from .summarize import TableSummarizer
from .types import SummaryConfig

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="surveytools", description="Headless survey summarizer")
    p.add_argument("--input", type=Path, required=True, help="Input Excel/CSV (supports dbfs:/...)")
    p.add_argument("--sheet", type=str, help="Excel sheet (if applicable)")
    p.add_argument("--output", type=Path, required=True, help="Output .xlsx path")
    p.add_argument("--mode", choices=["percent","fraction","count"], default="percent")
    p.add_argument("--decimals", type=int, default=1)
    p.add_argument("--groups", nargs="*", default=[], help="Group columns")
    p.add_argument("--question-prefix", default="Q", help="Question column prefix")
    p.add_argument("--delimiter", default="|", help="Multi-select delimiter")
    p.add_argument("--select", default="", help="Comma/regex filter for questions")
    p.add_argument("--charts-dir", type=Path, help="Optional dir to save overall charts")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)

def _dbfs_to_local(p: Path) -> Path:
    s = str(p)
    return Path(s.replace("dbfs:/","/dbfs/")) if s.startswith("dbfs:/") else p

def _sheet_name(name: str, seen: set[str]) -> str:
    safe = re.sub(r"[\\/*?:\\[\\]]", "", name)[:31]
    base, i = safe, 1
    while safe in seen:
        safe = (base[:28] + f"_{i}")[:31]; i += 1
    seen.add(safe); return safe

def main(argv=None) -> int:
    a = parse_args(argv)
    lvl = logging.WARNING if a.quiet else (logging.DEBUG if a.verbose else logging.INFO)
    logging.basicConfig(level=lvl, format="%(message)s")
    log = logging.getLogger("surveytools")

    in_path  = _dbfs_to_local(a.input)
    out_path = _dbfs_to_local(a.output)
    charts   = _dbfs_to_local(a.charts_dir) if a.charts_dir else None

    if not in_path.exists():
        print(f"Input not found: {a.input}", file=sys.stderr); return 2
    if out_path.suffix.lower() != ".xlsx":
        print("Only .xlsx outputs are supported.", file=sys.stderr); return 2
    if charts: charts.mkdir(parents=True, exist_ok=True)

    # Read
    if in_path.suffix.lower() in {".xlsx",".xls"}:
        df = pd.read_excel(in_path, sheet_name=a.sheet if a.sheet else 0)
    else:
        df = pd.read_csv(in_path)

    # Clean
    df = fill_client_tokens(fix_df_text(df))

    # Config + summarizer
    cfg = SummaryConfig(
        output_mode=a.mode,
        decimals=a.decimals,
        as_percent=(a.mode=="percent"),
        multiselect_delimiter=a.delimiter
    )
    summ = TableSummarizer(df, cfg)

    # Question selection
    questions = [c for c in df.columns if str(c).startswith(a.question_prefix) and c not in summ.group_options]
    if a.select:
        parts = [s.strip() for s in a.select.split(",") if s.strip()]
        rx = re.compile("|".join(map(re.escape, parts))) if "," in a.select else re.compile(a.select)
        questions = [q for q in questions if rx.search(str(q))]

    # Validate groups
    present_groups, missing = [], []
    for g in a.groups:
        (present_groups if g in df.columns else missing).append(g)
    if missing:
        log.warning("Missing group columns: %s (will proceed without them)", ", ".join(missing))

    # Write
    seen = set()
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        for q in questions:
            try:
                res = summ.summarize_overall_and_demo(q, present_groups)
                res.to_excel(w, sheet_name=_sheet_name(str(q), seen), index=False)
                if charts:
                    try:
                        from .report_assistant import save_overall_chart
                        png = charts / (re.sub(r"[^A-Za-z0-9_.-]", "_", str(q))[:80] + ".png")
                        save_overall_chart(df, q, str(png))
                    except Exception as e:
                        log.debug("Chart skip for %s: %s", q, e)
            except Exception as e:
                log.warning("Failed to summarize %s: %s", q, e)

    print(f"Wrote Excel: {out_path}")
    if charts: print(f"Charts dir: {charts}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''
path = "/Workspace/teams/survey_team/Core_automation/Correct code/surveytools/src/surveytools/cli.py"
with open(path, "w", encoding="utf-8") as f:
    f.write(cli_code)
print("✅ Wrote", path)
