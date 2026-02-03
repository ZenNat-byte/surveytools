# CLAUDE.md - AI Assistant Guide for surveytools

## Project Overview

**surveytools** is a Python package for processing survey data with Likert scale mapping, demographic breakouts, and report generation. It provides both CLI and interactive Jupyter/Databricks UI components.

**Version:** 0.1.0
**Python:** >= 3.9
**Total codebase:** ~1,331 lines across 9 core modules

## Quick Start Commands

```bash
# Install in development mode with widgets
pip install -e ".[widgets]"

# Run CLI to process survey data
surveytools --input survey.xlsx --output results.xlsx --groups gender age --charts-dir ./charts

# Run with all options
surveytools --input data.xlsx --output out.xlsx --groups gender education_group --mode percent --decimals 1 --charts-dir ./charts --verbose
```

## Repository Structure

```
/home/user/surveytools/
├── pyproject.toml                    # Build config, dependencies, CLI entry point
├── src/surveytools/
│   ├── __init__.py                   # Package exports
│   ├── types.py                      # SummaryConfig dataclass
│   ├── constants.py                  # Colors, presets, zero aliases
│   ├── cleaning.py                   # Text encoding fixes, token replacement
│   ├── summarize.py                  # Core TableSummarizer engine (403 lines)
│   ├── likert_mapper.py              # Interactive Likert mapping UI (292 lines)
│   ├── bulk_assistant.py             # Batch export assistant (166 lines)
│   ├── report_assistant.py           # Per-question reporting UI (215 lines)
│   └── cli.py                        # Command-line interface (156 lines)
├── artifacts/                        # Test data and example outputs
├── run_surveytools.ipynb             # Comprehensive usage guide notebook
└── bootstrap.ipynb                   # Databricks setup notebook
```

## Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >= 1.5 | Data manipulation, Excel I/O |
| numpy | >= 1.24, < 1.28 | Numerical ops (pinned for Databricks compatibility) |
| openpyxl | >= 3.1 | Excel file creation |
| matplotlib | >= 3.7 | Chart generation |
| python-pptx | >= 0.6.21 | PowerPoint support |
| ftfy | >= 6.1 | Text encoding fixes (optional) |
| ipywidgets | >= 8.0 | Interactive UIs (optional) |

## Architecture & Key Components

### 1. TableSummarizer (`summarize.py`)
The core engine that detects question types and generates summary tables.

**Question Types Detected:**
- `"open-ended"` - Columns ending with `(open-ended)`
- `"scale"` - Likert questions with numeric scores (format: `"Label: 5"`)
- `"choice"` - Multiple choice / multi-select questions

**Key Methods:**
- `detect_question_type(col)` - Returns question type string
- `summarize_overall_and_demo(question, group_cols)` - Generates summary DataFrame
- `_extract_score(x)` - Parses `"Label: 4"` or `"(4)"` format

### 2. Interactive UIs (require ipywidgets)
- **LikertMapper** - Maps textual Likert options to numeric scores
- **ReportAssistant** - Per-question charts and Excel export
- **ExcelBulkAssistant** - Bulk export all questions to single workbook

All UIs degrade gracefully without ipywidgets (headless alternatives available).

### 3. CLI (`cli.py`)
Entry point: `surveytools` command

**Exit Codes:**
- 0: Success
- 2: Input file error
- 3: No questions found matching criteria
- 4: No sheets written to output

## Column Naming Conventions

The codebase expects specific column naming patterns:

| Pattern | Example | Purpose |
|---------|---------|---------|
| `Q*` prefix | `Q1: Satisfaction (general)` | Question columns |
| `(demographics)` suffix | `Age (demographics)` | Demographic grouping columns |
| `(open-ended)` suffix | `Q5: Comments (open-ended)` | Open-ended questions |
| `(general)` suffix | `Q2: Agreement (general)` | General/Likert questions |
| `latest_weight` | — | Weight column for aggregation |
| `client_name` | — | Token replacement source |

**Standard Demographic Columns:**
- `polygon_name`, `age`, `ethnicity`, `education_group`, `gender`, `income_group`

## Data Format Expectations

**Scale Response Format:**
```
"Very satisfied: 5"    # Label: number format
"Neutral: 3"           # Parsed by _extract_score()
```

**Multi-Select Format:**
```
"Option A|Option B|Option C"   # Pipe-delimited
```

**Zero-Value Aliases (mapped to 0):**
- "prefer not to say", "prefer not to answer", "don't know", "n/a", "unsure", etc.

## Configuration Options

### SummaryConfig Dataclass (`types.py`)
```python
@dataclass
class SummaryConfig:
    as_percent: bool = True           # Output as percentages
    decimals: int = 1                 # Decimal places
    output_mode: str = "percent"      # "percent" | "fraction" | "count"
    suppress_below_n: int = 30        # Hide groups with N < this
    asterisk_from_n: int = 30         # Mark with "*" if N in range
    asterisk_to_n: int = 49           # Mark with "*" if N in range
    auto_group_scale: bool = True     # Group 5-pt to Neg/Neu/Pos
    highlight_threshold: float = 5.0  # Highlight threshold
```

### CLI Arguments
```
--input          Required. Excel/CSV path (supports dbfs:/)
--output         Required. Output .xlsx path
--sheet          Excel sheet name/index (default: 0)
--groups         Demographic columns for breakouts
--mode           "percent" | "fraction" | "count" (default: percent)
--decimals       Decimal places (default: 1)
--question-prefix Column prefix for questions (default: "Q")
--select         Filter questions by comma-list or regex
--charts-dir     Directory for PNG chart output
--charts-likert-only  Only chart scale questions
--verbose        Debug logging
--quiet          Suppress logging
```

## Code Patterns

### Graceful Widget Degradation
```python
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML
except ImportError:
    widgets = None
    def display(*args, **kwargs): pass
    def clear_output(*args, **kwargs): pass
    def HTML(x): return x
```

### Weight Handling
- Auto-detects `latest_weight` or `latest_weight_level_1` columns
- Falls back to weight = 1.0 if missing
- Supports per-group weights for polygon_name

### Excel Sheet Naming
- Max 31 characters (Excel limit)
- Invalid characters removed: `:\/*?[]`
- Collisions appended with `_1`, `_2`, etc.

## Development Workflow

### Local Development
```bash
# Clone and install
git clone <repo>
cd surveytools
pip install -e ".[widgets]"

# Run CLI
surveytools --input artifacts/mock_survey.xlsx --output test.xlsx --verbose
```

### Jupyter/Databricks
```python
# Standard imports
import pandas as pd
from surveytools.cleaning import fix_df_text, fill_client_tokens
from surveytools.summarize import TableSummarizer
from surveytools.likert_mapper import LikertMapper
from surveytools.report_assistant import ReportAssistant
from surveytools.bulk_assistant import ExcelBulkAssistant

# Load and clean data
df = pd.read_excel("data.xlsx")
df = fill_client_tokens(fix_df_text(df))

# Interactive UIs
LikertMapper(df)                                    # Map Likert options
ReportAssistant(df, preselected_groups=["gender"])  # Per-question reporting
ExcelBulkAssistant(df)                              # Bulk export
```

### Databricks-Specific
```python
# Find repo root and install
import subprocess
repo_root = "/Workspace/Repos/<user>/surveytools"
subprocess.run(["pip", "install", "-e", f"{repo_root}[widgets]"])

# Restart Python after install
dbutils.library.restartPython()

# DBFS paths supported
surveytools --input dbfs:/data/survey.xlsx --output dbfs:/output/results.xlsx
```

## Testing

**No automated test suite exists.** Testing is done via notebooks:
- `run_surveytools.ipynb` - Comprehensive usage examples and manual testing
- `artifacts/mock_survey.xlsx` - Sample test data

When adding features, test manually using the notebooks.

## Important Technical Notes

1. **NumPy Version**: Pinned to `< 1.28` for Databricks SciPy/thinc compatibility. Force reinstall if needed: `pip install "numpy<1.28" --force-reinstall`

2. **Text Encoding**: Uses `ftfy` library (optional). Gracefully skipped if unavailable.

3. **Multi-Select Delimiter**: Always pipe (`|`) - not configurable

4. **Weighting**: Automatic fallback to 1.0 if weight column missing

5. **Headless Mode**: All export functions work without ipywidgets. Core logic never depends on widget rendering.

## Common Tasks for AI Assistants

### Adding a New Question Type
1. Update `detect_question_type()` in `summarize.py`
2. Add handling in `summarize_overall_and_demo()`
3. Update chart logic in `report_assistant.py` if needed

### Adding CLI Options
1. Add argument in `cli.py` using argparse
2. Pass to `SummaryConfig` or handle directly in main()
3. Update docstrings

### Modifying Summary Output
1. Edit `summarize_overall_and_demo()` in `summarize.py`
2. Check `_fmt()` for output formatting
3. Update `SummaryConfig` if new options needed

### Adding New Demographic Groups
1. Add to `possible_groups` list in `TableSummarizer.__init__`
2. Ensure column ends with `(demographics)` or matches standard name

## File Modification Guidelines

| File | When to Modify |
|------|----------------|
| `types.py` | Adding new configuration options |
| `constants.py` | Adding colors, presets, or aliases |
| `cleaning.py` | Text processing utilities |
| `summarize.py` | Core summarization logic |
| `likert_mapper.py` | Likert mapping UI changes |
| `bulk_assistant.py` | Batch export features |
| `report_assistant.py` | Per-question reporting/charts |
| `cli.py` | Command-line interface changes |
| `pyproject.toml` | Dependencies, version, entry points |

## Known Limitations

- No automated tests (notebook-based testing only)
- Single-threaded processing (suitable for typical survey sizes)
- Hardcoded column naming conventions
- Limited to matplotlib charts (no interactive visualizations)
- python-pptx listed but minimally used
