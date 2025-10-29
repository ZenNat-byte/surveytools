# src/surveytools/bulk_assistant.py
from __future__ import annotations
import re
from typing import List, Optional
import pandas as pd

from .summarize import TableSummarizer
from .types import SummaryConfig

try:
    import ipywidgets as w
    from IPython.display import display, clear_output
except Exception:
    # Databricks Serverless often doesn't have working widgets; fall back to headless
    w = None
    def display(*_a, **_k):  # type: ignore
        pass
    def clear_output(*_a, **_k):  # type: ignore
        pass


def bulk_export_headless(
    df: pd.DataFrame,
    groups: List[str],
    filename: str = "All_Questions_Summary.xlsx",
    config: Optional[SummaryConfig] = None,
) -> str:
    """
    Headless bulk export: summarize every question and write to a single Excel file.
    Works without ipywidgets (e.g., on Serverless).
    """
    cfg = config or SummaryConfig()
    summ = TableSummarizer(df, cfg)

    if not groups:
        raise ValueError("groups cannot be empty; pick at least one demographic column.")

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        for i, q in enumerate(summ.question_options, start=1):
            try:
                res = summ.summarize_overall_and_demo(q, groups)
                sheet = re.sub(r"[\\/*?:\[\]]", "", q)[:31]
                res.to_excel(writer, index=False, sheet_name=sheet)
            except Exception as e:
                pd.DataFrame({"Error": [str(e)]}).to_excel(writer, sheet_name=f"Error_{i}", index=False)
    return filename


class ConfigPanel:
    """Small widget panel for configuring SummaryConfig (only used when widgets are available)."""
    def __init__(self):
        if w is None:
            # dummy attributes for headless mode
            self.output_mode_w = None
            self.decimals_w = None
            self.suppress_w = None
            self.ast_from_w = None
            self.ast_to_w = None
            self.auto_group_cb = None
            self.highlight_w = None
            return

        self.output_mode_w = w.ToggleButtons(
            options=[("Percent","percent"),("Fraction","fraction"),("Weighted count","count")],
            value="percent",
            description="Output:"
        )
        self.decimals_w = w.IntSlider(value=1, min=0, max=4, step=1, description="Decimals")
        self.suppress_w = w.IntText(value=30, description="Suppress N <")
        self.ast_from_w = w.IntText(value=30, description="* from N:")
        self.ast_to_w = w.IntText(value=49, description="* to N:")
        self.auto_group_cb = w.Checkbox(value=True, description="Auto-group 5-pt (Neg/Neu/Pos)")
        self.highlight_w = w.FloatSlider(value=5.0, min=0.0, max=10.0, step=0.1, description="Highlight ≥")

        def _sync(_=None):
            mode = self.output_mode_w.value
            if mode == "percent":
                self.highlight_w.max, self.highlight_w.step = 10.0, 0.1
            elif mode == "fraction":
                self.highlight_w.max, self.highlight_w.step = 0.10, 0.01
            else:
                self.highlight_w.max, self.highlight_w.step = 10.0, 1.0

        _sync()
        self.output_mode_w.observe(_sync, names="value")

    def to_config(self) -> SummaryConfig:
        if w is None:
            return SummaryConfig()
        return SummaryConfig(
            as_percent=(self.output_mode_w.value == "percent"),
            output_mode=self.output_mode_w.value,
            decimals=self.decimals_w.value,
            suppress_below_n=self.suppress_w.value,
            asterisk_from_n=self.ast_from_w.value,
            asterisk_to_n=self.ast_to_w.value,
            auto_group_scale=self.auto_group_cb.value,
            highlight_threshold=self.highlight_w.value,
        )


class ExcelBulkAssistant:
    """
    A light UI wrapper (when ipywidgets is available) around TableSummarizer to
    export all question summaries to a single Excel workbook.
    Falls back to headless mode automatically on environments without widgets.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.config_panel = ConfigPanel()
        self.summarizer = TableSummarizer(df, self.config_panel.to_config())
        self.group_options = self.summarizer.group_options
        self.question_options = self.summarizer.question_options

        if w is not None:
            self._build()

    # ---------- UI (widgets available) ----------
    def _build(self):
        header = w.HTML("<h3 style='margin:6px 0'>Survey Excel Bulk Assistant</h3>")
        self.file_input = w.Text(value="All_Questions_Summary.xlsx", description="File:")
        self.group_checks = [w.Checkbox(value=False, description=g) for g in self.group_options]
        self.run_btn = w.Button(description="Export", button_style="primary")
        self.out = w.Output()

        # Layout
        controls = w.VBox(
            [
                self.config_panel.output_mode_w,
                self.config_panel.decimals_w,
                self.config_panel.suppress_w,
                w.HBox([self.config_panel.ast_from_w, self.config_panel.ast_to_w]),
                self.config_panel.auto_group_cb,
                self.config_panel.highlight_w,
                w.HTML("<b>Groups:</b>"),
                w.VBox(self.group_checks),
                self.file_input,
                self.run_btn,
                self.out,
            ]
        )
        display(header, controls)
        self.run_btn.on_click(self._on_export_clicked)

    def _selected_groups(self) -> List[str]:
        if w is None:
            return []
        return [cb.description for cb in self.group_checks if cb.value]

    def _on_export_clicked(self, _):
        if w is None:
            return
        with self.out:
            clear_output()
            groups = self._selected_groups()
            if not groups:
                print("❌ Pick at least one group (demographic column).")
                return
            # Refresh config
            self.summarizer.config = self.config_panel.to_config()
            filename = self.file_input.value.strip() or "All_Questions_Summary.xlsx"
            try:
                path = bulk_export_headless(self.df, groups, filename, self.summarizer.config)
                print(f"✅ Saved {path}")
            except Exception as e:
                print("❌ Error:", e)
