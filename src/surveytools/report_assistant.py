# src/surveytools/report_assistant.py
from __future__ import annotations
from typing import List, Optional, Tuple
import re

import pandas as pd
import matplotlib.pyplot as plt

from .summarize import TableSummarizer
from .types import SummaryConfig

# Widgets are optional; Serverless often runs headless
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
except Exception:
    widgets = None
    def display(*_a, **_k):  # type: ignore
        pass
    def clear_output(*_a, **_k):  # type: ignore
        pass


# ------------------------------ Headless helpers ------------------------------

def export_single_question(
    df: pd.DataFrame,
    question: str,
    groups: List[str],
    *,
    filename: str = "Summary.xlsx",
    config: Optional[SummaryConfig] = None,
) -> Tuple[str, pd.DataFrame]:
    """
    Headless utility: summarize one question (overall + chosen groups) and
    append/overwrite a sheet in an Excel file. Returns (path, table).
    """
    summ = TableSummarizer(df, config or SummaryConfig())
    table = summ.summarize_overall_and_demo(question, groups)
    safe_sheet = re.sub(r"[\\/*?:\[\]]", "", question)[:31]
    summ.summary_table = table
    summ.export_summary_table(filename, safe_sheet)
    return filename, table


def save_overall_chart(
    df: pd.DataFrame,
    question: str,
    *,
    path: str = "overall.png",
    config: Optional[SummaryConfig] = None,
) -> str:
    """
    Headless utility: save a simple overall chart for a question.
    - For Likert/scale: horizontal stacked bar by score.
    - For choice: top N category bar chart.
    Uses matplotlib with default colors (no style set).
    """
    s = TableSummarizer(df, config or SummaryConfig())

    qtype = s.detect_question_type(question)
    if qtype == "scale":
        wcol = s._pick_weight_col(overall=True)
        temp = s._reshape_questions_to_long(df, question,
                                            group_col=s.group_options[0] if s.group_options else df.columns[0],
                                            weight_col=wcol)
        temp["score"] = temp["selected_choice"].apply(s._extract_score)
        temp = temp.dropna(subset=["score", wcol])
        grouped = temp.groupby("score")[wcol].sum()
        total = grouped.sum()
        keys = sorted(grouped.index, reverse=True)
        sizes = [(grouped.get(k, 0) / total) if total > 0 else 0 for k in keys]

        fig, ax = plt.subplots(figsize=(8, 2.2))
        left = 0.0
        for p in sizes:
            ax.barh(0, p, left=left, height=1.0)   # no explicit colors/styles
            if p > 0.04:
                ax.text(left + p / 2, 0, f"{int(round(p*100))}%", ha="center", va="center")
            left += p
        ax.set_xlim(0, 1)
        ax.axis("off")
        ax.set_title(question, pad=12)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    # choice / multiselect chart (top 8)
    table = s.summarize_overall_and_demo(question, [])
    # melt overall row (skip N/Row/Group)
    overall = table.iloc[0].drop(labels=["Row", "Group", "N"], errors="ignore")
    ser = overall.dropna().astype(float).sort_values(ascending=False).head(8)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(list(ser.index)[::-1], list(ser.values)[::-1])  # default colors
    for i, v in enumerate(list(ser.values)[::-1]):
        ax.text(v + max(ser.values) * 0.01, i, f"{v:.1f}", va="center")
    ax.set_xlabel("%" if (s.config.output_mode == "percent") else s.config.output_mode)
    ax.set_title(question)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------ Widget UI (optional) ------------------------------

class ReportAssistant(TableSummarizer):
    """
    Simple per-question UI:
      • Pick a question and groups
      • See an overall chart
      • See a summary table
      • Export the current question to Excel

    Falls back to headless helpers on environments without ipywidgets.
    """
    def __init__(self, df: pd.DataFrame, preselected_groups: Optional[List[str]] = None):
        super().__init__(df)
        self.preselected_groups = preselected_groups or []
        if widgets is not None:
            self._build()

    # ---- plotting (uses plain matplotlib, no styles/colors set) ----
    def _plot_overall(self, question: str):
        qtype = self.detect_question_type(question)
        if qtype == "scale":
            wcol = self._pick_weight_col(overall=True)
            temp = self._reshape_questions_to_long(self.df, question,
                                                   group_col=self.group_options[0] if self.group_options else self.df.columns[0],
                                                   weight_col=wcol)
            temp["score"] = temp["selected_choice"].apply(self._extract_score)
            temp = temp.dropna(subset=["score", wcol])
            grouped = temp.groupby("score")[wcol].sum()
            total = grouped.sum()
            keys = sorted(grouped.index, reverse=True)
            sizes = [(grouped.get(k, 0) / total) if total > 0 else 0 for k in keys]

            fig, ax = plt.subplots(figsize=(8, 2.2))
            left = 0.0
            for p in sizes:
                ax.barh(0, p, left=left, height=1.0)
                if p > 0.04:
                    ax.text(left + p / 2, 0, f"{int(round(p*100))}%", ha="center", va="center")
                left += p
            ax.set_xlim(0, 1)
            ax.axis("off")
            ax.set_title(question, pad=12)
            plt.show()
            return

        # choice / multiselect: bar of top 8 overall
        table = self.summarize_overall_and_demo(question, [])
        overall = table.iloc[0].drop(labels=["Row", "Group", "N"], errors="ignore")
        ser = overall.dropna().astype(float).sort_values(ascending=False).head(8)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.barh(list(ser.index)[::-1], list(ser.values)[::-1])
        for i, v in enumerate(list(ser.values)[::-1]):
            ax.text(v + max(ser.values) * 0.01, i, f"{v:.1f}", va="center")
        ax.set_xlabel("%" if (self.config.output_mode == "percent") else self.config.output_mode)
        ax.set_title(question)
        plt.show()

    # ---- widget UI scaffolding ----
    def _build(self):
        self.q_dd = widgets.Dropdown(options=self.question_options, description="Question:")
        self.grp_checks = [widgets.Checkbox(value=(g in self.preselected_groups), description=g)
                           for g in self.group_options]
        self.gout = widgets.Output()
        self.tout = widgets.Output()
        self.eout = widgets.Output()

        self.q_dd.observe(self._refresh, names="value")
        for cb in self.grp_checks:
            cb.observe(self._refresh, names="value")

        display(self.q_dd, widgets.VBox(self.grp_checks), self.gout, self.tout, self.eout)
        self._refresh()

    def _selected_groups(self) -> List[str]:
        return [cb.description for cb in self.grp_checks if cb.value]

    def _refresh(self, _=None):
        if widgets is None:
            return
        q = self.q_dd.value
        groups = self._selected_groups()

        with self.gout:
            clear_output()
            print("Overall:")
            self._plot_overall(q)

        with self.tout:
            clear_output()
            self.summary_table = self.summarize_overall_and_demo(q, groups or [])
            percent_cols = [c for c in self.summary_table.columns if c not in ["Row", "Group", "N"]]
            display(self.summary_table.style.format({c: "{:.1f}%" for c in percent_cols}))

        with self.eout:
            clear_output()
            btn = widgets.Button(description="Export this question to Excel")
            out = widgets.Output()

            def do_export(_b):
                safe = re.sub(r"[\\/*?:\[\]]", "", q)[:31] if q else "Summary"
                self.export_summary_table("Summary.xlsx", safe)
                with out:
                    clear_output()
                    print("Saved Summary.xlsx")

            btn.on_click(do_export)
            display(btn, out)
