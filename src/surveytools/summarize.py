# src/surveytools/summarize.py
from __future__ import annotations
import os, re, ast
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from .types import SummaryConfig

class TableSummarizer:
    """Core summarization shared by UIs and CLI."""
    def __init__(self, df: pd.DataFrame, config: Optional[SummaryConfig] = None):
        self.df = df
        self.config = config or SummaryConfig()
        self.last_note_message: Optional[str] = None

        # what we treat as demographic/group columns
        self.possible_groups = ["polygon_name", "age", "ethnicities",
                                "education_group", "gender", "income_group"]

        self.group_options = [
            c for c in df.columns
            if (str(c).strip().endswith("(demographics)") or c in self.possible_groups)
        ]
        self.question_options = [
            c for c in df.columns
            if str(c).startswith("Q") and c not in self.group_options
        ]

        self.user_custom_orders: Dict[str, Dict[str, bool] | List[str]] = {}
        self.summary_table: Optional[pd.DataFrame] = None

    # ---------- helpers ----------
    def _pick_weight_col(self, *, overall: bool = False, group_col: Optional[str] = None) -> str:
        if overall:
            return "latest_weight"
        if group_col == "polygon_name" and (
            "latest_weight_level_1" in self.df.columns
            and not self.df["latest_weight_level_1"].isna().all()
        ):
            return "latest_weight_level_1"
        return "latest_weight"

    def _ensure_valid_weight_col(self, df: pd.DataFrame, weight_col: str) -> pd.DataFrame:
        if weight_col not in df.columns or df[weight_col].isna().all():
            df = df.copy()
            df[weight_col] = 1.0
        return df
    
    def _reshape_questions_to_long(self, df: pd.DataFrame, question: str,
                                   group_col: str, weight_col: str) -> pd.DataFrame:
        cols = [group_col, weight_col, question]
        # Avoid duplicating 'response_id' if it's already the group column
        if "response_id" in df.columns and group_col != "response_id":
            cols.append("response_id")

        temp = df[cols].copy()
        temp = temp.rename(columns={question: "selected_choice"})
        temp = temp[temp[weight_col].notna() & (temp[weight_col] != 0)]
        temp = temp[temp["selected_choice"].notna() &
                    (temp["selected_choice"].astype(str).str.strip() != "")]
        temp["qid"] = question
        return temp



    def _get_df_for_group(self, df: pd.DataFrame, group_col: str) -> pd.DataFrame:
        if group_col == "ethnicities":
            return self._clean_ethnicities_with_hispanic(df)
        return df

    def _clean_ethnicities_with_hispanic(
        self, df: pd.DataFrame, ethnicity_col="ethnicities",
        hispanic_col="Q17: Are you of Hispanic, Latino, or Spanish origin?-1967 (demographics)"
    ) -> pd.DataFrame:
        df = df.copy()

        def parse_ethnicity(val):
            if isinstance(val, list):
                return val
            if isinstance(val, str):
                try:
                    return ast.literal_eval(val)
                except Exception:
                    return [val]
            return [val]

        if hispanic_col not in df.columns:
            df[ethnicity_col] = df[ethnicity_col].apply(parse_ethnicity)
            return df.explode(ethnicity_col)

        def augment(row):
            eth = parse_ethnicity(row[ethnicity_col])
            if pd.notna(row[hispanic_col]) and str(row[hispanic_col]).strip().lower() == "yes":
                if "Hispanic or Latino" not in eth:
                    eth.append("Hispanic or Latino")
            return eth

        df[ethnicity_col] = df.apply(augment, axis=1)
        return df.explode(ethnicity_col)

    # ---------- formatting / rules ----------
    def _fmt(self, value: float, denom: float | None = None) -> float:
        mode = getattr(self.config, "output_mode", "percent")
        if mode == "percent":
            v = value * 100.0
        elif mode == "fraction":
            v = value
        elif mode == "count":
            v = value if denom is None else value * float(denom)
        else:
            v = value
        return round(v, self.config.decimals)

    def _should_suppress(self, n: int) -> bool:
        return n < self.config.suppress_below_n

    def _label_with_asterisk(self, label: str, n: int) -> str:
        if self.config.asterisk_from_n <= n <= self.config.asterisk_to_n:
            return f"{label}*"
        return label

    def _group_5point(self, props_by_num: Dict[int, float]) -> Dict[str, float]:
        return {
            "Negative": props_by_num.get(1, 0.0) + props_by_num.get(2, 0.0),
            "Neutral":  props_by_num.get(3, 0.0),
            "Positive": props_by_num.get(4, 0.0) + props_by_num.get(5, 0.0),
        }

    def _extract_score(self, x) -> Optional[int]:
        """Parse 'label: 4' or 'label (4)' → 4; return None if not found."""
        if pd.isna(x):
            return None
        s = str(x).strip()
        m = re.search(r":\s*([0-9]+(?:\.[0-9]+)?)\s*$", s) or \
            re.search(r"([0-9]+(?:\.[0-9]+)?)\s*\)?\s*$", s)
        if not m:
            return None
        try:
            v = float(m.group(1))
            return int(round(v))
        except Exception:
            return None


    # ---------- heuristic question type ----------
    def detect_question_type(self, question_col: str) -> str:
        if question_col.strip().endswith("(open-ended)"):
            return "open-ended"
        series = self.df[question_col].dropna()
        if series.empty:
            return "choice"
        scores = series.astype(str).map(self._extract_score)
        parsed = scores.notna().sum()
        unique_scores = set(scores.dropna().astype(int).unique().tolist())
        if parsed >= 0.5 * len(series) and len(unique_scores) >= 2:
            return "scale"
        return "choice"

    # ---------- export helper ----------
    def export_summary_table(self, filename: str, sheet_name: str) -> None:
        if self.summary_table is None:
            raise RuntimeError("No summary table to export.")
        if os.path.exists(filename):
            with pd.ExcelWriter(filename, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
                self.summary_table.to_excel(w, index=False, sheet_name=sheet_name)
        else:
            with pd.ExcelWriter(filename, engine="openpyxl") as w:
                self.summary_table.to_excel(w, index=False, sheet_name=sheet_name)

    # ---------- main ----------
    def summarize_overall_and_demo(self, question: str, group_cols: List[str],
                                   grouped_choices: Optional[Dict[str, List[str]]] = None,
                                   base_mode: str = "all") -> pd.DataFrame:
        def ensure_resp(df_: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
            respondent_col = "response_id" if "response_id" in df_.columns else None
            if respondent_col is None:
                df_ = df_.reset_index().rename(columns={"index": "tmp_idx"})
                respondent_col = "tmp_idx"
            return df_, respondent_col

        q_type = self.detect_question_type(question)

        # --- open-ended: just a notice row
        if q_type == "open-ended":
            result = pd.DataFrame([{
                "Row": "Overall", "Group": "All respondents",
                "N": self.df[question].notnull().sum(),
                "Note": "Open-ended question. Not summarized.",
            }])
            self.summary_table = result
            return result

        # --- scale (Likert-style numbers parsed from strings)
        if q_type == "scale":
            weight_col = self._pick_weight_col(overall=True)
            df_w = self._ensure_valid_weight_col(self.df, weight_col)
            temp = self._reshape_questions_to_long(
                df_w, question,
                group_col="polygon_name" if "polygon_name" in self.df.columns else self.df.columns[0],
                weight_col=weight_col
            )
            temp["score"] = temp["selected_choice"].apply(self._extract_score)
            temp = temp.dropna(subset=["score", weight_col])
            temp, respondent_col = ensure_resp(temp)

            # labels per numeric score
            score_labels: Dict[int, str] = {}
            for x in temp["selected_choice"].dropna().unique():
                n = self._extract_score(x)
                if n is not None and n not in score_labels:
                    score_labels[n] = x

            all_nums = sorted(score_labels.keys())
            grouped = temp.groupby("score")[weight_col].sum()
            total = grouped.sum()
            n_resp = temp[respondent_col].nunique()

            # auto-group 5-pt scales
            core = {n for n in set(all_nums) if 1 <= n <= 5}
            use_grouping = self.config.auto_group_scale and len(core) > 0

            if use_grouping:
                cols = ["Negative", "Neutral", "Positive"]
                full_props = {k: (grouped.get(k, 0) / total) if total > 0 else 0 for k in range(1, 6)}
                gp = self._group_5point(full_props)
                overall_row = {"Row": "Overall", "Group": "All respondents", "N": n_resp}
                for c in cols:
                    overall_row[c] = self._fmt(gp[c], denom=total)
            else:
                cols = [score_labels[k] for k in all_nums]
                overall_row = {"Row": "Overall", "Group": "All respondents", "N": n_resp}
                for k in all_nums:
                    prop = (grouped.get(k, 0) / total) if total > 0 else 0
                    overall_row[score_labels[k]] = self._fmt(prop, denom=total)

            group_rows = []
            for group_col in (group_cols or []):
                weight_col = self._pick_weight_col(group_col=group_col)
                use_df = self._get_df_for_group(self.df, group_col)
                use_df_w = self._ensure_valid_weight_col(use_df, weight_col)
                long_df = self._reshape_questions_to_long(use_df_w, question, group_col, weight_col)
                long_df["score"] = long_df["selected_choice"].apply(self._extract_score)
                long_df = long_df.dropna(subset=["score", weight_col, group_col])

                # optional visibility filter
                show_dict = self.user_custom_orders.get(f"{group_col}_show", {})
                if isinstance(show_dict, dict) and show_dict:
                    visible = [g for g in show_dict if show_dict[g]]
                    long_df = long_df[long_df[group_col].isin(visible)]

                long_df, respondent_col = ensure_resp(long_df)
                group_n = long_df.groupby(group_col)[respondent_col].nunique().sort_index()

                for gval in group_n.index:
                    n = group_n[gval]
                    if self._should_suppress(n):
                        continue
                    gdata = long_df[long_df[group_col] == gval]
                    grouped_g = gdata.groupby("score")[weight_col].sum()
                    total_g = grouped_g.sum()
                    row = {"Row": group_col, "Group": self._label_with_asterisk(str(gval), n), "N": n}
                    if use_grouping:
                        full_props_g = {k: (grouped_g.get(k, 0) / total_g) if total_g > 0 else 0 for k in range(1, 6)}
                        gp_g = self._group_5point(full_props_g)
                        for c in ["Negative", "Neutral", "Positive"]:
                            row[c] = self._fmt(gp_g[c], denom=total_g)
                    else:
                        for k in all_nums:
                            prop = (grouped_g.get(k, 0) / total_g) if total_g > 0 else 0
                            row[score_labels[k]] = self._fmt(prop, denom=total_g)
                    group_rows.append(row)

            columns = ["Row", "Group", "N"] + cols
            rows = [overall_row] + group_rows
            result = pd.DataFrame(rows, columns=columns)

            self.summary_table = result
            return result

        # --- choice / multiselect
        weight_col = self._pick_weight_col(overall=True)
        df_w = self._ensure_valid_weight_col(self.df, weight_col)
        temp = self._reshape_questions_to_long(
            df_w, question,
            group_col="polygon_name" if "polygon_name" in self.df.columns else self.df.columns[0],
            weight_col=weight_col
        )
        temp, respondent_col = ensure_resp(temp)
        temp["selected_choice"] = temp["selected_choice"].astype(str)

        # split multi-select with '|' delimiter
        if temp["selected_choice"].str.contains("|", regex=False).any():
            temp = temp.assign(selected_choice=temp["selected_choice"].str.split("|", regex=False)).explode("selected_choice")
            temp["selected_choice"] = temp["selected_choice"].str.strip()

        temp = temp[temp["selected_choice"].notna() & (temp["selected_choice"] != "")]
        temp_unique = temp.drop_duplicates(subset=[respondent_col, "selected_choice"]).copy()
        temp_unique["selected_choice"] = temp_unique["selected_choice"].str.strip().replace(
            to_replace=r"^Other:.*", value="Other", regex=True
        )

        if grouped_choices:
            grouped_set = {x for ops in grouped_choices.values() for x in ops}
            all_atomic = set(temp_unique["selected_choice"].unique())
            ungrouped = all_atomic - grouped_set
            all_columns = list(grouped_choices.keys()) + sorted(ungrouped)
        else:
            all_columns = sorted(temp_unique["selected_choice"].dropna().unique().tolist())

        resp_w = temp[[respondent_col, weight_col]].drop_duplicates().set_index(respondent_col)[weight_col]
        total_w = resp_w.sum()
        n_resp = resp_w.count()
        overall_row = {"Row": "Overall", "Group": "All respondents", "N": n_resp}

        if grouped_choices:
            resp_group: Dict[object, set] = {}
            for _, row in temp_unique.iterrows():
                r = row[respondent_col]
                orig = row["selected_choice"]
                val = next((g for g, atoms in grouped_choices.items() if orig in atoms), orig)
                resp_group.setdefault(r, set()).add(val)
            for col in all_columns:
                rs = [r for r, vals in resp_group.items() if col in vals]
                w = resp_w.loc[rs].sum() if rs else 0
                overall_row[col] = self._fmt((w / total_w) if total_w > 0 else 0, denom=total_w)
        else:
            for col in all_columns:
                rs = temp_unique.loc[temp_unique["selected_choice"] == col, respondent_col].unique()
                w = resp_w.loc[rs].sum() if len(rs) > 0 else 0
                overall_row[col] = self._fmt((w / total_w) if total_w > 0 else 0, denom=total_w)

        # order columns by overall %
        
        all_columns = sorted(all_columns, key=lambda cat: overall_row.get(cat, 0), reverse=True)

        group_rows = []
        for group_col in (group_cols or []):
            weight_col = self._pick_weight_col(group_col=group_col)
            use_df = self._get_df_for_group(self.df, group_col)
            use_df_w = self._ensure_valid_weight_col(use_df, weight_col)
            temp_g = self._reshape_questions_to_long(use_df_w, question, group_col, weight_col)
            temp_g, respondent_col = ensure_resp(temp_g)
            temp_g["selected_choice"] = temp_g["selected_choice"].astype(str)
            if temp_g["selected_choice"].str.contains("|", regex=False).any():
                temp_g = temp_g.assign(selected_choice=temp_g["selected_choice"].str.split("|", regex=False)).explode("selected_choice")
                temp_g["selected_choice"] = temp_g["selected_choice"].str.strip()
            temp_g = temp_g[temp_g["selected_choice"].notna() & (temp_g["selected_choice"] != "")]
            temp_unique_g = temp_g.drop_duplicates(subset=[respondent_col, "selected_choice"]).copy()
            temp_unique_g["selected_choice"] = temp_unique_g["selected_choice"].str.strip().replace(
                to_replace=r"^Other:.*", value="Other", regex=True
            )

            # optional visibility
            show_dict = self.user_custom_orders.get(f"{group_col}_show", {})
            if isinstance(show_dict, dict) and show_dict:
                visible = [g for g in show_dict if show_dict[g]]
                temp_unique_g = temp_unique_g[temp_unique_g[group_col].isin(visible)]

            group_n = temp_unique_g.groupby(group_col)[respondent_col].nunique().sort_index()
            for gval in group_n.index:
                n = group_n[gval]
                if self._should_suppress(n):
                    continue
                gdata = temp_g[temp_g[group_col] == gval].drop_duplicates(subset=[respondent_col, "selected_choice"])
                rw = temp_g[[group_col, respondent_col, weight_col]].drop_duplicates()
                if base_mode == "visible":
                    valid_rs = gdata[respondent_col].unique()
                    resp_w_g = rw[(rw[group_col] == gval) & (rw[respondent_col].isin(valid_rs))] \
                                  .set_index(respondent_col)[weight_col]
                else:
                    resp_w_g = rw[rw[group_col] == gval].set_index(respondent_col)[weight_col]
                total_gw = resp_w_g.sum()
                row = {"Row": group_col, "Group": self._label_with_asterisk(str(gval), n),
                       "N": resp_w_g.count()}

                if grouped_choices:
                    resp_group = {}
                    for _, r2 in gdata.iterrows():
                        r = r2[respondent_col]
                        orig = r2["selected_choice"]
                        val = next((g for g, atoms in grouped_choices.items() if orig in atoms), orig)
                        resp_group.setdefault(r, set()).add(val)
                    for col in all_columns:
                        rs = [r for r, vals in resp_group.items() if col in vals]
                        w = resp_w_g.loc[rs].sum() if rs else 0
                        row[col] = self._fmt((w / total_gw) if total_gw > 0 else 0, denom=total_gw)
                else:
                    for col in all_columns:
                        rs = gdata.loc[gdata["selected_choice"] == col, respondent_col].unique()
                        w = resp_w_g.loc[rs].sum() if len(rs) > 0 else 0
                        row[col] = self._fmt((w / total_gw) if total_gw > 0 else 0, denom=total_gw)

                group_rows.append(row)

        columns = ["Row", "Group", "N"] + all_columns
        rows = [overall_row] + group_rows
        result = pd.DataFrame(rows, columns=columns)

        self.summary_table = result
        return result

