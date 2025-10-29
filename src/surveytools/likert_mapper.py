# src/surveytools/likert_mapper.py
from __future__ import annotations
from typing import Dict, List, Set
import re
import pandas as pd

from .cleaning import normalized, norm_key, is_likert_candidate
from .constants import ZERO_ALIASES, PRESETS

# Optional UI: if ipywidgets isn't available (e.g., headless/serverless), this still imports fine.
try:
    import ipywidgets as W
    from IPython.display import display, HTML, clear_output
except Exception:
    W = None
    def display(*_a, **_k):  # type: ignore
        pass
    def HTML(x):  # type: ignore
        return x
    def clear_output(*_a, **_k):  # type: ignore
        pass


class LikertMapper:
    """
    Notebook helper to map textual Likert options to numbers by writing back to the DataFrame as
    'Original Label: number'. Designed to be safe in environments without ipywidgets (imports OK).
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.SHARED_MAP: Dict[str, int] = {}
        self.PROCESSED: Set[str] = set()
        self.HISTORY: List[dict] = []
        self._state = {"queue": [], "idx": 0, "session_active": False}

        # Build UI only if widgets exist
        if W is not None:
            self._build_ui()

    # ---------------- UI scaffolding ----------------
    def _build_ui(self):
        df = self.df
        text_cols = [c for c in df.columns if df[c].dtype == object]
        # Pick only columns that look like Likert and end with "(general)" convention
        self.LIKERT_COLS = [
            c for c in text_cols
            if str(c).endswith("(general)") and is_likert_candidate(df[c])
        ]

        check_items = [
            W.Checkbox(value=False, description=c, indent=False, layout=W.Layout(width="100%"))
            for c in self.LIKERT_COLS
        ]
        self.checkbox_by_name = {cb.description: cb for cb in check_items}

        self.select_all_btn = W.Button(description="Select all", layout=W.Layout(width="120px"))
        self.clear_all_btn  = W.Button(description="Clear all",  layout=W.Layout(width="120px"))
        self.start_btn      = W.Button(description="Start mapping", button_style="primary", layout=W.Layout(width="160px"))
        self.undo_btn       = W.Button(description="Undo last Apply", layout=W.Layout(width="160px"))

        checklist_box = W.VBox(
            check_items,
            layout=W.Layout(max_height="260px", overflow="auto", border="1px solid #EEE2DB",
                            padding="8px", border_radius="8px")
        )
        controls_header = W.HBox([self.select_all_btn, self.clear_all_btn, self.start_btn, self.undo_btn])

        display(HTML("<b>Choose questions (only those ending with <code>(general)</code>):</b>"))
        display(checklist_box); display(controls_header)

        self.controls_out = W.Output()
        self.status_out = W.Output()
        display(self.controls_out); display(self.status_out)

        self.select_all_btn.on_click(lambda _b: [setattr(cb, "value", True) for cb in check_items if not cb.disabled])
        self.clear_all_btn.on_click(lambda _b: [setattr(cb, "value", False) for cb in check_items if not cb.disabled])
        self.start_btn.on_click(self._start_clicked)
        self.undo_btn.on_click(self._undo_clicked)

    # ---------------- core helpers ----------------
    def _can_auto_map(self, col: str) -> bool:
        df = self.df
        opts = list(pd.unique(df[col].dropna().astype(str).map(normalized)))
        opts_norm = {norm_key(o) for o in opts}
        return all(
            (o in self.SHARED_MAP) or (o in {norm_key(s) for s in ZERO_ALIASES}) or (o == "")
            for o in opts_norm
        )

    def _apply_mapping_to_column(self, col: str):
        df = self.df
        original = df[col].astype("object")
        norm = original.astype(object).where(~original.isna(), "").astype(str).map(normalized)
        out_vals: List[str] = []
        for orig_label, key in zip(original, norm):
            if pd.isna(orig_label) or key == "":
                out_vals.append("")
                continue
            k = norm_key(key)
            if k in self.SHARED_MAP:
                v = self.SHARED_MAP[k]
            elif k in {norm_key(s) for s in ZERO_ALIASES}:
                v = 0
            else:
                v = 0
            out_vals.append(f"{orig_label}: {v}")
        df[col] = out_vals
        self.PROCESSED.add(col)
        if hasattr(self, "checkbox_by_name") and col in self.checkbox_by_name:
            self.checkbox_by_name[col].disabled = True
            self.checkbox_by_name[col].value = False

    def _auto_apply_to_others(self, current_col: str):
        applied = []
        for other in getattr(self, "LIKERT_COLS", []):
            if other == current_col or other in self.PROCESSED:
                continue
            if self._can_auto_map(other):
                self._apply_mapping_to_column(other)
                applied.append(other)
                if other in self._state["queue"]:
                    self._state["queue"] = [q for q in self._state["queue"] if q != other]
        return applied

    # ---------------- undo support ----------------
    def _snapshot(self):
        general_cols_all = [c for c in self.df.columns if str(c).strip().endswith("(general)")]
        return {
            "SHARED_MAP": self.SHARED_MAP.copy(),
            "PROCESSED": self.PROCESSED.copy(),
            "queue": list(self._state.get("queue", [])),
            "idx": self._state.get("idx", 0),
            "session_active": self._state.get("session_active", False),
            "checkbox_states": {name: (cb.value, cb.disabled) for name, cb in getattr(self, "checkbox_by_name", {}).items()},
            "columns_backup": {c: self.df[c].copy() for c in general_cols_all},
        }

    def _undo_clicked(self, _b):
        if not self.HISTORY:
            if W is not None:
                with self.status_out:
                    clear_output(wait=True)
                    print("Nothing to undo.")
            return
        snap = self.HISTORY.pop()
        for c, series_backup in snap["columns_backup"].items():
            self.df[c] = series_backup
        self.SHARED_MAP.clear(); self.SHARED_MAP.update(snap["SHARED_MAP"])
        self.PROCESSED.clear(); self.PROCESSED.update(snap["PROCESSED"])
        self._state.update({"queue": snap["queue"], "idx": snap["idx"], "session_active": snap["session_active"]})
        if W is not None:
            for name, (val, disabled) in snap["checkbox_states"].items():
                if name in getattr(self, "checkbox_by_name", {}):
                    self.checkbox_by_name[name].value = val
                    self.checkbox_by_name[name].disabled = disabled
            with self.status_out:
                clear_output(wait=True)
                print("↩️ Undid last Apply.")

    # ---------------- UI flow ----------------
    def _start_clicked(self, _b):
        sel = [name for name, cb in getattr(self, "checkbox_by_name", {}).items()
               if cb.value and name not in self.PROCESSED]
        self._state["queue"] = sel
        self._state["idx"] = 0
        self._state["session_active"] = True
        for cb in getattr(self, "checkbox_by_name", {}).values():
            cb.disabled = True
        if hasattr(self, "select_all_btn"):
            self.select_all_btn.disabled = self.clear_all_btn.disabled = self.start_btn.disabled = True
        if W is not None:
            with self.status_out:
                clear_output()
                if not sel:
                    print("Pick at least one (general) question.")
                    return
                print(f"Selected {len(sel)} question(s).")
        if sel:
            self._render_for(sel[0])

    def _render_for(self, col: str):
        if W is None:
            return  # headless: nothing to render
        with self.controls_out:
            clear_output()
            df = self.df
            opts = list(pd.unique(df[col].dropna().astype(str).map(normalized)))
            # Sort with some heuristics so the “right” order appears top-to-bottom
            order_weight = {
                "extremely unsatisfactory": 1, "unsatisfactory": 2, "not very pleasant": 2,
                "not safe at all": 1, "not very safe": 2, "very difficult": 1, "somewhat difficult": 2,
                "neutral": 3, "satisfactory": 4, "somewhat pleasant": 4, "quite safe": 4,
                "very pleasant": 5, "extremely satisfactory": 5, "somewhat easy": 4, "very easy": 5,
            }
            sorted_opts = sorted(opts, key=lambda x: order_weight.get(norm_key(x), 999))
            n = len(sorted_opts)

            rows = []
            opt_controls: Dict[str, W.Dropdown] = {}
            for i, opt in enumerate(sorted_opts):
                nk = norm_key(opt)
                default_guess = self.SHARED_MAP.get(nk, None)
                default = default_guess if default_guess is not None else (
                    0 if nk in {norm_key(s) for s in ZERO_ALIASES} else min(i+1, n)
                )
                dd = W.Dropdown(options=list(range(0, n+1)), value=int(default), layout=W.Layout(width="100px"))
                opt_controls[opt] = dd
                rows.append(W.HBox([W.HTML(f"<div style='min-width:360px'>{opt}</div>"), dd]))

            preset_dd = W.Dropdown(options=[(k, k) for k in PRESETS.keys()],
                                   description="Preset:", layout=W.Layout(width="60%"))
            apply_preset_btn = W.Button(description="Apply preset to options", button_style="info")

            def do_apply_preset(_):
                pairs = PRESETS.get(preset_dd.value, [])
                for label, score in pairs:
                    p = norm_key(label)
                    for opt, dd in opt_controls.items():
                        o = norm_key(opt)
                        if o == p or re.search(rf"\b{re.escape(p)}\b", o):
                            dd.value = min(max(int(score), 0), n)
                for opt, dd in opt_controls.items():
                    if norm_key(opt) in {norm_key(s) for s in ZERO_ALIASES}:
                        dd.value = 0

            apply_preset_btn.on_click(do_apply_preset)

            apply_btn = W.Button(description="Apply to this question", button_style="success")
            next_btn  = W.Button(description="Next ▶")
            preview_btn = W.Button(description="Preview first 12 rows")
            msg = W.Output()

            def apply_clicked(_):
                with msg:
                    clear_output()
                    self.HISTORY.append(self._snapshot())
                    for opt, dd in opt_controls.items():
                        self.SHARED_MAP[norm_key(opt)] = int(dd.value)
                    self._apply_mapping_to_column(col)
                    auto_applied = self._auto_apply_to_others(col)
                    display(HTML(
                        "<div style='margin:8px 0;padding:8px;border-left:4px solid #148578;background:#FCF5F1'>"
                        f"<b>Applied</b> to <code>{col}</code>."
                        + (f"<br/>Auto-applied to: <code>{', '.join(auto_applied)}</code>" if auto_applied else "")
                        + "<br/><i>Use “Undo last Apply” to revert.</i></div>"
                    ))

            def next_clicked(_):
                self._state["idx"] += 1
                while self._state["idx"] < len(self._state["queue"]) and self._state["queue"][self._state["idx"]] in self.PROCESSED:
                    self._state["idx"] += 1
                if self._state["idx"] < len(self._state["queue"]):
                    self._render_for(self._state["queue"][self._state["idx"]])
                else:
                    with msg:
                        clear_output()
                        print("✅ Finished all selected questions.")
                    self._state["session_active"] = False
                    for cb in self.checkbox_by_name.values():
                        if cb.description not in self.PROCESSED:
                            cb.disabled = False
                    self.select_all_btn.disabled = self.clear_all_btn.disabled = self.start_btn.disabled = False

            def preview_clicked(_):
                with msg:
                    clear_output()
                    temp_vals = []
                    original = self.df[col].astype("object")
                    for orig_label in original:
                        if pd.isna(orig_label):
                            temp_vals.append("")
                        else:
                            match_key = norm_key(str(orig_label))
                            dd = None
                            for opt_label, _dd in opt_controls.items():
                                if norm_key(opt_label) == match_key:
                                    dd = _dd; break
                            temp_vals.append(f"{orig_label}: {dd.value if dd else 0}")
                    preview_df = pd.DataFrame({col: temp_vals})
                    display(HTML("<b>Preview (proposed mapping):</b>"))
                    display(preview_df.head(12))

            apply_btn.on_click(apply_clicked)
            next_btn.on_click(next_clicked)
            preview_btn.on_click(preview_clicked)

            display(HTML(f"<b>Question:</b> {col}"))
            display(W.HBox([preset_dd, apply_preset_btn]))
            grid = W.VBox(rows, layout=W.Layout(border="1px solid #EEE2DB", padding="8px", border_radius="8px"))
            display(HTML("<b>Assign numbers (0..N):</b>")); display(grid)
            display(W.HBox([apply_btn, next_btn, preview_btn])); display(msg)
