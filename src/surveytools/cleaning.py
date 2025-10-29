# src/surveytools/cleaning.py
from __future__ import annotations
import re
from typing import Iterable
import pandas as pd

# optional: if ftfy is installed, fix weird encodings; otherwise no-op
try:
    from ftfy import fix_text as _fix_text
except Exception:  # keep package runnable without ftfy
    def _fix_text(x: str) -> str:
        return x

def fix_df_text(df: pd.DataFrame) -> pd.DataFrame:
    """Idempotent: fixes encoding in headers and string cells."""
    df = df.copy()
    df.columns = [_fix_text(str(c)) for c in df.columns]
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].apply(lambda x: _fix_text(x) if isinstance(x, str) else x)
    return df

def fill_client_tokens(df: pd.DataFrame, col_name: str = "client_name") -> pd.DataFrame:
    """
    Replace %client_name% in headers/cells with the first non-null value in `col_name`.
    If the column is missing or all-null, returns df unchanged.
    """
    if col_name not in df.columns or not df[col_name].notna().any():
        return df
    df = df.copy()
    val = str(df[col_name].dropna().iloc[0])
    df.columns = [str(c).replace("%client_name%", val) for c in df.columns]
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].apply(lambda x: x.replace("%client_name%", val) if isinstance(x, str) else x)
    return df

def is_numeric_like_series(s: pd.Series) -> bool:
    vals = pd.unique(s.dropna())
    try:
        pd.to_numeric(pd.Series(vals), errors="raise")
        return True
    except Exception:
        return False

def is_likert_candidate(s: pd.Series, max_unique: int = 12, min_unique: int = 3) -> bool:
    """
    Heuristic: text column with small-ish unique values and not parseable as numbers.
    Useful for finding columns to map with LikertMapper.
    """
    if s.dtype == object:
        uniq = pd.unique(s.dropna().astype(str).str.strip())
        return (min_unique <= len(uniq) <= max_unique) and not is_numeric_like_series(s)
    return False

def norm_key(x: str) -> str:
    """Lowercase + collapse whitespace (for matching labels)."""
    return re.sub(r"\s+", " ", (x or "").strip().lower())

def normalized(x: str) -> str:
    """Basic strip to keep original case but remove leading/trailing spaces."""
    return (x or "").strip()
