# src/surveytools/types.py
from dataclasses import dataclass

@dataclass
class SummaryConfig:
    as_percent: bool = True
    decimals: int = 1
    output_mode: str = "percent"  # "percent" | "fraction" | "count"
    suppress_below_n: int = 30
    asterisk_from_n: int = 30
    asterisk_to_n: int = 49
    auto_group_scale: bool = True
    highlight_threshold: float = 5.0
