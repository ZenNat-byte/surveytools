# src/surveytools/constants.py
ZC_TURQUOISE = "#3BD1BB"; ZC_CREAM = "#FCF5F1"; ZC_BEIGE = "#EEE2DB"; ZC_YELLOW = "#FAD565"
ZC_NIGHT = "#191824"; ZC_TEAL = "#148578"; ZC_MINT = "#79E8D7"; ZC_BLUE = "#7385FF"
ZC_WHITE = "#FFFFFF"; ZC_CORAL = "#FDA891"; ZC_BG = ZC_CREAM

ZERO_ALIASES = {
    "prefer not to say","prefer not to answer","don't know","do not know","unsure",
    "not applicable","na","n/a","n.a.","none"
}
PRESETS = {
    "Satisfaction (Very→Very)": [
        ("very unsatisfied", 1), ("somewhat unsatisfied", 2),
        ("neither satisfied nor unsatisfied", 3), ("somewhat satisfied", 4), ("very satisfied", 5),
    ],
    "Agreement (Strongly→Strongly)": [
        ("strongly disagree", 1), ("disagree", 2), ("neutral", 3), ("agree", 4), ("strongly agree", 5),
    ],
    "Safety (Not safe→Very safe)": [
        ("not safe at all", 1), ("not very safe", 2), ("neutral", 3), ("quite safe", 4), ("very safe", 5),
    ],
    "Ease (Very difficult→Very easy)": [
        ("very difficult", 1), ("somewhat difficult", 2), ("neutral", 3), ("somewhat easy", 4), ("very easy", 5),
    ],
}
