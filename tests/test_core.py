"""Tests for surveytools core modules: cleaning and summarize."""
import pytest
import pandas as pd
import numpy as np

from surveytools.cleaning import (
    fix_df_text,
    fill_client_tokens,
    is_likert_candidate,
    is_numeric_like_series,
    norm_key,
    normalized,
)
from surveytools.summarize import TableSummarizer
from surveytools.types import SummaryConfig


# ============================================================================
# Tests for cleaning.py
# ============================================================================

class TestNormKey:
    def test_basic_lowercasing(self):
        assert norm_key("Hello World") == "hello world"

    def test_collapse_whitespace(self):
        assert norm_key("hello   world") == "hello world"
        assert norm_key("  hello   world  ") == "hello world"

    def test_none_handling(self):
        assert norm_key(None) == ""

    def test_empty_string(self):
        assert norm_key("") == ""


class TestNormalized:
    def test_strip_whitespace(self):
        assert normalized("  hello  ") == "hello"

    def test_preserve_case(self):
        assert normalized("  Hello World  ") == "Hello World"

    def test_none_handling(self):
        assert normalized(None) == ""

    def test_empty_string(self):
        assert normalized("") == ""


class TestIsNumericLikeSeries:
    def test_numeric_series(self):
        s = pd.Series([1, 2, 3, 4, 5])
        assert is_numeric_like_series(s) is True

    def test_string_numbers(self):
        s = pd.Series(["1", "2", "3"])
        assert is_numeric_like_series(s) is True

    def test_text_series(self):
        s = pd.Series(["Agree", "Disagree", "Neutral"])
        assert is_numeric_like_series(s) is False

    def test_mixed_series(self):
        s = pd.Series(["1", "two", "3"])
        assert is_numeric_like_series(s) is False

    def test_with_nan(self):
        s = pd.Series([1, 2, np.nan, 4])
        assert is_numeric_like_series(s) is True


class TestIsLikertCandidate:
    def test_likert_text_column(self):
        # dtype must be object for is_likert_candidate to work
        s = pd.Series(["Strongly Agree", "Agree", "Neutral", "Disagree", "Strongly Disagree"], dtype=object)
        assert is_likert_candidate(s) is True

    def test_numeric_column_not_candidate(self):
        s = pd.Series([1, 2, 3, 4, 5])
        assert is_likert_candidate(s) is False

    def test_too_few_unique_values(self):
        s = pd.Series(["Yes", "No"], dtype=object)
        assert is_likert_candidate(s) is False

    def test_too_many_unique_values(self):
        s = pd.Series([f"Value {i}" for i in range(20)], dtype=object)
        assert is_likert_candidate(s) is False

    def test_custom_thresholds(self):
        s = pd.Series(["A", "B"], dtype=object)
        assert is_likert_candidate(s, min_unique=2) is True


class TestFillClientTokens:
    def test_replace_in_headers(self):
        df = pd.DataFrame({
            "client_name": ["Acme Corp", "Acme Corp"],
            "Q1: How do you rate %client_name%?": ["Good", "Bad"],
        })
        result = fill_client_tokens(df)
        assert "Q1: How do you rate Acme Corp?" in result.columns

    def test_replace_in_cells(self):
        df = pd.DataFrame({
            "client_name": ["Acme Corp", "Acme Corp"],
            "feedback": ["I love %client_name%", "Great service at %client_name%"],
        })
        result = fill_client_tokens(df)
        assert result["feedback"].iloc[0] == "I love Acme Corp"

    def test_missing_column(self):
        df = pd.DataFrame({"Q1": ["A", "B"]})
        result = fill_client_tokens(df)
        pd.testing.assert_frame_equal(result, df)

    def test_all_null_column(self):
        df = pd.DataFrame({
            "client_name": [None, None],
            "Q1": ["A", "B"],
        })
        result = fill_client_tokens(df)
        pd.testing.assert_frame_equal(result, df)


class TestFixDfText:
    def test_basic_operation(self):
        df = pd.DataFrame({"col1": ["hello", "world"]})
        result = fix_df_text(df)
        assert list(result["col1"]) == ["hello", "world"]

    def test_preserves_non_string_columns(self):
        df = pd.DataFrame({"text": ["hello"], "num": [42]})
        result = fix_df_text(df)
        assert result["num"].iloc[0] == 42


# ============================================================================
# Tests for summarize.py - TableSummarizer
# ============================================================================

class TestExtractScore:
    @pytest.fixture
    def summarizer(self):
        df = pd.DataFrame({"Q1": ["test"]})
        return TableSummarizer(df)

    def test_colon_format(self, summarizer):
        assert summarizer._extract_score("Agree: 4") == 4
        assert summarizer._extract_score("Strongly Agree: 5") == 5

    def test_parenthesis_format(self, summarizer):
        assert summarizer._extract_score("Agree (4)") == 4
        assert summarizer._extract_score("Strongly Disagree (1)") == 1

    def test_plain_number(self, summarizer):
        assert summarizer._extract_score("4") == 4

    def test_float_rounding(self, summarizer):
        assert summarizer._extract_score("Score: 3.7") == 4
        assert summarizer._extract_score("Score: 3.2") == 3

    def test_no_score(self, summarizer):
        assert summarizer._extract_score("No number here") is None

    def test_nan_input(self, summarizer):
        assert summarizer._extract_score(np.nan) is None

    def test_none_input(self, summarizer):
        assert summarizer._extract_score(None) is None


class TestGroup5Point:
    @pytest.fixture
    def summarizer(self):
        df = pd.DataFrame({"Q1": ["test"]})
        return TableSummarizer(df)

    def test_standard_distribution(self, summarizer):
        props = {1: 0.1, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.3}
        result = summarizer._group_5point(props)
        assert result["Negative"] == 0.2  # 0.1 + 0.1
        assert result["Neutral"] == 0.2
        assert result["Positive"] == 0.6  # 0.3 + 0.3

    def test_missing_values(self, summarizer):
        props = {1: 0.5, 5: 0.5}  # Only extreme values
        result = summarizer._group_5point(props)
        assert result["Negative"] == 0.5
        assert result["Neutral"] == 0.0
        assert result["Positive"] == 0.5

    def test_empty_props(self, summarizer):
        result = summarizer._group_5point({})
        assert result["Negative"] == 0.0
        assert result["Neutral"] == 0.0
        assert result["Positive"] == 0.0


class TestDetectQuestionType:
    def test_open_ended_suffix(self):
        df = pd.DataFrame({"Q1 (open-ended)": ["comment1", "comment2"]})
        summarizer = TableSummarizer(df)
        assert summarizer.detect_question_type("Q1 (open-ended)") == "open-ended"

    def test_scale_type(self):
        df = pd.DataFrame({"Q1": ["Agree: 4", "Strongly Agree: 5", "Neutral: 3", "Disagree: 2"]})
        summarizer = TableSummarizer(df)
        assert summarizer.detect_question_type("Q1") == "scale"

    def test_choice_type(self):
        df = pd.DataFrame({"Q1": ["Option A", "Option B", "Option C", "Option A"]})
        summarizer = TableSummarizer(df)
        assert summarizer.detect_question_type("Q1") == "choice"

    def test_empty_column(self):
        df = pd.DataFrame({"Q1": [None, None, None]})
        summarizer = TableSummarizer(df)
        assert summarizer.detect_question_type("Q1") == "choice"


class TestFormatting:
    def test_percent_mode(self):
        df = pd.DataFrame({"Q1": ["test"]})
        config = SummaryConfig(output_mode="percent", decimals=1)
        summarizer = TableSummarizer(df, config)
        assert summarizer._fmt(0.75) == 75.0

    def test_fraction_mode(self):
        df = pd.DataFrame({"Q1": ["test"]})
        config = SummaryConfig(output_mode="fraction", decimals=2)
        summarizer = TableSummarizer(df, config)
        assert summarizer._fmt(0.753) == 0.75

    def test_count_mode(self):
        df = pd.DataFrame({"Q1": ["test"]})
        config = SummaryConfig(output_mode="count", decimals=0)
        summarizer = TableSummarizer(df, config)
        assert summarizer._fmt(0.5, denom=100) == 50


class TestSuppression:
    def test_suppress_below_threshold(self):
        df = pd.DataFrame({"Q1": ["test"]})
        config = SummaryConfig(suppress_below_n=10)
        summarizer = TableSummarizer(df, config)
        assert summarizer._should_suppress(5) is True
        assert summarizer._should_suppress(10) is False
        assert summarizer._should_suppress(15) is False


class TestAsteriskLabeling:
    def test_within_range(self):
        df = pd.DataFrame({"Q1": ["test"]})
        config = SummaryConfig(asterisk_from_n=10, asterisk_to_n=30)
        summarizer = TableSummarizer(df, config)
        assert summarizer._label_with_asterisk("Group A", 20) == "Group A*"

    def test_below_range(self):
        df = pd.DataFrame({"Q1": ["test"]})
        config = SummaryConfig(asterisk_from_n=10, asterisk_to_n=30)
        summarizer = TableSummarizer(df, config)
        assert summarizer._label_with_asterisk("Group A", 5) == "Group A"

    def test_above_range(self):
        df = pd.DataFrame({"Q1": ["test"]})
        config = SummaryConfig(asterisk_from_n=10, asterisk_to_n=30)
        summarizer = TableSummarizer(df, config)
        assert summarizer._label_with_asterisk("Group A", 50) == "Group A"


class TestQuestionOptions:
    def test_identifies_question_columns(self):
        df = pd.DataFrame({
            "Q1: First question": [1, 2],
            "Q2: Second question": [3, 4],
            "age": [25, 30],
            "gender": ["M", "F"],
        })
        summarizer = TableSummarizer(df)
        assert "Q1: First question" in summarizer.question_options
        assert "Q2: Second question" in summarizer.question_options
        assert "age" not in summarizer.question_options

    def test_identifies_demographic_columns(self):
        df = pd.DataFrame({
            "Q1": [1, 2],
            "age": [25, 30],
            "gender": ["M", "F"],
            "Custom (demographics)": ["A", "B"],
        })
        summarizer = TableSummarizer(df)
        assert "age" in summarizer.group_options
        assert "gender" in summarizer.group_options
        assert "Custom (demographics)" in summarizer.group_options


class TestSummarizeOverallAndDemo:
    def test_open_ended_returns_notice(self):
        df = pd.DataFrame({
            "Q1 (open-ended)": ["comment1", "comment2", "comment3"],
            "response_id": [1, 2, 3],
        })
        summarizer = TableSummarizer(df)
        result = summarizer.summarize_overall_and_demo("Q1 (open-ended)", [])
        assert "Note" in result.columns
        assert result.iloc[0]["Note"] == "Open-ended question. Not summarized."

    def test_choice_question_overall(self):
        df = pd.DataFrame({
            "Q1": ["Option A", "Option B", "Option A", "Option A"],
            "response_id": [1, 2, 3, 4],
            "polygon_name": ["Area1", "Area1", "Area2", "Area2"],
            "latest_weight": [1.0, 1.0, 1.0, 1.0],
        })
        summarizer = TableSummarizer(df)
        result = summarizer.summarize_overall_and_demo("Q1", [])
        assert result.iloc[0]["Row"] == "Overall"
        assert result.iloc[0]["N"] == 4
        # Option A: 3/4 = 75%, Option B: 1/4 = 25%
        assert "Option A" in result.columns
        assert "Option B" in result.columns

    def test_scale_question_grouping(self):
        df = pd.DataFrame({
            "Q1": ["Strongly Agree: 5", "Agree: 4", "Neutral: 3", "Disagree: 2"],
            "response_id": [1, 2, 3, 4],
            "polygon_name": ["Area1", "Area1", "Area2", "Area2"],
            "latest_weight": [1.0, 1.0, 1.0, 1.0],
        })
        config = SummaryConfig(auto_group_scale=True)
        summarizer = TableSummarizer(df, config)
        result = summarizer.summarize_overall_and_demo("Q1", [])
        assert "Positive" in result.columns
        assert "Neutral" in result.columns
        assert "Negative" in result.columns
