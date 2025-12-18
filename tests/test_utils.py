"""Unit tests for app/utils.py - Non-regression tests for core functions."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta


# Import the module under test
import sys
from pathlib import Path

# Add the parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils import (
    calculate_moving_average,
    detect_anomalies,
    filter_by_dates,
    get_date_range,
)


class TestCalculateMovingAverage:
    """Tests for calculate_moving_average function."""

    def test_simple_moving_average_basic(self):
        """Test simple moving average with basic data."""
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=10),
            "Poids (Kgs)": [80, 81, 79, 80, 82, 81, 80, 79, 78, 77],
        })
        result = calculate_moving_average(df, "Poids (Kgs)", window_size=3, method="Simple")
        
        assert len(result) == 10
        # First value should be just that value (min_periods=1)
        assert result.iloc[0] == 80.0
        # Third value should be average of first 3
        assert abs(result.iloc[2] - (80 + 81 + 79) / 3) < 0.01

    def test_exponential_moving_average(self):
        """Test exponential moving average."""
        df = pd.DataFrame({
            "Poids (Kgs)": [80, 82, 84, 86, 88],
        })
        result = calculate_moving_average(df, "Poids (Kgs)", window_size=3, method="Exponentielle")
        
        assert len(result) == 5
        # EWM should give more weight to recent values
        assert result.iloc[-1] > result.iloc[0]

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pd.DataFrame({"Poids (Kgs)": []})
        result = calculate_moving_average(df, "Poids (Kgs)", window_size=3, method="Simple")
        assert len(result) == 0


class TestDetectAnomalies:
    """Tests for detect_anomalies function."""

    def test_zscore_detection(self):
        """Test Z-score anomaly detection."""
        # Create data with one obvious outlier
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=10),
            "Poids (Kgs)": [80, 81, 79, 80, 150, 81, 80, 79, 80, 81],  # 150 is outlier
        })
        result = detect_anomalies(df, method="Z-score", z_threshold=2.0)
        
        assert "Anomalies" in result.columns
        # The 150 value should be detected as anomaly
        assert result[result["Poids (Kgs)"] == 150]["Anomalies"].iloc[0] == True

    def test_isolation_forest_detection(self):
        """Test IsolationForest anomaly detection."""
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=20),
            "Poids (Kgs)": [80 + np.random.randn() for _ in range(19)] + [150],  # 150 is outlier
        })
        result = detect_anomalies(df, method="IsolationForest", contamination=0.1)
        
        assert "Anomalies" in result.columns
        # Should detect some anomalies
        assert result["Anomalies"].sum() >= 1

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pd.DataFrame(columns=["Date", "Poids (Kgs)"])
        result = detect_anomalies(df, method="Z-score", z_threshold=2.0)
        assert "Anomalies" in result.columns
        assert len(result) == 0


class TestFilterByDates:
    """Tests for filter_by_dates function."""

    def test_filter_with_valid_range(self):
        """Test filtering with valid date range."""
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=30),
            "Poids (Kgs)": range(30),
        })
        date_range = (pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-20"))
        result = filter_by_dates(df, date_range)
        
        assert len(result) == 11  # 10th to 20th inclusive
        assert result["Date"].min() >= pd.Timestamp("2024-01-10")
        assert result["Date"].max() <= pd.Timestamp("2024-01-20")

    def test_filter_with_single_date(self):
        """Test that single date doesn't filter (returns original)."""
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=10),
            "Poids (Kgs)": range(10),
        })
        date_range = (pd.Timestamp("2024-01-05"),)  # Single element tuple
        result = filter_by_dates(df, date_range)
        
        # Should return original dataframe
        assert len(result) == 10

    def test_filter_empty_result(self):
        """Test filtering that results in empty dataframe."""
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=10),
            "Poids (Kgs)": range(10),
        })
        date_range = (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-10"))
        result = filter_by_dates(df, date_range)
        
        assert len(result) == 0


class TestGetDateRange:
    """Tests for get_date_range function."""

    def test_valid_date_range(self):
        """Test getting date range from valid dataframe."""
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=30),
            "Poids (Kgs)": range(30),
        })
        min_date, max_date = get_date_range(df)
        
        assert min_date == pd.Timestamp("2024-01-01")
        assert max_date == pd.Timestamp("2024-01-30")

    def test_empty_dataframe_raises(self):
        """Test that empty dataframe raises ValueError."""
        df = pd.DataFrame(columns=["Date", "Poids (Kgs)"])
        with pytest.raises(ValueError, match="DataFrame is empty"):
            get_date_range(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
