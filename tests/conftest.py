from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    rng = pd.date_range("2026-01-01", periods=60, freq="D")
    trend = np.linspace(92, 86, 60)
    noise = np.random.default_rng(42).normal(0, 0.25, 60)
    return pd.DataFrame({"Date": rng, "Poids (Kgs)": trend + noise})
