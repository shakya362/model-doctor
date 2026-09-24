"""Shared fixtures: small, fast, synthetic datasets with known defects."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _base_classification(n=600, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.choice(["a", "b", "c"], n)
    logit = 0.9 * x1 - 0.7 * x2 + np.where(x3 == "a", 0.6, -0.2)
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})


@pytest.fixture
def clean_classification():
    """A dataset with genuine, moderate signal and no planted defects."""
    return _base_classification()


@pytest.fixture
def clean_regression():
    rng = np.random.default_rng(1)
    n = 600
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    y = 3 * x1 - 2 * x2 + rng.normal(0, 2.0, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def leaky_classification():
    """A column that is the target with 2% label noise."""
    df = _base_classification()
    rng = np.random.default_rng(5)
    df["settlement_flag"] = np.where(rng.random(len(df)) < 0.02, 1 - df["y"], df["y"])
    return df


@pytest.fixture
def duplicated_classification():
    """A third of the rows are exact copies, as oversampling-before-split creates."""
    df = _base_classification(n=500)
    dupes = df[df["y"] == 1]
    return pd.concat([df, dupes, dupes], ignore_index=True).sample(
        frac=1.0, random_state=0
    ).reset_index(drop=True)


@pytest.fixture
def imbalanced_classification():
    """~2.5% positive rate — accuracy is meaningless here."""
    rng = np.random.default_rng(3)
    n = 3000
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    logit = -4.2 + 0.5 * x1 + 0.3 * x2
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture
def messy_quality():
    rng = np.random.default_rng(4)
    n = 400
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "region": rng.choice(["North", "north ", "NORTH", "South"], n),
        "notes": rng.choice(["ok", "unknown", "N/A", "fine"], n),
        "pipeline_version": "v1",
        "y": rng.binomial(1, 0.5, n),
    })
    df.loc[rng.random(n) < 0.3, "x1"] = np.nan
    return df
