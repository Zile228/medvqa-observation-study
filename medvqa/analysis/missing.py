"""Consistent treatment of scalar missing values in experiment tables."""

import pandas as pd


def is_missing_scalar(value) -> bool:
    """None, NaN and pd.NA are missing; strings and containers are not."""
    return pd.api.types.is_scalar(value) and bool(pd.isna(value))
