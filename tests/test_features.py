import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.features import single_transaction_to_vector, to_feature_matrix, validate_columns


def _full_row(**overrides):
    row = {"Time": 100.0, "Amount": 50.0, **{f"V{i}": 0.0 for i in range(1, 29)}}
    row.update(overrides)
    return row


def test_validate_columns_raises_on_missing_column():
    df = pd.DataFrame([_full_row()])
    df = df.drop(columns=["V5"])
    with pytest.raises(ValueError):
        validate_columns(df)


def test_to_feature_matrix_orders_columns_correctly():
    df = pd.DataFrame([_full_row(Amount=999.99)])
    matrix = to_feature_matrix(df)
    assert matrix.shape == (1, len(FEATURE_COLUMNS))
    assert matrix[0, FEATURE_COLUMNS.index("Amount")] == 999.99


def test_single_transaction_to_vector_negative_amount_allowed_at_feature_level():
    # Business-level validation (Amount >= 0) lives in the pydantic schema;
    # this lower-level helper just converts a dict to a vector.
    row = _full_row(Amount=-5.0)
    vec = single_transaction_to_vector(row)
    assert vec.shape == (1, len(FEATURE_COLUMNS))


def test_single_transaction_to_vector_extreme_values():
    row = _full_row(Amount=1e9, V14=-999.0)
    vec = single_transaction_to_vector(row)
    assert vec.shape == (1, len(FEATURE_COLUMNS))
    assert vec[0, FEATURE_COLUMNS.index("V14")] == -999.0
