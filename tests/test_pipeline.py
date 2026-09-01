import pandas as pd
import pytest

from credit_risk import split_features_target


def test_splits_binary_target():
    features, labels = split_features_target(pd.DataFrame({"income": [1, 2], "default": [0, 1]}))
    assert features.columns.tolist() == ["income"]
    assert labels.tolist() == [0, 1]


def test_rejects_non_binary_target():
    with pytest.raises(ValueError, match="binary"):
        split_features_target(pd.DataFrame({"default": [0, 2]}))
