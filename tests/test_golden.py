"""Golden-file test: the API's feature code must reproduce the training features.

models/golden_customers.json holds 5 hold-out customers saved on Kaggle (notebook 09):
their raw statements, the 897 features training computed for them, and the model's probability.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from app.features import make_features

MODELS = Path(__file__).resolve().parent.parent / "models"
TOLERANCE = 1e-5          # same tolerance as the skew check in notebook 01c


@pytest.fixture(scope="module")
def golden():
    return json.loads((MODELS / "golden_customers.json").read_text())


@pytest.fixture(scope="module")
def feature_names():
    return json.loads((MODELS / "feature_names.json").read_text())


def expected_matrix(golden, feature_names):
    rows = sorted(golden["customers"], key=lambda c: c["customer_ID"])
    return np.array([[np.nan if c["expected_features"][f] is None else c["expected_features"][f]
                      for f in feature_names] for c in rows], dtype=np.float64)


def test_feature_list_matches_model(golden, feature_names):
    assert len(feature_names) == 897
    assert set(golden["customers"][0]["expected_features"]) == set(feature_names)


def test_features_match_training(golden, feature_names):
    statements = [s for c in golden["customers"] for s in c["statements"]]
    got = make_features(statements, feature_names)

    expected_ids = sorted(c["customer_ID"] for c in golden["customers"])
    assert got["customer_ID"].to_list() == expected_ids

    a = got.select(feature_names).cast(float).to_numpy()
    b = expected_matrix(golden, feature_names)
    same = np.isclose(a, b, rtol=TOLERANCE, atol=TOLERANCE, equal_nan=True)
    bad = [feature_names[j] for j in np.where(~same.all(axis=0))[0]]
    assert same.all(), f"{(~same).sum()} values differ, e.g. in: {bad[:10]}"


def test_one_customer_at_a_time(golden, feature_names):
    """The API will usually score one customer per request: results must not depend on batching."""
    all_at_once = make_features([s for c in golden["customers"] for s in c["statements"]], feature_names)
    for c in golden["customers"]:
        alone = make_features(c["statements"], feature_names)
        together = all_at_once.filter(all_at_once["customer_ID"] == c["customer_ID"])
        assert alone.equals(together)
