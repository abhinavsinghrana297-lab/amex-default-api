"""The full path raw statements -> probability must reproduce what the model gave on Kaggle."""
import json
import random
from pathlib import Path

import pytest

from app.predict import DefaultModel

MODELS = Path(__file__).resolve().parent.parent / "models"


@pytest.fixture(scope="module")
def model():
    return DefaultModel()


@pytest.fixture(scope="module")
def golden():
    return json.loads((MODELS / "golden_customers.json").read_text())


def test_probabilities_match_training(model, golden):
    for c in golden["customers"]:
        [result] = model.predict(c["statements"])
        assert result["customer_ID"] == c["customer_ID"]
        assert result["default_probability"] == pytest.approx(c["expected_probability"], abs=1e-6)


def test_statement_order_does_not_matter(model, golden):
    """Clients may send statements in any order: features sort by date, so the answer must not change."""
    c = golden["customers"][1]
    shuffled = c["statements"][:]
    random.Random(0).shuffle(shuffled)
    assert model.predict(shuffled) == model.predict(c["statements"])


def test_output_is_a_probability(model, golden):
    statements = [s for c in golden["customers"] for s in c["statements"]]
    results = model.predict(statements)
    assert len(results) == len(golden["customers"])
    assert all(0.0 <= r["default_probability"] <= 1.0 for r in results)
