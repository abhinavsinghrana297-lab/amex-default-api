"""Send real HTTP requests to the API (in memory, no server needed) and check the replies."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

MODELS = Path(__file__).resolve().parent.parent / "models"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:          # "with" runs the startup step, which loads the model
        yield c


@pytest.fixture(scope="module")
def golden():
    return json.loads((MODELS / "golden_customers.json").read_text())


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_matches_training(client, golden):
    for c in golden["customers"]:
        r = client.post("/predict", json={"statements": c["statements"]})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["customer_ID"] == c["customer_ID"]
        assert body["default_probability"] == pytest.approx(c["expected_probability"], abs=1e-6)


def test_missing_values_are_allowed(client, golden):
    """A client may leave out columns it doesn't have: they become missing values, like in training."""
    s = dict(golden["customers"][1]["statements"][0])
    for col in ["P_2", "B_1", "D_39"]:
        s.pop(col, None)
    r = client.post("/predict", json={"statements": [s]})
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("change, reason", [
    ({"D_63": "ZZ"}, "unknown category"),
    ({"S_2": "15/03/2018"}, "wrong date format"),
    ({"P_2": "abc"}, "text instead of a number"),
    ({"P_22222": 1.0}, "unknown column (typo)"),
])
def test_bad_input_is_rejected(client, golden, change, reason):
    s = dict(golden["customers"][1]["statements"][0])
    s.update(change)
    r = client.post("/predict", json={"statements": [s]})
    assert r.status_code == 422, f"{reason} was accepted"


def test_two_customers_in_one_request_is_rejected(client, golden):
    mixed = golden["customers"][0]["statements"] + golden["customers"][1]["statements"]
    r = client.post("/predict", json={"statements": mixed})
    assert r.status_code == 422


def test_empty_request_is_rejected(client):
    assert client.post("/predict", json={"statements": []}).status_code == 422
