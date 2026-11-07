"""Smoke test for a RUNNING API (local uvicorn or a Docker container).

Sends every golden customer to /predict and checks the probability matches training.
Usage:  python scripts/smoke_test.py                 (default http://127.0.0.1:8000)
        python scripts/smoke_test.py http://host:port
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
GOLDEN = Path(__file__).resolve().parent.parent / "models" / "golden_customers.json"


def wait_until_healthy(timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{URL}/health") as r:
                return json.load(r)
        except OSError:
            time.sleep(2)
    raise SystemExit(f"API at {URL} not healthy after {timeout}s")


def main():
    print("health:", wait_until_healthy())
    failures = 0
    for c in json.loads(GOLDEN.read_text())["customers"]:
        req = urllib.request.Request(f"{URL}/predict", data=json.dumps({"statements": c["statements"]}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            got = json.load(r)["default_probability"]
        ok = abs(got - c["expected_probability"]) < 1e-6
        failures += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {c['customer_ID'][:12]}...  got {got:.6f}  expected {c['expected_probability']:.6f}")
    if failures:
        raise SystemExit(f"{failures} customer(s) gave a different probability")
    print("all golden customers match ✓")


if __name__ == "__main__":
    main()
