"""Write example_request.json: one golden customer's statements, ready to send to /predict."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
golden = json.loads((ROOT / "models" / "golden_customers.json").read_text())
customer = min(golden["customers"], key=lambda c: c["n_statements"])     # the shortest one

(ROOT / "example_request.json").write_text(json.dumps({"statements": customer["statements"]}, indent=1))
print(f"wrote example_request.json: customer {customer['customer_ID'][:12]}..., "
      f"{customer['n_statements']} statement(s)")
print(f"expected default_probability: {customer['expected_probability']:.6f}")
