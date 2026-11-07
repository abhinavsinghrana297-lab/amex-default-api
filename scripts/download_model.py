"""Download the model release (model + feature list + golden file) into models/.

Usage:  python scripts/download_model.py
The version is pinned below: changing it is how the API moves to a new model (or rolls back).
"""
import urllib.request
from pathlib import Path

GITHUB_REPO = "abhinavsinghrana297-lab/amex-default-training"
MODEL_VERSION = "v1.0.0"
FILES = ["lgbm_final.txt", "feature_names.json", "model_card.json", "golden_customers.json"]

MODELS = Path(__file__).resolve().parent.parent / "models"


def main():
    MODELS.mkdir(exist_ok=True)
    for name in FILES:
        url = f"https://github.com/{GITHUB_REPO}/releases/download/{MODEL_VERSION}/{name}"
        target = MODELS / name
        urllib.request.urlretrieve(url, target)
        print(f"{name:24s} {target.stat().st_size / 1e3:8.1f} KB")


if __name__ == "__main__":
    main()
