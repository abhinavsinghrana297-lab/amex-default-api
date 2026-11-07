"""Load the model once, then turn raw statements into default probabilities."""
import json
from pathlib import Path

import lightgbm as lgb
import polars as pl

from app.features import make_features

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class DefaultModel:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.booster = lgb.Booster(model_file=str(models_dir / "lgbm_final.txt"))
        self.feature_names = json.loads((models_dir / "feature_names.json").read_text())
        self.card = json.loads((models_dir / "model_card.json").read_text())
        # The model only knows "Column_0..Column_896": the count must match the name list
        assert self.booster.num_feature() == len(self.feature_names), "model and feature list do not match"

    @property
    def version(self) -> str:
        return self.card["version"]

    def predict(self, statements: list[dict]) -> list[dict]:
        """Raw statements of one or more customers -> one probability per customer."""
        features = make_features(statements, self.feature_names)
        X = features.select(self.feature_names).cast(pl.Float32).to_numpy()   # same as training
        probs = self.booster.predict(X)
        return [{"customer_ID": cid, "default_probability": float(p)}
                for cid, p in zip(features["customer_ID"].to_list(), probs)]
