"""What a request and a response look like. FastAPI uses these to check every request
(wrong types, unknown columns, bad dates -> a clear 422 error) and to draw the /docs page."""
import datetime as dt
import json
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator

from app.features import CAT_COLS, columns_from_feature_names
from app.predict import MODELS_DIR

FEATURE_NAMES = json.loads((MODELS_DIR / "feature_names.json").read_text())
RAW_COLS, _ = columns_from_feature_names(FEATURE_NAMES)


class _StatementBase(BaseModel):
    model_config = ConfigDict(extra="forbid")      # unknown column names (typos) are rejected
    customer_ID: str = Field(min_length=1)
    S_2: dt.date                                    # statement date, "YYYY-MM-DD"
    D_63: Optional[Literal["CO", "CR", "CL", "XZ", "XM", "XL"]] = None
    D_64: Optional[Literal["O", "U", "R", "-1"]] = None


# The other 186 columns are all "a number, or missing". Writing them out by hand would be
# 186 lines, so the model is built from the column list instead.
_numeric = {c: (Optional[float], None) for c in RAW_COLS if c not in ("customer_ID", "S_2", "D_63", "D_64")}
Statement = create_model("Statement", __base__=_StatementBase, **_numeric)


class PredictRequest(BaseModel):
    statements: list[Statement] = Field(min_length=1, max_length=100,
                                        description="Monthly statements of ONE customer, any order")

    @field_validator("statements")
    @classmethod
    def one_customer(cls, statements):
        ids = {s.customer_ID for s in statements}
        if len(ids) != 1:
            raise ValueError(f"all statements must belong to one customer, got {len(ids)}")
        dates = [s.S_2 for s in statements]
        if len(set(dates)) != len(dates):
            raise ValueError("two statements have the same date")
        return statements


class PredictResponse(BaseModel):
    customer_ID: str
    default_probability: float
    model_version: str


assert all(c in RAW_COLS for c in CAT_COLS)
