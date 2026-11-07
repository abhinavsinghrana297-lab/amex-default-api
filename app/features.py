"""Raw monthly statements -> the 897 model features.

This is the same logic as training, in two steps:
  1. encode   : the rules from notebook 01 (text -> numbers, Float32 / Int8 types)
  2. aggregate: build_e5_features from notebook 04a, copied exactly

tests/test_golden.py proves both steps reproduce the training features.
"""
import polars as pl

# ---- Step 1: encoding rules from notebook 01 --------------------------------------------
CAT_COLS = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120",
            "D_126", "D_63", "D_64", "D_66", "D_68"]
D63_MAP = {"CO": 0, "CR": 1, "CL": 2, "XZ": 3, "XM": 4, "XL": 5}
D64_MAP = {"O": 0, "U": 1, "R": 2, "-1": 3}
TEXT_COLS = ("customer_ID", "S_2", "D_63", "D_64")


def columns_from_feature_names(feature_names: list[str]) -> tuple[list[str], list[str]]:
    """The model's feature list starts with the raw columns (the 'last statement' features),
    in their original order, followed by 'n_statements'. So both lists can be derived from it."""
    raw_feature_cols = feature_names[: feature_names.index("n_statements")]
    raw_cols = ["customer_ID", "S_2"] + raw_feature_cols
    num_cols = [c for c in raw_feature_cols if c not in CAT_COLS]
    return raw_cols, num_cols


def encode(statements: list[dict], raw_cols: list[str]) -> pl.DataFrame:
    """Raw records (original CSV format) -> the typed table that train.parquet holds."""
    schema = {c: (pl.String if c in TEXT_COLS else pl.Float64) for c in raw_cols}
    df = pl.DataFrame(statements, schema=schema)          # missing keys -> null, column order fixed
    return df.with_columns(
        pl.col("S_2").str.to_date("%Y-%m-%d"),
        # unknown category text -> null (same lenient rule used for the test set)
        pl.col("D_63").replace_strict(D63_MAP, default=None, return_dtype=pl.Int8),
        pl.col("D_64").replace_strict(D64_MAP, default=None, return_dtype=pl.Int8),
        *[pl.col(c).cast(pl.Int8) for c in CAT_COLS if c not in ("D_63", "D_64")],
        *[pl.col(c).cast(pl.Float32) for c in raw_cols if c not in TEXT_COLS and c not in CAT_COLS],
    )


# ---- Step 2: feature building from notebook 04a (E5), unchanged --------------------------
def build_e5_features(statements: pl.LazyFrame, num_cols: list) -> pl.DataFrame:
    den = statements.with_columns([(pl.col(c) * 100).floor().cast(pl.Float32) for c in num_cols])
    last = den.group_by("customer_ID").agg(pl.all().exclude("S_2").sort_by(pl.col("S_2")).last())
    n_stat = den.group_by("customer_ID").agg(pl.len().alias("n_statements"))
    aggs = den.group_by("customer_ID").agg(
        *[pl.col(c).mean().alias(f"{c}_mean") for c in num_cols],
        *[pl.col(c).std().alias(f"{c}_std") for c in num_cols],
        *[pl.col(c).min().alias(f"{c}_min") for c in num_cols],
        *[pl.col(c).max().alias(f"{c}_max") for c in num_cols],
    ).with_columns(pl.exclude("customer_ID").cast(pl.Float32))
    return (last.join(n_stat, on="customer_ID").join(aggs, on="customer_ID")
                .collect().sort("customer_ID"))


# ---- Both steps together -----------------------------------------------------------------
def make_features(statements: list[dict], feature_names: list[str]) -> pl.DataFrame:
    """Raw statements of one or more customers -> customer_ID + features in the model's order."""
    raw_cols, num_cols = columns_from_feature_names(feature_names)
    encoded = encode(statements, raw_cols)
    features = build_e5_features(encoded.lazy(), num_cols)
    return features.select(["customer_ID"] + feature_names)
