"""Feature engineering helpers: date handling, aggregation, categorical encoding."""
import gc
import re
import time

import pandas as pd
import polars as pl
import polars.selectors as cs

from .config import FEAT_DIR
from .io import NON_FEATURES, files_of, set_dtypes

# Integer calendar fields (suffix T), e.g. dpdmaxdateyear_596T = year of the max-DPD event.
# Years are converted to "years before the decision date"; months (1-12) carry no
# information once the year is relative, so they are dropped.
YEAR_PAT = re.compile(r"(dateyear|_year_)", re.I)
MONTH_PAT = re.compile(r"(datemonth|_month_)", re.I)


def date_cols_of(lf: pl.LazyFrame):
    return [c for c, t in lf.collect_schema().items() if t == pl.Date and c not in NON_FEATURES]


def calendar_cols_of(lf: pl.LazyFrame):
    """Return (year_cols, month_cols), verified by value range to avoid false name matches."""
    schema = lf.collect_schema()
    cand_y = [c for c, t in schema.items() if YEAR_PAT.search(c) and t.is_numeric()]
    cand_m = [c for c, t in schema.items() if MONTH_PAT.search(c) and t.is_numeric()]
    if not cand_y + cand_m:
        return [], []
    med = lf.select([pl.col(c).cast(pl.Float64).median().alias(c) for c in cand_y + cand_m]).collect().row(0, named=True)
    years = [c for c in cand_y if med[c] is not None and 1950 <= med[c] <= 2035]
    months = [c for c in cand_m if med[c] is not None and 1 <= med[c] <= 12]
    return years, months


def to_relative_time(lf: pl.LazyFrame, date_cols, year_cols) -> pl.LazyFrame:
    """Dates -> days relative to date_decision (negative = before decision);
    calendar years -> years before the decision (fractional decision year minus the year)."""
    dec_year = pl.col("date_decision").dt.year() + (pl.col("date_decision").dt.ordinal_day() - 1) / 365.25
    exprs = [(pl.col(c) - pl.col("date_decision")).dt.total_days().cast(pl.Float32).alias(c) for c in date_cols]
    exprs += [(dec_year - pl.col(c).cast(pl.Float64)).cast(pl.Float32).alias(f"{c}_yrsago") for c in year_cols]
    return lf.with_columns(exprs).drop(year_cols)


def build_aggs(lf: pl.LazyFrame, name: str, date_cols):
    """Aggregation expressions to case_id level.
    numeric: max, mean | dates (days): max, min | boolean: mean |
    string: number of distinct values + value of the primary record (num_group1 == 0)."""
    schema = lf.collect_schema()
    has_g1 = "num_group1" in schema
    exprs = [pl.len().cast(pl.Int32).alias(f"{name}__cnt")]
    for c, t in schema.items():
        if c in NON_FEATURES:
            continue
        if c in date_cols:
            exprs += [pl.col(c).max().alias(f"{c}__max"), pl.col(c).min().alias(f"{c}__min")]
        elif t == pl.Boolean:
            exprs.append(pl.col(c).cast(pl.Float32).mean().alias(f"{c}__mean"))
        elif t.is_numeric():
            exprs += [pl.col(c).max().alias(f"{c}__max"), pl.col(c).mean().alias(f"{c}__mean")]
        elif t == pl.String:
            exprs.append(pl.col(c).n_unique().cast(pl.Int32).alias(f"{c}__nuniq"))
            if has_g1:
                exprs.append(pl.col(c).filter(pl.col("num_group1") == 0).first().alias(f"{c}__first"))
    return exprs


def process_table(name: str, dates: pl.LazyFrame, overwrite: bool = False) -> None:
    """Aggregate one depth-1/2 table to case_id level, chunk by chunk, and write it to FEAT_DIR.
    Each chunk is aggregated with the streaming engine; results go straight to disk."""
    out = FEAT_DIR / f"{name}.parquet"
    if out.exists() and not overwrite:
        print(f"{name:20s} exists, skipped")
        return
    t0, parts = time.time(), []
    for f in files_of(name):
        lf = set_dtypes(pl.scan_parquet(f))
        dcols = date_cols_of(lf)
        ycols, mcols = calendar_cols_of(lf)
        lf = lf.drop(mcols)
        if dcols or ycols:
            lf = to_relative_time(lf.join(dates, on="case_id", how="left"), dcols, ycols)
        part = (lf.group_by("case_id").agg(build_aggs(lf, name, dcols))
                  .collect(engine="streaming")
                  .with_columns(cs.float().cast(pl.Float32)))
        parts.append(part)
        del lf
        gc.collect()
    df = pl.concat(parts, how="diagonal_relaxed")
    dup = df.height - df["case_id"].n_unique()
    if dup:
        print(f"WARNING {name}: {dup} case_ids appear in more than one chunk")
    df.write_parquet(out)
    print(f"{name:20s} rows={df.height:>10,} cols={df.width:>4}  {time.time() - t0:6.1f}s")


OTHER = "__OTHER__"


def encode_cats(X: pd.DataFrame, spec: dict, merged: bool = True) -> pd.DataFrame:
    """Encode string columns as pandas categoricals with a fixed category list (from training data).
    Values outside the list become '__OTHER__' (merged=True) or missing (merged=False)."""
    for col, cats in spec.items():
        s = X[col].astype(object)
        known = s.isin(cats)
        other = s.notna() & ~known
        s = s.where(known, None)
        if merged:
            s[other] = OTHER
        X[col] = pd.Categorical(s, categories=list(cats) + ([OTHER] if merged else []))
    return X


def to_model_frame(df: pl.DataFrame, features, categorical, spec, merged=True) -> pd.DataFrame:
    """Polars -> pandas frame ready for LightGBM / XGBoost (float32 numerics, fixed categoricals)."""
    X = df.select(features).with_columns(pl.col(pl.Boolean).cast(pl.Float32)).to_pandas()
    num = [c for c in features if c not in categorical]
    X[num] = X[num].astype("float32")
    return encode_cats(X, {c: spec[c] for c in categorical}, merged)
