"""Reading the raw Kaggle parquet files with consistent dtypes."""
import polars as pl

from .config import RAW_DIR

# Column-name suffixes in this dataset encode the variable type:
#   P = days past due, A = amount, D = date, M = masked category, T/L = other
ID_COLS = {"case_id", "WEEK_NUM", "num_group1", "num_group2", "MONTH"}
NON_FEATURES = ID_COLS | {"date_decision", "target"}


def set_dtypes(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Cast columns according to their suffix."""
    casts = []
    for col in lf.collect_schema().names():
        if col in ID_COLS:
            casts.append(pl.col(col).cast(pl.Int64))
        elif col == "date_decision" or col.endswith("D"):
            casts.append(pl.col(col).cast(pl.Date, strict=False))
        elif col.endswith(("P", "A")):
            casts.append(pl.col(col).cast(pl.Float64, strict=False))
        elif col.endswith("M"):
            casts.append(pl.col(col).cast(pl.String))
    return lf.with_columns(casts)


def files_of(name: str, split: str = "train"):
    """All parquet chunks of a table, e.g. static_0 -> static_0_0, static_0_1."""
    d = RAW_DIR if split == "train" else RAW_DIR.parent / "test"
    files = sorted(d.glob(f"{split}_{name}.parquet")) + sorted(d.glob(f"{split}_{name}_[0-9]*.parquet"))
    if not files:
        raise FileNotFoundError(f"{split}_{name}*.parquet not found in {d}")
    return files


def table_exists(name: str) -> bool:
    try:
        files_of(name)
        return True
    except FileNotFoundError:
        return False


def scan(name: str, split: str = "train") -> pl.LazyFrame:
    """Lazily read one table (all chunks) with dtypes applied. Nothing is loaded into memory."""
    return pl.concat([set_dtypes(pl.scan_parquet(f)) for f in files_of(name, split)], how="diagonal_relaxed")
