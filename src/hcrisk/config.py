"""Project paths and global settings.

Data location
-------------
By default the project expects the Kaggle data under ``<repo>/data/``::

    data/
    └── home-credit-credit-risk-model-stability/
        └── parquet_files/train/*.parquet

To keep the data elsewhere, set the environment variable ``HCRISK_DATA_DIR``
to the folder that contains ``home-credit-credit-risk-model-stability/``.
Intermediate outputs are written to ``features/`` and ``results/`` inside the
same data folder, so nothing large is ever written into the repository.
"""
import os
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("HCRISK_DATA_DIR", REPO_ROOT / "data")).expanduser()
KAGGLE_DIR = DATA_DIR / "home-credit-credit-risk-model-stability"
RAW_DIR = KAGGLE_DIR / "parquet_files" / "train"
FEAT_DIR = DATA_DIR / "features"      # per-table aggregated features (notebook 02 onwards)
RES_DIR = DATA_DIR / "results"        # analysis outputs, models, predictions
FIG_DIR = REPO_ROOT / "reports" / "figures"

for _d in (FEAT_DIR, RES_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Sample design (WEEK_NUM boundaries) -------------------------------------
TRAIN_END, VALID_END, OOT1_END = 40, 50, 62      # train 0-40 | valid 41-50 | OOT1 51-62 | OOT2 63-91
PERIODS = ["1_train", "2_valid", "3_oot1", "4_oot2"]
PERIOD_LABEL = {"1_train": "Train", "2_valid": "Valid", "3_oot1": "OOT1", "4_oot2": "OOT2"}
BOUNDS = (TRAIN_END + 0.5, VALID_END + 0.5, OOT1_END + 0.5)   # vertical lines on weekly charts
SEED = 42


def period_expr() -> pl.Expr:
    """Map WEEK_NUM to the four sample periods."""
    w = pl.col("WEEK_NUM")
    return (pl.when(w <= TRAIN_END).then(pl.lit("1_train"))
              .when(w <= VALID_END).then(pl.lit("2_valid"))
              .when(w <= OOT1_END).then(pl.lit("3_oot1"))
              .otherwise(pl.lit("4_oot2")).alias("period"))


def savefig(fig, name: str) -> None:
    """Save a figure to reports/figures/ (PNG, 150 dpi)."""
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


def check_data() -> None:
    """Fail early with a clear message if the Kaggle data cannot be found."""
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Kaggle training data not found at {RAW_DIR}.\n"
            "Download it (see README) or set HCRISK_DATA_DIR to the folder that "
            "contains 'home-credit-credit-risk-model-stability/'.")
