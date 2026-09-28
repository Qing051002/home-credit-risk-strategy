"""Scoring with a saved scorecard (optbinning BinningProcess + logistic-regression coefficients)."""
import json
import pickle

import numpy as np

from .config import RES_DIR


def load_scorecard(tag: str):
    """Load the shared binning process and one scorecard version ('A' or 'B')."""
    with open(RES_DIR / "06_binning_process.pkl", "rb") as fh:
        bp = pickle.load(fh)
    model = json.loads((RES_DIR / f"06_scorecard_model_{tag}.json").read_text())
    return bp, model


def woe_transform(bp, feature, values, categorical):
    x = values.astype(object) if categorical else values.astype("float64")
    return bp.get_binned_variable(feature).transform(
        x, metric="woe", metric_missing="empirical", metric_special="empirical")


def scorecard_score(bp, model, df):
    """Integer scorecard points (higher = safer). `df` is a pandas DataFrame with the model features.
    optbinning WOE = ln(%good / %bad), so all logistic coefficients are negative."""
    k = model["scaling"]["factor"]
    s = np.full(len(df), float(model["scaling"]["base_points"]))
    for f, b in zip(model["features"], model["coef"]):
        s += np.round(-k * b * woe_transform(bp, f, df[f].to_numpy(), f in model["categorical"]))
    return s


def score_to_pd(score, scaling):
    """Invert score = offset + factor * ln(odds_good) to a probability of default."""
    return 1 / (1 + np.exp((score - scaling["offset"]) / scaling["factor"]))
