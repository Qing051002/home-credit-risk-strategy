"""Discrimination, stability and binning metrics."""
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score, roc_curve


def ks_score(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metrics(y, p) -> dict:
    """AUC / Gini / KS. `p` must increase with risk."""
    auc = roc_auc_score(y, p)
    return {"AUC": round(auc, 4), "Gini": round(2 * auc - 1, 4), "KS": round(ks_score(y, p), 4)}


def weekly_gini(week, y, p) -> pd.Series:
    df = pd.DataFrame({"w": week, "y": y, "p": p})
    return df.groupby("w").apply(lambda t: 2 * roc_auc_score(t["y"], t["p"]) - 1, include_groups=False)


def stability_metric(g: pd.Series) -> dict:
    """Official competition metric on a weekly Gini series:
    mean(gini) + 88 * min(0, slope) - 0.5 * std(residuals), with x = 0, 1, 2, ..."""
    x = np.arange(len(g))
    a, b = np.polyfit(x, g.values, 1)
    res = g.values - (a * x + b)
    return {"score": g.mean() + 88 * min(0.0, a) - 0.5 * res.std(), "gini_mean": g.mean(),
            "slope": a, "res_std": res.std()}


def psi(p_exp, p_act, eps=1e-6) -> float:
    p_exp, p_act = np.maximum(p_exp, eps), np.maximum(p_act, eps)
    return float(((p_act - p_exp) * np.log(p_act / p_exp)).sum())


def psi_by_deciles(ref, cur) -> float:
    """PSI of `cur` against `ref`, using the deciles of `ref` as bins."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)[1:-1]))
    k = len(edges) + 1
    a = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=k) / len(ref)
    b = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=k) / len(cur)
    return psi(a, b)


# ---- Quick binning used for IV screening (the scorecard itself uses optbinning) ----
def bin_numeric(x_tr, x_va, n_bins=10):
    """Equal-frequency bins fitted on training data, reused for validation; bin 0 = missing."""
    v = x_tr[~np.isnan(x_tr)]
    if v.size == 0:
        return None
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1]))

    def f(x):
        b = np.searchsorted(edges, x, side="right") + 1
        b[np.isnan(x)] = 0
        return b
    return f(x_tr), f(x_va), len(edges) + 2


def bin_categorical(s_tr, s_va, min_share=0.005):
    """Categories below 0.5% of training rows are merged into 'other' (bin 1); bin 0 = missing."""
    vc = s_tr.drop_nulls().value_counts()
    cats = sorted(vc.filter(pl.col("count") >= min_share * len(s_tr))[s_tr.name].to_list())
    mapping = {c: i + 2 for i, c in enumerate(cats)}

    def f(s):
        return (pl.DataFrame({"x": s})
                  .select(pl.when(pl.col("x").is_null()).then(0)
                            .otherwise(pl.col("x").replace_strict(mapping, default=1, return_dtype=pl.Int64)))
                  .to_series().to_numpy())
    return f(s_tr), f(s_va), len(cats) + 2


def woe_iv(b, y, k):
    """WOE = ln(%bad / %good) per bin (+0.5 smoothing), IV, and bin shares."""
    tot = np.bincount(b, minlength=k).astype(float)
    bad = np.bincount(b, weights=y, minlength=k)
    good = tot - bad
    pb = (bad + 0.5) / (bad.sum() + 0.5 * k)
    pg = (good + 0.5) / (good.sum() + 0.5 * k)
    woe = np.log(pb / pg)
    m = tot > 0
    return woe, float(((pb - pg) * woe)[m].sum()), tot / tot.sum()


def woe_consistency(w_tr, w_va, s_tr, s_va, min_share=0.01):
    """Correlation of per-bin WOE between two periods (1 = identical risk pattern)."""
    m = (s_tr >= min_share) & (s_va >= min_share)
    if m.sum() < 3 or np.std(w_tr[m]) == 0 or np.std(w_va[m]) == 0:
        return None
    return float(np.corrcoef(w_tr[m], w_va[m])[0, 1])
