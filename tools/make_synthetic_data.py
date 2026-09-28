"""Generate a small synthetic dataset with the same file layout and quirks as the Kaggle data.

Purpose: a smoke test of the full pipeline (notebooks 01-10) without downloading the real data.
The numbers produced from this data are meaningless; only the mechanics are exercised.

Quirks reproduced on purpose:
  * bad rate rising in weeks 40-62 and falling after a volume collapse at week ~63
  * tax registry vendor switch c -> a -> b (a duplicates c during the overlap; b = a x 8.1)
  * a credit-bureau field that only exists after week 22 (coverage change)
  * integer calendar fields (dpdmaxdateyear / month)
  * a "days since fixed campaign date" time proxy, gender fields, high-cardinality districts

Usage:  python tools/make_synthetic_data.py [--n 40000] [--out data_synthetic/]
        HCRISK_DATA_DIR=data_synthetic bash tools/run_pipeline.sh
"""
import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs


def _write(df: pl.DataFrame, path: Path) -> None:
    """Write with NaN -> null, as in the real data."""
    df.with_columns(cs.float().fill_nan(None)).write_parquet(path)


def main(n: int, out: Path, seed: int = 0):
    rng = np.random.default_rng(seed)
    raw = out / "home-credit-credit-risk-model-stability" / "parquet_files" / "train"
    raw.mkdir(parents=True, exist_ok=True)

    # ---- base: volume collapses after week 63, bad rate rises in 40-62 then falls
    wts = np.where(np.arange(92) < 63, 1.0, 0.45)
    wk = rng.choice(92, size=n, p=wts / wts.sum())
    wk.sort()
    dd = [date(2019, 1, 1) + timedelta(days=int(w * 7 + rng.integers(0, 7))) for w in wk]
    risk = rng.normal(0, 1, n)
    age = np.clip(rng.normal(40, 11, n), 19, 75)
    level = np.where(wk < 40, -3.6, np.where(wk <= 62, -3.6 + (wk - 40) * 0.025, -3.9))
    y = rng.binomial(1, 1 / (1 + np.exp(-(level + 0.9 * risk - 0.02 * (age - 40)))))
    case = np.arange(n)
    pl.DataFrame({"case_id": case, "date_decision": [str(d) for d in dd], "MONTH": [d.year * 100 + d.month for d in dd],
                  "WEEK_NUM": wk, "target": y}).pipe(_write, raw / "train_base.parquet")

    def miss(x, p):
        x = np.asarray(x, dtype=float).copy()
        x[rng.random(len(x)) < p] = np.nan
        return x

    # ---- static_0 (two chunks)
    c_era = wk < 35
    s0 = pl.DataFrame({
        "case_id": case,
        "maxdpdlast24m_143P": miss(np.maximum(0, risk * 12 + rng.normal(0, 8, n)), 0.2),
        "avgdpdtolclosure24_3658938P": miss(np.maximum(0, risk * 6 + rng.normal(0, 5, n)), 0.1),
        "credamount_770A": rng.lognormal(10, 0.7, n),
        "annuity_780A": rng.lognormal(8, 0.5, n),
        "eir_270L": 0.2 + 0.03 * risk + rng.normal(0, 0.02, n),
        "interestrate_311L": 0.18 + 0.03 * risk + rng.normal(0, 0.02, n),
        "mobilephncnt_593L": rng.integers(0, 4, n).astype(float),
        "pmtssum_45A": np.where(c_era, np.abs(rng.normal(3000, 800, n)), np.nan),
        "pmtaverage_3A": np.where(c_era, np.abs(rng.normal(500, 100, n)), np.nan),
        "firstclxcampaign_1125D": [str(date(2018, 1, 1) + timedelta(days=int(rng.integers(0, 20)))) for _ in range(n)],
        "lastrejectdate_50D": [None if rng.random() < 0.4 else str(d - timedelta(days=int(rng.integers(5, 900)))) for d in dd],
        "constant_0L": np.ones(n),
        "mostlynull_1A": miss(rng.random(n), 0.99),
    })
    half = n // 2
    s0[:half].pipe(_write, raw / "train_static_0_0.parquet")
    s0[half:].pipe(_write, raw / "train_static_0_1.parquet")

    # ---- static_cb_0: birth date + an education field whose codes change after week 45
    edu = np.where(risk + rng.normal(0, 1.5, n) > 0, "low", "high")
    edu = np.where(wk > 45, np.char.add(edu, "_v2"), edu)
    pl.DataFrame({"case_id": case,
                  "dateofbirth_337D": [str(d - timedelta(days=int(a * 365.25))) for d, a in zip(dd, age)],
                  "education_1103M": edu}).pipe(_write, raw / "train_static_cb_0.parquet")

    # ---- person_1: applicant (num_group1 == 0) + one contact person for 40% of cases
    sex = rng.choice(["F", "M"], n, p=[0.62, 0.38])
    inc = np.exp(10 - 0.2 * risk + rng.normal(0, 0.4, n))
    app = pl.DataFrame({
        "case_id": case, "num_group1": np.zeros(n, dtype=np.int64),
        "birth_259D": [str(d - timedelta(days=int(a * 365.25))) for d, a in zip(dd, age)],
        "sex_738L": sex, "gender_992L": pl.Series([None] * n, dtype=pl.String),
        "mainoccupationinc_384A": inc,
        "incometype_1044T": rng.choice(["EMPLOYED", "SELF", "PENSION"], n),
        "education_927M": np.where(risk + rng.normal(0, 1.5, n) > 0, "sec", "uni"),
        "contaddr_district_15M": [f"D{v}" for v in rng.integers(0, 400, n)],
        "registaddr_zipcode_184M": [f"Z{v}" for v in rng.integers(0, 3000, n)],
    })
    k = rng.random(n) < 0.4
    con = app.filter(pl.Series(k)).with_columns(pl.lit(1, pl.Int64).alias("num_group1"),
                                                pl.lit(None, pl.String).alias("sex_738L"),
                                                pl.lit("F").alias("gender_992L"))
    pl.concat([app, con]).pipe(_write, raw / "train_person_1.parquet")

    # ---- credit_bureau_a_1: coverage 60% before week 22, 90% after; one field only after week 22
    cov = np.where(wk < 22, 0.6, 0.9)
    ids = case[rng.random(n) < cov]
    reps = rng.integers(1, 5, len(ids))
    cid = np.repeat(ids, reps)
    m = len(cid)
    g1 = np.concatenate([np.arange(r) for r in reps])
    dcase = np.array(dd, dtype="datetime64[D]")[cid]
    yrs_ago = rng.uniform(0, 5, m)
    ev = dcase - (yrs_ago * 365.25).astype("timedelta64[D]")
    cb1 = pl.DataFrame({
        "case_id": cid, "num_group1": g1,
        "dpdmaxdateyear_596T": ev.astype("datetime64[Y]").astype(int) + 1970,
        "dpdmaxdatemonth_442T": (ev.astype("datetime64[M]").astype(int) % 12) + 1,
        "overdueamountmax_155A": np.maximum(0, risk[cid] * 2000 + rng.normal(0, 1500, m)),
        "dateofcredstart_739D": (dcase - rng.integers(30, 3000, m).astype("timedelta64[D]")).astype(str),
        "totalamount_6A": np.where(wk[cid] >= 22, rng.lognormal(10, 1, m) * np.exp(-0.2 * risk[cid]), np.nan),
        "classificationofcontr_400M": rng.choice([f"C{i}" for i in range(30)], m),
    })
    h = cb1["case_id"] < half
    cb1.filter(h).pipe(_write, raw / "train_credit_bureau_a_1_0.parquet")
    cb1.filter(~h).pipe(_write, raw / "train_credit_bureau_a_1_1.parquet")

    # ---- credit_bureau_a_2 (depth 2) with payment year / month
    cid2 = np.repeat(cid, 3)
    m2 = len(cid2)
    ev2 = np.array(dd, dtype="datetime64[D]")[cid2] - rng.integers(0, 1500, m2).astype("timedelta64[D]")
    pl.DataFrame({"case_id": cid2, "num_group1": np.repeat(g1, 3), "num_group2": np.tile([0, 1, 2], m),
                  "pmts_dpd_1073P": np.maximum(0, risk[cid2] * 5 + rng.normal(0, 4, m2)),
                  "pmts_overdue_1152A": np.maximum(0, risk[cid2] * 300 + rng.normal(0, 300, m2)),
                  "pmts_year_1139T": ev2.astype("datetime64[Y]").astype(int) + 1970,
                  "pmts_month_158T": (ev2.astype("datetime64[M]").astype(int) % 12) + 1,
                  }).pipe(_write, raw / "train_credit_bureau_a_2.parquet")

    # ---- applprev_1: previous applications (80% of cases), with a district field
    ida = case[rng.random(n) < 0.8]
    ra = rng.integers(1, 4, len(ida))
    ca = np.repeat(ida, ra)
    pl.DataFrame({"case_id": ca, "num_group1": np.concatenate([np.arange(r) for r in ra]),
                  "actualdpd_943P": np.maximum(0, risk[ca] * 4 + rng.normal(0, 4, len(ca))),
                  "credamount_590A": rng.lognormal(9.5, 0.8, len(ca)),
                  "district_544M": [f"P{v}" for v in rng.integers(0, 300, len(ca))],
                  }).pipe(_write, raw / "train_applprev_1.parquet")

    # ---- tax registries: one record set per case, exposed by vendors c (<40), a (34-66), b (>62, x8.1)
    idt = case[rng.random(n) < 0.7]
    rt = rng.integers(1, 9, len(idt))
    ct = np.repeat(idt, rt)
    mt = len(ct)
    amt = np.abs(rng.normal(1600, 450, mt) - risk[ct] * 200)
    emp = rng.choice(["E1", "E2", "E3", "E4"], mt)
    g1t = np.concatenate([np.arange(r) for r in rt])
    wt = wk[ct]
    rec_date = [str(dd[c] + timedelta(days=int(rng.integers(3, 15)))) for c in ct]      # after decision (record date)
    for name, mask, cols, scale in [
        ("tax_registry_c_1", wt < 40, ("pmtamount_36A", "processingdate_168D", "employername_160M"), 1.0),
        ("tax_registry_a_1", (wt >= 34) & (wt <= 66), ("amount_4527230A", "recorddate_4527225D", "name_4527232M"), 1.0),
        ("tax_registry_b_1", wt > 62, ("amount_4917619A", "deductiondate_4917603D", "name_4917606M"), 8.1),
    ]:
        pl.DataFrame({"case_id": ct[mask], "num_group1": g1t[mask], cols[0]: amt[mask] * scale,
                      cols[1]: [rec_date[i] for i in np.where(mask)[0]], cols[2]: emp[mask]}).pipe(_write, raw / f"train_{name}.parquet")

    # ---- small side tables
    idd = case[rng.random(n) < 0.07]
    pl.DataFrame({"case_id": idd, "num_group1": np.zeros(len(idd), dtype=int),
                  "amount_416A": rng.lognormal(8, 1, len(idd))}).pipe(_write, raw / "train_deposit_1.parquet")
    ido = case[(rng.random(n) < 0.12) & (wk > 53)]
    pl.DataFrame({"case_id": ido, "num_group1": np.zeros(len(ido), dtype=int),
                  "amtdebitincoming_4809443A": rng.lognormal(8, 1, len(ido))}).pipe(_write, raw / "train_other_1.parquet")
    print(f"synthetic data written to {raw}  ({n:,} cases, bad rate {y.mean():.2%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40_000)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data_synthetic")
    a = ap.parse_args()
    main(a.n, a.out)
