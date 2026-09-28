# Credit Risk Scorecard & Approval Strategy under Distribution Shift

An end-to-end credit-risk project on the **Kaggle Home Credit – Credit Risk Model Stability (2024)** data:
from raw multi-table data to an interpretable scorecard, gradient-boosting challengers, an approval strategy and a
monitoring design — evaluated out-of-time through a risk deterioration and the COVID-19 population shift.

The focus is **strategy deployment rather than leaderboard score**: every modelling choice is judged by whether it
survives time, can be explained, and can be operated.

---

## Key findings

*Full Kaggle training data (≈1.53M applications, 92 weeks). Figures are from the development run and may differ
slightly after re-running.*

| | Train (wk 0–40) | Valid (41–50) | OOT1 (51–62) | OOT2 (63–91) |
|---|---|---|---|---|
| Applications | 779k | 251k | 238k | 259k |
| Bad rate | 2.90% | 3.68% | 4.27% | 2.34% |
| AUC — scorecard (18 features) | 0.788 | 0.789 | 0.782 | 0.815 |
| AUC — LightGBM | 0.857 | 0.829 | 0.822 | 0.842 |

**Data**
- **Tax-registry vendor switch.** Three tax tables take over from each other (c → a → b). Customers present at two
  vendors during hand-overs revealed that *a* duplicates *c* exactly (ratio 1.00, a system migration) and *b = 8.10 × a*
  (a unit change). De-duplication plus an estimated fixed factor produced vendor-agnostic features — no distributional
  assumption and no out-of-time data needed.
- **Hidden time proxies.** Integer calendar years (e.g. year of the worst delinquency) drift upward with application
  time; converting them to *years before the decision* kept their signal and removed the drift.

**Models**
- The scorecard ranks stably through both shocks; LightGBM adds **+0.04 AUC** in-period, narrowing to **+0.027** in OOT2.
- **Districts were memorised, not learned:** three district fields took 27% of LightGBM's gain with IV ≈ 0.03–0.05;
  removing them *raised* validation AUC (0.825 → 0.832). The compliance-driven exclusion was also the right technical one.
- **Regularisation experiment:** shrinking LightGBM from 32 to 8 leaves left validation AUC flat (0.828–0.829) while
  the train–valid gap halved (0.057 → 0.029) — the gap was in-sample optimism, not harmful overfitting.
- **Scorecard A vs B** (keep vs drop features whose distribution drifts while their risk relationship is stable):
  A wins on the competition stability metric *and* on bad rate at equal approval rates, but its OOT2 score PSI reaches
  0.145 (B: 0.088). CSI traces the drift to three credit-bureau fields whose missingness changed.
- **Calibration:** OOT1 under-predicts risk by ~45% (macro deterioration is invisible in applicant features), while
  OOT2 calibration recovers (the post-COVID population shift *is* visible). Ranking is stable; the score-to-PD mapping is not.

**Strategy**
- Decision matrix (age ≥ 22 → grades A–E → auto-approve / review / decline): **~75% direct approval at a 1.65% bad
  rate vs 3.68% for all booked loans (−55%)**, ~15% routed to manual review.
- **Swap-set analysis:** at 80% approval, LightGBM would approve **15–18% fewer bad loans** in every period.
- **Challenger roll-out:** (1) LightGBM triages the manual-review queue; (2) a dual-score matrix cuts auto-approved bads
  per 10k applications by **~21% (valid) and ~14% (OOT1)** at equal volume.
- **Cut-off maintenance:** a rolling cut-off with a risk floor holds volume when the population improves and holds risk
  when it deteriorates. A label-triggered tightening layer fired *after* the deterioration had passed — **a lagging
  signal is pro-cyclical**; leading indicators are needed before adopting it.
- **Monitoring:** weekly score PSI never crossed 0.10, yet approval rates under a fixed cut-off drifted 5–6 pp.
  Business metrics near the cut-off (grade D+E share, approval rate) must be the primary monitor.

**Compliance choices:** gender and proxy features (Cramér's V / correlation > 0.5) excluded; age used only as an
admission rule; ZIP codes dropped; districts allowed in GBDT only and ultimately removed; the application's own pricing
fields (`eir_270L`, `interestrate_311L`) excluded — under risk-based pricing they are set after the risk decision.

---

## Methodology

**Sample design** — time-based throughout: train (0–40), internal validation (41–50) for every selection decision,
two out-of-time windows that no step touches until notebook 08. Decision rules for model choices were fixed before
seeing the results they decide on.

**Feature selection funnel** — compliance → data quality → IV → stability. Stability distinguishes
*distribution drift* (PSI) from *relationship drift* (WOE-pattern consistency between periods): only the latter is
dropped; the former is kept and monitored. Adversarial validation cross-checks the result.

**Scorecard** — monotonic optimal binning (optbinning), WOE, forward/backward stepwise logistic regression with sign,
VIF and validation-AUC gates, points scaling (600 at 50:1 odds, PDO 20).

**Challengers** — LightGBM and XGBoost with matched complexity; LightGBM chosen for equal accuracy at ~2× speed.

**Evaluation** — AUC / KS by period, weekly Gini, the competition stability metric, score PSI, CSI, decile calibration.

---

## Repository structure

```
├── notebooks/
│   ├── 00_run_all.ipynb                 runs 01–10 in order, one fresh kernel per notebook
│   ├── 01_eda.ipynb                     volume & bad rate over time, table sizes
│   ├── 02_aggregation.ipynb             memory-aware aggregation to one row per application
│   ├── 03_data_quality.ipynb            sample split, source coverage, quality rules, age profile
│   ├── 04_features_iv.ipynb             gender-proxy check, unified tax features, model table, IV
│   ├── 05_stability_correlation.ipynb   stability rules, adversarial validation, correlation pruning
│   ├── 06_scorecard.ipynb               time-proxy check, optimal binning, stepwise LR, scorecards A/B
│   ├── 07_gbdt_challengers.ipynb        LightGBM / XGBoost + categorical & regularisation experiments
│   ├── 08_oot_evaluation.ipynb          out-of-time evaluation, A/B decision, calibration, CSI
│   ├── 09_strategy.ipynb                admission, grades, swap-set, cut-off maintenance, monitoring
│   └── 10_strategy_iterations.ipynb     review-queue triage, dual-score matrix, roll-out plan
├── src/hcrisk/                          shared code (config, IO, features, metrics, scorecard, strategy)
├── tools/
│   ├── run_pipeline.sh                  execute notebooks in order
│   └── make_synthetic_data.py           synthetic data with the same layout, for a smoke test
├── reports/figures/                     key charts written by the notebooks
└── environment.yml
```

---

## How to run

**1. Environment**

```bash
conda env create -f environment.yml
conda activate homecredit
python -m ipykernel install --user --name homecredit
```

**2. Data** — accept the competition rules on Kaggle, then:

```bash
kaggle competitions download -c home-credit-credit-risk-model-stability -p data/
unzip data/home-credit-credit-risk-model-stability.zip -d data/home-credit-credit-risk-model-stability
```

Expected: `data/home-credit-credit-risk-model-stability/parquet_files/train/*.parquet`. To keep the data elsewhere,
set `HCRISK_DATA_DIR` to the folder containing `home-credit-credit-risk-model-stability/`. All intermediate outputs go
to `features/` and `results/` next to the data, never into the repository.

**3. Run** — three equivalent options:

- **One click:** open `notebooks/00_run_all.ipynb` and run it. Each notebook is executed in its own fresh kernel
  (memory is released between steps) and its outputs are saved back into the notebook. Set `FROM` / `TO` to re-run
  part of the pipeline.
- **Command line:**
  ```bash
  bash tools/run_pipeline.sh          # all notebooks, outputs saved in place
  bash tools/run_pipeline.sh 06 10    # a range
  ```
- **Step by step:** open 01 → 10 in order. Shut down each kernel before starting the next one.

All notebooks use the kernel `homecredit` created in step 1.

Runtime on a laptop (8 GB RAM) is roughly 1.5–2 hours, most of it in notebooks 02 and 07. Peak memory stays within
8 GB thanks to lazy / streaming processing.

**Smoke test without Kaggle data**

```bash
python tools/make_synthetic_data.py                       # writes data_synthetic/
HCRISK_DATA_DIR=data_synthetic bash tools/run_pipeline.sh
```

The synthetic data reproduces the structural quirks (vendor switch with duplicates and an 8.1× unit change, calendar
fields, a field added mid-way, time proxies, gender and district fields) so every code path is exercised. Its numbers
are meaningless.

---

## Limitations

- **Booked loans only.** Outcomes of declined applicants are unobserved (no reject inference); "approval rate" is
  relative to historically approved applicants.
- **No leading indicators.** Only the final default flag is available, so early-warning and tightening rules rely on
  lagging labels.
- **Unknown field semantics.** Column meanings are masked; conclusions such as "record-write date" are inferred from data.
- **Year-level event fields.** Fields that store only the year of an event are converted to *years before the decision*
  using the fractional decision date (the best unbiased estimate when the event month is unknown). Because all
  applications in the same week share the same fractional offset, equal-frequency PSI on these fields is inflated;
  their WOE patterns remain stable (consistency 0.92–0.95). Pairing each year with its month field would remove this.
- **No profitability layer.** Cut-offs are set on volume and bad rate; pricing, LGD and funding cost are not modelled.

## Data licence

The data belongs to Home Credit and is provided under the Kaggle competition rules; it is not included in this repository.
