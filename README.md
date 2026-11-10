# To The Moon 2: Pair Relative-Value Signal Research

QFGB-890T / FNGB-74DD, Fordham, Fall 2026. Team: **Heet Mehta** (CTO), **Rishi Chauhan** (COO).

## The question
BTC, ETH, SOL, XRP and DOGE perpetuals usually move together. When one drifts away from another for a short time, does that gap close? Can we forecast the gap's next move well enough to trade it after costs?

## Data
Course Binance USDT-M perpetual data, 1-minute bars (`level1_1min`, `trade_1min`), 1 Jan 2023 – 30 Nov 2025, 5 coins.
- **Development:** Jan 2023 – Jul 2025.
- **Clean holdout:** Aug – Nov 2025. No script has touched it yet.

## How to run (Windows PowerShell)
```
pip install -r requirements.txt
python step1_inventory.py        # lists what raw data exists
python step2_build_panel.py      # raw .csv.gz -> data/<COIN>_1min.pkl   (~4 min; set ROOT at the top first)
python step3_combine_panel.py    # -> work/panel.pkl
python step4_correlations.py     # milestone plots 01-04 (correlations, high-vol day, index)
python step5_features.py         # cointegration scan (plot 05) + 15-min pair features
python step6_feature_survey.py   # feature stats, distributions, cross-correlation, VIF, IC by year (plots 06-08)
python step7_model.py            # walk-forward model, baselines, improvement loop, cost staging (~2 min)
python step8_plots.py            # plots 09-10
python step9_placebo.py          # sign-flip placebo test
```
Plots are saved to `figs/`, and tables and intermediate files to `work/`.

## Milestone 1 results (development period, walk-forward out-of-sample)
| Model | OOS IC (next 1h) | Gross Sharpe | Net Sharpe (maker / taker) |
|---|---|---|---|
| Best single feature (`last_bar`) | 0.91% | 1.77 | -34 / -59 |
| Final combined signal + trading layer | 1.77% | 1.49 | -1.4 / -5.1 |
| Textbook rule (\|z\|>2, exit at 0) | – | -0.96 | – / -1.29 |

- **The combined signal roughly doubles the best single feature's forecast quality.** It is significant: t = 5.5 on non-overlapping hourly samples. It earns about +19%/yr before costs, against a placebo gross Sharpe of 0.0 ± 0.6.
- **Trading costs still exceed the gross edge.** Moving to longer horizons and lower turnover is the next step.
- **Price-level cointegration is rare in this universe.** Only 1.5–3% of 30-day windows pass a 5% Engle-Granger test, which is below chance. So the project uses beta-hedged spreads, with cointegration strength used as a conditioning feature.

## AI use
Generative AI (Anthropic's Claude) helped write the Python pipeline, the plots and the first draft of the milestone write-up, as the course AI policy allows. The team reviewed and ran all code and checked the results.
