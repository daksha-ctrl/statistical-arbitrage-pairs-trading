# Statistical Arbitrage Pairs Trading

A rigorous daily-frequency statistical arbitrage research project using cointegration-based pair selection, per-fold EM Kalman dynamic hedge ratios, leg-level transaction costs, walk-forward validation, non-overlapping fold inference, HMM regime decomposition, random-pair null benchmarking, and a sector ETF residual mean-reversion benchmark.

## Key Finding

This project finds that daily-frequency cointegration pairs trading on liquid US equities is not a universal alpha source, but it remains conditionally useful when combined with disciplined pair selection and regime awareness.

Under the primary specification — per-fold EM Kalman hedge ratios, FDR q ≤ 0.20, leg-level transaction costs, and walk-forward validation — the strategy achieves a mean Sharpe of 0.68 at 5 bps across 20 active folds. On non-overlapping folds, the 5 bps result is suggestive but not conventionally significant, with mean Sharpe of 0.80 and p = 0.091. Gross-of-cost performance is significant at the 5% level.

The clearest result is regime-dependent: at 5 bps, the strategy achieves a mean Sharpe of 0.82 when the market enters the test window in a low-volatility HMM state, versus approximately zero in high-volatility states. FDR-selected pairs also outperform top-K fallback pairs, and the selected portfolio ranks at the 62nd percentile of an EM-fit random-pair null benchmark.

The project does not present a deployable live trading system. It presents a conservative quant research pipeline and a clear empirical conclusion: classical pair-level cointegration signals are sparse and regime-sensitive, while broader sector ETF residual mean reversion provides a more tradeable complementary signal.

## Overview

This project develops and evaluates a cointegration-based pairs trading framework on liquid US equities. The goal is to test whether classical statistical arbitrage signals survive modern research standards: realistic cost accounting, walk-forward validation, multiple-testing control, ablation analysis, regime decomposition, and null benchmarking.

The framework:

- defines a falsifiable trading hypothesis;
- forms candidate pairs only within economically plausible sector groups;
- selects pairs using cointegration tests, economic filters, FDR control, and a transparent top-K fallback;
- estimates time-varying hedge ratios using per-fold EM-estimated Kalman filters;
- constructs rolling z-score mean-reversion signals;
- chooses entry thresholds on validation data only;
- evaluates out-of-sample test performance using walk-forward splits;
- applies leg-level transaction costs rather than spread-level approximations;
- reports cross-fold inference on non-overlapping test windows;
- decomposes results by signal component, selection mode, and pre-specified HMM volatility regime;
- benchmarks selected portfolios against same-fold random-pair null distributions;
- compares pair-level cointegration with a sector ETF residual mean-reversion benchmark.

The project evolved through 26 research iterations. Each version addressed a specific methodology concern: per-fold parameter estimation, leg-level transaction cost accounting, FDR multiple-testing correction, top-K fallback tagging, NaN-safe threshold selection, non-overlapping fold inference, HMM regime decomposition, and EM-fit random-pair benchmarking.

## Research Questions

1. Does a cointegration-based pairs trading framework with dynamic hedge ratios produce risk-adjusted returns after realistic transaction costs?
2. Does pair selection via FDR control and economic filters extract signal beyond random pair selection?
3. What is the marginal contribution of each signal component: static OLS hedge ratios, Kalman dynamic hedge ratios, uncertainty filtering, and confidence scaling?
4. Are returns concentrated in market regimes that can be identified ex ante using an HMM volatility-state classifier?
5. How does pair-level cointegration compare to a sector ETF residual mean-reversion strategy on the same universe?

## Key Results

All results below use the primary specification unless otherwise stated:

- `kalman_full` signal
- FDR q ≤ 0.20
- leg-level transaction costs
- walk-forward train / validation / test design
- current-ticker liquid US equity universe
- daily adjusted close data

### Walk-Forward Performance

| Transaction cost | Mean Sharpe | Median Sharpe | Active folds |
|---:|---:|---:|---:|
| 0 bps | 0.99 | 0.88 | 20 |
| 5 bps | 0.68 | 0.63 | 20 |
| 10 bps | 0.39 | 0.37 | 20 |
| 25 bps | -0.41 | -0.43 | 20 |

Performance is positive at low and moderate transaction cost assumptions, but erodes quickly at high costs. This is consistent with a daily-frequency mean-reversion strategy where edge is present but thin.

### Non-Overlapping Fold Inference

Adjacent test folds overlap because 6-month test windows step forward every 3 months. For inference, the project reports t-tests on a non-overlapping every-other-fold subset.

| Transaction cost | Mean Sharpe | t-statistic | p-value | 95% CI |
|---:|---:|---:|---:|---:|
| 0 bps | 1.05 | 2.44 | 0.041 | [0.06, 2.04] |
| 5 bps | 0.80 | 1.92 | 0.091 | [-0.16, 1.76] |
| 25 bps | -0.33 | -0.86 | 0.420 | [-1.23, 0.56] |

Gross-of-cost performance is statistically significant at the 5% level. At 5 bps, the result is economically meaningful but statistically borderline.

### HMM Regime Decomposition

A 2-state Gaussian HMM is fit on S&P 500 daily returns using only information available before each test window. The regime label describes the dominant latent state in the 60 trading days immediately before test start.

| Regime at test start | Folds | Mean Sharpe at 5 bps |
|---|---:|---:|
| Low volatility | 16 | 0.82 |
| High volatility | 4 | -0.005 |

The strategy's performance is concentrated in low-volatility regimes. This is the cleanest empirical finding in the project and is consistent with the idea that pair-level relationships are more likely to break down during volatility spikes.

### Ablation Summary

| Specification | Mean Sharpe | Median Sharpe |
|---|---:|---:|
| `static_ols` | 0.51 | 0.24 |
| `kalman_base` | 0.55 | 0.49 |
| `kalman_filter` | 0.68 | 0.63 |
| `kalman_full` | 0.68 | 0.63 |

Dynamic Kalman hedge ratios outperform static OLS hedge ratios. The beta-uncertainty filter improves performance, while the final confidence scaling layer provides little additional benefit over `kalman_filter`.

### Selection Mode Decomposition

| Selection mode | Pair-folds | Mean Sharpe | Median Sharpe |
|---|---:|---:|---:|
| FDR-pass | 130 | 0.68 | 1.37 |
| Top-K fallback | 32 | 0.26 | 0.42 |

FDR-passing pairs outperform fallback pairs, especially in median Sharpe. This supports the view that the selection pipeline extracts signal, while the fallback rule mainly preserves fold coverage when the FDR screen is too sparse.

### Random-Pair Null Benchmark

For each fold, 100 random pair portfolios are sampled from the same universe and matched to the selected portfolio's composition. Random pairs are EM-fit on the same training window and backtested on the same test window.

Across 18 folds with well-defined null distributions, the selected portfolio sits at the 62nd percentile of the same-fold random-pair null on average. Under random selection, the expected percentile is 50. This indicates that the FDR and economic filter pipeline extracts signal beyond random cointegration-like pair selection, although the edge is heterogeneous and not present in every fold.

### Sector ETF Residual Benchmark

A parallel sector ETF residual mean-reversion strategy is included as a benchmark inspired by the residual-trading literature. Instead of selecting pair-level cointegration relationships, this benchmark trades stock residuals relative to sector ETF exposures.

The sector ETF residual strategy produces positive Sharpe in 14 of 20 active folds. Its performance differs from the pair-level cointegration strategy in several periods, suggesting that the two approaches capture related but not identical mean-reversion signals.

## Data

The project uses daily adjusted close prices from Yahoo Finance via `yfinance`.

The universe consists of 151 liquid US equities grouped into ten economic sectors:

- Banks
- Energy
- Healthcare
- Industrials
- Consumer Staples
- Consumer Discretionary
- Mega-Cap Technology
- Semiconductors
- Telecom and Media
- Payments and Credit

The framework also uses sector ETFs for stock-ETF candidate pairs and the residual benchmark:

- XLF
- XLE
- XLV
- XLI
- XLP
- XLY
- XLK
- SMH
- IYZ

The universe yields approximately 1,095 candidate pairs per fold. Pairs are formed only within sectors or between a stock and its sector ETF. Cross-sector pairs are excluded a priori as economically implausible.

## Walk-Forward Design

Each fold uses:

- 36 months of training data
- 6 months of validation data
- 6 months of test data
- 3-month step size between folds

The test period spans 2020–2025. Because adjacent test windows overlap, the project reports both:

1. descriptive performance across all active folds;
2. inferential statistics on non-overlapping fold subsets.

This avoids treating overlapping test windows as independent observations in cross-fold inference.

## Methodology

### Pair Selection

Candidate pairs are evaluated using the Engle-Granger cointegration test on training data only. Pairs must also pass economic and stability filters:

- positive hedge ratio;
- minimum R² threshold;
- half-life within a plausible mean-reversion range;
- subsample cointegration consistency;
- half-life stability;
- rolling beta stability.

The primary specification uses Benjamini-Hochberg FDR control at q ≤ 0.20, with q ≤ 0.10 reported as a stricter sensitivity. When fewer than three pairs survive FDR in a fold, the framework uses a tagged top-K fallback based on raw p-values and economic filters. FDR-pass and fallback results are reported separately.

### Per-Fold EM Kalman Hedge Ratios

For each selected pair and fold, the Kalman process and observation noise parameters are estimated using Expectation-Maximization on that fold's training data only. The resulting parameters are then used to run the Kalman filter forward through validation and test.

The Kalman cache is keyed by:

```python
(stock_a, stock_b, fold)
```

This prevents cross-fold parameter leakage.

### Signal Construction

The trading signal is based on the z-score of the Kalman spread:

```python
spread_t = log_price_A_t - alpha_t - beta_t * log_price_B_t
z_t = (spread_t - rolling_mean(spread_t)) / rolling_std(spread_t)
```

Positions are opened when the absolute z-score exceeds a validation-selected threshold and closed when the spread reverts toward zero, hits a holding-period limit, or breaches a stop-loss threshold.

### Validation

Entry thresholds are selected on validation data only from a pre-specified grid:

```python
[2.0, 2.5, 3.0, 3.5]
```

Threshold selection is NaN-safe, with secondary sorting on total return and trade count to avoid selecting dead thresholds with zero trades.

### Transaction Costs

The project uses leg-level transaction costs:

```python
cost_t = (abs(delta_weight_A_t) + abs(delta_weight_B_t)) * transaction_cost
```

This is more conservative than charging costs only on spread-position turnover, which understates trading cost when hedge ratios differ from one.

The primary tables report flat per-leg transaction costs of 0, 5, 10, and 25 bps. Roll spread-cost infrastructure is implemented but not enabled in the primary run.

### Risk Controls

The framework includes:

- maximum holding period;
- stop-loss on spread z-score;
- beta-uncertainty filter based on Kalman posterior variance;
- inverse-volatility portfolio weighting across selected pairs;
- capped number of final validation-approved pairs per fold.

## Project Structure

```text
statistical-arbitrage-pairs-trading/
  README.md
  requirements.txt
  .gitignore

  stat_arb_v26.py

  results_v26_full/
    tables/
      v26_full_fold_summary.csv
      v26_full_ablation_summary.csv
      v26_full_tc_summary_all_folds_fdr*.csv
      v26_full_tc_summary_nonoverlapping_fdr*.csv
      v26_full_cross_fold_inference_nonoverlapping_fdr*.csv
      v26_full_null_benchmark.csv
      v26_full_null_benchmark_summary.csv
      v26_full_hmm_regimes.csv
      v26_full_hmm_regime_summary.csv
      v26_full_sector_etf_residual_results.csv
      v26_full_kalman_em_parameters_by_fold.csv
      v26_full_pair_metrics.csv

    figures/
      *.png
```

## Reproducing the Results

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python stat_arb_v26.py
```

The full 21-fold run takes approximately 60–90 minutes on a laptop.

Key configuration flags:

```python
USE_EM_KALMAN = True
RUN_ABLATION = True
USE_FDR_GRID = True
RUN_NULL_BENCHMARK = True
USE_HMM_REGIMES = True
RUN_SECTOR_ETF_RESIDUAL_STRATEGY = True
USE_ROLL_SPREAD_COST = False
MAX_FOLDS_OVERRIDE = None
```

For smoke testing, set:

```python
MAX_FOLDS_OVERRIDE = 4
```

## Main Limitations

This is a research pipeline, not a live trading system. Important limitations remain:

- Yahoo Finance data is not institutional-grade.
- The universe is based on current tickers, so survivorship bias is present.
- Borrow costs, short-sale constraints, slippage, and market impact are not modeled.
- Daily data may miss intraday execution and liquidity dynamics.
- The HMM regime classifier is intentionally simple.
- The sector ETF residual benchmark is a research comparison, not a fully optimized production strategy.
- The top-K fallback improves fold coverage but adds weaker selections; this is why fallback results are tagged and decomposed separately.

## Future Extensions

- Enable Roll spread-based dynamic transaction costs in the primary run.
- Add borrow-cost and short-availability assumptions.
- Test signal-strength-based position sizing using z-score magnitude and estimated half-life.
- Extend the HMM to VIX-conditioned or 3-state volatility regimes.
- Test weekly horizons to reduce turnover and transaction cost sensitivity.
- Expand to European equities or ADR universes.
- Implement a live paper-trading version with production-quality data and execution assumptions.

## References

- Avellaneda, M. and Lee, J. H. (2010). *Statistical Arbitrage in the U.S. Equities Market.*
- Benjamini, Y. and Hochberg, Y. (1995). *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing.*
- Gatev, E., Goetzmann, W. N., and Rouwenhorst, K. G. (2006). *Pairs Trading: Performance of a Relative-Value Arbitrage Rule.*
- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis.*

## Bottom Line

The main result is not "this strategy prints money." The result is more useful: a classical cointegration strategy becomes sparse and regime-dependent under conservative testing, but selected pairs still outperform random-pair nulls, dynamic Kalman hedging improves on static OLS, and low-volatility regimes contain most of the remaining edge. The project demonstrates a rigorous research process for separating fragile backtest performance from interpretable statistical arbitrage signal.

- define a trading hypothesis;
- select economically plausible pairs;
- estimate hedge relationships;
- construct mean-reversion signals;
- backtest out of sample;
- include transaction costs and risk controls;
- diagnose failure modes;
- avoid overfitting through train / validation / test design.

The project began with a naive cointegration screen and progressed through increasingly disciplined versions: economic grouping, multi-pair portfolio testing, rolling hedge ratios, max-holding and stop-loss rules, train / validation / test splits, validation-based filtering, and a ridge basket extension.
