# Statistical Arbitrage Pairs Trading

Cointegration-based statistical arbitrage research project with rolling hedge ratios, validation filtering, transaction costs, and a ridge regression basket extension.

## Overview

This project develops and evaluates a simple statistical arbitrage framework for equity pairs trading. The goal is not to claim a live trading strategy, but to build a realistic quant research workflow:

- define a trading hypothesis;
- select economically plausible pairs;
- estimate hedge relationships;
- construct mean-reversion signals;
- backtest out of sample;
- include transaction costs and risk controls;
- diagnose failure modes;
- avoid overfitting through train / validation / test design.

The project began with a naive cointegration screen and progressed through increasingly disciplined versions: economic grouping, multi-pair portfolio testing, rolling hedge ratios, max-holding and stop-loss rules, train / validation / test splits, validation-based filtering, and a ridge basket extension.

## Main Interpretation

The project is not presented as a profitable trading system. The main finding is that apparent mean-reversion edges weakened as the research design became more rigorous.

Exploratory versions showed modest positive performance after adding rolling hedge ratios and risk controls. However, once threshold selection and final testing were separated through a train / validation / test split, the strategy no longer produced robust positive returns.

This is itself an important result. It shows that cointegration-based pairs trading can look promising under flexible backtests but is highly sensitive to pair selection, regime changes, hedge-ratio stability, transaction costs, and validation design.

## Key Results

The main validation-filtered cointegration strategy produced a final held-out test return of **-1.07%**, with a net Sharpe of **-0.03** and max drawdown of **-9.53%**. This suggests that validation filtering reduced losses relative to the unfiltered portfolio but did not produce a robust positive edge.

The ridge basket extension performed worse at the portfolio level, with a net return of **-10.27%** and Sharpe of **-0.61**. However, the WFC financials basket produced a positive out-of-sample return of **+11.5%** with a Sharpe of **0.53**.

## Research Question

Can an economically constrained cointegration-based pairs trading strategy generate out-of-sample mean-reversion signals across liquid equities?

More specifically:

- Do cointegrated pairs selected in a training period remain tradable out of sample?
- Does adding economic grouping reduce spurious pair selection?
- Do rolling hedge ratios and risk controls improve performance?
- Does validation-based filtering improve final test results?
- Can ridge regression construct more stable synthetic spreads using baskets of related assets?

## Data

The project uses daily adjusted close prices from Yahoo Finance via `yfinance`.

The initial universe contains 25 liquid U.S. equities grouped into broad economic categories:

- Banks
- Energy
- Semiconductors
- Consumer
- Technology

The universe is intentionally small and interpretable. This makes the research process easier to audit and reduces the temptation to search across hundreds of pairs until something looks profitable.

### Data limitations

- Yahoo Finance data is not institutional-grade.
- The universe uses current tickers, so survivorship bias is possible.
- Raw price data is not included in the repository, but can be downloaded by running the scripts.
- Borrow costs, short-sale constraints, slippage, and market impact are not modelled.

## Project Structure

```text
statistical-arbitrage-pairs-trading/
  README.md
  requirements.txt
  .gitignore

  scripts/
    stat_arb_main.py
    ridge_basket_extension.py

  results/
    figures/
    tables/