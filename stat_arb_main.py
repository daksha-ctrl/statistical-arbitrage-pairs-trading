#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 18:49:54 2026

@author: dakshayanipinninti
"""

"""
Statistical Arbitrage Project 
Train / Validation / Test + Half-Life Filter

Author: Dakshayani Pinninti

Purpose:
    Make the stat-arb research design more credible by:
    - selecting pairs only on training data
    - estimating mean-reversion half-life
    - choosing thresholds on validation data
    - evaluating final performance on a held-out test period
    - avoiding threshold choice based on final test performance
"""

from itertools import combinations
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIG
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_DIR / "results_v6"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

for folder in [DATA_DIR, RAW_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


SECTOR_GROUPS = {
    "Banks": ["JPM", "BAC", "C", "WFC", "GS", "MS"],
    "Energy": ["XOM", "CVX", "COP", "SLB"],
    "Semiconductors": ["NVDA", "AMD", "INTC", "QCOM", "AVGO"],
    "Consumer": ["KO", "PEP", "PG", "WMT", "COST"],
    "Tech": ["MSFT", "AAPL", "GOOGL", "META", "AMZN"]
}

TICKERS = sorted(set([ticker for group in SECTOR_GROUPS.values() for ticker in group]))

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

# V5 split
TRAIN_START = "2015-01-01"
TRAIN_END = "2019-12-31"

VALID_START = "2020-01-01"
VALID_END = "2021-12-31"

TEST_START = "2022-01-01"
TEST_END = "2025-12-31"

COINTEGRATION_PVALUE_THRESHOLD = 0.10
MIN_R_SQUARED = 0.50

MIN_HALF_LIFE = 5
MAX_HALF_LIFE = 60

MAX_PAIRS_TO_TRADE = 10

ROLLING_HEDGE_WINDOW = 252
ROLLING_Z_WINDOW = 60

THRESHOLDS_TO_TEST = [1.5, 2.0, 2.5]
EXIT_THRESHOLD = 0.0

MAX_HOLDING_DAYS = 30
STOP_LOSS_Z = 4.0

TRANSACTION_COST_BPS = 5
TRANSACTION_COST = TRANSACTION_COST_BPS / 10_000

# Validation filters
MIN_VALIDATION_TRADES = 5
MAX_VALIDATION_DRAWDOWN = -0.20  # exclude if worse than -35%
MIN_VALIDATION_ROLLING_R2 = 0.30


# =============================================================================
# 2. DATA
# =============================================================================

def download_adjusted_prices(tickers, start, end):
    print("Downloading adjusted price data...")

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column"
    )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = tickers

    missing_share = prices.isna().mean()
    keep_cols = missing_share[missing_share < 0.05].index.tolist()
    prices = prices[keep_cols]

    prices = prices.ffill().dropna()
    prices.index = pd.to_datetime(prices.index)

    return prices


prices = download_adjusted_prices(TICKERS, START_DATE, END_DATE)
prices.to_csv(RAW_DIR / "adjusted_close_prices.csv")

log_prices = np.log(prices)

train_prices = prices.loc[TRAIN_START:TRAIN_END].copy()
valid_prices = prices.loc[VALID_START:VALID_END].copy()
test_prices = prices.loc[TEST_START:TEST_END].copy()

log_train_prices = log_prices.loc[TRAIN_START:TRAIN_END].copy()
log_valid_prices = log_prices.loc[VALID_START:VALID_END].copy()
log_test_prices = log_prices.loc[TEST_START:TEST_END].copy()

print("\nData shape:", prices.shape)
print("Training:", train_prices.index.min().date(), "to", train_prices.index.max().date())
print("Validation:", valid_prices.index.min().date(), "to", valid_prices.index.max().date())
print("Test:", test_prices.index.min().date(), "to", test_prices.index.max().date())


# =============================================================================
# 3. CORE FUNCTIONS
# =============================================================================

def estimate_hedge_ratio(y, x):
    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const).fit()
    alpha = model.params["const"]
    beta = model.params[x.name]
    return alpha, beta, model


def compute_static_spread(log_price_df, stock_a, stock_b, alpha, beta):
    return log_price_df[stock_a] - alpha - beta * log_price_df[stock_b]


def estimate_half_life(spread):
    """
    Estimate mean-reversion half-life using:
        delta_spread_t = a + b * spread_{t-1} + error_t

    Half-life:
        -ln(2) / b

    Valid only when b < 0.
    """

    spread = spread.dropna()
    lagged_spread = spread.shift(1).dropna()
    delta_spread = spread.diff().dropna()

    aligned = pd.concat([delta_spread, lagged_spread], axis=1).dropna()
    aligned.columns = ["delta_spread", "lagged_spread"]

    if len(aligned) < 30:
        return np.nan

    x = sm.add_constant(aligned["lagged_spread"])
    model = sm.OLS(aligned["delta_spread"], x).fit()

    beta = model.params["lagged_spread"]

    if beta >= 0:
        return np.nan

    half_life = -np.log(2) / beta

    return half_life


def find_cointegrated_pairs_by_group(log_price_df, sector_groups):
    results = []

    for group_name, tickers in sector_groups.items():
        available = [t for t in tickers if t in log_price_df.columns]

        for stock_a, stock_b in combinations(available, 2):
            y = log_price_df[stock_a]
            x = log_price_df[stock_b]

            try:
                test_stat, pvalue, critical_values = coint(y, x)
                alpha, beta, model = estimate_hedge_ratio(y, x)
                spread = compute_static_spread(log_price_df, stock_a, stock_b, alpha, beta)
                half_life = estimate_half_life(spread)

                results.append({
                    "group": group_name,
                    "stock_a": stock_a,
                    "stock_b": stock_b,
                    "pvalue": pvalue,
                    "test_stat": test_stat,
                    "alpha": alpha,
                    "beta": beta,
                    "r_squared": model.rsquared,
                    "half_life": half_life,
                    "crit_1pct": critical_values[0],
                    "crit_5pct": critical_values[1],
                    "crit_10pct": critical_values[2]
                })

            except Exception as e:
                print(f"Skipping {stock_a}-{stock_b}: {e}")

    results_df = pd.DataFrame(results).sort_values("pvalue").reset_index(drop=True)

    selected_df = results_df[
        (results_df["pvalue"] < COINTEGRATION_PVALUE_THRESHOLD) &
        (results_df["beta"] > 0) &
        (results_df["r_squared"] > MIN_R_SQUARED) &
        (results_df["half_life"] >= MIN_HALF_LIFE) &
        (results_df["half_life"] <= MAX_HALF_LIFE)
    ].copy()

    return results_df, selected_df


def rolling_hedge_ratio(log_price_df, stock_a, stock_b, window=252):
    y = log_price_df[stock_a]
    x = log_price_df[stock_b]

    alphas = pd.Series(index=log_price_df.index, dtype=float)
    betas = pd.Series(index=log_price_df.index, dtype=float)
    r2s = pd.Series(index=log_price_df.index, dtype=float)

    for i in range(window, len(log_price_df)):
        y_window = y.iloc[i-window:i]
        x_window = x.iloc[i-window:i]

        x_const = sm.add_constant(x_window)
        model = sm.OLS(y_window, x_const).fit()

        alphas.iloc[i] = model.params["const"]
        betas.iloc[i] = model.params[stock_b]
        r2s.iloc[i] = model.rsquared

    return alphas, betas, r2s


def compute_rolling_spread(log_price_df, stock_a, stock_b, alphas, betas):
    return log_price_df[stock_a] - alphas - betas * log_price_df[stock_b]


def compute_zscore(spread, window=60):
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    return (spread - rolling_mean) / rolling_std


def generate_positions_with_risk_rules(
    zscore,
    entry_threshold=2.0,
    exit_threshold=0.0,
    max_holding_days=30,
    stop_loss_z=4.0
):
    positions = pd.Series(index=zscore.index, dtype=float)
    holding_days = pd.Series(index=zscore.index, dtype=float)
    exit_reasons = pd.Series(index=zscore.index, dtype=object)

    current_position = 0
    current_holding_days = 0

    for date, z in zscore.items():
        exit_reason = None

        if np.isnan(z):
            positions.loc[date] = 0
            holding_days.loc[date] = 0
            exit_reasons.loc[date] = None
            continue

        if current_position == 0:
            current_holding_days = 0

            if z < -entry_threshold:
                current_position = 1
                current_holding_days = 1

            elif z > entry_threshold:
                current_position = -1
                current_holding_days = 1

        else:
            current_holding_days += 1

            if current_position == 1 and z >= exit_threshold:
                current_position = 0
                exit_reason = "mean_reversion"

            elif current_position == -1 and z <= -exit_threshold:
                current_position = 0
                exit_reason = "mean_reversion"

            elif current_holding_days >= max_holding_days:
                current_position = 0
                exit_reason = "max_holding"

            elif stop_loss_z is not None:
                if current_position == 1 and z < -stop_loss_z:
                    current_position = 0
                    exit_reason = "stop_loss"
                elif current_position == -1 and z > stop_loss_z:
                    current_position = 0
                    exit_reason = "stop_loss"

            if current_position == 0:
                current_holding_days = 0

        positions.loc[date] = current_position
        holding_days.loc[date] = current_holding_days
        exit_reasons.loc[date] = exit_reason

    return positions, holding_days, exit_reasons


def calculate_strategy_returns(price_df, stock_a, stock_b, positions, betas):
    returns = price_df[[stock_a, stock_b]].pct_change().dropna()

    aligned_positions = positions.reindex(returns.index).fillna(0)
    aligned_betas = betas.reindex(returns.index).ffill()

    lagged_positions = aligned_positions.shift(1).fillna(0)
    lagged_betas = aligned_betas.shift(1).ffill()

    gross_exposure = 1 + lagged_betas.abs()

    weight_a = lagged_positions * (1 / gross_exposure)
    weight_b = lagged_positions * (-lagged_betas / gross_exposure)

    gross_returns = weight_a * returns[stock_a] + weight_b * returns[stock_b]
    gross_returns.name = "gross_strategy_return"

    return gross_returns, lagged_positions, lagged_betas


def apply_transaction_costs(gross_returns, positions, transaction_cost=0.0005):
    aligned_positions = positions.reindex(gross_returns.index).fillna(0)
    position_change = aligned_positions.diff().abs().fillna(0)

    costs = position_change * transaction_cost
    costs.name = "transaction_cost"

    net_returns = gross_returns - costs
    net_returns.name = "net_strategy_return"

    return net_returns, costs


def performance_metrics(returns, periods_per_year=252):
    returns = returns.dropna()

    if len(returns) == 0:
        return {
            "Total Return": np.nan,
            "Annual Return": np.nan,
            "Annual Volatility": np.nan,
            "Sharpe Ratio": np.nan,
            "Max Drawdown": np.nan,
            "Hit Rate": np.nan,
            "Number of Observations": 0
        }

    cumulative = (1 + returns).cumprod()

    total_return = cumulative.iloc[-1] - 1
    annual_return = returns.mean() * periods_per_year
    annual_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = annual_return / annual_vol if annual_vol != 0 else np.nan

    drawdown = cumulative / cumulative.cummax() - 1
    max_drawdown = drawdown.min()

    hit_rate = (returns > 0).mean()

    return {
        "Total Return": total_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
        "Hit Rate": hit_rate,
        "Number of Observations": len(returns)
    }


def summarize_trades(positions, exit_reasons):
    pos = positions.fillna(0)

    entries = ((pos.shift(1) == 0) & (pos != 0)).sum()
    exits = ((pos.shift(1) != 0) & (pos == 0)).sum()

    exit_counts = exit_reasons.dropna().value_counts().to_dict()

    return {
        "Number of Entries": int(entries),
        "Number of Exits": int(exits),
        "Share of Days Invested": float((pos != 0).sum() / len(pos)),
        "Mean-Reversion Exits": int(exit_counts.get("mean_reversion", 0)),
        "Max-Holding Exits": int(exit_counts.get("max_holding", 0)),
        "Stop-Loss Exits": int(exit_counts.get("stop_loss", 0))
    }


def backtest_pair(
    stock_a,
    stock_b,
    group,
    full_log_prices,
    eval_prices,
    eval_start,
    eval_end,
    entry_threshold
):
    """
    Backtest a pair over a specified evaluation window.
    Rolling hedge ratio uses all available prior data but strategy evaluation
    is restricted to eval_start:eval_end.
    """

    alphas, betas, rolling_r2 = rolling_hedge_ratio(
        full_log_prices[[stock_a, stock_b]],
        stock_a,
        stock_b,
        window=ROLLING_HEDGE_WINDOW
    )

    spread = compute_rolling_spread(
        full_log_prices[[stock_a, stock_b]],
        stock_a,
        stock_b,
        alphas,
        betas
    )

    zscore = compute_zscore(spread, window=ROLLING_Z_WINDOW)

    eval_zscore = zscore.loc[eval_start:eval_end].copy()
    eval_betas = betas.loc[eval_start:eval_end].copy()
    eval_r2 = rolling_r2.loc[eval_start:eval_end].copy()

    positions, holding_days, exit_reasons = generate_positions_with_risk_rules(
        eval_zscore,
        entry_threshold=entry_threshold,
        exit_threshold=EXIT_THRESHOLD,
        max_holding_days=MAX_HOLDING_DAYS,
        stop_loss_z=STOP_LOSS_Z
    )

    gross_returns, lagged_positions, lagged_betas = calculate_strategy_returns(
        eval_prices,
        stock_a,
        stock_b,
        positions,
        eval_betas
    )

    net_returns, costs = apply_transaction_costs(
        gross_returns,
        lagged_positions,
        transaction_cost=TRANSACTION_COST
    )

    gross_stats = performance_metrics(gross_returns)
    net_stats = performance_metrics(net_returns)
    trade_stats = summarize_trades(lagged_positions, exit_reasons)

    result = {
        "group": group,
        "stock_a": stock_a,
        "stock_b": stock_b,
        "entry_threshold": entry_threshold,

        "avg_rolling_beta": eval_betas.mean(),
        "std_rolling_beta": eval_betas.std(),
        "avg_rolling_r2": eval_r2.mean(),

        "gross_total_return": gross_stats["Total Return"],
        "gross_annual_return": gross_stats["Annual Return"],
        "gross_annual_volatility": gross_stats["Annual Volatility"],
        "gross_sharpe": gross_stats["Sharpe Ratio"],
        "gross_max_drawdown": gross_stats["Max Drawdown"],
        "gross_hit_rate": gross_stats["Hit Rate"],

        "net_total_return": net_stats["Total Return"],
        "net_annual_return": net_stats["Annual Return"],
        "net_annual_volatility": net_stats["Annual Volatility"],
        "net_sharpe": net_stats["Sharpe Ratio"],
        "net_max_drawdown": net_stats["Max Drawdown"],
        "net_hit_rate": net_stats["Hit Rate"],

        "num_entries": trade_stats["Number of Entries"],
        "num_exits": trade_stats["Number of Exits"],
        "share_days_invested": trade_stats["Share of Days Invested"],
        "mean_reversion_exits": trade_stats["Mean-Reversion Exits"],
        "max_holding_exits": trade_stats["Max-Holding Exits"],
        "stop_loss_exits": trade_stats["Stop-Loss Exits"]
    }

    pair_returns = pd.DataFrame({
        f"{stock_a}_{stock_b}_gross": gross_returns,
        f"{stock_a}_{stock_b}_net": net_returns,
        f"{stock_a}_{stock_b}_position": lagged_positions
    })

    return result, pair_returns


def portfolio_backtest(
    selected_pairs_to_trade,
    full_log_prices,
    eval_prices,
    eval_start,
    eval_end,
    threshold
):
    pair_metrics = []
    all_gross_returns = []
    all_net_returns = []

    for _, row in selected_pairs_to_trade.iterrows():
        stock_a = row["stock_a"]
        stock_b = row["stock_b"]
        group = row["group"]

        result, pair_returns = backtest_pair(
            stock_a=stock_a,
            stock_b=stock_b,
            group=group,
            full_log_prices=full_log_prices,
            eval_prices=eval_prices,
            eval_start=eval_start,
            eval_end=eval_end,
            entry_threshold=threshold
        )

        result["selection_pvalue"] = row["pvalue"]
        result["selection_beta"] = row["beta"]
        result["selection_r_squared"] = row["r_squared"]
        result["selection_half_life"] = row["half_life"]

        pair_metrics.append(result)

        all_gross_returns.append(pair_returns[f"{stock_a}_{stock_b}_gross"])
        all_net_returns.append(pair_returns[f"{stock_a}_{stock_b}_net"])

    pair_metrics_df = pd.DataFrame(pair_metrics)

    gross_returns_df = pd.concat(all_gross_returns, axis=1).dropna(how="all")
    net_returns_df = pd.concat(all_net_returns, axis=1).dropna(how="all")

    gross_portfolio_returns = gross_returns_df.fillna(0).mean(axis=1)
    net_portfolio_returns = net_returns_df.fillna(0).mean(axis=1)

    gross_stats = performance_metrics(gross_portfolio_returns)
    net_stats = performance_metrics(net_portfolio_returns)

    portfolio_result = {
        "entry_threshold": threshold,
        "num_pairs": len(selected_pairs_to_trade),

        "gross_total_return": gross_stats["Total Return"],
        "gross_annual_return": gross_stats["Annual Return"],
        "gross_annual_volatility": gross_stats["Annual Volatility"],
        "gross_sharpe": gross_stats["Sharpe Ratio"],
        "gross_max_drawdown": gross_stats["Max Drawdown"],
        "gross_hit_rate": gross_stats["Hit Rate"],

        "net_total_return": net_stats["Total Return"],
        "net_annual_return": net_stats["Annual Return"],
        "net_annual_volatility": net_stats["Annual Volatility"],
        "net_sharpe": net_stats["Sharpe Ratio"],
        "net_max_drawdown": net_stats["Max Drawdown"],
        "net_hit_rate": net_stats["Hit Rate"]
    }

    return portfolio_result, pair_metrics_df, gross_portfolio_returns, net_portfolio_returns


# =============================================================================
# 4. TRAINING PAIR SELECTION
# =============================================================================

print("\nSelecting pairs on training data only...")

all_pairs, selected_pairs = find_cointegrated_pairs_by_group(
    log_train_prices,
    SECTOR_GROUPS
)

all_pairs.to_csv(TABLES_DIR / "v5_all_training_pair_tests.csv", index=False)
selected_pairs.to_csv(TABLES_DIR / "v5_selected_training_pairs.csv", index=False)

print("\nTop training pairs:")
print(all_pairs[[
    "group", "stock_a", "stock_b", "pvalue", "beta", "r_squared", "half_life"
]].head(20))

print("\nSelected training pairs after half-life filter:")
print(selected_pairs[[
    "group", "stock_a", "stock_b", "pvalue", "beta", "r_squared", "half_life"
]].head(20))

if selected_pairs.empty:
    raise ValueError("No pairs passed training filters. Loosen half-life/p-value filters.")

selected_pairs_to_trade = selected_pairs.head(MAX_PAIRS_TO_TRADE).copy()


# =============================================================================
# 5. VALIDATION: CHOOSE THRESHOLD
# =============================================================================

validation_results = []
validation_pair_results = {}

print("\nRunning validation backtests...")

for threshold in THRESHOLDS_TO_TEST:
    portfolio_result, pair_metrics_df, gross_port, net_port = portfolio_backtest(
        selected_pairs_to_trade=selected_pairs_to_trade,
        full_log_prices=log_prices,
        eval_prices=valid_prices,
        eval_start=VALID_START,
        eval_end=VALID_END,
        threshold=threshold
    )

    validation_results.append(portfolio_result)
    validation_pair_results[threshold] = pair_metrics_df

    pair_metrics_df.to_csv(
        TABLES_DIR / f"v5_validation_pair_metrics_threshold_{threshold}.csv",
        index=False
    )

validation_results_df = pd.DataFrame(validation_results)
validation_results_df.to_csv(TABLES_DIR / "v5_validation_threshold_comparison.csv", index=False)

print("\nValidation threshold comparison:")
print(validation_results_df)

best_valid_row = validation_results_df.sort_values("net_sharpe", ascending=False).iloc[0]
BEST_THRESHOLD = best_valid_row["entry_threshold"]

print(f"\nChosen threshold based on validation net Sharpe: {BEST_THRESHOLD}")


# =============================================================================
# 6. VALIDATION PAIR FILTER
# =============================================================================

valid_pair_metrics = validation_pair_results[BEST_THRESHOLD].copy()

# Keep pairs that behave reasonably in validation.
valid_pair_metrics["passes_validation_filter"] = (
    (valid_pair_metrics["num_entries"] >= MIN_VALIDATION_TRADES) &
    (valid_pair_metrics["net_total_return"] > 0) &
    (valid_pair_metrics["net_sharpe"] > 0) &
    (valid_pair_metrics["net_max_drawdown"] >= -0.20) &
    (valid_pair_metrics["avg_rolling_r2"] >= 0.30)
)

passing_pairs = valid_pair_metrics[valid_pair_metrics["passes_validation_filter"]].copy()

print("\nValidation pair-level metrics at chosen threshold:")
print(valid_pair_metrics[[
    "group", "stock_a", "stock_b", "net_total_return", "net_sharpe",
    "net_max_drawdown", "num_entries", "avg_rolling_r2",
    "passes_validation_filter"
]].sort_values("net_sharpe", ascending=False))

if passing_pairs.empty:
    print("\nNo pairs passed validation filters. Using all selected training pairs.")
    final_pairs_to_trade = selected_pairs_to_trade.copy()
else:
    keep_keys = passing_pairs[["stock_a", "stock_b"]]
    final_pairs_to_trade = selected_pairs_to_trade.merge(
        keep_keys,
        on=["stock_a", "stock_b"],
        how="inner"
    )

print("\nFinal pairs selected for test after stricter validation filter:")
print(final_pairs_to_trade[[
    "group", "stock_a", "stock_b", "pvalue", "beta", "r_squared", "half_life"
]])


# =============================================================================
# 7. FINAL TEST BACKTEST
# =============================================================================

print("\nRunning final held-out test backtest...")

test_result, test_pair_metrics, gross_test_port, net_test_port = portfolio_backtest(
    selected_pairs_to_trade=final_pairs_to_trade,
    full_log_prices=log_prices,
    eval_prices=test_prices,
    eval_start=TEST_START,
    eval_end=TEST_END,
    threshold=BEST_THRESHOLD
)

test_pair_metrics.to_csv(TABLES_DIR / "v6_test_pair_level_metrics.csv", index=False)

test_results_df = pd.DataFrame([test_result])
test_results_df.to_csv(TABLES_DIR / "v6_final_test_portfolio_metrics.csv", index=False)

print("\nFinal test portfolio metrics:")
print(test_results_df)

print("\nFinal test pair-level metrics:")
print(test_pair_metrics[[
    "group", "stock_a", "stock_b", "net_total_return", "net_sharpe",
    "net_max_drawdown", "num_entries", "avg_rolling_r2",
    "mean_reversion_exits", "max_holding_exits", "stop_loss_exits"
]].sort_values("net_sharpe", ascending=False))


# =============================================================================
# 8. PLOTS
# =============================================================================

# Validation Sharpe by threshold
plt.figure(figsize=(8, 5))
plt.plot(
    validation_results_df["entry_threshold"],
    validation_results_df["net_sharpe"],
    marker="o"
)
plt.title("Validation Net Sharpe by Entry Threshold")
plt.xlabel("Entry Threshold")
plt.ylabel("Validation Net Sharpe")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "v6_validation_net_sharpe_by_threshold.png", dpi=300)
plt.show()


# Final test cumulative returns
gross_cum = (1 + gross_test_port.dropna()).cumprod()
net_cum = (1 + net_test_port.dropna()).cumprod()

plt.figure(figsize=(11, 5))
plt.plot(gross_cum.index, gross_cum, label="Gross")
plt.plot(net_cum.index, net_cum, label="Net")
plt.title("V5 Final Test Portfolio Returns")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / "v6_final_test_cumulative_returns.png", dpi=300)
plt.show()


# Final test drawdown
net_drawdown = net_cum / net_cum.cummax() - 1

plt.figure(figsize=(11, 5))
plt.plot(net_drawdown.index, net_drawdown)
plt.title("V5 Final Test Portfolio Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "v6_final_test_drawdown.png", dpi=300)
plt.show()


# =============================================================================
# 9. SUMMARY
# =============================================================================

summary_text = f"""
Statistical Arbitrage Project — Version 6
Train / Validation / Test + Half-Life Filter

Research Question:
    Can an economically constrained cointegration-based pairs trading strategy
    produce out-of-sample mean-reversion signals when pair selection, threshold
    choice and final testing are separated?

Data:
    - 25 liquid U.S. equities from Yahoo Finance.
    - Groups: {", ".join(SECTOR_GROUPS.keys())}

Periods:
    Training: {TRAIN_START} to {TRAIN_END}
    Validation: {VALID_START} to {VALID_END}
    Test: {TEST_START} to {TEST_END}

Training Pair Selection:
    - Pairs tested only within economic groups.
    - Cointegration tested on log prices.
    - Static hedge regression estimated on training data.
    - Half-life of spread mean reversion estimated from training spread.
    - Filters:
        p-value < {COINTEGRATION_PVALUE_THRESHOLD}
        beta > 0
        R-squared > {MIN_R_SQUARED}
        {MIN_HALF_LIFE} <= half-life <= {MAX_HALF_LIFE}

Validation filters:
    minimum trades: {MIN_VALIDATION_TRADES}
    validation net return > 0
    validation net Sharpe > 0
    max drawdown no worse than: -0.20
    avg rolling R² at least: 0.30

Final Test:
    Number of final pairs: {len(final_pairs_to_trade)}
    Net Total Return: {test_result['net_total_return']:.4f}
    Net Annual Return: {test_result['net_annual_return']:.4f}
    Net Annual Volatility: {test_result['net_annual_volatility']:.4f}
    Net Sharpe Ratio: {test_result['net_sharpe']:.4f}
    Net Max Drawdown: {test_result['net_max_drawdown']:.4f}
    Net Hit Rate: {test_result['net_hit_rate']:.4f}

Interpretation:
    Version 5 improves the research design by separating pair selection,
    threshold choice and final performance evaluation. This reduces overfitting
    relative to choosing parameters directly on the final test period.

Limitations:
    - Yahoo Finance data is not institutional-grade.
    - Current ticker universe introduces survivorship bias.
    - Universe is small and manually selected.
    - Transaction costs are simplified.
    - Borrow costs, slippage, market impact and shorting constraints are not modelled.
    - Half-life and validation filters are simple research heuristics.
    - Results are for research demonstration only.
"""

with open(RESULTS_DIR / "project_summary_v6.txt", "w") as f:
    f.write(summary_text)

print("\nProject summary saved.")
print(summary_text)

print("\nDONE.")
print(f"Results saved in: {RESULTS_DIR}")
print(f"Chosen threshold from validation: {BEST_THRESHOLD}")