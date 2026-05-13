#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 20:02:12 2026

@author: dakshayanipinninti
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Arbitrage Project 
Ridge Regression Basket Spread Extension

Author: Dakshayani Pinninti

Purpose:
    Test whether ridge regression can construct more stable synthetic spreads
    than simple pairwise hedge regressions.

    Instead of:
        log(A) = alpha + beta * log(B)

    This version estimates:
        log(target) = alpha + beta_1*log(asset_1) + ... + beta_k*log(asset_k)

    The residual becomes the spread:
        spread = log(target) - predicted_log(target)

    Then trade mean reversion in this residual spread.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIG
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_DIR / "results_v7_ridge"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

for folder in [DATA_DIR, RAW_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

TRAIN_START = "2015-01-01"
TRAIN_END = "2019-12-31"

VALID_START = "2020-01-01"
VALID_END = "2021-12-31"

TEST_START = "2022-01-01"
TEST_END = "2025-12-31"

ROLLING_Z_WINDOW = 60
ENTRY_THRESHOLD = 2.0
EXIT_THRESHOLD = 0.0
MAX_HOLDING_DAYS = 30
STOP_LOSS_Z = 4.0

TRANSACTION_COST_BPS = 5
TRANSACTION_COST = TRANSACTION_COST_BPS / 10_000

RIDGE_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


BASKETS = {
    "consumer_pep": {
        "target": "PEP",
        "basket": ["KO", "PG", "WMT", "COST"]
    },
    "tech_googl": {
        "target": "GOOGL",
        "basket": ["AMZN", "META", "MSFT", "AAPL"]
    },
    "banks_wfc": {
        "target": "WFC",
        "basket": ["JPM", "BAC", "GS", "MS"]
    },
    "semis_amd": {
        "target": "AMD",
        "basket": ["NVDA", "AVGO", "INTC", "QCOM"]
    }
}

TICKERS = sorted(set(
    [v["target"] for v in BASKETS.values()] +
    [ticker for v in BASKETS.values() for ticker in v["basket"]]
))


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

    prices = prices.ffill().dropna()
    prices.index = pd.to_datetime(prices.index)

    return prices


prices = download_adjusted_prices(TICKERS, START_DATE, END_DATE)
prices.to_csv(RAW_DIR / "ridge_basket_adjusted_close_prices.csv")

log_prices = np.log(prices)

print("\nData shape:", prices.shape)
print("Tickers:", list(prices.columns))


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def fit_ridge_spread(log_price_df, target, basket, alpha):
    """
    Fit ridge model:
        log(target) ~ log(basket assets)

    Returns fitted model and in-sample residual spread.
    """

    y = log_price_df[target]
    X = log_price_df[basket]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha))
    ])

    model.fit(X, y)

    fitted = pd.Series(model.predict(X), index=X.index, name="fitted")
    spread = y - fitted
    spread.name = "spread"

    r2 = r2_score(y, fitted)

    return model, spread, r2


def predict_ridge_spread(model, log_price_df, target, basket):
    y = log_price_df[target]
    X = log_price_df[basket]

    fitted = pd.Series(model.predict(X), index=X.index, name="fitted")
    spread = y - fitted
    spread.name = "spread"

    return spread, fitted


def compute_zscore(spread, window=60):
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
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


def get_ridge_coefficients(model, basket):
    """
    Convert ridge coefficients back to a readable table.

    Note:
        Since the model uses standardized X, coefficients are on standardized scale
    """

    ridge = model.named_steps["ridge"]
    coefs = pd.Series(ridge.coef_, index=basket, name="standardized_coefficient")
    return coefs


def calculate_basket_strategy_returns(
    price_df,
    target,
    basket,
    positions,
    model
):
    """
    Approximate beta/basket-neutral returns.

    If long spread:
        long target
        short basket using normalized absolute ridge coefficients

    If short spread:
        short target
        long basket

    simplified implementation because ridge coefficients are estimated
    on standardized log prices
    """

    returns = price_df[[target] + basket].pct_change().dropna()

    aligned_positions = positions.reindex(returns.index).fillna(0)
    lagged_positions = aligned_positions.shift(1).fillna(0)

    coefs = get_ridge_coefficients(model, basket)

    if coefs.abs().sum() == 0:
        basket_weights = pd.Series(1 / len(basket), index=basket)
    else:
        basket_weights = coefs / coefs.abs().sum()

    # Gross exposure normalized:
    # 50% target leg, 50% basket leg.
    target_weight = 0.5 * lagged_positions
    basket_position_multiplier = -0.5 * lagged_positions

    basket_return = returns[basket].dot(basket_weights)

    gross_returns = target_weight * returns[target] + basket_position_multiplier * basket_return
    gross_returns.name = "gross_strategy_return"

    return gross_returns, lagged_positions, basket_weights


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


def backtest_ridge_basket(
    name,
    target,
    basket,
    alpha,
    eval_start,
    eval_end,
    train_start=TRAIN_START,
    train_end=TRAIN_END
):
    """
    Fit ridge on training period, evaluate on selected period.
    """

    train_log = log_prices.loc[train_start:train_end, [target] + basket]
    eval_log = log_prices.loc[eval_start:eval_end, [target] + basket]
    eval_prices = prices.loc[eval_start:eval_end, [target] + basket]

    model, train_spread, train_r2 = fit_ridge_spread(
        train_log,
        target,
        basket,
        alpha
    )

    eval_spread, eval_fitted = predict_ridge_spread(
        model,
        eval_log,
        target,
        basket
    )

    zscore = compute_zscore(eval_spread, window=ROLLING_Z_WINDOW)

    positions, holding_days, exit_reasons = generate_positions_with_risk_rules(
        zscore,
        entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=EXIT_THRESHOLD,
        max_holding_days=MAX_HOLDING_DAYS,
        stop_loss_z=STOP_LOSS_Z
    )

    gross_returns, lagged_positions, basket_weights = calculate_basket_strategy_returns(
        eval_prices,
        target,
        basket,
        positions,
        model
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
        "basket_name": name,
        "target": target,
        "basket": ",".join(basket),
        "alpha": alpha,
        "train_r2": train_r2,

        "net_total_return": net_stats["Total Return"],
        "net_annual_return": net_stats["Annual Return"],
        "net_annual_volatility": net_stats["Annual Volatility"],
        "net_sharpe": net_stats["Sharpe Ratio"],
        "net_max_drawdown": net_stats["Max Drawdown"],
        "net_hit_rate": net_stats["Hit Rate"],

        "gross_total_return": gross_stats["Total Return"],
        "gross_sharpe": gross_stats["Sharpe Ratio"],

        "num_entries": trade_stats["Number of Entries"],
        "num_exits": trade_stats["Number of Exits"],
        "share_days_invested": trade_stats["Share of Days Invested"],
        "mean_reversion_exits": trade_stats["Mean-Reversion Exits"],
        "max_holding_exits": trade_stats["Max-Holding Exits"],
        "stop_loss_exits": trade_stats["Stop-Loss Exits"]
    }

    diagnostics = pd.DataFrame({
        "spread": eval_spread,
        "zscore": zscore,
        "position": lagged_positions,
        "gross_return": gross_returns,
        "net_return": net_returns,
        "cost": costs
    })

    coef_table = get_ridge_coefficients(model, basket).reset_index()
    coef_table.columns = ["asset", "standardized_coefficient"]
    coef_table["basket_name"] = name
    coef_table["alpha"] = alpha

    return result, diagnostics, coef_table, net_returns


# =============================================================================
# 4. VALIDATION: CHOOSE RIDGE ALPHA PER BASKET
# =============================================================================

validation_results = []

print("\nRunning ridge validation backtests...")

for name, config in BASKETS.items():
    target = config["target"]
    basket = config["basket"]

    for alpha in RIDGE_ALPHAS:
        result, diagnostics, coef_table, net_returns = backtest_ridge_basket(
            name=name,
            target=target,
            basket=basket,
            alpha=alpha,
            eval_start=VALID_START,
            eval_end=VALID_END
        )

        validation_results.append(result)

validation_df = pd.DataFrame(validation_results)
validation_df.to_csv(TABLES_DIR / "ridge_validation_results.csv", index=False)

print("\nValidation results:")
print(validation_df.sort_values("net_sharpe", ascending=False))


# Choose best alpha for each basket based on validation Sharpe
best_alpha_by_basket = (
    validation_df.sort_values("net_sharpe", ascending=False)
    .groupby("basket_name")
    .head(1)
    .reset_index(drop=True)
)

best_alpha_by_basket.to_csv(TABLES_DIR / "best_alpha_by_basket.csv", index=False)

print("\nBest alpha by basket:")
print(best_alpha_by_basket[[
    "basket_name", "target", "alpha", "train_r2",
    "net_total_return", "net_sharpe", "net_max_drawdown"
]])


# =============================================================================
# 5. FINAL TEST
# =============================================================================

test_results = []
all_net_returns = []
all_coef_tables = []

print("\nRunning final ridge basket test...")

for _, row in best_alpha_by_basket.iterrows():
    name = row["basket_name"]
    target = row["target"]
    basket = BASKETS[name]["basket"]
    alpha = row["alpha"]

    result, diagnostics, coef_table, net_returns = backtest_ridge_basket(
        name=name,
        target=target,
        basket=basket,
        alpha=alpha,
        eval_start=TEST_START,
        eval_end=TEST_END
    )

    test_results.append(result)
    all_net_returns.append(net_returns.rename(name))
    all_coef_tables.append(coef_table)

    diagnostics.to_csv(TABLES_DIR / f"ridge_test_diagnostics_{name}.csv")
    coef_table.to_csv(TABLES_DIR / f"ridge_coefficients_{name}.csv", index=False)

test_df = pd.DataFrame(test_results)
test_df.to_csv(TABLES_DIR / "ridge_test_results.csv", index=False)

coef_df = pd.concat(all_coef_tables, axis=0)
coef_df.to_csv(TABLES_DIR / "ridge_coefficients_all_baskets.csv", index=False)

print("\nFinal test results by basket:")
print(test_df.sort_values("net_sharpe", ascending=False))


# Equal-weight portfolio across ridge baskets
net_returns_df = pd.concat(all_net_returns, axis=1).dropna(how="all")
portfolio_net_returns = net_returns_df.fillna(0).mean(axis=1)
portfolio_stats = performance_metrics(portfolio_net_returns)

portfolio_stats_df = pd.DataFrame([portfolio_stats])
portfolio_stats_df.to_csv(TABLES_DIR / "ridge_portfolio_test_metrics.csv", index=False)

print("\nRidge basket portfolio test metrics:")
print(portfolio_stats_df)


# =============================================================================
# 6. PLOTS
# =============================================================================

# Portfolio cumulative returns
portfolio_cum = (1 + portfolio_net_returns.dropna()).cumprod()

plt.figure(figsize=(11, 5))
plt.plot(portfolio_cum.index, portfolio_cum)
plt.title("Ridge Basket Strategy — Final Test Cumulative Returns")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "ridge_basket_final_test_cumulative_returns.png", dpi=300)
plt.show()


# Portfolio drawdown
portfolio_drawdown = portfolio_cum / portfolio_cum.cummax() - 1

plt.figure(figsize=(11, 5))
plt.plot(portfolio_drawdown.index, portfolio_drawdown)
plt.title("Ridge Basket Strategy — Final Test Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "ridge_basket_final_test_drawdown.png", dpi=300)
plt.show()


# Bar chart of basket Sharpes
plt.figure(figsize=(9, 5))
plt.bar(test_df["basket_name"], test_df["net_sharpe"])
plt.title("Ridge Basket Final Test Net Sharpe by Basket")
plt.xlabel("Basket")
plt.ylabel("Net Sharpe")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "ridge_basket_test_sharpe_by_basket.png", dpi=300)
plt.show()


# =============================================================================
# 7. SUMMARY
# =============================================================================

summary_text = f"""
Statistical Arbitrage Project 
Ridge Regression Basket Spread Extension

Research Question:
    Can ridge regression construct more stable synthetic spreads using baskets
    of related assets, compared with simple pairwise hedge regressions?

Method:
    - For each basket, fit ridge regression on training data:
        log(target) ~ log(related assets)
    - Use residual as synthetic spread.
    - Choose ridge alpha on validation period.
    - Evaluate selected alpha on held-out test period.
    - Trade spread mean reversion using z-score signals.
    - Apply max holding period, stop-loss rules and transaction costs.

Baskets:
    {list(BASKETS.keys())}

Ridge Alphas Tested:
    {RIDGE_ALPHAS}

Final Test Portfolio:
    Net Total Return: {portfolio_stats['Total Return']:.4f}
    Net Annual Return: {portfolio_stats['Annual Return']:.4f}
    Net Annual Volatility: {portfolio_stats['Annual Volatility']:.4f}
    Net Sharpe Ratio: {portfolio_stats['Sharpe Ratio']:.4f}
    Net Max Drawdown: {portfolio_stats['Max Drawdown']:.4f}
    Net Hit Rate: {portfolio_stats['Hit Rate']:.4f}

Interpretation:
    This extension tests whether shrinkage can stabilize hedge relationships
    when constructing synthetic spreads from multiple related assets. Ridge is
    especially relevant when basket constituents are correlated, making OLS hedge
    coefficients unstable.

Limitations:
    - Ridge coefficients are estimated on standardized log prices, so mapping
      coefficients into exact dollar hedge weights is approximate.
    - Basket construction is manually specified.
    - Transaction costs are simplified.
    - Borrow costs, slippage and market impact are not modelled.
    - Results are for research demonstration only.
"""

with open(RESULTS_DIR / "project_summary_v7_ridge.txt", "w") as f:
    f.write(summary_text)

print("\nProject summary saved.")
print(summary_text)

print("\nDONE.")
print(f"Results saved in: {RESULTS_DIR}")