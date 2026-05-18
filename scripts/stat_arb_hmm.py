#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 00:45:50 2026

@author: dakshayanipinninti
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from itertools import combinations
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.stats.multitest import multipletests
from scipy import stats

try:
    import pandas_datareader.data as web
    PANDAS_DATAREADER_AVAILABLE = True
except ImportError:
    PANDAS_DATAREADER_AVAILABLE = False

try:
    from pykalman import KalmanFilter
    PYKALMAN_AVAILABLE = True
except ImportError:
    PYKALMAN_AVAILABLE = False
    print("WARNING: pykalman not installed. EM estimation will be skipped.")
    print("         Install with: pip install pykalman")

try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIG
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR    = PROJECT_DIR / "data"
RAW_DIR     = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_DIR / "results_v26_full"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR  = RESULTS_DIR / "tables"

for folder in [DATA_DIR, RAW_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Feature flags
# -----------------------------------------------------------------------------

# Tier 1 (default ON; defining the new baseline)
USE_EM_KALMAN          = True   # Per-fold EM estimation of Q, R
RUN_ABLATION           = True   # Run all four signal specifications
USE_FDR_GRID           = True   # Run both q ≤ 0.10 AND q ≤ 0.25 as parallel specs

# Tier 2 (default OFF)
RUN_NULL_BENCHMARK     = True
USE_ROLL_SPREAD_COST   = False
USE_HMM_REGIMES        = True
RUN_SECTOR_ETF_RESIDUAL_STRATEGY = True

NULL_BENCHMARK_REPS    = 100
NULL_BENCHMARK_SEED    = 42
MIN_BPS_FLOOR          = 2.0

# FDR grid
FDR_Q_PRIMARY          = 0.20
FDR_Q_SECONDARY        = 0.10
FDR_Q_VALUES           = [FDR_Q_PRIMARY, FDR_Q_SECONDARY] if USE_FDR_GRID else [FDR_Q_PRIMARY]

MAX_FOLDS_OVERRIDE = None

# -----------------------------------------------------------------------------
# Universe (unchanged)
# -----------------------------------------------------------------------------

SECTOR_GROUPS = {
    "Banks": [
        "JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "PNC",
        "TFC", "FITB", "KEY", "RF", "HBAN", "CFG", "MTB", "NTRS",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX",
        "HES", "OXY", "FANG", "DVN", "KMI", "OKE", "WMB",
    ],
    "Semiconductors": [
        "NVDA", "AMD", "INTC", "QCOM", "AVGO", "MU", "TXN", "ADI",
        "LRCX", "KLAC", "ON", "MCHP", "MRVL", "MPWR", "SWKS", "NXPI",
    ],
    "MegaCapTech": [
        "MSFT", "AAPL", "GOOGL", "META", "AMZN", "ORCL", "CRM", "ADBE",
        "IBM", "INTU", "NOW", "ACN", "CSCO",
    ],
    "ConsumerStaples": [
        "KO", "PEP", "PG", "WMT", "COST", "CL", "KMB", "MDLZ",
        "GIS", "HSY", "CHD", "CLX", "EL", "STZ", "SYY"
    ],
    "ConsumerDiscretionary": [
        "HD", "LOW", "NKE", "SBUX", "MCD", "TJX", "TGT",
        "BKNG", "ROST", "LVS", "MAR", "HLT", "DRI",
    ],
    "Healthcare": [
        "JNJ", "PFE", "MRK", "ABBV", "BMY", "LLY", "AMGN", "GILD",
        "TMO", "ABT", "DHR", "MDT", "ISRG", "CI", "HUM", "ELV", "CVS",
    ],
    "PaymentsCredit": [
        "V", "MA", "AXP", "PYPL", "COF",
        "SYF", "FIS", "GPN", "JKHY", "BR", "FLT"
    ],
    "Industrials": [
        "CAT", "DE", "HON", "GE", "MMM", "UNP", "UPS",
        "NSC", "CSX", "LMT", "RTX", "BA", "EMR", "ETN", "ITW", "PH", "ROK",
    ],
    "TelecomMedia": [
        "T", "VZ", "CMCSA", "DIS", "NFLX",
        "TMUS", "CHTR", "WBD", "EA", "TTWO",
    ],
}

SECTOR_ETFS = {
    "Banks":                 "XLF",
    "Energy":                "XLE",
    "Semiconductors":        "SMH",
    "MegaCapTech":           "XLK",
    "ConsumerStaples":       "XLP",
    "ConsumerDiscretionary": "XLY",
    "Healthcare":            "XLV",
    "PaymentsCredit":        "XLF",
    "Industrials":           "XLI",
    "TelecomMedia":          "IYZ",
}

MARKET_TICKER = "SPY"

STOCK_TICKERS = sorted(set(t for g in SECTOR_GROUPS.values() for t in g))
ETF_TICKERS   = sorted(set(list(SECTOR_ETFS.values()) + [MARKET_TICKER]))
TICKERS       = sorted(set(STOCK_TICKERS + ETF_TICKERS))

START_DATE = "2015-01-01"
END_DATE   = "2025-12-31"


# -----------------------------------------------------------------------------
# Walk-forward folds (unchanged)
# -----------------------------------------------------------------------------

def make_rolling_folds(
    first_train_start="2015-01-01",
    first_valid_start="2020-01-01",
    last_test_end="2025-12-31",
    train_years=4,
    valid_months=6,
    test_months=6,
    step_months=3,
):
    folds = []
    valid_start = pd.Timestamp(first_valid_start)
    last_test_end = pd.Timestamp(last_test_end)
    fold_num = 1

    while True:
        train_start = valid_start - pd.DateOffset(years=train_years)
        train_end = valid_start - pd.DateOffset(days=1)
        valid_end = valid_start + pd.DateOffset(months=valid_months) - pd.DateOffset(days=1)
        test_start = valid_end + pd.DateOffset(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.DateOffset(days=1)

        if test_end > last_test_end:
            break

        folds.append({
            "fold": f"fold_{fold_num:02d}_{test_start.strftime('%Y%m')}_{test_end.strftime('%Y%m')}",
            "train_start": train_start.strftime("%Y-%m-%d"),
            "train_end": train_end.strftime("%Y-%m-%d"),
            "valid_start": valid_start.strftime("%Y-%m-%d"),
            "valid_end": valid_end.strftime("%Y-%m-%d"),
            "test_start": test_start.strftime("%Y-%m-%d"),
            "test_end": test_end.strftime("%Y-%m-%d"),
        })

        valid_start = valid_start + pd.DateOffset(months=step_months)
        fold_num += 1

    return folds


WALK_FORWARD_FOLDS = make_rolling_folds()

# Apply debug fold limit if set
if MAX_FOLDS_OVERRIDE is not None:
    WALK_FORWARD_FOLDS = WALK_FORWARD_FOLDS[:MAX_FOLDS_OVERRIDE]
    print(f"DEBUG: limiting to first {MAX_FOLDS_OVERRIDE} folds")

print(f"Number of rolling folds created: {len(WALK_FORWARD_FOLDS)}")


# -----------------------------------------------------------------------------
# Pair selection thresholds
# -----------------------------------------------------------------------------

COINTEGRATION_PVALUE_THRESHOLD = 0.10
MIN_R_SQUARED                  = 0.35
MIN_HALF_LIFE                  = 5
MAX_HALF_LIFE                  = 80
MAX_PAIRS_TO_TRADE             = 20
MAX_VALIDATION_PAIRS           = 5

REQUIRE_FDR_PASS = True


# -----------------------------------------------------------------------------
# Signal parameters
# -----------------------------------------------------------------------------

ROLLING_Z_WINDOW   = 60
THRESHOLDS_TO_TEST = [2.0, 2.5, 3.0, 3.5] #lowered from 2.5-4 to 2-3.5
EXIT_THRESHOLD     = 0.0
MAX_HOLDING_DAYS   = 45
STOP_LOSS_Z        = 4.0
TRANSACTION_COSTS_BPS = [0, 5, 10, 25]


# -----------------------------------------------------------------------------
# Validation scoring
# -----------------------------------------------------------------------------

MIN_VALIDATION_TRADES = 2
RELAXED_MAX_DRAWDOWN  = -0.35
RELAXED_MIN_SHARPE    = 0.00

MIN_VALIDATION_TRADES_STRICT = 3
MIN_VALIDATION_SHARPE_STRICT = 0.0
MIN_VALIDATION_TRADES_FALLBACK = 1
MIN_VALIDATION_SHARPE_FALLBACK = 0.75

ROLLING_BETA_STABILITY_WINDOW = 126
MAX_BETA_STABILITY            = 1.75 #changed from 1.25
MAX_HALF_LIFE_RATIO           = 5.00


# -----------------------------------------------------------------------------
# Kalman defaults (fallback only)
# -----------------------------------------------------------------------------

KALMAN_TRANSITION_COV    = 1e-3
KALMAN_OBSERVATION_COV   = 1e-3
KALMAN_INITIAL_STATE_COV = 1.0

USE_BETA_UNCERTAINTY_FILTER     = True
BETA_UNCERTAINTY_ROLLING_WINDOW = 252
BETA_UNCERTAINTY_QUANTILE       = 0.75

USE_CONFIDENCE_SCALING = True
CONFIDENCE_FLOOR       = 0.25
CONFIDENCE_CAP         = 1.00

MAX_PAIR_WEIGHT = 0.35
MIN_PAIRS_PER_FOLD     = 3      # below this, trigger top-K fallback
TOP_K_FALLBACK         = 10     # number of pairs in the fallback pool


# =============================================================================
# 2. DATA DOWNLOAD
# =============================================================================

def download_adjusted_prices(tickers, start, end):
    print("Downloading adjusted price data...")
    data = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, group_by="column",
    )
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = tickers

    missing_share = prices.isna().mean()
    keep_cols = missing_share[missing_share < 0.05].index.tolist()
    prices = prices[keep_cols].ffill().dropna()
    prices.index = pd.to_datetime(prices.index)
    return prices


prices     = download_adjusted_prices(TICKERS, START_DATE, END_DATE)
log_prices = np.log(prices)

if MARKET_TICKER in prices.columns:
    market_returns = prices[MARKET_TICKER].pct_change().dropna()
else:
    market_returns = pd.Series(dtype=float)
    print(f"WARNING: {MARKET_TICKER} not in price data. HMM regimes will fall back.")

prices.to_csv(RAW_DIR / "adjusted_close_prices_v26_full.csv")
print(f"\nTickers available after cleaning: {len(prices.columns)}")


# =============================================================================
# 3. STATISTICAL HELPERS (unchanged)
# =============================================================================

def estimate_hedge_ratio(y, x):
    x_const = sm.add_constant(x)
    model   = sm.OLS(y, x_const).fit()
    return model.params["const"], model.params[x.name], model


def compute_static_spread(log_price_df, stock_a, stock_b, alpha, beta):
    spread = log_price_df[stock_a] - alpha - beta * log_price_df[stock_b]
    spread.name = "spread"
    return spread


def estimate_half_life(spread):
    spread = spread.dropna()
    if len(spread) < 60:
        return np.nan
    aligned = pd.concat([spread.diff(), spread.shift(1)], axis=1).dropna()
    aligned.columns = ["delta_spread", "lagged_spread"]
    if len(aligned) < 30:
        return np.nan
    model = sm.OLS(aligned["delta_spread"],
                   sm.add_constant(aligned["lagged_spread"])).fit()
    beta = model.params["lagged_spread"]
    return np.nan if beta >= 0 else -np.log(2) / beta


def subsample_cointegration_check(log_price_df, stock_a, stock_b,
                                   pvalue_cutoff=0.20):

    data = log_price_df[[stock_a, stock_b]].dropna()
    if len(data) < 252:
        return False, np.nan, np.nan
    n = len(data)
    try:
        _, p1, _ = coint(data.iloc[:n//2][stock_a], data.iloc[:n//2][stock_b])
        _, p2, _ = coint(data.iloc[n//2:][stock_a], data.iloc[n//2:][stock_b])
        return (p1 < pvalue_cutoff) and (p2 < pvalue_cutoff), p1, p2
    except Exception:
        return False, np.nan, np.nan


def half_life_consistency_check(log_price_df, stock_a, stock_b, max_ratio=5.0):
    data = log_price_df[[stock_a, stock_b]].dropna()
    if len(data) < 252:
        return False, np.nan, np.nan, np.nan
    n = len(data)
    try:
        a1, b1, _ = estimate_hedge_ratio(data.iloc[:n//2][stock_a],
                                          data.iloc[:n//2][stock_b])
        hl1 = estimate_half_life(
            compute_static_spread(data.iloc[:n//2], stock_a, stock_b, a1, b1))
        a2, b2, _ = estimate_hedge_ratio(data.iloc[n//2:][stock_a],
                                          data.iloc[n//2:][stock_b])
        hl2 = estimate_half_life(
            compute_static_spread(data.iloc[n//2:], stock_a, stock_b, a2, b2))
        if pd.isna(hl1) or pd.isna(hl2) or hl1 <= 0 or hl2 <= 0:
            return False, hl1, hl2, np.nan
        ratio = max(hl1, hl2) / min(hl1, hl2)
        return ratio <= max_ratio, hl1, hl2, ratio
    except Exception:
        return False, np.nan, np.nan, np.nan


def estimate_training_beta_stability(log_price_df, stock_a, stock_b, window=126):
    data = log_price_df[[stock_a, stock_b]].dropna()
    if len(data) < window + 30:
        return np.nan
    betas = []
    for i in range(window, len(data)):
        try:
            _, beta, _ = estimate_hedge_ratio(
                data[stock_a].iloc[i-window:i],
                data[stock_b].iloc[i-window:i])
            betas.append(beta)
        except Exception:
            continue
    if len(betas) < 30:
        return np.nan
    s = pd.Series(betas).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty or abs(s.mean()) < 1e-8:
        return np.nan
    return s.std() / abs(s.mean())


def compute_zscore(spread, window=60):
    z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    z.name = "zscore"
    return z


def performance_metrics(returns, periods_per_year=252):
    returns = returns.dropna()
    if len(returns) == 0:
        return dict(Total_Return=np.nan, Annual_Return=np.nan,
                    Annual_Volatility=np.nan, Sharpe_Ratio=np.nan,
                    Sharpe_SE=np.nan, Max_Drawdown=np.nan,
                    Daily_Hit_Rate=np.nan, N_Obs=0)
    cum   = (1 + returns).cumprod()
    tr    = cum.iloc[-1] - 1
    ar    = returns.mean() * periods_per_year
    av    = returns.std()  * np.sqrt(periods_per_year)
    sh    = ar / av if av and av != 0 else np.nan
    n     = len(returns)
    sh_se = (1/np.sqrt(n)) * np.sqrt(1 + 0.5*sh**2) if pd.notna(sh) and n > 1 else np.nan
    dd    = cum / cum.cummax() - 1
    return dict(Total_Return=tr, Annual_Return=ar, Annual_Volatility=av,
                Sharpe_Ratio=sh, Sharpe_SE=sh_se,
                Max_Drawdown=dd.min(),
                Daily_Hit_Rate=(returns > 0).mean(), N_Obs=n)


# =============================================================================
# 3B. ROLL EFFECTIVE SPREAD
# =============================================================================

def roll_effective_spread_bps(price_series, window=60, min_bps=MIN_BPS_FLOOR):
    log_p = np.log(price_series)
    dp = log_p.diff()
    cov = dp.rolling(window).apply(
        lambda x: pd.Series(x).cov(pd.Series(x).shift(1)),
        raw=False
    )
    spread_frac = np.where(cov < 0, 2 * np.sqrt(-cov), np.nan)
    spread_bps = pd.Series(spread_frac * 10_000, index=price_series.index)
    spread_bps = spread_bps.clip(lower=min_bps).fillna(min_bps)
    return spread_bps


def pair_roll_cost_bps(price_df, stock_a, stock_b, window=60):
    sa = roll_effective_spread_bps(price_df[stock_a], window=window)
    sb = roll_effective_spread_bps(price_df[stock_b], window=window)
    return (sa + sb) / 2


# =============================================================================
# 4. KALMAN FILTER
# =============================================================================

def kalman_dynamic_hedge_ratio(
    log_price_df, stock_a, stock_b,
    transition_cov=KALMAN_TRANSITION_COV,
    observation_cov=KALMAN_OBSERVATION_COV,
    initial_state_cov=KALMAN_INITIAL_STATE_COV,
):
    data  = log_price_df[[stock_a, stock_b]].dropna().copy()
    y     = data[stock_a].values
    x     = data[stock_b].values
    dates = data.index
    n     = len(data)

    betas     = np.full(n, np.nan)
    alphas    = np.full(n, np.nan)
    beta_var  = np.full(n, np.nan)
    alpha_var = np.full(n, np.nan)

    if n < 30:
        return (pd.Series(betas, index=dates, name="beta"),
                pd.Series(alphas, index=dates, name="alpha"),
                pd.Series(beta_var, index=dates, name="beta_var"),
                pd.Series(alpha_var, index=dates, name="alpha_var"))

    init_window = min(60, n)
    try:
        init_y = pd.Series(y[:init_window], name=stock_a)
        init_x = pd.Series(x[:init_window], name=stock_b)
        init_alpha, init_beta, _ = estimate_hedge_ratio(init_y, init_x)
        theta = np.array([[init_beta], [init_alpha]])
    except Exception:
        theta = np.array([[1.0], [0.0]])

    P = initial_state_cov * np.eye(2)
    Q = transition_cov    * np.eye(2)
    R = np.array([[observation_cov]])
    I = np.eye(2)

    for t in range(n):
        P_pred = P + Q
        H      = np.array([[x[t], 1.0]])
        innov  = y[t] - (H @ theta)[0, 0]
        S      = H @ P_pred @ H.T + R
        try:
            K = P_pred @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = P_pred @ H.T @ np.linalg.pinv(S)
        theta = theta + K * innov
        P     = (I - K @ H) @ P_pred

        betas[t]     = theta[0, 0]
        alphas[t]    = theta[1, 0]
        beta_var[t]  = P[0, 0]
        alpha_var[t] = P[1, 1]

    return (pd.Series(betas,     index=dates, name="beta"),
            pd.Series(alphas,    index=dates, name="alpha"),
            pd.Series(beta_var,  index=dates, name="beta_var"),
            pd.Series(alpha_var, index=dates, name="alpha_var"))


def kalman_em_estimate(log_price_df, stock_a, stock_b, train_start, train_end,
                       max_em_iter=10,
                       fallback_transition=KALMAN_TRANSITION_COV,
                       fallback_observation=KALMAN_OBSERVATION_COV):
    """v26_full: EM estimation of Q and R using only data up to train_end.

    Called once per (pair, fold) — see kalman_cache_for_fold below.
    """
    if not PYKALMAN_AVAILABLE:
        return fallback_transition, fallback_observation

    data = log_price_df[[stock_a, stock_b]].loc[train_start:train_end].dropna()
    if len(data) < 60:
        return fallback_transition, fallback_observation

    y = data[stock_a].values
    x = data[stock_b].values
    obs_mats = np.array([[[xi, 1.0]] for xi in x])

    try:
        a0, b0, _ = estimate_hedge_ratio(
            pd.Series(y[:60], name=stock_a),
            pd.Series(x[:60], name=stock_b),
        )
        kf = KalmanFilter(
            transition_matrices=np.eye(2),
            observation_matrices=obs_mats,
            initial_state_mean=[b0, a0],
            initial_state_covariance=KALMAN_INITIAL_STATE_COV * np.eye(2),
            transition_covariance=fallback_transition * np.eye(2),
            observation_covariance=np.array([[fallback_observation]]),
            n_dim_state=2,
            n_dim_obs=1,
        )
        kf = kf.em(
            X=y.reshape(-1, 1),
            n_iter=max_em_iter,
            em_vars=["transition_covariance", "observation_covariance"],
        )
        Q_scalar = float(np.mean(np.diag(kf.transition_covariance)))
        R_scalar = float(kf.observation_covariance[0, 0])

        if (not np.isfinite(Q_scalar) or not np.isfinite(R_scalar)
                or Q_scalar <= 0 or R_scalar <= 0
                or Q_scalar > 1.0 or R_scalar > 1.0):
            return fallback_transition, fallback_observation

        return Q_scalar, R_scalar
    except Exception:
        return fallback_transition, fallback_observation


def compute_kalman_spread(log_price_df, stock_a, stock_b, alphas, betas):
    aligned = log_price_df[[stock_a, stock_b]].copy()
    spread  = aligned[stock_a] - alphas.reindex(aligned.index).ffill() \
                               - betas.reindex(aligned.index).ffill() \
                               * aligned[stock_b]
    spread.name = "spread"
    return spread


def compute_kalman_confidence(beta_var):
    rolling_mean = beta_var.rolling(BETA_UNCERTAINTY_ROLLING_WINDOW,
                                     min_periods=30).mean()
    conf = 1 / (1 + beta_var / rolling_mean)
    conf = conf.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    conf = conf.clip(lower=CONFIDENCE_FLOOR, upper=CONFIDENCE_CAP)
    conf.name = "confidence"
    return conf


def compute_beta_uncertainty_allowed(beta_var):
    threshold = beta_var.rolling(BETA_UNCERTAINTY_ROLLING_WINDOW,
                                  min_periods=30).quantile(BETA_UNCERTAINTY_QUANTILE)
    allowed = (beta_var <= threshold).fillna(True)
    allowed.name = "beta_uncertainty_allowed"
    return allowed


# =============================================================================
# 4B. v26_full: PER-FOLD KALMAN CACHE
# =============================================================================

def get_or_compute_kalman_for_fold(
    kalman_cache, stock_a, stock_b, fold_name, train_start, train_end,
    log_price_df, use_em=True,
):
    """v26_full: Kalman cache keyed by (stock_a, stock_b, fold_name).

    If EM enabled, Q and R are estimated from data ending at train_end of this
    fold, so the same pair has different Kalman parameters across folds.
    """
    key = (stock_a, stock_b, fold_name)
    if key in kalman_cache:
        return kalman_cache[key]

    if use_em and PYKALMAN_AVAILABLE:
        Q_est, R_est = kalman_em_estimate(
            log_price_df, stock_a, stock_b,
            train_start=train_start,
            train_end=train_end,
            fallback_transition=KALMAN_TRANSITION_COV,
            fallback_observation=KALMAN_OBSERVATION_COV,
        )
        em_params = {"Q": Q_est, "R": R_est}
        kalman_output = kalman_dynamic_hedge_ratio(
            log_price_df, stock_a, stock_b,
            transition_cov=Q_est, observation_cov=R_est,
        )
    else:
        em_params = {"Q": KALMAN_TRANSITION_COV, "R": KALMAN_OBSERVATION_COV}
        kalman_output = kalman_dynamic_hedge_ratio(log_price_df, stock_a, stock_b)

    kalman_cache[key] = (kalman_output, em_params)
    return kalman_cache[key]


# =============================================================================
# 5. PAIR SELECTION (FDR threshold passed in)
# =============================================================================

def build_candidate_pairs(sector_groups, sector_etfs, available_columns):
    candidates = []
    for group_name, tickers in sector_groups.items():
        avail = [t for t in tickers if t in available_columns]
        for stock_a, stock_b in combinations(avail, 2):
            candidates.append(dict(group=group_name, stock_a=stock_a,
                                   stock_b=stock_b, pair_type="stock_stock"))
        etf = sector_etfs.get(group_name)
        if etf and etf in available_columns:
            for stock in avail:
                if stock != etf:
                    candidates.append(dict(group=group_name, stock_a=stock,
                                           stock_b=etf, pair_type="stock_etf"))
    return (pd.DataFrame(candidates)
            .drop_duplicates(subset=["stock_a", "stock_b", "pair_type"])
            .reset_index(drop=True))


def find_cointegrated_pairs(log_price_df, sector_groups, sector_etfs,
                            fdr_q_threshold=FDR_Q_PRIMARY):
    """v26_full: FDR threshold is now a function argument so we can grid over it."""
    candidate_pairs = build_candidate_pairs(
        sector_groups, sector_etfs, log_price_df.columns)
    print(f"Testing {len(candidate_pairs)} candidate pairs.")
    results = []

    for _, pair in candidate_pairs.iterrows():
        group_name = pair["group"]
        stock_a    = pair["stock_a"]
        stock_b    = pair["stock_b"]
        pair_type  = pair["pair_type"]

        aligned = pd.concat([log_price_df[stock_a], log_price_df[stock_b]],
                             axis=1).dropna()
        if len(aligned) < 252:
            continue

        try:
            test_stat, pvalue, crits = coint(aligned[stock_a], aligned[stock_b])
            alpha, beta, model = estimate_hedge_ratio(
                aligned[stock_a], aligned[stock_b])
            spread    = compute_static_spread(aligned, stock_a, stock_b, alpha, beta)
            half_life = estimate_half_life(spread)

            sub_ok, sub_p1, sub_p2 = subsample_cointegration_check(
                aligned, stock_a, stock_b)
            hl_ok, hl1, hl2, hl_ratio = half_life_consistency_check(
                aligned, stock_a, stock_b, max_ratio=MAX_HALF_LIFE_RATIO)
            beta_stab = estimate_training_beta_stability(
                aligned, stock_a, stock_b,
                window=ROLLING_BETA_STABILITY_WINDOW)

            results.append(dict(
                group=group_name, pair_type=pair_type,
                stock_a=stock_a, stock_b=stock_b,
                pvalue=pvalue, test_stat=test_stat,
                alpha=alpha, beta=beta, r_squared=model.rsquared,
                half_life=half_life,
                crit_1pct=crits[0], crit_5pct=crits[1], crit_10pct=crits[2],
                subsample_coint_ok=sub_ok, subsample_p1=sub_p1, subsample_p2=sub_p2,
                half_life_first=hl1, half_life_second=hl2,
                half_life_ratio=hl_ratio, half_life_consistent=hl_ok,
                beta_stability=beta_stab,
            ))
        except Exception:
            continue

    results_df = pd.DataFrame(results)
    if results_df.empty:
        return results_df, results_df

    results_df = results_df.sort_values("pvalue").reset_index(drop=True)
    results_df["pvalue_bh_adj"] = multipletests(
        results_df["pvalue"].fillna(1.0), method="fdr_bh")[1]
    results_df["passes_fdr"] = results_df["pvalue_bh_adj"] <= fdr_q_threshold
    results_df["fdr_q_threshold_used"] = fdr_q_threshold
    
    print("Filter diagnostics:")
    print("raw p < 0.10:", (results_df["pvalue"] < COINTEGRATION_PVALUE_THRESHOLD).sum())
    print("passes FDR:", results_df["passes_fdr"].sum())
    print("beta > 0:", (results_df["beta"] > 0).sum())
    print("R2 pass:", (results_df["r_squared"] > MIN_R_SQUARED).sum())
    print("half-life pass:", ((results_df["half_life"] >= MIN_HALF_LIFE) & (results_df["half_life"] <= MAX_HALF_LIFE)).sum())
    print("subsample pass:", results_df["subsample_coint_ok"].sum())
    print("half-life consistency pass:", results_df["half_life_consistent"].sum())
    print("beta stability pass:", (results_df["beta_stability"].fillna(999) <= MAX_BETA_STABILITY).sum())
    
    mask_p = results_df["pvalue"] < COINTEGRATION_PVALUE_THRESHOLD
    mask_fdr = results_df["passes_fdr"]
    mask_beta = results_df["beta"] > 0
    mask_r2 = results_df["r_squared"] > MIN_R_SQUARED
    mask_hl = (
        (results_df["half_life"] >= MIN_HALF_LIFE) &
        (results_df["half_life"] <= MAX_HALF_LIFE)
    )
    
    print("\nCumulative filter funnel:")
    current = pd.Series(True, index=results_df.index)
    
    for name, mask in [
        ("raw p-value", mask_p),
        ("FDR", mask_fdr),
        ("beta > 0", mask_beta),
        ("R2", mask_r2),
        ("half-life", mask_hl),
    ]:
        current = current & mask
        print(f"after {name}: {current.sum()}")
    
    print("\nTop 10 pairs by p-value:")
    print(
        results_df[
            ["group", "pair_type", "stock_a", "stock_b",
             "pvalue", "pvalue_bh_adj", "passes_fdr",
             "r_squared", "half_life", "beta",
             "subsample_coint_ok", "half_life_consistent", "beta_stability"]
        ].head(10).to_string(index=False)
    )

    # Economic filter (FDR-agnostic): pairs that pass all non-significance gates
    base_mask = (
        (results_df["pvalue"] < COINTEGRATION_PVALUE_THRESHOLD) &
        (results_df["beta"]   > 0) &
        (results_df["r_squared"]  > MIN_R_SQUARED) &
        (results_df["half_life"] >= MIN_HALF_LIFE) &
        (results_df["half_life"] <= MAX_HALF_LIFE)
    )

    # Primary selection: economic filter + FDR pass
    sel = results_df[base_mask & results_df["passes_fdr"]].copy()
    if not sel.empty:
        sel["selection_mode"] = "fdr_pass"

    # Top-K fallback: if FDR selection produces fewer than MIN_PAIRS_PER_FOLD,
    # fall back to top-K pairs that pass the economic filter, sorted by p-value.
    # FDR-pass pairs are kept and tagged "fdr_pass"; the rest are tagged
    # "top_raw_p_fallback" and reported separately.
    if len(sel) < MIN_PAIRS_PER_FOLD:
        fallback_pool = results_df[base_mask].copy()
        if not fallback_pool.empty:
            fallback_pool["selection_mode"] = np.where(
                fallback_pool["passes_fdr"],
                "fdr_pass",
                "top_raw_p_fallback"
            )
            fallback_pool = fallback_pool.sort_values(
                ["passes_fdr", "pvalue", "pvalue_bh_adj", "r_squared"],
                ascending=[False, True, True, False]
            ).head(TOP_K_FALLBACK)
            sel = fallback_pool

    if not sel.empty:
        sel["training_stability_score"] = (
            sel["subsample_coint_ok"].astype(int)
            + sel["half_life_consistent"].astype(int)
            + (sel["beta_stability"].fillna(999) <= MAX_BETA_STABILITY).astype(int)
            + sel["passes_fdr"].astype(int)
        )
        sel = sel.sort_values(
            ["training_stability_score", "passes_fdr", "pvalue",
             "pvalue_bh_adj", "r_squared"],
            ascending=[False, False, True, True, False],
        ).reset_index(drop=True)

    return results_df, sel


# =============================================================================
# 6. SIGNALS, RETURNS, TRADES (v26_full: leg-level returns + costs)
# =============================================================================

def generate_positions_with_risk_rules(
    zscore,
    entry_threshold=2.0,
    exit_threshold=0.0,
    max_holding_days=30,
    stop_loss_z=4.0,
    beta_allowed=None,
    confidence=None,
):
    positions    = pd.Series(index=zscore.index, dtype=float)
    holding_days = pd.Series(index=zscore.index, dtype=float)
    exit_reasons = pd.Series(index=zscore.index, dtype=object)

    if beta_allowed is None:
        beta_allowed = pd.Series(True, index=zscore.index)
    else:
        beta_allowed = beta_allowed.reindex(zscore.index).fillna(True)

    if confidence is None:
        confidence = pd.Series(1.0, index=zscore.index)
    else:
        confidence = confidence.reindex(zscore.index).fillna(1.0)

    cur_dir  = 0
    cur_hold = 0

    for date, z in zscore.items():
        exit_reason     = None
        allowed_today   = bool(beta_allowed.loc[date])
        confidence_today = float(confidence.loc[date])

        if np.isnan(z):
            positions.loc[date]    = cur_dir * confidence_today
            holding_days.loc[date] = cur_hold
            exit_reasons.loc[date] = None
            continue

        if not allowed_today:
            if cur_dir != 0:
                cur_dir  = 0
                cur_hold = 0
                exit_reason = "beta_uncertainty_filter"
            positions.loc[date]    = 0.0
            holding_days.loc[date] = cur_hold
            exit_reasons.loc[date] = exit_reason
            continue

        if cur_dir == 0:
            cur_hold = 0
            if   z < -entry_threshold: cur_dir =  1; cur_hold = 1
            elif z >  entry_threshold: cur_dir = -1; cur_hold = 1
        else:
            cur_hold += 1
            if cur_dir == 1 and z >= exit_threshold:
                cur_dir = 0; exit_reason = "mean_reversion"
            elif cur_dir == -1 and z <= -exit_threshold:
                cur_dir = 0; exit_reason = "mean_reversion"
            elif cur_hold >= max_holding_days:
                cur_dir = 0; exit_reason = "max_holding"
            elif stop_loss_z is not None:
                if cur_dir ==  1 and z < -stop_loss_z:
                    cur_dir = 0; exit_reason = "stop_loss"
                elif cur_dir == -1 and z >  stop_loss_z:
                    cur_dir = 0; exit_reason = "stop_loss"
            if cur_dir == 0:
                cur_hold = 0

        positions.loc[date]    = cur_dir * confidence_today
        holding_days.loc[date] = cur_hold
        exit_reasons.loc[date] = exit_reason

    positions.name    = "position"
    holding_days.name = "holding_days"
    exit_reasons.name = "exit_reason"
    return positions, holding_days, exit_reasons


def calculate_strategy_returns(price_df, stock_a, stock_b, positions, betas):
    """v26_full: now returns weight_a and weight_b for leg-level cost accounting."""
    returns          = price_df[[stock_a, stock_b]].pct_change().dropna()
    lagged_positions = positions.reindex(returns.index).fillna(0).shift(1).fillna(0)
    lagged_betas     = betas.reindex(returns.index).ffill().shift(1).ffill()

    gross_exposure = (1 + lagged_betas.abs()).replace(0, np.nan)
    weight_a       = lagged_positions *  (1          / gross_exposure)
    weight_b       = lagged_positions * (-lagged_betas / gross_exposure)

    weight_a = weight_a.fillna(0)
    weight_b = weight_b.fillna(0)

    gross_returns = weight_a * returns[stock_a] + weight_b * returns[stock_b]
    gross_returns.name = "gross_strategy_return"
    return gross_returns, lagged_positions, lagged_betas, weight_a, weight_b


def apply_transaction_costs_leg_level(gross_returns, weight_a, weight_b,
                                       transaction_cost):
    """v26_full FIX 2: cost based on leg-level weight changes, not spread position.

    cost_t = (|Δweight_A_t| + |Δweight_B_t|) × tc_t

    transaction_cost can be scalar or pd.Series.
    """
    if isinstance(transaction_cost, pd.Series):
        tc = transaction_cost.reindex(gross_returns.index).ffill().fillna(0)
    else:
        tc = pd.Series(transaction_cost, index=gross_returns.index)

    turnover_leg = (weight_a.diff().abs().fillna(0)
                    + weight_b.diff().abs().fillna(0))
    costs = turnover_leg * tc
    costs.name = "transaction_cost_leg_level"
    net = gross_returns - costs
    net.name = "net_strategy_return"
    return net, costs, turnover_leg


def extract_trade_diagnostics(positions, net_returns, gross_returns, exit_reasons):
    pos          = positions.reindex(net_returns.index).fillna(0)
    net_returns  = net_returns.reindex(pos.index).fillna(0)
    gross_returns = gross_returns.reindex(pos.index).fillna(0)
    exit_reasons = exit_reasons.reindex(pos.index)
    dir_sign     = np.sign(pos)

    trades, in_trade = [], False
    entry_date = direction = None
    tnr = tgr = []

    for date in pos.index:
        p_sign = dir_sign.loc[date]
        if not in_trade and p_sign != 0:
            in_trade   = True
            entry_date = date
            direction  = p_sign
            tnr = []; tgr = []
        if in_trade:
            tnr.append(net_returns.loc[date])
            tgr.append(gross_returns.loc[date])

        next_idx = pos.index.get_loc(date) + 1
        next_sign = 0 if next_idx >= len(pos.index) else dir_sign.iloc[next_idx]

        if in_trade and (next_sign == 0 or np.sign(next_sign) != np.sign(direction)):
            er = exit_reasons.loc[date]
            if pd.isna(er):
                er = "position_closed_or_period_end"
            trades.append(dict(
                entry_date=entry_date, exit_date=date,
                direction="long_spread" if direction == 1 else "short_spread",
                holding_days=len(tnr),
                net_trade_return=np.prod(1 + np.array(tnr)) - 1,
                gross_trade_return=np.prod(1 + np.array(tgr)) - 1,
                exit_reason=er,
            ))
            in_trade = False

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        return tdf, dict(num_trades=0, trade_win_rate=np.nan,
                         avg_net_trade_return=np.nan,
                         median_net_trade_return=np.nan,
                         avg_holding_days=np.nan,
                         best_trade=np.nan, worst_trade=np.nan)

    summary = dict(
        num_trades=len(tdf),
        trade_win_rate=(tdf["net_trade_return"] > 0).mean(),
        avg_net_trade_return=tdf["net_trade_return"].mean(),
        median_net_trade_return=tdf["net_trade_return"].median(),
        avg_holding_days=tdf["holding_days"].mean(),
        best_trade=tdf["net_trade_return"].max(),
        worst_trade=tdf["net_trade_return"].min(),
    )
    return tdf, summary


# =============================================================================
# 6B. ABLATION SPEC HELPERS
# =============================================================================

ABLATION_SPECS = ["static_ols", "kalman_base", "kalman_filter", "kalman_full"]


def get_signal_components_for_spec(spec, log_price_df, stock_a, stock_b,
                                   train_start, train_end, kalman_cache, fold_name):
    """v26_full: spec dispatch reads from per-fold Kalman cache."""
    if spec == "static_ols":
        train_data = log_price_df[[stock_a, stock_b]].loc[train_start:train_end].dropna()
        full_index = log_price_df.index
        if len(train_data) < 30:
            return (pd.Series(0.0, index=full_index, name="alpha"),
                    pd.Series(1.0, index=full_index, name="beta"),
                    pd.Series(np.nan, index=full_index, name="beta_var"),
                    pd.Series(True, index=full_index, name="beta_allowed"),
                    pd.Series(1.0, index=full_index, name="confidence"))
        alpha, beta, _ = estimate_hedge_ratio(
            train_data[stock_a], train_data[stock_b])
        alphas = pd.Series(alpha, index=full_index, name="alpha")
        betas  = pd.Series(beta,  index=full_index, name="beta")
        beta_var = pd.Series(0.0, index=full_index, name="beta_var")
        beta_allowed = pd.Series(True, index=full_index, name="beta_allowed")
        confidence = pd.Series(1.0, index=full_index, name="confidence")
        return alphas, betas, beta_var, beta_allowed, confidence

    # Kalman specs from per-fold cache
    kalman_output, _ = get_or_compute_kalman_for_fold(
        kalman_cache, stock_a, stock_b, fold_name, train_start, train_end,
        log_price_df, use_em=USE_EM_KALMAN,
    )
    betas, alphas, beta_var, _ = kalman_output

    if spec == "kalman_base":
        beta_allowed = pd.Series(True, index=beta_var.index, name="beta_allowed")
        confidence   = pd.Series(1.0,  index=beta_var.index, name="confidence")
    elif spec == "kalman_filter":
        beta_allowed = compute_beta_uncertainty_allowed(beta_var)
        confidence   = pd.Series(1.0, index=beta_var.index, name="confidence")
    elif spec == "kalman_full":
        beta_allowed = compute_beta_uncertainty_allowed(beta_var)
        confidence   = compute_kalman_confidence(beta_var)
    else:
        raise ValueError(f"Unknown spec: {spec}")

    return alphas, betas, beta_var, beta_allowed, confidence


# =============================================================================
# 7. BACKTEST FUNCTIONS (v26_full)
# =============================================================================

def backtest_pair_with_spec(
    stock_a, stock_b, group, pair_type, spec,
    eval_prices, eval_start, eval_end, train_start, train_end, fold_name,
    entry_threshold, transaction_cost,
    kalman_cache, log_price_df,
    cost_series_pair=None,
):
    alphas, betas, beta_var, beta_allowed, confidence = \
        get_signal_components_for_spec(
            spec, log_price_df, stock_a, stock_b, train_start, train_end,
            kalman_cache, fold_name,
        )

    spread = compute_kalman_spread(log_price_df, stock_a, stock_b, alphas, betas)
    zscore = compute_zscore(spread, window=ROLLING_Z_WINDOW)

    eval_zscore       = zscore.loc[eval_start:eval_end]
    eval_betas        = betas.loc[eval_start:eval_end]
    eval_beta_var     = beta_var.loc[eval_start:eval_end]
    eval_beta_allowed = beta_allowed.loc[eval_start:eval_end]
    eval_confidence   = confidence.loc[eval_start:eval_end]

    positions, holding_days, exit_reasons = generate_positions_with_risk_rules(
        eval_zscore, entry_threshold=entry_threshold,
        exit_threshold=EXIT_THRESHOLD, max_holding_days=MAX_HOLDING_DAYS,
        stop_loss_z=STOP_LOSS_Z,
        beta_allowed=eval_beta_allowed, confidence=eval_confidence,
    )

    # v26_full: leg-level returns
    gross_returns, lagged_positions, lagged_betas, weight_a, weight_b = \
        calculate_strategy_returns(
            eval_prices, stock_a, stock_b, positions, eval_betas,
        )

    if cost_series_pair is not None:
        cost_input = cost_series_pair
    else:
        cost_input = transaction_cost

    # v26_full FIX 2: leg-level transaction costs
    net_returns, costs, turnover_leg = apply_transaction_costs_leg_level(
        gross_returns, weight_a, weight_b, cost_input,
    )

    gross_stats  = performance_metrics(gross_returns)
    net_stats    = performance_metrics(net_returns)
    trades_df, trade_summary = extract_trade_diagnostics(
        lagged_positions, net_returns, gross_returns, exit_reasons,
    )

    spread_turnover  = lagged_positions.diff().abs().fillna(0).mean()
    leg_turnover_avg = turnover_leg.mean()
    cost_drag = costs.sum()

    result = dict(
        spec=spec,
        group=group, pair_type=pair_type,
        stock_a=stock_a, stock_b=stock_b,
        entry_threshold=entry_threshold,
        avg_beta=eval_betas.mean(),
        std_beta=eval_betas.std(),
        avg_beta_uncertainty=eval_beta_var.mean(),
        beta_uncertainty_block_share=1 - eval_beta_allowed.mean(),
        avg_confidence=eval_confidence.mean(),
        avg_abs_position=lagged_positions.abs().mean(),
        avg_spread_turnover=spread_turnover,
        avg_leg_turnover=leg_turnover_avg,
        total_cost_drag=cost_drag,
        net_total_return=net_stats["Total_Return"],
        net_annual_return=net_stats["Annual_Return"],
        net_annual_volatility=net_stats["Annual_Volatility"],
        net_sharpe=net_stats["Sharpe_Ratio"],
        net_sharpe_se=net_stats["Sharpe_SE"],
        net_max_drawdown=net_stats["Max_Drawdown"],
        net_daily_hit_rate=net_stats["Daily_Hit_Rate"],
        gross_total_return=gross_stats["Total_Return"],
        gross_sharpe=gross_stats["Sharpe_Ratio"],
        gross_sharpe_se=gross_stats["Sharpe_SE"],
        num_trades=trade_summary["num_trades"],
        trade_win_rate=trade_summary["trade_win_rate"],
        avg_net_trade_return=trade_summary["avg_net_trade_return"],
        median_net_trade_return=trade_summary["median_net_trade_return"],
        avg_holding_days=trade_summary["avg_holding_days"],
        best_trade=trade_summary["best_trade"],
        worst_trade=trade_summary["worst_trade"],
    )

    pair_returns = pd.DataFrame({
        f"{stock_a}_{stock_b}_gross":    gross_returns,
        f"{stock_a}_{stock_b}_net":      net_returns,
        f"{stock_a}_{stock_b}_position": lagged_positions,
    })
    return result, pair_returns, trades_df


def portfolio_backtest_spec(
    selected_pairs, spec, eval_prices, eval_start, eval_end, train_start, train_end,
    fold_name, threshold, transaction_cost, kalman_cache, log_price_df,
    use_roll_costs=False,
):
    pair_metrics, all_net, all_gross, all_trades = [], [], [], []

    for _, row in selected_pairs.iterrows():
        stock_a   = row["stock_a"]
        stock_b   = row["stock_b"]
        group     = row["group"]
        pair_type = row.get("pair_type", "unknown")

        if (stock_a not in log_price_df.columns
                or stock_b not in log_price_df.columns):
            continue

        cost_series_pair = None
        if use_roll_costs:
            cost_bps_series = pair_roll_cost_bps(
                eval_prices, stock_a, stock_b, window=60)
            cost_series_pair = cost_bps_series / 10_000.0

        result, pair_returns, trades_df = backtest_pair_with_spec(
            stock_a=stock_a, stock_b=stock_b,
            group=group, pair_type=pair_type, spec=spec,
            eval_prices=eval_prices,
            eval_start=eval_start, eval_end=eval_end, train_start=train_start,
            train_end=train_end,
            fold_name=fold_name,
            entry_threshold=threshold,
            transaction_cost=transaction_cost,
            kalman_cache=kalman_cache,
            log_price_df=log_price_df,
            cost_series_pair=cost_series_pair,
        )

        for col in ["pvalue", "pvalue_bh_adj", "passes_fdr", "beta",
                    "r_squared", "half_life", "training_stability_score",
                    "beta_stability", "half_life_ratio",
                    "fdr_q_threshold_used",
                    "selection_mode"]:
            if col in row:
                result[f"selection_{col}"] = row[col]

        pair_metrics.append(result)
        all_net.append(pair_returns[f"{stock_a}_{stock_b}_net"])
        all_gross.append(pair_returns[f"{stock_a}_{stock_b}_gross"])

        if not trades_df.empty:
            trades_df["stock_a"]   = stock_a
            trades_df["stock_b"]   = stock_b
            trades_df["group"]     = group
            trades_df["pair_type"] = pair_type
            trades_df["spec"]      = spec
            all_trades.append(trades_df)

    pair_metrics_df = pd.DataFrame(pair_metrics)

    if len(all_net) == 0:
        empty = dict(spec=spec, entry_threshold=threshold,
                     transaction_cost=transaction_cost,
                     num_pairs=0, net_total_return=np.nan, net_annual_return=np.nan,
                     net_annual_volatility=np.nan, net_sharpe=np.nan,
                     net_sharpe_se=np.nan, net_max_drawdown=np.nan,
                     net_daily_hit_rate=np.nan, trade_win_rate=np.nan,
                     num_trades=0)
        return empty, pair_metrics_df, pd.Series(dtype=float), pd.DataFrame()


    net_df   = pd.concat(all_net,   axis=1).dropna(how="all")
    gross_df = pd.concat(all_gross, axis=1).dropna(how="all")

    pair_vols = net_df.std().replace(0, np.nan)
    if pair_vols.notna().sum() == 0:
        weights = pd.Series(1 / net_df.shape[1], index=net_df.columns)
    else:
        inv_vol = 1 / pair_vols
        weights = (inv_vol / inv_vol.sum()).fillna(0)

    net_portfolio   = net_df.fillna(0).dot(
        weights.reindex(net_df.columns).fillna(0))
    gross_portfolio = gross_df.fillna(0).dot(
        weights.reindex(gross_df.columns).fillna(0))

    net_stats   = performance_metrics(net_portfolio)
    gross_stats = performance_metrics(gross_portfolio)

    if all_trades:
        trades_all       = pd.concat(all_trades, axis=0, ignore_index=True)
        trade_win_rate   = (trades_all["net_trade_return"] > 0).mean()
        avg_trade_return = trades_all["net_trade_return"].mean()
        avg_hold_days    = trades_all["holding_days"].mean()
        num_trades       = len(trades_all)
    else:
        trades_all     = pd.DataFrame()
        trade_win_rate = avg_trade_return = avg_hold_days = np.nan
        num_trades     = 0

    portfolio_result = dict(
        spec=spec,
        entry_threshold=threshold, transaction_cost=transaction_cost,
        num_pairs=len(selected_pairs),
        portfolio_weighting="inverse_volatility",
        max_pair_weight=weights.max(), min_pair_weight=weights.min(),
        gross_total_return=gross_stats["Total_Return"],
        gross_sharpe=gross_stats["Sharpe_Ratio"],
        gross_sharpe_se=gross_stats["Sharpe_SE"],
        net_total_return=net_stats["Total_Return"],
        net_annual_return=net_stats["Annual_Return"],
        net_annual_volatility=net_stats["Annual_Volatility"],
        net_sharpe=net_stats["Sharpe_Ratio"],
        net_sharpe_se=net_stats["Sharpe_SE"],
        net_max_drawdown=net_stats["Max_Drawdown"],
        net_daily_hit_rate=net_stats["Daily_Hit_Rate"],
        num_trades=num_trades,
        trade_win_rate=trade_win_rate,
        avg_trade_return=avg_trade_return,
        avg_holding_days=avg_hold_days,
    )

    return portfolio_result, pair_metrics_df, net_portfolio, trades_all


# =============================================================================
# 8. NULL BENCHMARK (v26_full FIX 5: uses EM Kalman for random pairs)
# =============================================================================

def run_null_benchmark_for_fold(
    fold_cfg, log_price_df, eval_prices, kalman_cache,
    sector_groups, sector_etfs, selected_pairs, threshold, transaction_cost,
    n_reps=NULL_BENCHMARK_REPS, seed=NULL_BENCHMARK_SEED,
):
    """v26_full: random pairs use EM Kalman from this fold's training window,
    matching the selected portfolio's specification."""
    rng = np.random.default_rng(seed)

    if selected_pairs.empty:
        return pd.DataFrame()

    composition = (selected_pairs.groupby(["group", "pair_type"]).size()
                                  .reset_index(name="n"))

    test_start = fold_cfg["test_start"]
    test_end   = fold_cfg["test_end"]
    train_start = fold_cfg["train_start"]
    train_end  = fold_cfg["train_end"]
    fold_name  = fold_cfg["fold"]

    null_sharpes = []
    for rep in range(n_reps):
        random_pairs = []
        for _, row in composition.iterrows():
            group = row["group"]
            pair_type = row["pair_type"]
            n_needed = row["n"]
            avail_in_group = [t for t in sector_groups.get(group, [])
                              if t in log_price_df.columns]
            etf = sector_etfs.get(group)

            if pair_type == "stock_stock":
                all_pairs = list(combinations(avail_in_group, 2))
            elif pair_type == "stock_etf" and etf and etf in log_price_df.columns:
                all_pairs = [(s, etf) for s in avail_in_group if s != etf]
            else:
                continue

            if len(all_pairs) < n_needed:
                chosen = all_pairs
            else:
                idxs = rng.choice(len(all_pairs), size=n_needed, replace=False)
                chosen = [all_pairs[i] for i in idxs]

            for sa, sb in chosen:
                random_pairs.append({
                    "group": group, "pair_type": pair_type,
                    "stock_a": sa, "stock_b": sb,
                })

        if not random_pairs:
            continue

        random_df = pd.DataFrame(random_pairs)

        # v26_full FIX 4: ensure all random pairs have EM Kalman for this fold,
        # matching the selected portfolio's Kalman calibration.
        for _, rrow in random_df.iterrows():
            sa, sb = rrow["stock_a"], rrow["stock_b"]
            if (sa, sb, fold_name) not in kalman_cache:
                if sa in log_price_df.columns and sb in log_price_df.columns:
                    get_or_compute_kalman_for_fold(
                        kalman_cache, sa, sb, fold_name, train_start, train_end,
                        log_price_df, use_em=USE_EM_KALMAN,
                    )

        try:
            null_result, _, _, _ = portfolio_backtest_spec(
                selected_pairs=random_df, spec="kalman_full",
                eval_prices=eval_prices,
                eval_start=test_start, eval_end=test_end, train_start=train_start,
                train_end=train_end,
                fold_name=fold_name,
                threshold=threshold, transaction_cost=transaction_cost,
                kalman_cache=kalman_cache, log_price_df=log_price_df,
                use_roll_costs=False,
            )
            null_sharpes.append({
                "rep": rep,
                "null_sharpe": null_result["net_sharpe"],
                "null_return": null_result["net_total_return"],
                "null_num_trades": null_result["num_trades"],
            })
        except Exception:
            continue

    return pd.DataFrame(null_sharpes)


# =============================================================================
# 9. HMM REGIME CLASSIFIER
# =============================================================================

def classify_regime_hmm_at_test_start(market_returns_series, test_start_date):
    """v26_full FIX 7: explicit naming.

    Fit HMM on market returns up to test_start_date, then identify the
    dominant latent state in the 60 trading days immediately before
    test_start_date. The label describes the regime at test entry, not the
    within-test-window regime.
    """
    if not HMMLEARN_AVAILABLE or market_returns_series.empty:
        return "unclassified"

    train_returns = market_returns_series.loc[:test_start_date].dropna()
    if len(train_returns) < 252:
        return "unclassified"

    X = train_returns.values.reshape(-1, 1)
    try:
        hmm = GaussianHMM(n_components=2, covariance_type="full",
                           n_iter=200, random_state=42)
        hmm.fit(X)
        variances = hmm.covars_.flatten()
        high_vol_state = int(np.argmax(variances))

        recent = train_returns.tail(60).values.reshape(-1, 1)
        preds = hmm.predict(recent)
        if (preds == high_vol_state).mean() > 0.5:
            return "high_vol_at_test_start"
        else:
            return "low_vol_at_test_start"
    except Exception:
        return "unclassified"


# =============================================================================
# 10. AVELLANEDA-LEE ETF-RESIDUAL STRATEGY (unchanged)
# =============================================================================

def sector_etf_residual_signals(
    eval_prices, sector_groups, sector_etfs,
    z_window=60, train_window_days=252,
):
    """Sector ETF residual mean-reversion signals (Avellaneda-Lee style).
 
    For each stock, rolling-regress returns on the sector ETF return.
    Compute residuals, take the cumulative sum (an OU process proxy), and
    z-score. Trade when the z-score exceeds threshold.
 
    Note: this is the ETF-residual variant of Avellaneda-Lee (2010), not the
    PCA-residual variant. PCA would extract principal components across the
    universe; here we use a single sector ETF as the explanatory factor.
    """

    returns = eval_prices.pct_change().dropna()
    signals = {}

    for group, tickers in sector_groups.items():
        etf = sector_etfs.get(group)
        if etf is None or etf not in returns.columns:
            continue
        avail = [t for t in tickers if t in returns.columns and t != etf]
        for stock in avail:
            y = returns[stock]
            x = returns[etf]
            cov_xy = y.rolling(train_window_days).cov(x)
            var_x  = x.rolling(train_window_days).var()
            beta   = cov_xy / var_x
            alpha  = (y.rolling(train_window_days).mean()
                      - beta * x.rolling(train_window_days).mean())
            resid = y - (alpha + beta * x)
            cum_resid = resid.cumsum()
            zs = (cum_resid - cum_resid.rolling(z_window).mean()) \
                 / cum_resid.rolling(z_window).std()
            signals[stock] = zs

    if not signals:
        return pd.DataFrame()
    return pd.DataFrame(signals)


def sector_etf_residual_portfolio_backtest(
    eval_prices, sector_groups, sector_etfs,
    eval_start, eval_end, entry_threshold=2.0,
    exit_threshold=0.5, max_holding_days=30,
    transaction_cost=5/10_000,
):
    signals = sector_etf_residual_signals(eval_prices, sector_groups,
                                              sector_etfs)
    if signals.empty:
        return dict(num_trades=0, net_sharpe=np.nan, net_total_return=np.nan), \
               pd.Series(dtype=float), pd.DataFrame()

    signals = signals.loc[eval_start:eval_end]
    returns = eval_prices.pct_change().dropna().loc[eval_start:eval_end]

    all_stock_returns = []
    pair_records = []

    for group, tickers in sector_groups.items():
        etf = sector_etfs.get(group)
        if etf is None or etf not in returns.columns:
            continue
        avail = [t for t in tickers if t in signals.columns and t in returns.columns]

        for stock in avail:
            zs = signals[stock].dropna()
            if zs.empty:
                continue

            positions = pd.Series(0.0, index=zs.index)
            cur_dir, cur_hold = 0, 0
            for date, z in zs.items():
                if np.isnan(z):
                    positions.loc[date] = cur_dir
                    continue
                if cur_dir == 0:
                    if z >  entry_threshold: cur_dir = -1; cur_hold = 1
                    elif z < -entry_threshold: cur_dir =  1; cur_hold = 1
                else:
                    cur_hold += 1
                    if cur_dir == 1 and z >= -exit_threshold:
                        cur_dir = 0
                    elif cur_dir == -1 and z <= exit_threshold:
                        cur_dir = 0
                    elif cur_hold >= max_holding_days:
                        cur_dir = 0
                    if cur_dir == 0:
                        cur_hold = 0
                positions.loc[date] = cur_dir

            common = returns[[stock, etf]].dropna()
            if len(common) < 30:
                continue
            beta = common[stock].cov(common[etf]) / common[etf].var()

            lagged_pos = positions.reindex(returns.index).fillna(0).shift(1).fillna(0)
            gross_exposure = 1 + abs(beta)
            weight_stock = lagged_pos * (1 / gross_exposure)
            weight_etf   = lagged_pos * (-beta / gross_exposure)

            stock_return = (weight_stock * returns[stock]
                            + weight_etf * returns[etf])
            # v26_full: leg-level cost here too for consistency
            costs = (weight_stock.diff().abs().fillna(0)
                     + weight_etf.diff().abs().fillna(0)) * transaction_cost
            stock_net = stock_return - costs

            all_stock_returns.append(stock_net.rename(stock))
            pair_records.append({
                "stock": stock, "etf": etf, "group": group,
                "beta_hedge": beta,
                "n_trades": int(((lagged_pos.diff().abs() > 0)).sum()),
            })

    if not all_stock_returns:
        return dict(num_trades=0, net_sharpe=np.nan, net_total_return=np.nan), \
               pd.Series(dtype=float), pd.DataFrame()

    portfolio = pd.concat(all_stock_returns, axis=1).fillna(0)
    vols = portfolio.std().replace(0, np.nan)
    inv_vol = 1 / vols
    weights = (inv_vol / inv_vol.sum()).fillna(0)
    portfolio_returns = portfolio.dot(weights)

    perf = performance_metrics(portfolio_returns)
    return (
        dict(num_trades=sum(r["n_trades"] for r in pair_records),
             net_sharpe=perf["Sharpe_Ratio"],
             net_total_return=perf["Total_Return"],
             net_max_drawdown=perf["Max_Drawdown"],
             num_assets=len(all_stock_returns)),
        portfolio_returns,
        pd.DataFrame(pair_records),
    )


# =============================================================================
# 11. WALK-FORWARD LOOP (v26_full: per-fold EM, FDR grid, leg-level costs)
# =============================================================================

# v26_full: Kalman cache keyed by (stock_a, stock_b, fold_name).
# Populated lazily as folds run.
KALMAN_CACHE = {}

# v26_full FIX 4: ablation results now include kalman_full directly
all_fold_summaries             = []
all_fold_pair_metrics          = []
all_fold_trades                = []
all_fold_returns               = []
all_fold_training_pair_summaries = []
all_fold_ablation               = []
all_fold_null_benchmark         = []
all_fold_hmm_regimes            = []
all_fold_sector_etf_residual           = []
all_em_params_records           = []


def select_best_threshold_nan_safe(validation_df):
    """v26_full FIX 6: explicit NaN handling so dead thresholds cannot be selected.

    If all thresholds produce NaN Sharpe, return the lowest threshold as a
    deterministic fallback.
    """
    if validation_df.empty:
        return THRESHOLDS_TO_TEST[0]

    df = validation_df.copy()
    df["net_sharpe_rank"]   = df["net_sharpe"].fillna(-np.inf)
    df["num_trades_safe"]   = df["num_trades"].fillna(0)
    df["net_total_return_safe"] = df["net_total_return"].fillna(-np.inf)

    df_sorted = df.sort_values(
        ["net_sharpe_rank", "net_total_return_safe", "num_trades_safe"],
        ascending=[False, False, False],
    )
    return df_sorted.iloc[0]["entry_threshold"]


# =============================================================================
# Walk-forward loop: outer over FDR thresholds, inner over folds.
# Each (fold, fdr_q) combination is a separate run, results tagged accordingly.
# =============================================================================

for fdr_q in FDR_Q_VALUES:
    print("\n" + "#"*90)
    print(f"# FDR sweep: q <= {fdr_q}")
    print("#"*90)

    for fold_cfg in WALK_FORWARD_FOLDS:

        fold_name    = fold_cfg["fold"]
        train_start  = fold_cfg["train_start"]
        train_end    = fold_cfg["train_end"]
        valid_start  = fold_cfg["valid_start"]
        valid_end    = fold_cfg["valid_end"]
        test_start   = fold_cfg["test_start"]
        test_end     = fold_cfg["test_end"]

        print("\n" + "="*90)
        print(f"Running {fold_name}  (FDR q ≤ {fdr_q})")
        print("="*90)

        train_log    = log_prices.loc[train_start:train_end].copy()
        valid_prices = prices.loc[valid_start:valid_end].copy()
        test_prices  = prices.loc[test_start:test_end].copy()

        hmm_regime = "disabled"
        if USE_HMM_REGIMES:
            hmm_regime = classify_regime_hmm_at_test_start(market_returns,
                                                           test_start)
        all_fold_hmm_regimes.append({"fold": fold_name,
                                      "fdr_q": fdr_q,
                                      "hmm_regime_at_test_start": hmm_regime,
                                      "test_start": test_start,
                                      "test_end": test_end})

        all_pairs, selected_pairs = find_cointegrated_pairs(
            train_log, SECTOR_GROUPS, SECTOR_ETFS, fdr_q_threshold=fdr_q)

        # Save training selection per (fold, fdr_q)
        all_pairs.to_csv(
            TABLES_DIR / f"{fold_name}_fdr{int(fdr_q*100):02d}_all_training_pairs.csv",
            index=False)
        selected_pairs.to_csv(
            TABLES_DIR / f"{fold_name}_fdr{int(fdr_q*100):02d}_selected_training_pairs.csv",
            index=False)

        print(f"Candidate pairs tested:  {len(all_pairs)}")
        print(f"Training-selected pairs: {len(selected_pairs)}")

        all_fold_training_pair_summaries.append(dict(
            fold=fold_name, fdr_q=fdr_q, hmm_regime_at_test_start=hmm_regime,
            num_candidate_pairs_tested=len(all_pairs),
            num_raw_pvalue_pass=int((all_pairs["pvalue"] < COINTEGRATION_PVALUE_THRESHOLD).sum())
                if not all_pairs.empty else 0,
            num_fdr_pass=int(all_pairs["passes_fdr"].sum()) if not all_pairs.empty else 0,
            num_training_selected=len(selected_pairs),
            num_selected_fdr_pass=int((selected_pairs["selection_mode"] == "fdr_pass").sum())
                if not selected_pairs.empty else 0,                                          # NEW
            num_selected_fallback=int((selected_pairs["selection_mode"] == "top_raw_p_fallback").sum())
                if not selected_pairs.empty else 0, 
            num_stock_stock_selected=int((selected_pairs["pair_type"] == "stock_stock").sum())
                if not selected_pairs.empty else 0,
            num_stock_etf_selected=int((selected_pairs["pair_type"] == "stock_etf").sum())
                if not selected_pairs.empty else 0,
        ))

        if selected_pairs.empty:
            print("  No pairs selected. Skipping fold.")
            continue

        # v26_full FIX 1: per-fold EM Kalman estimation for every selected pair
        for _, row in selected_pairs.iterrows():
            sa, sb = row["stock_a"], row["stock_b"]
            if sa in log_prices.columns and sb in log_prices.columns:
                _, em_params = get_or_compute_kalman_for_fold(
                    KALMAN_CACHE, sa, sb, fold_name, train_start, train_end,
                    log_prices, use_em=USE_EM_KALMAN,
                )
                all_em_params_records.append({
                    "fold": fold_name, "fdr_q": fdr_q,
                    "stock_a": sa, "stock_b": sb,
                    "Q_estimated": em_params["Q"],
                    "R_estimated": em_params["R"],
                })

        selected_pairs_to_trade = selected_pairs.head(MAX_PAIRS_TO_TRADE).copy()

        # ---------------------------------------------------------------------
        # Validation: choose threshold under kalman_full at 5 bps
        # ---------------------------------------------------------------------
        validation_results = []
        validation_pair_results = {}
        baseline_tc = 5 / 10_000

        for threshold in THRESHOLDS_TO_TEST:
            valid_result, valid_pair_metrics, _, _ = portfolio_backtest_spec(
                selected_pairs=selected_pairs_to_trade, spec="kalman_full",
                eval_prices=valid_prices,
                eval_start=valid_start, eval_end=valid_end, train_start=train_start,
                train_end=train_end,
                fold_name=fold_name,
                threshold=threshold, transaction_cost=baseline_tc,
                kalman_cache=KALMAN_CACHE, log_price_df=log_prices,
                use_roll_costs=False,
            )
            valid_result["fold"] = fold_name
            valid_result["fdr_q"] = fdr_q
            validation_results.append(valid_result)
            validation_pair_results[threshold] = valid_pair_metrics

        validation_df = pd.DataFrame(validation_results)
        validation_df.to_csv(
            TABLES_DIR / f"{fold_name}_fdr{int(fdr_q*100):02d}_validation_thresholds.csv",
            index=False)

        # v26_full FIX 6: NaN-safe threshold selection
        best_threshold = select_best_threshold_nan_safe(validation_df)
        print(f"Chosen threshold: {best_threshold}")

        # ---------------------------------------------------------------------
        # Validation pair scoring
        # ---------------------------------------------------------------------
        valid_pair_metrics = validation_pair_results[best_threshold].copy()
        if valid_pair_metrics.empty:
            print("  No validation pair metrics. Skipping fold.")
            continue

        def validation_score(row):
            score  = 2 * int(row["num_trades"] >= MIN_VALIDATION_TRADES)
            score += 1 * int(row["net_sharpe"] > RELAXED_MIN_SHARPE)
            score += 1 * int(row["net_max_drawdown"] >= RELAXED_MAX_DRAWDOWN)
            score += 1 * int(row.get("avg_beta_uncertainty", 1.0) < 0.05)
            score += 1 * int(row["net_total_return"] > 0)
            score += 1 * int(row["net_sharpe"] > 0)
            score += 1 * int(bool(row.get("selection_passes_fdr", False)))
            return score

        valid_pair_metrics["validation_score"] = valid_pair_metrics.apply(
            validation_score, axis=1)
        valid_pair_metrics = valid_pair_metrics.sort_values(
            ["validation_score", "net_sharpe", "net_total_return",
             "avg_beta_uncertainty"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        
        # Remove validation pair-folds that are too thin to be considered real signals.
        # This prevents 1-day / 1-trade artifacts from entering final_pairs.
        MIN_VALIDATION_TRADES_FOR_SELECTION = 2
        MIN_VALIDATION_AVG_HOLDING_DAYS = 2
        
# =============================================================================
#         valid_pair_metrics_tradeable = valid_pair_metrics[
#             (valid_pair_metrics["num_trades"] >= MIN_VALIDATION_TRADES_FOR_SELECTION) &
#             (valid_pair_metrics["avg_holding_days"] >= MIN_VALIDATION_AVG_HOLDING_DAYS)
#         ].copy()
# =============================================================================
        
        print(f"Validation pair candidates: {len(valid_pair_metrics)}")
        print(
            "Validation pair candidates with >=2 trades and >=2 avg holding days:",
            (
                (valid_pair_metrics["num_trades"] >= 2) &
                (valid_pair_metrics["avg_holding_days"] >= 2)
            ).sum()
        )
        primary = valid_pair_metrics[
            (valid_pair_metrics["num_trades"] >= MIN_VALIDATION_TRADES_STRICT) &
            (valid_pair_metrics["net_sharpe"] > MIN_VALIDATION_SHARPE_STRICT) &
            (valid_pair_metrics["net_total_return"] > 0)
        ].copy()
        
        fallback = valid_pair_metrics[
            (valid_pair_metrics["num_trades"] >= MIN_VALIDATION_TRADES_FALLBACK) &
            (valid_pair_metrics["net_sharpe"] > MIN_VALIDATION_SHARPE_FALLBACK) &
            (valid_pair_metrics["net_total_return"] > 0)
        ].copy()

        valid_pair_metrics_filtered = (
            pd.concat([primary, fallback])
            .drop_duplicates(subset=["stock_a", "stock_b", "pair_type"])
            .sort_values(
                ["validation_score", "net_sharpe", "net_total_return"],
                ascending=[False, False, False]
            )
        )

        if valid_pair_metrics_filtered.empty:
            print("  No validation-approved pairs. Skipping fold.")
            continue

        keep_keys = valid_pair_metrics_filtered.head(MAX_VALIDATION_PAIRS)[
            ["stock_a", "stock_b", "pair_type"]
        ]
        final_pairs = selected_pairs_to_trade.merge(
            keep_keys, on=["stock_a", "stock_b", "pair_type"], how="inner")

        # ---------------------------------------------------------------------
        # TEST: TC sensitivity + ablation
        # v26_full FIX 3: kalman_full now appended to ablation results too
        # ---------------------------------------------------------------------
        for tc_bps in TRANSACTION_COSTS_BPS:
            tc = tc_bps / 10_000

            test_result, test_pair_metrics, test_net_returns, test_trades = \
                portfolio_backtest_spec(
                    selected_pairs=final_pairs, spec="kalman_full",
                    eval_prices=test_prices,
                    eval_start=test_start, eval_end=test_end, train_start=train_start,
                    train_end=train_end,
                    fold_name=fold_name,
                    threshold=best_threshold, transaction_cost=tc,
                    kalman_cache=KALMAN_CACHE, log_price_df=log_prices,
                    use_roll_costs=USE_ROLL_SPREAD_COST,
                )

            test_result.update(dict(
                fold=fold_name, fdr_q=fdr_q,
                hmm_regime_at_test_start=hmm_regime,
                test_start=test_start, test_end=test_end,
                chosen_threshold=best_threshold,
                transaction_cost_bps=tc_bps,
                num_training_pairs=len(selected_pairs_to_trade),
                num_final_pairs=len(final_pairs),
                num_final_stock_stock=int((final_pairs["pair_type"] == "stock_stock").sum()),
                num_final_stock_etf=int((final_pairs["pair_type"] == "stock_etf").sum()),
                num_final_fdr_pass=int(final_pairs["passes_fdr"].sum()),
                cost_model=("roll" if USE_ROLL_SPREAD_COST else "flat_bps"),
                cost_accounting="leg_level",
            ))
            all_fold_summaries.append(test_result)

            # v26_full FIX 3: also store kalman_full in the ablation table
            ablation_kalman_full = {k: v for k, v in test_result.items()}
            ablation_kalman_full["spec"] = "kalman_full"
            all_fold_ablation.append(ablation_kalman_full)

            test_pair_metrics["fold"]                 = fold_name
            test_pair_metrics["fdr_q"]                = fdr_q
            test_pair_metrics["transaction_cost_bps"] = tc_bps
            test_pair_metrics["hmm_regime_at_test_start"] = hmm_regime
            all_fold_pair_metrics.append(test_pair_metrics)

            if not test_trades.empty:
                test_trades["fold"]                 = fold_name
                test_trades["fdr_q"]                = fdr_q
                test_trades["transaction_cost_bps"] = tc_bps
                all_fold_trades.append(test_trades)

            all_fold_returns.append(pd.DataFrame({
                "date":                test_net_returns.index,
                "net_return":          test_net_returns.values,
                "fold":                fold_name,
                "fdr_q":               fdr_q,
                "transaction_cost_bps": tc_bps,
                "spec":                "kalman_full",
            }))

            # Other ablation specs
            if RUN_ABLATION:
                for spec in ["static_ols", "kalman_base", "kalman_filter"]:
                    ablation_result, _, _, _ = portfolio_backtest_spec(
                        selected_pairs=final_pairs, spec=spec,
                        eval_prices=test_prices,
                        eval_start=test_start, eval_end=test_end,train_start=train_start,
                        train_end=train_end, fold_name=fold_name,
                        threshold=best_threshold, transaction_cost=tc,
                        kalman_cache=KALMAN_CACHE, log_price_df=log_prices,
                        use_roll_costs=USE_ROLL_SPREAD_COST,
                    )
                    ablation_result.update(dict(
                        fold=fold_name, fdr_q=fdr_q,
                        hmm_regime_at_test_start=hmm_regime,
                        transaction_cost_bps=tc_bps,
                    ))
                    all_fold_ablation.append(ablation_result)

        # ---- Null benchmark (5 bps only, primary FDR only) ----
        if RUN_NULL_BENCHMARK and fdr_q == FDR_Q_PRIMARY:
            print(f"  Running null benchmark ({NULL_BENCHMARK_REPS} reps)...")
            null_df = run_null_benchmark_for_fold(
                fold_cfg, log_prices, test_prices, KALMAN_CACHE,
                SECTOR_GROUPS, SECTOR_ETFS,
                selected_pairs=final_pairs,
                threshold=best_threshold,
                transaction_cost=5/10_000,
                n_reps=NULL_BENCHMARK_REPS, seed=NULL_BENCHMARK_SEED,
            )
            if not null_df.empty:
                null_df["fold"] = fold_name
                null_df["fdr_q"] = fdr_q
                null_df["selected_sharpe"] = next(
                    (r["net_sharpe"] for r in all_fold_summaries
                     if r["fold"] == fold_name
                     and r["fdr_q"] == fdr_q
                     and r["transaction_cost_bps"] == 5),
                    np.nan
                )
                all_fold_null_benchmark.append(null_df)

        if RUN_SECTOR_ETF_RESIDUAL_STRATEGY and fdr_q == FDR_Q_PRIMARY:
            residual_prices = prices.loc[train_start:test_end].copy()
        
            etf_summary, etf_returns, etf_pairs = sector_etf_residual_portfolio_backtest(
                residual_prices,
                SECTOR_GROUPS,
                SECTOR_ETFS,
                eval_start=test_start,
                eval_end=test_end,
                entry_threshold=2.0,
                transaction_cost=5/10_000,
            )
        
            etf_summary.update(dict(
                fold=fold_name,
                fdr_q=fdr_q,
                hmm_regime_at_test_start=hmm_regime,
                transaction_cost_bps=5,
                benchmark_type="sector_etf_residual",
            ))
        
            all_fold_sector_etf_residual.append(etf_summary)


# =============================================================================
# 12. SAVE COMBINED OUTPUTS
# =============================================================================

fold_summary_df = pd.DataFrame(all_fold_summaries)
fold_summary_df.to_csv(TABLES_DIR / "v26_full_fold_summary.csv", index=False)

pair_metrics_df = (pd.concat(all_fold_pair_metrics, ignore_index=True)
                   if all_fold_pair_metrics else pd.DataFrame())
pair_metrics_df.to_csv(TABLES_DIR / "v26_full_pair_metrics.csv", index=False)


# =============================================================================
# CLEAN PAIR-METRICS TABLE FOR HONEST PAIR-LEVEL AGGREGATES
# =============================================================================

MIN_TRADES_FOR_AGGREGATE = 2
MIN_AVG_HOLDING_DAYS = 2

for col in ["num_trades", "avg_holding_days", "net_sharpe", "net_total_return"]:
    if col in pair_metrics_df.columns:
        pair_metrics_df[col] = pd.to_numeric(pair_metrics_df[col], errors="coerce")

pair_metrics_df_clean = pair_metrics_df[
    (pair_metrics_df["num_trades"] >= MIN_TRADES_FOR_AGGREGATE) &
    (pair_metrics_df["avg_holding_days"] >= MIN_AVG_HOLDING_DAYS)
].copy()

pair_metrics_df_clean.to_csv(
    TABLES_DIR / "v26_full_pair_metrics_clean.csv", index=False)

print(f"Pair-folds before filter: {len(pair_metrics_df)}")
print(f"Pair-folds after filter:  {len(pair_metrics_df_clean)}")

# =============================================================================
# CLEAN VS RAW PAIR-LEVEL DIAGNOSTICS
# =============================================================================

def pair_level_trade_density_summary(df, label):
    if df.empty:
        return pd.DataFrame()

    return (
        df.groupby(["fdr_q", "transaction_cost_bps"])
        .agg(
            n_pair_folds=("net_sharpe", "count"),
            mean_trades=("num_trades", "mean"),
            median_trades=("num_trades", "median"),
            mean_holding=("avg_holding_days", "mean"),
            median_holding=("avg_holding_days", "median"),
            mean_sharpe=("net_sharpe", "mean"),
            median_sharpe=("net_sharpe", "median"),
            mean_return=("net_total_return", "mean"),
        )
        .reset_index()
        .assign(sample=label)
    )

raw_pair_density = pair_level_trade_density_summary(pair_metrics_df, "raw")
clean_pair_density = pair_level_trade_density_summary(pair_metrics_df_clean, "clean")

pair_density_diagnostics = pd.concat(
    [raw_pair_density, clean_pair_density],
    ignore_index=True
)

pair_density_diagnostics.to_csv(
    TABLES_DIR / "v26_full_pair_trade_density_diagnostics.csv",
    index=False
)

print("\n" + "="*90)
print("PAIR-LEVEL TRADE DENSITY DIAGNOSTICS")
print("="*90)
print(pair_density_diagnostics.to_string(index=False))

# =============================================================================
# SELECTION MODE DECOMPOSITION — RAW VS CLEAN
# =============================================================================

selection_mode_diagnostics = []

for label, df in [("raw", pair_metrics_df), ("clean", pair_metrics_df_clean)]:
    if df.empty or "selection_selection_mode" not in df.columns:
        continue

    tmp = (
        df.groupby(["fdr_q", "transaction_cost_bps", "selection_selection_mode"])
        .agg(
            n_pair_folds=("net_sharpe", "count"),
            mean_sharpe=("net_sharpe", "mean"),
            median_sharpe=("net_sharpe", "median"),
            mean_trades=("num_trades", "mean"),
            median_trades=("num_trades", "median"),
            mean_holding_days=("avg_holding_days", "mean"),
            median_holding_days=("avg_holding_days", "median"),
            mean_return=("net_total_return", "mean"),
        )
        .reset_index()
    )
    tmp["sample"] = label
    selection_mode_diagnostics.append(tmp)

if selection_mode_diagnostics:
    selection_mode_diagnostics_df = pd.concat(
        selection_mode_diagnostics,
        ignore_index=True
    )

    selection_mode_diagnostics_df.to_csv(
        TABLES_DIR / "v26_full_selection_mode_diagnostics_raw_vs_clean.csv",
        index=False
    )

    print("\n" + "="*90)
    print("SELECTION MODE DIAGNOSTICS — RAW VS CLEAN")
    print("="*90)
    print(selection_mode_diagnostics_df.to_string(index=False))

trades_df = (pd.concat(all_fold_trades, ignore_index=True)
             if all_fold_trades else pd.DataFrame())
trades_df.to_csv(TABLES_DIR / "v26_full_trade_level_diagnostics.csv", index=False)

returns_all_df = (pd.concat(all_fold_returns, ignore_index=True)
                  if all_fold_returns else pd.DataFrame())
returns_all_df.to_csv(TABLES_DIR / "v26_full_walkforward_returns.csv", index=False)

training_pair_summary_df = pd.DataFrame(all_fold_training_pair_summaries)
training_pair_summary_df.to_csv(
    TABLES_DIR / "v26_full_training_pair_selection_summary.csv", index=False)

ablation_df = pd.DataFrame(all_fold_ablation)
ablation_df.to_csv(TABLES_DIR / "v26_full_ablation_results.csv", index=False)

em_params_df = pd.DataFrame(all_em_params_records)
em_params_df.to_csv(TABLES_DIR / "v26_full_kalman_em_parameters_by_fold.csv",
                    index=False)

if all_fold_null_benchmark:
    null_benchmark_df = pd.concat(all_fold_null_benchmark, ignore_index=True)
    null_benchmark_df.to_csv(TABLES_DIR / "v26_full_null_benchmark.csv", index=False)
else:
    null_benchmark_df = pd.DataFrame()

hmm_regimes_df = pd.DataFrame(all_fold_hmm_regimes)
hmm_regimes_df.to_csv(TABLES_DIR / "v26_full_hmm_regimes.csv", index=False)

sector_etf_residual_df = pd.DataFrame(all_fold_sector_etf_residual)
sector_etf_residual_df.to_csv(TABLES_DIR / "v26_full_sector_etf_residual_results.csv", index=False)


# =============================================================================
# 13. AGGREGATED RESULTS (v26_full: by FDR threshold + non-overlapping inference)
# =============================================================================

def non_overlapping_folds_only(fold_summary_df):
    """v26_full FIX 8: keep every other fold to make cross-fold inference honest.

    6-month test windows step every 3 months, so adjacent folds share 3 months
    of test data. Keeping every other fold gives non-overlapping test windows.
    """
    if fold_summary_df.empty:
        return fold_summary_df

    df = fold_summary_df.copy()
    df["test_start_dt"] = pd.to_datetime(df["test_start"])
    df = df.sort_values("test_start_dt")
    unique_folds_ordered = df["fold"].drop_duplicates().tolist()
    keep_folds = unique_folds_ordered[::2]
    return df[df["fold"].isin(keep_folds)].copy()


def cross_fold_sharpe_inference(fold_summary_subset, transaction_cost_bps=5):
    sharpes = fold_summary_subset[
        fold_summary_subset["transaction_cost_bps"] == transaction_cost_bps
    ]["net_sharpe"].dropna()
    if len(sharpes) < 2:
        return dict(transaction_cost_bps=transaction_cost_bps,
                    n_folds=len(sharpes), mean_sharpe=np.nan,
                    median_sharpe=np.nan, std_sharpe=np.nan,
                    t_stat=np.nan, p_value=np.nan,
                    ci_low_95=np.nan, ci_high_95=np.nan)
    t_stat, p_value = stats.ttest_1samp(sharpes, 0)
    ci_low, ci_high = stats.t.interval(
        0.95, df=len(sharpes)-1, loc=sharpes.mean(), scale=stats.sem(sharpes))
    return dict(transaction_cost_bps=transaction_cost_bps, n_folds=len(sharpes),
                mean_sharpe=sharpes.mean(), median_sharpe=sharpes.median(),
                std_sharpe=sharpes.std(), t_stat=t_stat, p_value=p_value,
                ci_low_95=ci_low, ci_high_95=ci_high)


print("\n" + "="*90)
print("v26_full WALK-FORWARD SUMMARY (kalman_full spec)")
print("="*90)

# Per-FDR summary
for fdr_q in FDR_Q_VALUES:
    sub = fold_summary_df[fold_summary_df["fdr_q"] == fdr_q]
    label = f"FDR q ≤ {fdr_q}"
    if sub.empty:
        print(f"\n{label}: no folds produced results.")
        continue

    print(f"\n--- {label} (n_folds={sub['fold'].nunique()}) ---")
    print(sub[[
        "fold", "transaction_cost_bps", "chosen_threshold",
        "num_final_pairs", "net_total_return", "net_sharpe",
        "net_max_drawdown", "num_trades", "trade_win_rate",
    ]].to_string(index=False))

    # Full set summary
    tc_summary_full = (
        sub.groupby("transaction_cost_bps")
        .agg(avg_net_total_return=("net_total_return", "mean"),
             median_net_total_return=("net_total_return", "median"),
             avg_net_sharpe=("net_sharpe", "mean"),
             median_net_sharpe=("net_sharpe", "median"),
             avg_max_drawdown=("net_max_drawdown", "mean"),
             avg_trade_win_rate=("trade_win_rate", "mean"),
             avg_num_trades=("num_trades", "mean"),
             avg_num_final_pairs=("num_final_pairs", "mean"),
             folds=("fold", "nunique"))
        .reset_index()
    )
    tc_summary_full["fdr_q"] = fdr_q
    tc_summary_full["fold_set"] = "all_folds"
    tc_summary_full.to_csv(
        TABLES_DIR / f"v26_full_tc_summary_all_folds_fdr{int(fdr_q*100):02d}.csv",
        index=False)
    print("\nTC summary (all folds, descriptive):")
    print(tc_summary_full.to_string(index=False))

    # Non-overlapping subset summary
    sub_nonoverlap = non_overlapping_folds_only(sub)
    tc_summary_nonoverlap = (
        sub_nonoverlap.groupby("transaction_cost_bps")
        .agg(avg_net_total_return=("net_total_return", "mean"),
             median_net_total_return=("net_total_return", "median"),
             avg_net_sharpe=("net_sharpe", "mean"),
             median_net_sharpe=("net_sharpe", "median"),
             avg_max_drawdown=("net_max_drawdown", "mean"),
             folds=("fold", "nunique"))
        .reset_index()
    )
    tc_summary_nonoverlap["fdr_q"] = fdr_q
    tc_summary_nonoverlap["fold_set"] = "non_overlapping"
    tc_summary_nonoverlap.to_csv(
        TABLES_DIR / f"v26_full_tc_summary_nonoverlapping_fdr{int(fdr_q*100):02d}.csv",
        index=False)
    print("\nTC summary (non-overlapping folds, inferential):")
    print(tc_summary_nonoverlap.to_string(index=False))

    # Cross-fold inference on non-overlapping subset
    inference_rows = [
        cross_fold_sharpe_inference(sub_nonoverlap, tc)
        for tc in TRANSACTION_COSTS_BPS
    ]
    inference_df = pd.DataFrame(inference_rows)
    inference_df["fdr_q"] = fdr_q
    inference_df["fold_set"] = "non_overlapping"
    inference_df.to_csv(
        TABLES_DIR / f"v26_full_cross_fold_inference_nonoverlapping_fdr{int(fdr_q*100):02d}.csv",
        index=False)
    print("\nCross-fold Sharpe inference (non-overlapping folds):")
    print(inference_df.to_string(index=False))
    
    # -----------------------------------------------------------------------------
# Selection mode decomposition (FDR pass vs top-K fallback)
# Filtered to pair-folds with meaningful trade density.
# -----------------------------------------------------------------------------

if not pair_metrics_df_clean.empty:
    print("\n" + "="*90)
    print("SELECTION MODE DECOMPOSITION (filtered: ≥2 trades, ≥2 avg holding days)")
    print("="*90)

    for fdr_q in FDR_Q_VALUES:
        for tc_bps in [0, 5]:  # baseline costs only to avoid clutter
            sub = pair_metrics_df_clean[
                (pair_metrics_df_clean["fdr_q"] == fdr_q)
                & (pair_metrics_df_clean["transaction_cost_bps"] == tc_bps)
            ]

            if sub.empty or "selection_selection_mode" not in sub.columns:
                continue

            decomp = sub.groupby("selection_selection_mode").agg(
                n_pair_folds=("net_sharpe", "count"),
                mean_sharpe=("net_sharpe", "mean"),
                median_sharpe=("net_sharpe", "median"),
                mean_trades=("num_trades", "mean"),
                mean_holding_days=("avg_holding_days", "mean"),
                mean_return=("net_total_return", "mean"),
            ).reset_index()

            decomp["fdr_q"] = fdr_q
            decomp["transaction_cost_bps"] = tc_bps

            print(f"\nFDR q ≤ {fdr_q}, {tc_bps} bps:")
            print(decomp.to_string(index=False))

            decomp.to_csv(
                TABLES_DIR / f"v22_selection_mode_decomp_fdr{int(fdr_q*100):02d}_tc{tc_bps:02d}.csv",
                index=False
            )


# -----------------------------------------------------------------------------
# Ablation summary (v26_full: now self-contained, includes kalman_full)
# -----------------------------------------------------------------------------

if not ablation_df.empty:
    print("\n" + "="*90)
    print("ABLATION SUMMARY")
    print("="*90)
    ablation_summary = (
        ablation_df.groupby(["fdr_q", "spec", "transaction_cost_bps"])
        .agg(avg_net_sharpe=("net_sharpe", "mean"),
             median_net_sharpe=("net_sharpe", "median"),
             avg_net_total_return=("net_total_return", "mean"),
             avg_max_drawdown=("net_max_drawdown", "mean"),
             avg_num_trades=("num_trades", "mean"),
             folds=("fold", "nunique"))
        .reset_index()
        .sort_values(["fdr_q", "transaction_cost_bps", "spec"])
    )
    ablation_summary.to_csv(TABLES_DIR / "v26_full_ablation_summary.csv", index=False)
    print(ablation_summary.to_string(index=False))


# -----------------------------------------------------------------------------
# Null benchmark summary
# -----------------------------------------------------------------------------

if not null_benchmark_df.empty:
    print("\n" + "="*90)
    print("NULL BENCHMARK SUMMARY (random pairs vs selected, FDR primary only)")
    print("="*90)
    null_summary = (
        null_benchmark_df.groupby("fold")
        .agg(
            null_mean_sharpe=("null_sharpe", "mean"),
            null_median_sharpe=("null_sharpe", "median"),
            null_p10=("null_sharpe", lambda s: s.quantile(0.10)),
            null_p90=("null_sharpe", lambda s: s.quantile(0.90)),
            selected_sharpe=("selected_sharpe", "first"),
            n_reps=("null_sharpe", "count"),
        )
        .reset_index()
    )
    null_summary["selected_percentile"] = null_summary.apply(
        lambda r: (null_benchmark_df[
            null_benchmark_df["fold"] == r["fold"]
        ]["null_sharpe"] < r["selected_sharpe"]).mean()
        if pd.notna(r["selected_sharpe"]) else np.nan,
        axis=1
    )
    null_summary.to_csv(TABLES_DIR / "v26_full_null_benchmark_summary.csv", index=False)
    print(null_summary.to_string(index=False))


# -----------------------------------------------------------------------------
# HMM regime decomposition
# -----------------------------------------------------------------------------

if USE_HMM_REGIMES and not fold_summary_df.empty:
    hmm_regime_summary = (
        fold_summary_df.groupby(
            ["fdr_q", "transaction_cost_bps", "hmm_regime_at_test_start"]
        )
        .agg(avg_net_sharpe=("net_sharpe", "mean"),
             avg_net_total_return=("net_total_return", "mean"),
             avg_max_drawdown=("net_max_drawdown", "mean"),
             folds=("fold", "nunique"))
        .reset_index()
    )
    hmm_regime_summary.to_csv(
        TABLES_DIR / "v26_full_hmm_regime_summary.csv", index=False)
    print("\nHMM regime decomposition (at test start, 5 bps, primary FDR):")
    print(hmm_regime_summary[
        (hmm_regime_summary["transaction_cost_bps"] == 5)
        & (hmm_regime_summary["fdr_q"] == FDR_Q_PRIMARY)
    ].to_string(index=False))


# -----------------------------------------------------------------------------
# ETF residual strategy summary
# -----------------------------------------------------------------------------

if not sector_etf_residual_df.empty:
    print("\n" + "="*90)
    print("ETF RESIDUAL STRATEGY (parallel benchmark)")
    print("="*90)
    print(sector_etf_residual_df.to_string(index=False))


# =============================================================================
# 14. EM-ESTIMATED PARAMETER DIAGNOSTICS
# =============================================================================

if not em_params_df.empty:
    print("\n" + "="*90)
    print("EM-ESTIMATED KALMAN PARAMETERS — distribution by fold")
    print("="*90)
    em_summary = (
        em_params_df.groupby("fold")
        .agg(median_Q=("Q_estimated", "median"),
             mean_Q=("Q_estimated", "mean"),
             median_R=("R_estimated", "median"),
             mean_R=("R_estimated", "mean"),
             n_pairs=("stock_a", "count"))
        .reset_index()
    )
    em_summary.to_csv(TABLES_DIR / "v26_full_em_parameters_by_fold_summary.csv",
                      index=False)
    print(em_summary.to_string(index=False))


# =============================================================================
# 15. PLOTS
# =============================================================================

# Sharpe by TC, by FDR
if not fold_summary_df.empty:
    for fdr_q in FDR_Q_VALUES:
        sub = fold_summary_df[fold_summary_df["fdr_q"] == fdr_q]
        if sub.empty:
            continue
        tc_plot = sub.groupby("transaction_cost_bps")["net_sharpe"].mean()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(tc_plot.index, tc_plot.values, marker="o",
                label=f"All folds (n={sub['fold'].nunique()})")

        # Non-overlapping overlay
        sub_no = non_overlapping_folds_only(sub)
        if not sub_no.empty:
            tc_plot_no = sub_no.groupby("transaction_cost_bps")["net_sharpe"].mean()
            ax.plot(tc_plot_no.index, tc_plot_no.values, marker="s",
                    linestyle="--",
                    label=f"Non-overlapping (n={sub_no['fold'].nunique()})")

        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(f"v26_full Avg Net Sharpe by TC (FDR q ≤ {fdr_q})")
        ax.set_xlabel("Transaction cost, bps")
        ax.set_ylabel("Avg net Sharpe across folds")
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"v26_full_sharpe_by_tc_fdr{int(fdr_q*100):02d}.png",
            dpi=300)
        plt.close()

# Ablation comparison (primary FDR, 5 bps)
if not ablation_df.empty:
    plot_data = (
        ablation_df[(ablation_df["transaction_cost_bps"] == 5)
                    & (ablation_df["fdr_q"] == FDR_Q_PRIMARY)]
        .groupby("spec")["net_sharpe"].mean().reset_index()
    )
    spec_order = ["static_ols", "kalman_base", "kalman_filter", "kalman_full"]
    plot_data["spec"] = pd.Categorical(plot_data["spec"], categories=spec_order)
    plot_data = plot_data.sort_values("spec")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:green" if v >= 0 else "tab:red"
              for v in plot_data["net_sharpe"]]
    ax.bar(plot_data["spec"].astype(str), plot_data["net_sharpe"], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(f"v26_full Ablation: Avg Net Sharpe by Spec (5 bps, FDR ≤ {FDR_Q_PRIMARY})")
    ax.set_xlabel("Specification")
    ax.set_ylabel("Average net Sharpe across folds")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "v26_full_ablation_sharpe_primary_fdr.png", dpi=300)
    plt.close()

# EM parameter distributions across folds
if not em_params_df.empty:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    em_params_df.boxplot(column="Q_estimated", by="fold", ax=axes[0],
                          rot=45, fontsize=7)
    axes[0].set_yscale("log")
    axes[0].set_title("EM-estimated Q by fold (log scale)")
    axes[0].set_xlabel("")
    em_params_df.boxplot(column="R_estimated", by="fold", ax=axes[1],
                          rot=45, fontsize=7)
    axes[1].set_yscale("log")
    axes[1].set_title("EM-estimated R by fold (log scale)")
    axes[1].set_xlabel("")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "v26_full_em_params_distribution.png", dpi=300)
    plt.close()


# =============================================================================
# 16. SUMMARY TEXT
# =============================================================================

summary_text = f"""
================================================================================
Statistical Arbitrage Project — Version 25
Per-Fold EM Kalman + Leg-Level Costs + Non-Overlapping Fold Inference
+ FDR Grid + Self-Contained Ablation + NaN-Safe Threshold Selection
================================================================================

Author: Dakshayani Pinninti

REVIEWER-DRIVEN CHANGES FROM V21
================================================================================
1. PER-FOLD EM KALMAN.  Q and R are re-estimated inside each fold using only
   that fold's training data. V21 used data ending 2019 for all folds. The
   Kalman cache is now keyed by (stock_a, stock_b, fold).

2. LEG-LEVEL TRANSACTION COSTS.  Cost = (|Δweight_A| + |Δweight_B|) × tc.
   V21 cost = |Δspread_position| × tc understated true cost by ~(1+|β|)/2.

3. NON-OVERLAPPING FOLD INFERENCE.  Test windows step every 3 months but are
   6 months long, so adjacent folds share data. v26_full reports cross-fold
   statistics on every-other-fold subsets in addition to the full set.

4. FDR GRID.  Both q ≤ {FDR_Q_PRIMARY} (primary) and q ≤ {FDR_Q_SECONDARY}
   (sensitivity) are reported. Sparser primary specification is treated as a
   robustness finding rather than a project failure.

5. NULL BENCHMARK USES EM.  Random pairs are now EM-fit on the same fold's
   training data as the selected portfolio, removing apples-to-oranges bias.

6. SELF-CONTAINED ABLATION.  kalman_full appended to ablation table directly.

7. NAN-SAFE THRESHOLD SELECTION.  Dead thresholds (zero trades, NaN Sharpe)
   cannot be chosen accidentally; explicit secondary sort on total return and
   trade count.

8. HMM REGIME LABELS.  Renamed hmm_regime_at_test_start. Describes the
   dominant latent state in the 60 trading days immediately before test start,
   not the within-window regime.

CONFIGURATION FLAGS (THIS RUN)
================================================================================
USE_EM_KALMAN:              {USE_EM_KALMAN} (pykalman available: {PYKALMAN_AVAILABLE})
RUN_ABLATION:               {RUN_ABLATION}
USE_FDR_GRID:               {USE_FDR_GRID}  ({FDR_Q_VALUES})
RUN_NULL_BENCHMARK:         {RUN_NULL_BENCHMARK}
USE_ROLL_SPREAD_COST:       {USE_ROLL_SPREAD_COST}
USE_HMM_REGIMES:            {USE_HMM_REGIMES} (hmmlearn available: {HMMLEARN_AVAILABLE})
RUN_SECTOR_ETF_RESIDUAL_STRATEGY:  {RUN_SECTOR_ETF_RESIDUAL_STRATEGY}

RESEARCH QUESTION
================================================================================
Does a cointegration-based pairs-trading framework with Kalman dynamic hedge
ratios produce risk-adjusted returns that survive:
  (a) realistic leg-level transaction costs,
  (b) standard multiple-testing correction (FDR ≤ 0.10),
  (c) honest cross-fold inference on non-overlapping test windows,
  (d) and a random-pair null distribution?

What is the marginal contribution of each signal component (Kalman vs static
OLS, uncertainty filter, confidence scaling), and how does the framework
compare to an Avellaneda-Lee 2010, ETF variant strategy on the same universe?

LIMITATIONS RETAINED
================================================================================
- Yahoo Finance data, survivorship bias from current-ticker universe.
- Borrow costs, slippage, market impact omitted.
- 2-state HMM is a simple regime model; richer HMM specifications, or volatility
  index-based labels, would be a natural extension.
- EM noise covariance estimation is constrained to diagonal/isotropic priors;
  full-covariance EM is a future extension.

OUTPUTS
================================================================================
All tables in: {TABLES_DIR}
All figures in: {FIGURES_DIR}

Key tables:
  v26_full_fold_summary.csv                        primary results (kalman_full)
  v26_full_ablation_summary.csv                    self-contained ablation
  v26_full_kalman_em_parameters_by_fold.csv        per-fold EM Q, R per pair
  v26_full_em_parameters_by_fold_summary.csv       distribution summary
  v26_full_tc_summary_all_folds_fdr*.csv           descriptive TC sensitivity
  v26_full_tc_summary_nonoverlapping_fdr*.csv      inferential TC sensitivity
  v26_full_cross_fold_inference_nonoverlapping_fdr*.csv  cross-fold t-tests (honest)
  v26_full_null_benchmark.csv                      (if enabled) random pair null
  v26_full_null_benchmark_summary.csv              (if enabled) percentiles
  v26_full_hmm_regimes.csv                         (if enabled) regime labels
  v26_full_hmm_regime_summary.csv                  (if enabled) results by regime
  v26_full_sector_etf_residual_results.csv                (if enabled) ETF strategy
================================================================================
"""

with open(RESULTS_DIR / "project_summary_v26_full.txt", "w") as f:
    f.write(summary_text)

print(summary_text)
print("\nDONE.")
print(f"Results saved in: {RESULTS_DIR}")

print("\nSMOKE TEST CHECKS")
print("=" * 80)

if not fold_summary_df.empty:
    print("\nPortfolio folds by FDR and TC:")
    print(
        fold_summary_df.groupby(["fdr_q", "transaction_cost_bps"])
        .agg(
            n_folds=("fold", "nunique"),
            mean_pairs=("num_final_pairs", "mean"),
            median_pairs=("num_final_pairs", "median"),
            mean_trades=("num_trades", "mean"),
            mean_sharpe=("net_sharpe", "mean"),
        )
        .reset_index()
        .to_string(index=False)
    )

if not pair_metrics_df.empty:
    print("\nRaw pair metrics:")
    print(
        pair_metrics_df.groupby(["fdr_q", "transaction_cost_bps"])
        .agg(
            n_pair_folds=("net_sharpe", "count"),
            mean_trades=("num_trades", "mean"),
            median_trades=("num_trades", "median"),
            mean_holding=("avg_holding_days", "mean"),
            median_holding=("avg_holding_days", "median"),
            mean_sharpe=("net_sharpe", "mean"),
        )
        .reset_index()
        .to_string(index=False)
    )

if "pair_metrics_df_clean" in globals() and not pair_metrics_df_clean.empty:
    print("\nClean pair metrics:")
    print(
        pair_metrics_df_clean.groupby(["fdr_q", "transaction_cost_bps"])
        .agg(
            n_pair_folds=("net_sharpe", "count"),
            mean_trades=("num_trades", "mean"),
            median_trades=("num_trades", "median"),
            mean_holding=("avg_holding_days", "mean"),
            median_holding=("avg_holding_days", "median"),
            mean_sharpe=("net_sharpe", "mean"),
        )
        .reset_index()
        .to_string(index=False)
    )
else:
    print("\nWARNING: pair_metrics_df_clean is empty or not created.")

# =============================================================================
# import pandas as pd
# 
# pm = pd.read_csv("/Users/dakshayanipinninti/Desktop/stat_arb_project/results_v26_full/tables/v26_full_pair_metrics.csv")
# 
# # Decomposition at primary FDR, 5 bps
# sub = pm[(pm["fdr_q"] == 0.20) & (pm["transaction_cost_bps"] == 5)]
# 
# print(sub.groupby("selection_selection_mode").agg(
#     n_pair_folds=("net_sharpe", "count"),
#     mean_sharpe=("net_sharpe", "mean"),
#     median_sharpe=("net_sharpe", "median"),
#     mean_trades=("num_trades", "mean"),
#     mean_return=("net_total_return", "mean"),
# ))
# =============================================================================

# =============================================================================
# import pandas as pd
# v23 = pd.read_csv("/Users/dakshayanipinninti/Desktop/stat_arb_project/results_v23/tables/v23_pair_metrics.csv")
# v26_full = pd.read_csv("/Users/dakshayanipinninti/Desktop/stat_arb_project/results_v26_full/tables/v26_full_pair_metrics.csv")
# 
# # Fold 3, q ≤ 0.20, kalman_full, 5 bps
# def slice_fold3(df):
#     return df[
#         (df["fold"] == "fold_03_202101_202106") &
#         (df["fdr_q"] == 0.20) &
#         (df["transaction_cost_bps"] == 5)
#     ][["stock_a", "stock_b", "num_trades", "avg_holding_days", "net_sharpe"]]
# 
# print("V23 fold 3:")
# print(slice_fold3(v23))
# print("\nv26_full fold 3:")
# print(slice_fold3(v26_full))
# =============================================================================
