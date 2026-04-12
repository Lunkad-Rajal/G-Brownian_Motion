"""
Monte Carlo Forecast with Hybrid Drift and GARCH Volatility
-----------------------------------------------------------
Saves plots to disk. Drift combines long-term mean and short-term EWMA.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os
from typing import Optional

from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# ===========================
# CONFIGURATION
# ===========================
CONFIG = {
    # Stock & dates
    'ticker': 'NVDA',
    'train_start': '2000-01-01',
    'train_end': '2024-12-31',
    'forecast_start': '2025-01-01',
    'forecast_end': '2025-12-31',

    # Simulation
    'num_simulations': 20000,

    # Drift settings
    'drift_method': 'hybrid',        # 'hybrid', 'ewma', 'rolling', 'constant'
    'long_term_weight': 0.7,         # weight on long-term mean (only for hybrid)
    'ewma_span': 60,                 # half-life in days for short-term EWMA
    'drift_reversion_speed': 0.05,   # daily reversion to long-term mean (0 = none, 1 = instant)
    'min_daily_drift': 0.0001,       # floor (0.01% per day) to avoid zero/negative drift
    'max_daily_drift': 0.01,         # cap at 1% per day

    # GARCH
    'garch_p': 1,
    'garch_q': 1,
    'garch_dist': 't',               # 'normal' or 't' (Student's t for fat tails)

    # Plotting
    'output_dir': r'C:\A - Personal\User\Quant\RegimeStrat\output monte-carlo',
    'plot_ci': True,
    'ci_lower': 5,
    'ci_upper': 95,
}

# ===========================
# HELPER FUNCTIONS
# ===========================
def get_stock_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        if 'Adj Close' in data.columns:
            prices = data['Adj Close']
        elif 'Close' in data.columns:
            prices = data['Close']
        else:
            return None
        return prices.dropna()
    except Exception as e:
        print(f"Data error: {e}")
        return None

def compute_log_returns(prices):
    return np.log(1 + prices.pct_change()).dropna()

def predict_volatility_garch(log_returns, forecast_days, p, q, dist):
    """GARCH(1,1) with optional Student's t distribution."""
    try:
        returns_pct = log_returns * 100
        model = arch_model(returns_pct, vol='Garch', p=p, q=q, dist=dist, rescale=False)
        model_fit = model.fit(disp='off', show_warning=False)
        forecast = model_fit.forecast(horizon=forecast_days)
        var_forecast = forecast.variance.values[-1, :]
        vol_pct = np.sqrt(var_forecast)
        return vol_pct / 100
    except Exception as e:
        print(f"GARCH failed: {e}. Using historical std.")
        return np.full(forecast_days, log_returns.std())

def compute_hybrid_drift(log_returns, forecast_days, long_term_weight, ewma_span,
                         reversion_speed, min_drift, max_drift):
    """
    Returns an array of length forecast_days with time-varying drift.
    Drift starts at the hybrid value (weighted average of long-term mean and EWMA)
    and reverts exponentially to the long-term mean over the forecast horizon.
    """
    # Long-term mean (entire history)
    long_term_mean = log_returns.mean()

    # Short-term EWMA (exponential weighted mean)
    # pandas ewm: span = half-life? Actually span ~ 2*half_life. We'll use half-life directly.
    # For a half-life of N days, decay factor = exp(-ln(2)/N). But simpler: use span = ewma_span
    ewma_mean = log_returns.ewm(span=ewma_span, adjust=False).mean().iloc[-1]

    # Hybrid initial drift
    initial_drift = long_term_weight * long_term_mean + (1 - long_term_weight) * ewma_mean
    # Apply caps and floor
    initial_drift = np.clip(initial_drift, -max_drift, max_drift)
    if initial_drift < min_drift and long_term_mean > 0:
        # If initial drift is too low but long-term is positive, raise it
        initial_drift = min_drift

    # Generate drift path that reverts to long-term mean
    drift_series = np.zeros(forecast_days)
    current = initial_drift
    for t in range(forecast_days):
        drift_series[t] = current
        # Revert toward long-term mean
        current = current + reversion_speed * (long_term_mean - current)
        current = np.clip(current, -max_drift, max_drift)
        if current < min_drift and long_term_mean > 0:
            current = min_drift
    return drift_series

def compute_ewma_drift(log_returns, forecast_days, ewma_span, min_drift, max_drift):
    """Constant drift equal to the last EWMA value (repeated)."""
    ewma_mean = log_returns.ewm(span=ewma_span, adjust=False).mean().iloc[-1]
    drift = np.clip(ewma_mean, -max_drift, max_drift)
    if drift < min_drift and log_returns.mean() > 0:
        drift = min_drift
    return np.full(forecast_days, drift)

def compute_rolling_drift(log_returns, forecast_days, window, min_drift, max_drift):
    """Constant drift equal to rolling mean of last `window` days."""
    if len(log_returns) < window:
        drift_val = log_returns.mean()
    else:
        drift_val = log_returns.iloc[-window:].mean()
    drift_val = np.clip(drift_val, -max_drift, max_drift)
    if drift_val < min_drift and log_returns.mean() > 0:
        drift_val = min_drift
    return np.full(forecast_days, drift_val)

def monte_carlo_simulation(start_price, days, simulations, dynamic_vol, dynamic_drift):
    """Geometric Brownian motion with time-varying parameters."""
    dt = 1
    paths = np.zeros((days, simulations))
    paths[0] = start_price
    shocks = np.random.normal(0, 1, (days, simulations))
    for t in range(1, days):
        mu_t = dynamic_drift[t-1]
        sigma_t = dynamic_vol[t-1]
        drift = (mu_t - 0.5 * sigma_t**2) * dt
        diffusion = sigma_t * np.sqrt(dt) * shocks[t]
        paths[t] = paths[t-1] * np.exp(drift + diffusion)
    return paths

def evaluate(predicted, actual):
    mae = np.mean(np.abs(predicted - actual))
    rmse = np.sqrt(np.mean((predicted - actual)**2))
    mape = np.mean(np.abs((predicted - actual) / actual)) * 100
    pred_changes = np.sign(np.diff(predicted, prepend=predicted[0]))
    actual_changes = np.sign(np.diff(actual, prepend=actual[0]))
    dir_acc = np.mean(pred_changes == actual_changes) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'DirAcc': dir_acc}

# ===========================
# MAIN PIPELINE
# ===========================
def run_forecast():
    cfg = CONFIG
    os.makedirs(cfg['output_dir'], exist_ok=True)

    print(f"Loading {cfg['ticker']} from {cfg['train_start']} to {cfg['train_end']}...")
    prices_train = get_stock_data(cfg['ticker'], cfg['train_start'], cfg['train_end'])
    if prices_train is None or len(prices_train) < 100:
        print("Insufficient data.")
        return

    log_returns = compute_log_returns(prices_train)
    last_price = prices_train.iloc[-1]

    forecast_dates = pd.bdate_range(start=cfg['forecast_start'], end=cfg['forecast_end'])
    forecast_days = len(forecast_dates)
    if forecast_days == 0:
        print("No forecast days.")
        return

    # 1. Volatility forecast (GARCH)
    print("Running GARCH volatility forecast...")
    vol_forecast = predict_volatility_garch(
        log_returns, forecast_days,
        cfg['garch_p'], cfg['garch_q'], cfg['garch_dist']
    )

    # 2. Drift forecast (hybrid, EWMA, or rolling)
    print(f"Drift method: {cfg['drift_method']}")
    if cfg['drift_method'] == 'hybrid':
        drift_forecast = compute_hybrid_drift(
            log_returns, forecast_days,
            cfg['long_term_weight'], cfg['ewma_span'],
            cfg['drift_reversion_speed'],
            cfg['min_daily_drift'], cfg['max_daily_drift']
        )
    elif cfg['drift_method'] == 'ewma':
        drift_forecast = compute_ewma_drift(
            log_returns, forecast_days, cfg['ewma_span'],
            cfg['min_daily_drift'], cfg['max_daily_drift']
        )
    elif cfg['drift_method'] == 'rolling':
        drift_forecast = compute_rolling_drift(
            log_returns, forecast_days, cfg['ewma_span'],
            cfg['min_daily_drift'], cfg['max_daily_drift']
        )
    else:  # constant = historical mean
        drift_val = np.clip(log_returns.mean(), -cfg['max_daily_drift'], cfg['max_daily_drift'])
        drift_forecast = np.full(forecast_days, drift_val)

    print(f"Initial drift: {drift_forecast[0]:.6f} (daily)")
    print(f"Long-term historical drift: {log_returns.mean():.6f}")

    # 3. Monte Carlo simulation
    print(f"Running {cfg['num_simulations']} simulations...")
    paths = monte_carlo_simulation(
        last_price, forecast_days, cfg['num_simulations'],
        vol_forecast, drift_forecast
    )
    expected = np.mean(paths, axis=1)
    lower = np.percentile(paths, cfg['ci_lower'], axis=1)
    upper = np.percentile(paths, cfg['ci_upper'], axis=1)

    # 4. Backtest with actual data
    actual_prices = get_stock_data(cfg['ticker'], cfg['forecast_start'], cfg['forecast_end'])
    if actual_prices is not None and not actual_prices.empty:
        common = forecast_dates.intersection(actual_prices.index)
        if len(common) > 0:
            actual_vals = actual_prices.loc[common].values
            pred_vals = expected[forecast_dates.get_indexer(common)]
            metrics = evaluate(pred_vals, actual_vals)
            print("\n=== BACKTEST RESULTS ===")
            print(f"MAE:  {metrics['MAE']:.2f}")
            print(f"RMSE: {metrics['RMSE']:.2f}")
            print(f"MAPE: {metrics['MAPE']:.2f}%")
            print(f"Directional Accuracy: {metrics['DirAcc']:.1f}%")
        else:
            actual_prices = None
    else:
        print("No actual data for backtest.")

    # 5. Plot and save
    plt.figure(figsize=(12, 6))
    # Show a subset of simulated paths
    plt.plot(paths[:, :min(100, cfg['num_simulations'])], color='blue', alpha=0.05, linewidth=0.8)
    plt.plot(expected, color='red', linewidth=2, label='Expected Price (Hybrid Drift + GARCH)')
    if cfg['plot_ci']:
        plt.fill_between(range(forecast_days), lower, upper, color='red', alpha=0.2,
                         label=f'{cfg["ci_lower"]}–{cfg["ci_upper"]} percentile')
    if actual_prices is not None:
        idx = [forecast_dates.get_loc(d) for d in common]
        plt.plot(idx, actual_vals, color='green', linewidth=2, label='Actual Price')

    plt.title(f"{cfg['ticker']} Forecast ({cfg['forecast_start']} to {cfg['forecast_end']})\n"
              f"Drift: {cfg['drift_method']} (long-term weight={cfg.get('long_term_weight',0)})")
    plt.xlabel("Trading Day")
    plt.ylabel("Price (₹)")
    step = max(1, forecast_days // 10)
    tick_pos = list(range(0, forecast_days, step))
    plt.xticks(tick_pos, [forecast_dates[i].strftime('%b %d') for i in tick_pos], rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_file = os.path.join(cfg['output_dir'],
                            f"{cfg['ticker']}_forecast_{cfg['forecast_start']}_to_{cfg['forecast_end']}.png")
    plt.savefig(out_file, dpi=150)
    print(f"\nPlot saved: {out_file}")
    plt.close()

if __name__ == "__main__":
    run_forecast()
