import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def get_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data from Yahoo Finance.
    Fixed to handle yfinance updates and MultiIndex errors.
    """
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        
        if stock_data.empty:
            return None
            
        #Check if the columns are a "MultiIndex"
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.droplevel(1)

        #Look for 'Adj Close', if missing, use 'Close'
        if 'Adj Close' in stock_data.columns:
            return stock_data['Adj Close']
        elif 'Close' in stock_data.columns:
            print("Note: 'Adj Close' was not found. Using 'Close' prices instead.")
            return stock_data['Close']
        else:
            return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def monte_carlo_simulation(start_price, mu, sigma, days, simulations):
    """
    Performs Monte Carlo Simulation using Geometric Brownian Motion.
    """
    dt = 1
    
    stochastic_component = np.random.normal(0, 1, (days, simulations))
    price_paths = np.zeros((days, simulations))
    price_paths[0] = start_price
    
    for t in range(1, days):
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * stochastic_component[t]
        price_paths[t] = price_paths[t-1] * np.exp(drift + diffusion)
        
    return price_paths

def get_currency_symbol(ticker):
    if ticker.endswith('.NS') or ticker.endswith('.BO'):
        return '₹'
    else:
        return '$'

def run_forecast_project():
    """
    Main execution function with User Inputs for generic use.
    """
    print("--- Stock Price Forecasting Tool ---")
    ticker = input("Enter Stock Ticker (e.g., RELIANCE.NS, AAPL): ").strip().upper()
    
    print("\n[Step 1] Define Historical Training Data Range")
    history_start = input("Enter Start Date (YYYY-MM-DD): ").strip()
    history_end = input("Enter End Date (YYYY-MM-DD): ").strip()

    print("\n[Step 2] Define Forecasting Window")
    forecast_start = input("Enter Forecast Start Date (YYYY-MM-DD): ").strip()
    forecast_end = input("Enter Forecast End Date (YYYY-MM-DD): ").strip()
    
    currency = get_currency_symbol(ticker)
    num_simulations = 1000

    print(f"\n--- Fetching Data for {ticker} ---")
    prices = get_stock_data(ticker, history_start, history_end)
    
    if prices is None:
        print(f"Error: Could not fetch data. Check ticker or internet.")
        return

    log_returns = np.log(1 + prices.pct_change())
    mu = log_returns.mean()
    sigma = log_returns.std()
    
    last_actual_price = float(prices.iloc[-1])
    print(f"Last Training Price: {currency}{last_actual_price:.2f}")
    print(f"Daily Volatility: {sigma:.6f}")

    forecast_dates = pd.bdate_range(start=forecast_start, end=forecast_end)
    num_forecast_days = len(forecast_dates)
    
    if num_forecast_days == 0:
        print("Error: Invalid forecast date range (0 business days).")
        return

    print(f"\n--- Simulating {num_forecast_days} Trading Days ---")
    
    simulated_paths = monte_carlo_simulation(last_actual_price, mu, sigma, num_forecast_days, num_simulations)
    mean_price_path = np.mean(simulated_paths, axis=1)
    
    print("\n" + "="*40)
    print(f"PREDICTED VALUES ({ticker})")
    print("="*40)
    print(f"{'Date':<15} | {'Expected Price':<15}")
    print("-" * 33)
    
    for i, date in enumerate(forecast_dates):
        price = mean_price_path[i]
        date_str = date.strftime('%Y-%m-%d')
        print(f"{date_str:<15} | {currency} {price:.2f}")
    
    # Extend range by 1 day because ofmany timezones
    end_check_date = (pd.to_datetime(forecast_end) + timedelta(days=5)).strftime('%Y-%m-%d')
    actual_data = get_stock_data(ticker, forecast_start, end_check_date)

    if actual_data is not None and not actual_data.empty:
        print("\n" + "="*40)
        print("ACCURACY REPORT (Predicted vs Actual)")
        print("="*40)
        print(f"{'Date':<12} | {'Predicted':<10} | {'Actual':<10} | {'Diff':<10}")
        print("-" * 50)

        actual_prices = []
        predicted_prices_aligned = []

        for i, date in enumerate(forecast_dates):
            if date in actual_data.index:
                actual_val = float(actual_data.loc[date])
                pred_val = mean_price_path[i]

                actual_prices.append(actual_val)
                predicted_prices_aligned.append(pred_val)
                
                diff = pred_val - actual_val
                date_str = date.strftime('%Y-%m-%d')
                print(f"{date_str:<12} | {pred_val:.2f}      | {actual_val:.2f}      | {diff:+.2f}")

        if len(actual_prices) > 0:
            actuals = np.array(actual_prices)
            preds = np.array(predicted_prices_aligned)
            
            mae = np.mean(np.abs(preds - actuals))
            rmse = np.sqrt(np.mean((preds - actuals)**2))
            mape = np.mean(np.abs((preds - actuals) / actuals)) * 100
            
            print("-" * 50)
            print(f"Mean Absolute Error (MAE):   {currency}{mae:.2f}")
            print(f"Root Mean Sq Error (RMSE):   {currency}{rmse:.2f}")
            print(f"Mean Abs % Error (MAPE):     {mape:.2f}%")
            print(f"Directional Accuracy:        {100 - mape:.2f}% (Approx)")
            print("="*40)
    else:
        print("\nNote: Actual data for the forecast range is unavailable (likely future dates). Skipping accuracy check.")

    plt.figure(figsize=(10, 6))
    plt.plot(simulated_paths[:, :100], alpha=0.1, color='blue', linewidth=1)
    plt.plot(mean_price_path, color='red', linewidth=3, label='Expected Price')
    
    if actual_data is not None and not actual_data.empty:
         # Filter actual data to only show the relevant part on the graph
         relevant_actuals = actual_data[actual_data.index.isin(forecast_dates)]
         if not relevant_actuals.empty:
            plt.plot(relevant_actuals.index, relevant_actuals.values, color='green', linewidth=3, label='ACTUAL Price')

    plt.title(f"Forecast: {ticker} ({forecast_start} to {forecast_end})")
    plt.xticks(range(len(forecast_dates)), [d.strftime('%d-%b') for d in forecast_dates], rotation=45)
    plt.ylabel(f"Price ({currency})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_forecast_project()
