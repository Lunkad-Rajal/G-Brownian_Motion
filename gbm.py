import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


from arch import arch_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

def get_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        
        if stock_data.empty:
            return None
            
        #Checks if the columns are MultiIndex
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.droplevel(1)

        #Looks for 'Adj Close'. if missing, use 'Close'
        if 'Adj Close' in stock_data.columns:
            return stock_data['Adj Close']
        elif 'Close' in stock_data.columns:
            return stock_data['Close']
        else:
            return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def predict_volatility_garch(log_returns, forecast_days):
    try:
        #Cleans the data
        returns_clean = log_returns.dropna() * 100 
        
        #Fits GARCH(1,1)
        model = arch_model(returns_clean, vol='Garch', p=1, q=1, rescale=False)
        model_fit = model.fit(disp='off', show_warning=False)
        
        #Forecast volatility
        forecast = model_fit.forecast(horizon=forecast_days)
        predicted_variance = forecast.variance.values[-1, :]
        predicted_volatility = np.sqrt(predicted_variance) / 100
        
        return predicted_volatility
    except Exception as e:
        print(f"GARCH fitting failed: {e}. Using constant volatility.")
        return None


def predict_drift_lstm(prices, log_returns, forecast_days, lookback=60):
    try:
        #Prepare data
        returns_array = log_returns.dropna().values
        
        if len(returns_array) < lookback + 30:
            print("Not enough data for LSTM.")
            return None
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        returns_scaled = scaler.fit_transform(returns_array.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(returns_scaled)):
            X.append(returns_scaled[i-lookback:i, 0])
            y.append(returns_scaled[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split data
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, activation='tanh', return_sequences=True, input_shape=(lookback, 1)),
            Dropout(0.2),
            LSTM(50, activation='tanh'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train with early stopping
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=50, batch_size=32, 
                  validation_data=(X_test, y_test), 
                  callbacks=[early_stop], verbose=0)
        
        # Forecast
        predicted_returns = []
        current_sequence = returns_scaled[-lookback:]
        
        for _ in range(forecast_days):
            current_sequence_reshaped = current_sequence.reshape(1, lookback, 1)
            next_return = model.predict(current_sequence_reshaped, verbose=0)[0, 0]
            predicted_returns.append(next_return)
            current_sequence = np.append(current_sequence[1:], next_return)
        
        # Inverse transform
        predicted_returns = scaler.inverse_transform(np.array(predicted_returns).reshape(-1, 1))
        
        return predicted_returns.flatten()
    
    except Exception as e:
        print(f"LSTM training failed: {e}")
        return None

def monte_carlo_simulation(start_price, mu, sigma, days, simulations, 
                           dynamic_volatility=None, dynamic_drift=None):
    """
    Performs Monte Carlo Simulation with optional time-varying parameters.
    
    Parameters:
    - dynamic_volatility: Array of predicted volatilities (one per day)
    - dynamic_drift: Array of predicted drift values (one per day)
    """
    dt = 1
    
    stochastic_component = np.random.normal(0, 1, (days, simulations))
    price_paths = np.zeros((days, simulations))
    price_paths[0] = start_price
    
    for t in range(1, days):
        # Use dynamic parameters if available. if unavailable then use standard
        current_sigma = dynamic_volatility[t-1] if dynamic_volatility is not None else sigma
        current_mu = dynamic_drift[t-1] if dynamic_drift is not None else mu
        
        drift = (current_mu - 0.5 * current_sigma**2) * dt
        diffusion = current_sigma * np.sqrt(dt) * stochastic_component[t]
        price_paths[t] = price_paths[t-1] * np.exp(drift + diffusion)
        
    return price_paths

def get_currency_symbol(ticker):
    if ticker.endswith('.NS') or ticker.endswith('.BO'):
        return '₹'
    else:
        return '$'

def run_forecast_project():
    
    ticker = input("\nEnter Stock Ticker (e.g., RELIANCE.NS, AAPL): ").strip().upper()
    
    history_start = input("Enter Start Date (YYYY-MM-DD): ").strip()
    history_end = input("Enter End Date (YYYY-MM-DD): ").strip()

    forecast_start = input("Enter Forecast Start Date (YYYY-MM-DD): ").strip()
    forecast_end = input("Enter Forecast End Date (YYYY-MM-DD): ").strip()
    
    currency = get_currency_symbol(ticker)
    num_simulations = 1000

    prices = get_stock_data(ticker, history_start, history_end)
    
    if prices is None:
        print(f"Error: Could not fetch data")
        return

    log_returns = np.log(1 + prices.pct_change())
    mu = log_returns.mean()
    sigma = log_returns.std()
    
    last_actual_price = float(prices.iloc[-1])
    print(f"Last Training Price: {currency}{last_actual_price:.2f}")
    print(f"Historical Daily Volatility: {sigma:.6f}")
    print(f"Historical Daily Drift: {mu:.6f}")

    forecast_dates = pd.bdate_range(start=forecast_start, end=forecast_end)
    num_forecast_days = len(forecast_dates)
    
    if num_forecast_days == 0:
        print("Error: Invalid forecast date range")
        return

    # Predict time-varying volatility using GARCH & lstm
    predicted_volatility = predict_volatility_garch(log_returns, num_forecast_days)

    predicted_drift = predict_drift_lstm(prices, log_returns, num_forecast_days)
    
    if predicted_volatility is not None:
        dynamic_vol = predicted_volatility
    else:
        dynamic_vol = None

    if predicted_drift is not None:
        dynamic_mu = predicted_drift
    else:
        print(f"LSTM Drift: FAILED")
        dynamic_mu = None

    # Run simulation with dynamic parameters
    simulated_paths = monte_carlo_simulation(
        last_actual_price, mu, sigma, num_forecast_days, num_simulations,
        dynamic_volatility=dynamic_vol,
        dynamic_drift=dynamic_mu
    )
    
    mean_price_path = np.mean(simulated_paths, axis=1)
    
    print(f"PREDICTED VALUES ({ticker})")
    print(f"{'Date':<15} | {'Expected Price':<15}")
    
    for i, date in enumerate(forecast_dates):
        price = mean_price_path[i]
        date_str = date.strftime('%Y-%m-%d')
        print(f"{date_str:<15} | {currency} {price:.2f}")
    
    # Extend range by 1 day. without this errors came up. needed due to possible timezone changes
    end_check_date = (pd.to_datetime(forecast_end) + timedelta(days=5)).strftime('%Y-%m-%d')
    actual_data = get_stock_data(ticker, forecast_start, end_check_date)

    if actual_data is not None and not actual_data.empty:
        print(f"{'Date':<12} | {'Predicted':<10} | {'Actual':<10} | {'Diff':<10}")

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
            
            print(f"Mean Absolute Error (MAE):   {currency}{mae:.2f}")
            print(f"Root Mean Sq Error (RMSE):   {currency}{rmse:.2f}")
            print(f"Mean Abs % Error (MAPE):     {mape:.2f}%")
            print(f"Directional Accuracy:        {100 - mape:.2f}% (Approx)")

    else:
        print("Error")

    plt.figure(figsize=(10, 6))
    plt.plot(simulated_paths[:, :100], alpha=0.1, color='blue', linewidth=1)
    plt.plot(mean_price_path, color='red', linewidth=3, label='Expected Price (ML-Enhanced)')
    
    if actual_data is not None and not actual_data.empty:
         # Filter actual data to only show the relevant part on the graph
         relevant_actuals = actual_data[actual_data.index.isin(forecast_dates)]
         if not relevant_actuals.empty:
            plt.plot(relevant_actuals.index, relevant_actuals.values, color='green', linewidth=3, label='ACTUAL Price')

    plt.title(f"ML-Enhanced Forecast: {ticker} ({forecast_start} to {forecast_end})")
    plt.xticks(range(len(forecast_dates)), [d.strftime('%d-%b') for d in forecast_dates], rotation=45)
    plt.ylabel(f"Price ({currency})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_forecast_project()
