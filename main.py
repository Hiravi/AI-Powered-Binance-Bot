import requests
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import time
import warnings
import datetime
import csv
from buy_sell import open_buy_position, open_sell_position, close_position, get_balance
from tabulate import tabulate
from lpt_model import pred
import os

warnings.filterwarnings("ignore")

# Load API credentials from environment variables for security
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

# Binance API endpoint URLs
BASE_URL = "https://api.binance.com"
KLINES_URL = f"{BASE_URL}/api/v3/klines"

# Trading parameters
SYMBOL = "LPTUSDT"
INTERVAL = "1h"
LIMIT = 1440
POSITION_OPEN_VALUE = 0
BUY_AMOUNT = 30
SELL_AMOUNT = 30
START_TRAILING_AT = 0.003  # 0.30%
STOP_LOSS_TRAILING_RANGE = 0.1  # 0.10%

start_time = time.time()

# Initialize Binance API session
session = requests.Session()
session.headers.update({'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'})

# Global variables for tracking
highest_profit = float('-inf')
lowest_profit = float('inf')
total_profit = 0
profits = []

def fetch_historical_data():
    """Fetch historical trading data from Binance."""
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": LIMIT}
    response = session.get(KLINES_URL, params=params)
    response.raise_for_status()  # Raise an error for bad responses
    data = response.json()
    df = pd.DataFrame(data)
    df = df.iloc[:, :6]  # Extract only the OHLCV columns
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.astype(float)  # Convert all columns to float
    return df

def prepare_dataset(df):
    """Prepare the dataset for model training."""
    df["volume_before"] = df["volume"].shift(1)
    df["previous_price"] = df["close"].shift(1)
    df["price_change"] = df["close"].shift(-1) - df["close"]
    return df.dropna()

def train_model(df):
    """Train the machine learning model."""
    X = df[["volume", "volume_before", "previous_price"]]
    y = df["price_change"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Create a pipeline with feature scaling and ensemble model
    pipeline = make_pipeline(StandardScaler(), GradientBoostingRegressor())
    param_grid = {
        "gradientboostingregressor__n_estimators": [100, 200, 300],
        "gradientboostingregressor__learning_rate": [0.1, 0.05, 0.01],
        "gradientboostingregressor__max_depth": [3, 4, 5]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Evaluate the model on the test data
    accuracy = evaluate_model(grid_search.best_estimator_, X_test, y_test)
    print("Model Accuracy:", accuracy)
    return grid_search.best_estimator_, accuracy

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and calculate accuracy."""
    y_pred = model.predict(X_test)
    acc = []
    for pred, true in zip(y_pred, y_test):
        if pred > POSITION_OPEN_VALUE and true > 0:
            acc.append(1)
        elif pred < -POSITION_OPEN_VALUE and true < 0:
            acc.append(1)
        elif pred < -POSITION_OPEN_VALUE and true > 0:
            acc.append(0)
        elif pred > POSITION_OPEN_VALUE and true < 0:
            acc.append(0)
        else:
            acc.append(1)
    return sum(acc) / len(acc)

def fetch_real_time_data():
    """Fetch real-time volume data from Binance."""
    params = {"symbol": SYMBOL, "interval": INTERVAL, "limit": 2}
    response = session.get(KLINES_URL, params=params)
    response.raise_for_status()  # Raise an error for bad responses
    data = response.json()
    latest_data = data[-1]
    current_volume = float(latest_data[5])
    volume_before = float(data[-2][5])
    current_price = float(latest_data[4])
    previous_price = float(data[-2][4])
    return current_volume, volume_before, previous_price, current_price

def log_to_csv(data):
    """Log trading data to CSV."""
    csv_log_file = 'log_file_model1_twoDecisionsLogged_trailing_lptTUSD.csv'
    with open(csv_log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def bot_loop():
    """Main trading bot loop."""
    global total_profit
    while True:
        try:
            df = fetch_historical_data()
            df = prepare_dataset(df)
            current_price = df['close'][0]
            action = determine_action(df)

            # Trading logic here...
            print(f"Action determined: {action}")

            # Sleep for a period to throttle the API requests
            time.sleep(60)

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            time.sleep(1)  # Backoff before retrying

def determine_action(df):
    """Determine whether to buy, sell, or wait based on predictions."""
    result = pred(df)
    predicted_change_pct = ((result - df['close']) / df['close']) * 100
    predicted_change_pct = predicted_change_pct[0]

    if predicted_change_pct > 0.5:
        return 'BUY'
    elif predicted_change_pct < -0.5:
        return 'SELL'
    else:
        return 'WAIT'

if __name__ == "__main__":
    bot_loop()
