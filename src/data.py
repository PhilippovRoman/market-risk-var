# Загрузка рыночных данных и подготовка логарифмических доходностей.
# Используем AAPL и MSFT как два примера для сравнения методов оценки VaR.

from pathlib import Path
import numpy as np
import yfinance as yf

TICKERS = ["AAPL", "MSFT"]
START_DATE = "2010-01-01"
END_DATE = "2026-01-01"
PRICES_FILE = Path("data/raw/prices.csv")
RETURNS_FILE = Path("data/processed/log_returns.csv")

# Скачивает скорректированные цены закрытия.
def download_prices():
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    return data["Close"].dropna()

# Считает дневные логарифмические доходности.
def calculate_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()

def main():
    PRICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    RETURNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    prices = download_prices()
    returns = calculate_log_returns(prices)
    prices.to_csv(PRICES_FILE)
    returns.to_csv(RETURNS_FILE)
    print(f"Prices: {len(prices)} observations")
    print(f"Returns: {len(returns)} observations")
    print("\nReturn statistics:")
    print(returns.describe().T[["mean", "std", "min", "max"]])
    print(f"\nSaved prices to: {PRICES_FILE}")
    print(f"Saved returns to: {RETURNS_FILE}")

if __name__ == "__main__":
    main()