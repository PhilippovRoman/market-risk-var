# Оценка динамической волатильности RiskMetrics и стандартизованных остатков.

from pathlib import Path
import numpy as np
import pandas as pd

RETURNS_FILE = Path("data/processed/log_returns.csv")
VOLATILITY_FILE = Path("data/processed/riskmetrics_volatility.csv")
RESIDUALS_FILE = Path("data/processed/standardized_residuals.csv")
LAMBDA = 0.94
INIT_WINDOW = 30

# Строит RiskMetrics volatility по истории доходностей.
def riskmetrics_volatility(returns, lam=LAMBDA, init_window=INIT_WINDOW):
    returns = np.asarray(returns, dtype=float)
    variance = np.empty(len(returns))
    variance[:init_window] = np.var(returns[:init_window], ddof=1)
    for t in range(init_window, len(returns)):
        variance[t] = lam * variance[t - 1] + (1 - lam) * returns[t - 1]**2
    return np.sqrt(variance)

# Делит доходности на текущую оценку волатильности.
def standardized_residuals(returns, volatility, init_window=INIT_WINDOW):
    return returns[init_window:] / volatility[init_window:]

def main():
    returns = pd.read_csv(RETURNS_FILE, index_col=0, parse_dates=True)
    volatility = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    residuals = pd.DataFrame(index=returns.index[INIT_WINDOW:], columns=returns.columns, dtype=float)
    for ticker in returns.columns:
        sigma = riskmetrics_volatility(returns[ticker].values)
        eps = standardized_residuals(returns[ticker].values, sigma)
        volatility[ticker] = sigma
        residuals[ticker] = eps
    volatility.to_csv(VOLATILITY_FILE)
    residuals.to_csv(RESIDUALS_FILE)
    print("Residual statistics:")
    print(residuals.describe().T[["mean", "std", "min", "max"]])
    print(f"\nSaved volatility to: {VOLATILITY_FILE}")
    print(f"Saved residuals to: {RESIDUALS_FILE}")

if __name__ == "__main__":
    main()