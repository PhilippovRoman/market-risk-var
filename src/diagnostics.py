# Диагностика стандартизованных остатков RiskMetrics.
# Строим гистограммы и QQ-plots, чтобы проверить отклонение от Normal.

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, probplot

RESIDUALS_FILE = Path("data/processed/standardized_residuals.csv")
FIGURES_DIR = Path("figures")

# Строит гистограмму residuals и плотность нормального распределения.
def plot_histogram(residuals, ticker):
    mean = residuals.mean()
    std = residuals.std(ddof=1)
    x = np.linspace(residuals.min(), residuals.max(), 500)
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=60, density=True, alpha=0.6, label="Residuals")
    plt.plot(x, norm.pdf(x, mean, std), linewidth=2, label="Normal")
    plt.xlabel("Standardized residual")
    plt.ylabel("Density")
    plt.title(f"{ticker}: RiskMetrics residuals")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{ticker.lower()}_residual_hist.png", dpi=150)
    plt.close()

# Строит QQ-plot относительно нормального распределения.
def plot_qq(residuals, ticker):
    plt.figure(figsize=(6, 6))
    probplot(residuals, dist="norm", plot=plt)
    plt.title(f"{ticker}: Normal QQ-plot")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{ticker.lower()}_residual_qq.png", dpi=150)
    plt.close()

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    residuals = pd.read_csv(RESIDUALS_FILE, index_col=0, parse_dates=True)
    for ticker in residuals.columns:
        values = residuals[ticker].dropna().values
        plot_histogram(values, ticker)
        plot_qq(values, ticker)
        print(f"Saved diagnostics for {ticker}")

if __name__ == "__main__":
    main()