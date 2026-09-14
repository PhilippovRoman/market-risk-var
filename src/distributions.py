# Подгонка Normal и Generalized Hyperbolic распределений к residuals RiskMetrics.
# Сравниваем модели по AIC и сохраняем параметры GH для дальнейшего расчета VaR.

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import genhyperbolic, norm, skew, kurtosis

RESIDUALS_FILE = Path("data/processed/standardized_residuals.csv")
PARAMETERS_FILE = Path("data/processed/residual_fit_parameters.csv")
FIGURES_DIR = Path("figures")

# Фитит Normal и GH к одной выборке residuals.
def fit_distributions(values):
    normal_loc, normal_scale = norm.fit(values)
    p, a, b, gh_loc, gh_scale = genhyperbolic.fit(values)
    normal_ll = np.sum(norm.logpdf(values, normal_loc, normal_scale))
    gh_ll = np.sum(genhyperbolic.logpdf(values, p, a, b, loc=gh_loc, scale=gh_scale))
    normal_aic = 2 * 2 - 2 * normal_ll
    gh_aic = 2 * 5 - 2 * gh_ll
    return normal_loc, normal_scale, p, a, b, gh_loc, gh_scale, normal_aic, gh_aic

# Показывает, как Normal и GH описывают empirical residuals.
def plot_fit(values, ticker, normal_loc, normal_scale, p, a, b, gh_loc, gh_scale):
    x = np.linspace(np.quantile(values, 0.001), np.quantile(values, 0.999), 600)
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=60, density=True, alpha=0.5, label="Residuals")
    plt.plot(x, norm.pdf(x, normal_loc, normal_scale), linewidth=2, label="Normal")
    plt.plot(x, genhyperbolic.pdf(x, p, a, b, loc=gh_loc, scale=gh_scale), linewidth=2, label="GH")
    plt.xlabel("Standardized residual")
    plt.ylabel("Density")
    plt.title(f"{ticker}: Normal vs GH")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{ticker.lower()}_residual_fit.png", dpi=150)
    plt.close()

def main():
    residuals = pd.read_csv(RESIDUALS_FILE, index_col=0, parse_dates=True)
    results = []
    for ticker in residuals.columns:
        values = residuals[ticker].dropna().values
        normal_loc, normal_scale, p, a, b, gh_loc, gh_scale, normal_aic, gh_aic = fit_distributions(values)
        results.append({
            "ticker": ticker,
            "skewness": skew(values),
            "excess_kurtosis": kurtosis(values),
            "normal_loc": normal_loc,
            "normal_scale": normal_scale,
            "normal_aic": normal_aic,
            "gh_p": p,
            "gh_a": a,
            "gh_b": b,
            "gh_loc": gh_loc,
            "gh_scale": gh_scale,
            "gh_aic": gh_aic,
            "aic_improvement": normal_aic - gh_aic,
        })
        plot_fit(values, ticker, normal_loc, normal_scale, p, a, b, gh_loc, gh_scale)
        print(f"\n{ticker}")
        print(f"Excess kurtosis: {kurtosis(values):.3f}")
        print(f"Normal AIC: {normal_aic:.2f}")
        print(f"GH AIC: {gh_aic:.2f}")
        print(f"AIC improvement: {normal_aic - gh_aic:.2f}")
    pd.DataFrame(results).to_csv(PARAMETERS_FILE, index=False)
    print(f"\nSaved parameters to: {PARAMETERS_FILE}")

if __name__ == "__main__":
    main()