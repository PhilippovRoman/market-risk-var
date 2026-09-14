# Исторический backtesting шести вариантов VaR.
# На каждой дате модель использует только прошлые данные и сравнивает прогноз
# с фактической доходностью следующих 10 торговых дней.

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import chi2, genhyperbolic
from riskmetrics import riskmetrics_volatility, standardized_residuals
from estimators import simple_var
from var_models import simulate_gh_mc, simulate_gh_is, simulate_residual_mc, simulate_residual_is
from var_models import simulate_empirical_mc, simulate_empirical_is
from var_models import estimate_mc_var, estimate_is_var, estimate_smoothed_is_var

RETURNS_FILE = Path("data/processed/log_returns.csv")
FORECASTS_FILE = Path("results/backtest_forecasts.csv")
SUMMARY_FILE = Path("results/backtest_summary.csv")

ALPHA = 0.01
HORIZON = 10
LOOKBACK = 1000
STEP = 10
N_SIMULATIONS = 5000
SEED = 42

# Фитит GH заново на residuals текущего исторического окна.
def fit_gh(residuals):
    p, a, b, loc, scale = genhyperbolic.fit(residuals)
    return {"p": p, "a": a, "b": b, "loc": loc, "scale": scale}

# Kupiec test проверяет, соответствует ли частота breaches заявленному уровню 1%.
def kupiec_test(breaches):
    breaches = np.asarray(breaches, dtype=int)
    n = len(breaches)
    x = breaches.sum()
    observed = x / n
    null_ll = xlogy(x, ALPHA) + xlogy(n - x, 1 - ALPHA)
    alternative_ll = xlogy(x, observed) + xlogy(n - x, 1 - observed)
    statistic = -2 * (null_ll - alternative_ll)
    return statistic, chi2.sf(statistic, 1)

# Christoffersen test проверяет, не группируются ли breaches во времени.
def christoffersen_test(breaches):
    breaches = np.asarray(breaches, dtype=int)
    previous = breaches[:-1]
    current = breaches[1:]
    n00 = np.sum((previous == 0) & (current == 0))
    n01 = np.sum((previous == 0) & (current == 1))
    n10 = np.sum((previous == 1) & (current == 0))
    n11 = np.sum((previous == 1) & (current == 1))
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    pi01 = n01 / (n00 + n01) if n00 + n01 > 0 else 0
    pi11 = n11 / (n10 + n11) if n10 + n11 > 0 else 0
    null_ll = xlogy(n01 + n11, pi) + xlogy(n00 + n10, 1 - pi)
    alternative_ll = xlogy(n01, pi01) + xlogy(n00, 1 - pi01)
    alternative_ll += xlogy(n11, pi11) + xlogy(n10, 1 - pi11)
    statistic = -2 * (null_ll - alternative_ll)
    return statistic, chi2.sf(statistic, 1)

# Максимальное число breaches подряд.
def max_breach_streak(breaches):
    longest = 0
    current = 0
    for breach in breaches:
        current = current + 1 if breach else 0
        longest = max(longest, current)
    return longest

# Считает шесть прогнозов VaR для одного исторического окна.
def forecast_var(training_returns, seed):
    volatility = riskmetrics_volatility(training_returns)
    residuals = standardized_residuals(training_returns, volatility)
    gh = fit_gh(residuals)
    last_return = training_returns[-1]
    last_volatility = volatility[-1]

    gh_mc = simulate_gh_mc(last_return, last_volatility, gh, N_SIMULATIONS, seed)
    gh_is, gh_weights = simulate_gh_is(last_return, last_volatility, gh, N_SIMULATIONS, seed + 1)
    residual_mc = simulate_residual_mc(last_return, last_volatility, residuals, N_SIMULATIONS, seed + 2)
    residual_is, residual_weights = simulate_residual_is(last_return, last_volatility, residuals, N_SIMULATIONS, seed + 3)
    empirical_mc = simulate_empirical_mc(training_returns, N_SIMULATIONS, seed + 4)
    empirical_is, empirical_weights = simulate_empirical_is(training_returns, N_SIMULATIONS, seed + 5)

    gh_mc_var = estimate_mc_var(gh_mc)
    gh_is_var, _ = estimate_smoothed_is_var(gh_is, gh_weights)
    residual_mc_var = estimate_mc_var(residual_mc)
    residual_is_var, _ = estimate_is_var(residual_is, residual_weights)
    empirical_mc_var = estimate_mc_var(empirical_mc)
    empirical_is_var, _ = estimate_is_var(empirical_is, empirical_weights)

    return {
        "rm_gh_mc": gh_mc_var,
        "rm_gh_is": gh_is_var,
        "rm_residual_mc": residual_mc_var,
        "rm_residual_is": residual_is_var,
        "empirical_mc": empirical_mc_var,
        "empirical_is": empirical_is_var,
    }

# Собирает итоговые статистики backtest для одного метода.
def summarize(group):
    breaches = group["breach"].astype(int).to_numpy()
    kupiec_lr, kupiec_p = kupiec_test(breaches)
    christoffersen_lr, christoffersen_p = christoffersen_test(breaches)
    conditional_lr = kupiec_lr + christoffersen_lr
    return pd.Series({
        "n_forecasts": len(breaches),
        "breaches": breaches.sum(),
        "expected_breaches": ALPHA * len(breaches),
        "breach_rate": breaches.mean(),
        "kupiec_p_value": kupiec_p,
        "christoffersen_p_value": christoffersen_p,
        "conditional_coverage_p_value": chi2.sf(conditional_lr, 2),
        "max_breach_streak": max_breach_streak(breaches),
        "mean_var_pct": group["var_pct"].mean(),
    })

def main():
    RESULTS_FILE = FORECASTS_FILE.parent
    RESULTS_FILE.mkdir(parents=True, exist_ok=True)
    returns = pd.read_csv(RETURNS_FILE, index_col=0, parse_dates=True)
    forecasts = []
    for ticker_number, ticker in enumerate(returns.columns):
        series = returns[ticker].dropna()
        values = series.to_numpy()
        dates = series.index
        origins = range(LOOKBACK, len(values) - HORIZON + 1, STEP)
        origins = list(origins)
        print(f"\n{ticker}: {len(origins)} backtest periods")
        for number, position in enumerate(origins):
            training_returns = values[position - LOOKBACK:position]
            future_return = values[position:position + HORIZON].sum()
            realized_loss = simple_var(future_return)
            seed = SEED + ticker_number * 100000 + number * 10
            predictions = forecast_var(training_returns, seed)
            for method, var in predictions.items():
                forecasts.append({
                    "ticker": ticker,
                    "date": dates[position - 1],
                    "method": method,
                    "var_pct": 100 * var,
                    "realized_loss_pct": 100 * realized_loss,
                    "breach": realized_loss > var,
                })
            if number == 0 or (number + 1) % 10 == 0 or number == len(origins) - 1:
                print(f"{ticker}: {number + 1}/{len(origins)}")

    forecasts = pd.DataFrame(forecasts)
    forecasts.to_csv(FORECASTS_FILE, index=False)
    summary = forecasts.groupby(["ticker", "method"], sort=False).apply(summarize, include_groups=False).reset_index()
    summary.to_csv(SUMMARY_FILE, index=False)

    print(f"\nSaved forecasts to: {FORECASTS_FILE}")
    print(f"Saved summary to: {SUMMARY_FILE}")
    print("\nBacktest summary:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()