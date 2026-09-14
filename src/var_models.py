# Три модели 10-дневного VaR:
# 1) RiskMetrics + fitted GH;
# 2) RiskMetrics + empirical residuals;
# 3) empirical historical returns.
# Для каждой модели считаем обычный Monte Carlo и Importance Sampling.

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import genhyperbolic
from estimators import normalize_log_weights, effective_sample_size, weighted_quantile, smoothed_quantile, simple_var

RETURNS_FILE = Path("data/processed/log_returns.csv")
VOLATILITY_FILE = Path("data/processed/riskmetrics_volatility.csv")
RESIDUALS_FILE = Path("data/processed/standardized_residuals.csv")
PARAMETERS_FILE = Path("data/processed/residual_fit_parameters.csv")
RESULTS_FILE = Path("results/current_var.csv")

HORIZON = 10
N_SIMULATIONS = 100_000
ALPHA = 0.01
LAMBDA = 0.94

SHIFT = 0.3
SCALE_FACTOR = 1.4
RESIDUAL_TILT = 0.4
EMPIRICAL_TILT = 0.4
BANDWIDTH_FACTOR = 0.5
SEED = 42

# Вероятности для weighted empirical sampling с повышенным весом левого хвоста.
def tilted_probabilities(values, tilt, scale=1.0):
    scores = -tilt * values / scale
    scores -= np.max(scores)
    probabilities = np.exp(scores)
    return probabilities / probabilities.sum()

# RiskMetrics + GH, обычный Monte Carlo.
def simulate_gh_mc(last_return, last_volatility, gh, n_simulations=N_SIMULATIONS, seed=SEED):
    rng = np.random.default_rng(seed)
    sigma2 = np.full(n_simulations, last_volatility**2)
    previous_return = np.full(n_simulations, last_return)
    total_return = np.zeros(n_simulations)
    for _ in range(HORIZON):
        sigma2 = LAMBDA * sigma2 + (1 - LAMBDA) * previous_return**2
        sigma = np.sqrt(sigma2)
        epsilon = genhyperbolic.rvs(gh["p"], gh["a"], gh["b"], loc=gh["loc"], scale=gh["scale"], size=n_simulations, random_state=rng)
        simulated_return = sigma * epsilon
        total_return += simulated_return
        previous_return = simulated_return
    return total_return

# RiskMetrics + GH, Importance Sampling через shift + scale proposal.
def simulate_gh_is(last_return, last_volatility, gh, n_simulations=N_SIMULATIONS, seed=SEED):
    rng = np.random.default_rng(seed)
    sigma2 = np.full(n_simulations, last_volatility**2)
    previous_return = np.full(n_simulations, last_return)
    total_return = np.zeros(n_simulations)
    log_weights = np.zeros(n_simulations)
    proposal_loc = gh["loc"] - SHIFT
    proposal_scale = gh["scale"] * SCALE_FACTOR
    for _ in range(HORIZON):
        sigma2 = LAMBDA * sigma2 + (1 - LAMBDA) * previous_return**2
        sigma = np.sqrt(sigma2)
        epsilon = genhyperbolic.rvs(gh["p"], gh["a"], gh["b"], loc=proposal_loc, scale=proposal_scale, size=n_simulations, random_state=rng)
        target_logpdf = genhyperbolic.logpdf(epsilon, gh["p"], gh["a"], gh["b"], loc=gh["loc"], scale=gh["scale"])
        proposal_logpdf = genhyperbolic.logpdf(epsilon, gh["p"], gh["a"], gh["b"], loc=proposal_loc, scale=proposal_scale)
        log_weights += target_logpdf - proposal_logpdf
        simulated_return = sigma * epsilon
        total_return += simulated_return
        previous_return = simulated_return
    return total_return, log_weights

# RiskMetrics + реальные стандартизованные residuals, обычный resampling.
def simulate_residual_mc(last_return, last_volatility, residuals, n_simulations=N_SIMULATIONS, seed=SEED):
    rng = np.random.default_rng(seed)
    sigma2 = np.full(n_simulations, last_volatility**2)
    previous_return = np.full(n_simulations, last_return)
    total_return = np.zeros(n_simulations)
    for _ in range(HORIZON):
        sigma2 = LAMBDA * sigma2 + (1 - LAMBDA) * previous_return**2
        sigma = np.sqrt(sigma2)
        epsilon = rng.choice(residuals, size=n_simulations)
        simulated_return = sigma * epsilon
        total_return += simulated_return
        previous_return = simulated_return
    return total_return

# RiskMetrics + residuals, weighted sampling для Importance Sampling.
def simulate_residual_is(last_return, last_volatility, residuals, n_simulations=N_SIMULATIONS, seed=SEED):
    rng = np.random.default_rng(seed)
    probabilities = tilted_probabilities(residuals, RESIDUAL_TILT)
    sigma2 = np.full(n_simulations, last_volatility**2)
    previous_return = np.full(n_simulations, last_return)
    total_return = np.zeros(n_simulations)
    log_weights = np.zeros(n_simulations)
    log_target_probability = -np.log(len(residuals))
    for _ in range(HORIZON):
        sigma2 = LAMBDA * sigma2 + (1 - LAMBDA) * previous_return**2
        sigma = np.sqrt(sigma2)
        indices = rng.choice(len(residuals), size=n_simulations, p=probabilities)
        epsilon = residuals[indices]
        log_weights += log_target_probability - np.log(probabilities[indices])
        simulated_return = sigma * epsilon
        total_return += simulated_return
        previous_return = simulated_return
    return total_return, log_weights

# Empirical model: независимо пересэмплируем исторические дневные доходности.
def simulate_empirical_mc(returns, n_simulations=N_SIMULATIONS, seed=SEED):
    rng = np.random.default_rng(seed)
    samples = rng.choice(returns, size=(n_simulations, HORIZON))
    return samples.sum(axis=1)

# Empirical model с weighted sampling в сторону отрицательного хвоста.
def simulate_empirical_is(returns, n_simulations=N_SIMULATIONS, seed=SEED):
    rng = np.random.default_rng(seed)
    scale = np.std(returns, ddof=1)
    probabilities = tilted_probabilities(returns, EMPIRICAL_TILT, scale)
    total_return = np.zeros(n_simulations)
    log_weights = np.zeros(n_simulations)
    log_target_probability = -np.log(len(returns))
    for _ in range(HORIZON):
        indices = rng.choice(len(returns), size=n_simulations, p=probabilities)
        total_return += returns[indices]
        log_weights += log_target_probability - np.log(probabilities[indices])
    return total_return, log_weights

# Обычный Monte Carlo VaR.
def estimate_mc_var(simulated_returns):
    quantile = np.quantile(simulated_returns, ALPHA)
    return simple_var(quantile)

# IS VaR по взвешенному квантилю.
def estimate_is_var(simulated_returns, log_weights):
    weights = normalize_log_weights(log_weights)
    quantile = weighted_quantile(simulated_returns, weights, ALPHA)
    ess = effective_sample_size(weights)
    return simple_var(quantile), ess

# GH IS дополнительно использует сглаженную оценку квантиля.
def estimate_smoothed_is_var(simulated_returns, log_weights):
    weights = normalize_log_weights(log_weights)
    quantile = smoothed_quantile(simulated_returns, weights, ALPHA, BANDWIDTH_FACTOR)
    ess = effective_sample_size(weights)
    return simple_var(quantile), ess

# Берет параметры fitted GH для конкретного тикера.
def get_gh_parameters(parameters, ticker):
    row = parameters.loc[parameters["ticker"] == ticker].iloc[0]
    return {"p": row["gh_p"], "a": row["gh_a"], "b": row["gh_b"], "loc": row["gh_loc"], "scale": row["gh_scale"]}

def main():
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    returns = pd.read_csv(RETURNS_FILE, index_col=0, parse_dates=True)
    volatility = pd.read_csv(VOLATILITY_FILE, index_col=0, parse_dates=True)
    residuals = pd.read_csv(RESIDUALS_FILE, index_col=0, parse_dates=True)
    parameters = pd.read_csv(PARAMETERS_FILE)
    results = []
    for ticker in returns.columns:
        historical_returns = returns[ticker].dropna().values
        historical_residuals = residuals[ticker].dropna().values
        last_return = historical_returns[-1]
        last_volatility = volatility[ticker].dropna().iloc[-1]
        gh = get_gh_parameters(parameters, ticker)

        gh_mc = simulate_gh_mc(last_return, last_volatility, gh)
        gh_is, gh_weights = simulate_gh_is(last_return, last_volatility, gh)
        residual_mc = simulate_residual_mc(last_return, last_volatility, historical_residuals)
        residual_is, residual_weights = simulate_residual_is(last_return, last_volatility, historical_residuals)
        empirical_mc = simulate_empirical_mc(historical_returns)
        empirical_is, empirical_weights = simulate_empirical_is(historical_returns)

        gh_mc_var = estimate_mc_var(gh_mc)
        gh_is_var, gh_ess = estimate_smoothed_is_var(gh_is, gh_weights)
        residual_mc_var = estimate_mc_var(residual_mc)
        residual_is_var, residual_ess = estimate_is_var(residual_is, residual_weights)
        empirical_mc_var = estimate_mc_var(empirical_mc)
        empirical_is_var, empirical_ess = estimate_is_var(empirical_is, empirical_weights)

        ticker_results = [
            ("RiskMetrics + GH", "MC", gh_mc_var, N_SIMULATIONS),
            ("RiskMetrics + GH", "IS", gh_is_var, gh_ess),
            ("RiskMetrics + residuals", "MC", residual_mc_var, N_SIMULATIONS),
            ("RiskMetrics + residuals", "IS", residual_is_var, residual_ess),
            ("Empirical returns", "MC", empirical_mc_var, N_SIMULATIONS),
            ("Empirical returns", "IS", empirical_is_var, empirical_ess),
        ]

        print(f"\n{ticker}")
        for model, method, var, ess in ticker_results:
            print(f"{model} | {method}: VaR = {100 * var:.4f}%, ESS = {ess:.0f}")
            results.append({"ticker": ticker, "model": model, "method": method, "var": var, "var_pct": 100 * var, "ess": ess})

    pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)
    print(f"\nSaved results to: {RESULTS_FILE}")

if __name__ == "__main__":
    main()