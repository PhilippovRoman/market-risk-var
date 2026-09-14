# Вспомогательные функции для оценки VaR с Importance Sampling.
# Здесь считаются IS-веса, ESS, взвешенные и сглаженные квантильные оценки.

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

# Переводит логарифмы IS-весов в нормированные веса.
def normalize_log_weights(log_weights):
    log_weights = np.asarray(log_weights, dtype=float)
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    return weights / weights.sum()

# Показывает эффективный размер выборки после перевзвешивания.
def effective_sample_size(weights):
    weights = np.asarray(weights, dtype=float)
    return 1.0 / np.sum(weights**2)

# Считает квантиль по взвешенной выборке.
def weighted_quantile(values, weights, probability):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative_weights = np.cumsum(sorted_weights)
    index = np.searchsorted(cumulative_weights, probability)
    return sorted_values[min(index, len(sorted_values) - 1)]

# Переводит квантиль лог-доходности в VaR как долю стоимости актива.
def simple_var(log_quantile):
    return 1.0 - np.exp(log_quantile)

# Оценивает масштаб выборки через std и межквартильный размах.
def weighted_scale(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mean = np.sum(weights * values)
    variance = np.sum(weights * (values - mean)**2)
    std = np.sqrt(variance)
    q25 = weighted_quantile(values, weights, 0.25)
    q75 = weighted_quantile(values, weights, 0.75)
    robust_std = (q75 - q25) / 1.349
    return min(std, robust_std) if robust_std > 0 else std

# Считает квантиль через сглаженную взвешенную функцию распределения.
def smoothed_quantile(values, weights, probability, bandwidth_factor=0.5):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    ess = effective_sample_size(weights)
    scale = weighted_scale(values, weights)
    bandwidth = bandwidth_factor * scale * ess**(-1 / 3)
    if bandwidth <= 0:
        return weighted_quantile(values, weights, probability)
    def smoothed_cdf(x):
        return np.sum(weights * norm.cdf((x - values) / bandwidth))
    lower = np.min(values) - 10 * bandwidth
    upper = np.max(values) + 10 * bandwidth
    return brentq(lambda x: smoothed_cdf(x) - probability, lower, upper)