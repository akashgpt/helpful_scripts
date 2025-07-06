import numpy as np

def monte_carlo_error(func, x0, xerr, N=10_000, seed=None, return_ci=False):
    """
    Monte Carlo propagation of uncertainty.

    Parameters
    ----------
    func : callable
        Function f(x0[0], x0[1], …) → scalar.
    x0 : sequence of float
        Nominal input values.
    xerr : sequence of float
        1σ uncertainties of inputs.
    N : int
        Number of samples (default 1e5).
    seed : int or None
        RNG seed.
    return_ci : bool
        If True, also return the 95% CI from the samples.

    Returns
    -------
    mean : float
        Sample mean of f.
    sigma : float
        Sample standard deviation (1σ).
    ci_lower, ci_upper : float, float, optional
        The 2.5th and 97.5th percentiles (95% CI), only if return_ci=True.
    """
    rng = np.random.default_rng(seed)
    x0   = np.array(x0,   float)
    xerr = np.array(xerr, float)

    # draw samples
    S = rng.normal(loc=x0, scale=xerr, size=(N, len(x0)))
    # evaluate
    y = np.array([func(*s) for s in S])

    mean  = y.mean()
    sigma = y.std(ddof=1)

    if return_ci:
        ci_low, ci_high = np.percentile(y, [2.5, 97.5])
        return mean, sigma, ci_low, ci_high
    else:
        return mean, sigma



# Example usage:
# function and inputs
# f = lambda H, T: H / T**2
# H0, H_err = 100.0, 0.5
# T0, T_err = 300.0, 2.0

# # only 1σ
# mean1, err1 = monte_carlo_error(f, [H0, T0], [H_err, T_err], N=200_000, seed=1)
# print(f"Mean: {mean1:.6f}, 1σ: ±{err1:.6f}")

# # plus 95% CI
# mean2, err2, ci_lo, ci_hi = monte_carlo_error(
#     f, [H0, T0], [H_err, T_err],
#     N=200_000, seed=1, return_ci=True
# )
# print(
#     f"Mean: {mean2:.6f}, 1σ: ±{err2:.6f}, "
#     f"95% CI: [{ci_lo:.6f}, {ci_hi:.6f}]"
# )








def monte_carlo_error_asymmetric(
    func, x0, x_low, x_high,
    N=10_000, seed=None, return_ci=False
):
    """
    Monte Carlo propagation of uncertainty when inputs have asymmetric errors.
    Samples each input from a split-normal distribution defined by x_low < x0 < x_high.
    """
    rng = np.random.default_rng(seed)
    x0     = np.array(x0,     float)
    x_low  = np.array(x_low,  float)
    x_high = np.array(x_high, float)

    n_vars = len(x0)
    S = np.empty((N, n_vars), float)

    # for each variable, draw N samples from a split‐normal
    for j in range(n_vars):
        mu   = x0[j]
        sig_l = mu - x_low[j]
        sig_h = x_high[j] - mu
        if sig_l < 0 or sig_h < 0:
            raise ValueError(f"Bounds must bracket x0: var {j}")
        if sig_l == 0 and sig_h == 0:
            # no uncertainty, just use the nominal value
            S[:, j] = mu
            continue

        # mixing probability proportional to area under each half
        p_low = sig_l / (sig_l + sig_h)

        # draw uniform [0,1) to choose which side to sample
        U = rng.random(N)
        # draw full sets of left‐ and right‐side normals
        left_samples  = rng.normal(loc=mu, scale=sig_l, size=N)
        right_samples = rng.normal(loc=mu, scale=sig_h, size=N)

        # pick per sample
        S[:, j] = np.where(U < p_low, left_samples, right_samples)

    # evaluate the function
    y = np.array([func(*s) for s in S])

    # compute statistics
    mean  = y.mean()
    lower, upper = np.percentile(y, [16, 84])
    sigma = (upper - lower) / 2.0

    if return_ci:
        ci_low, ci_high = np.percentile(y, [2.5, 97.5])
        return mean, sigma, lower, upper, ci_low, ci_high
    else:
        return mean, sigma, lower, upper