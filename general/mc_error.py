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
            raise ValueError(f"Bounds must bracket x0: var {j}. sig_l={sig_l}, sig_h={sig_h}, x0={mu}")
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
    median = np.median(y)
    lower, upper = np.percentile(y, [16, 84])
    sigma = (upper - lower) / 2.0
    chosen_avg = median  # or use mean

    if return_ci:
        ci_low, ci_high = np.percentile(y, [2.5, 97.5])
        return chosen_avg, sigma, lower, upper, ci_low, ci_high
    else:
        return chosen_avg, sigma, lower, upper























import numpy as np

def monte_carlo_error_asymmetric_w_bounds(
    func,
    x0,
    x_low,
    x_high,
    N=10_000,
    seed=None,
    return_ci=False,
    bounds=None,
    bound_mode='truncate',   # 'truncate' | 'clip' | 'reflect'
    max_resample_factor=1000,
    return_samples=False,
    ABS_EPS=1e-20,           # absolute tolerance for near-zero checks
    REL_EPS=1e-10,           # relative tolerance for near-zero checks
):
    """
    Monte Carlo uncertainty propagation for variables with asymmetric errors.

    Each variable j has nominal value x0[j] with lower/upper "1σ-like" bounds
    x_low[j] < x0[j] < x_high[j]. A split normal is formed with:
        sigma_left  = x0[j] - x_low[j]
        sigma_right = x_high[j] - x0[j]
    Probability mass on each side is proportional to its sigma (so that the
    PDF is continuous at the mean and areas on each side match their widths).

    Optional hard bounds can be applied:

        bounds:
            - None (default): unbounded split normal.
            - (L, U): same numeric bounds for all variables.
            - array-like shape (n_vars, 2): per-variable bounds [[L1,U1], ...].

        bound_mode:
            - 'truncate': rejection sample until all draws lie in [L, U]
                          (statistically correct for truncation). If too many
                          rejections, raises an error (controlled by
                          max_resample_factor).
            - 'clip':     draw from original distribution then np.clip; fast
                          but density piles at edges (NOT a true truncated normal).
            - 'reflect':  reflect values outside (L,U) back into interval
                          (can create multi-modal / folded tails).

    Parameters
    ----------
    func : callable
        Function of n_vars positional arguments. Will be evaluated N times.
        (If you can vectorize func yourself, adapt code for speed.)
    x0, x_low, x_high : array-like
        Nominal values and their asymmetric bounds (same length).
    N : int
        Number of Monte Carlo samples.
    seed : int or None
        RNG seed.
    return_ci : bool
        If True, also return central 95% interval (2.5%, 97.5%).
    bounds : None, (L,U), or array-like (n_vars,2)
        Physical bounds (see above).
    bound_mode : {'truncate','clip','reflect'}
        How to enforce bounds.
    max_resample_factor : int
        Safety cap: we allow at most max_resample_factor * N attempted draws
        per variable when truncating before raising an error.
    return_samples : bool
        If True, include the raw y samples in the return tuple (last element).

    Returns
    -------
    mean, sigma, lower_16, upper_84  [, ci_low_2p5, ci_high_97p5] [, y_samples]

        sigma = (84th - 16th)/2

    Notes
    -----
    - For symmetric uncertainties set x_low = x0 - σ, x_high = x0 + σ.
    - 'truncate' keeps the correct truncated distribution; 'clip'/'reflect'
      alter the target distribution—use only if approximate handling is OK.
    """
    rng = np.random.default_rng(seed)

    x0     = np.asarray(x0,     dtype=float)
    x_low  = np.asarray(x_low,  dtype=float)
    x_high = np.asarray(x_high, dtype=float)

    if not (x0.shape == x_low.shape == x_high.shape):
        raise ValueError("x0, x_low, x_high must have identical shapes.")

    n_vars = x0.size
    S = np.empty((N, n_vars), dtype=float)

    # Normalize bounds input
    if bounds is None:
        bounds_arr = None
    else:
        b = np.asarray(bounds, dtype=float)
        if b.ndim == 1 and b.size == 2:
            bounds_arr = np.repeat(b.reshape(1,2), n_vars, axis=0)
        elif b.shape == (n_vars, 2):
            bounds_arr = b
        else:
            raise ValueError("bounds must be None, (low, high), or shape (n_vars,2).")

    if bounds_arr is not None:
        if np.any(bounds_arr[:,0] >= bounds_arr[:,1]):
            raise ValueError("Each bounds row must satisfy low < high.")
        if bound_mode not in {'truncate','clip','reflect'}:
            raise ValueError("bound_mode must be 'truncate', 'clip', or 'reflect'.")

    for j in range(n_vars):
        mu    = x0[j]
        sig_l = mu - x_low[j]
        sig_r = x_high[j] - mu
        if sig_l < 0 or sig_r < 0:
            raise ValueError(f"Asymmetric bounds must bracket x0 for var {j}: "
                             f"x_low={x_low[j]}, x0={mu}, x_high={x_high[j]}")
        if sig_l == 0 and sig_r == 0:
            S[:, j] = mu
            continue

        p_left = sig_l / (sig_l + sig_r)

        if bounds_arr is None:
            # Unbounded: simple mixture sample
            U = rng.random(N)
            left  = rng.normal(loc=mu, scale=sig_l, size=N)
            right = rng.normal(loc=mu, scale=sig_r, size=N)
            S[:, j] = np.where(U < p_left, left, right)
            continue

        # With bounds
        L, U_bound = bounds_arr[j]

        if bound_mode == 'truncate':
            # Rejection sampling
            samples = []
            needed = N
            attempts_left = max_resample_factor * N
            while needed > 0 and attempts_left > 0:
                batch = max(1024, needed)  # draw in chunks
                u = rng.random(batch)
                left  = rng.normal(loc=mu, scale=sig_l, size=batch)
                right = rng.normal(loc=mu, scale=sig_r, size=batch)
                draw  = np.where(u < p_left, left, right)
                accept = draw[(draw >= L) & (draw <= U_bound)]
                if accept.size:
                    take = min(accept.size, needed)
                    samples.append(accept[:take])
                    needed -= take
                attempts_left -= batch
            if needed > 0:
                raise RuntimeError(
                    f"Truncation rejection sampling exhausted attempts for var {j}. "
                    f"Consider increasing max_resample_factor or using 'clip'."
                )
            S[:, j] = np.concatenate(samples)
        else:
            # First draw unbounded mixture
            u = rng.random(N)
            left  = rng.normal(loc=mu, scale=sig_l, size=N)
            right = rng.normal(loc=mu, scale=sig_r, size=N)
            draw = np.where(u < p_left, left, right)

            if bound_mode == 'clip':
                draw = np.clip(draw, L, U_bound)
            elif bound_mode == 'reflect':
                # Reflect repeatedly until inside bounds
                # (handles multiple reflections for outliers)
                while True:
                    low_mask  = draw < L
                    high_mask = draw > U_bound
                    if not (low_mask.any() or high_mask.any()):
                        break
                    draw[low_mask]  = 2*L - draw[low_mask]
                    draw[high_mask] = 2*U_bound - draw[high_mask]
                # If reflections overshoot (rare), final clip safeguard
                draw = np.clip(draw, L, U_bound)
            S[:, j] = draw

    # Evaluate function (non-vectorized func)
    # If func can accept arrays, you could vectorize for speed.
    y = np.fromiter((func(*row) for row in S), dtype=float, count=N)

    mean = y.mean()
    median = np.median(y)
    lower_16, upper_84 = np.percentile(y, [16, 84])
    sigma = 0.5 * (upper_84 - lower_16)

    chosen_avg = median # mean

    # check if lower_16 and upper_84 are valid wrt chosen_avg : lower_16 < chosen_avg < upper_84
    if not (lower_16 <= chosen_avg <= upper_84):
        # if fractional difference between them is 1E-10, make them all equal to chosen_avg
        spread_1 = upper_84 - chosen_avg
        spread_2 = chosen_avg - lower_16
        center1 = 0.5*(upper_84 + chosen_avg)
        center2 = 0.5*(chosen_avg + lower_16)
        if spread_1 <= max(ABS_EPS, REL_EPS * abs(center1)):
            upper_84 = chosen_avg * (1 + REL_EPS)  # ensure upper_84 > chosen_avg
        elif spread_2 <= max(ABS_EPS, REL_EPS * abs(center2)):
            lower_16 = chosen_avg * (1 - REL_EPS)  # ensure lower_16 < chosen_avg
        else:
            raise ValueError(f"Invalid percentiles?: {lower_16}, {chosen_avg}, {upper_84}")

    results = [chosen_avg, sigma, lower_16, upper_84]
    if return_ci:
        ci_low, ci_high = np.percentile(y, [2.5, 97.5])
        results.extend([ci_low, ci_high])
    if return_samples:
        results.append(y)
    return tuple(results)
































import numpy as np

def monte_carlo_error_asymmetric_w_io_bounds(
    func,
    x0,
    x_low,
    x_high,
    N=10_000,
    seed=None,
    input_bounds=None,       # None, (L,U) or shape (n_vars,2)
    output_bounds=None,      # None or (Y_low, Y_high)
    batch_size=None,
    max_input_resample_factor=100000,
    max_output_resample_factor=200000,
    return_ci=False,
    return_samples=False,    # return (S, y) at end
    return_rates=False,       # return acceptance rates
    ABS_EPS = 1e-20,          # or tuned to your problem scale
    REL_EPS = 1e-10
):
    """
    Monte Carlo uncertainty propagation with asymmetric (split-normal) input errors
    and *output distribution truncation*.

    Each variable j has nominal value x0[j] with lower / upper "1σ-like" bounds:
        sigma_left  = x0[j] - x_low[j]
        sigma_right = x_high[j] - x0[j]
    We sample a mixture of two half normals (left/right) weighted by their sigmas.

    Input bounds (if given) are enforced by *input-level truncation* (rejection).
    Output bounds (if given) are enforced by *output-level truncation*:
        Only samples where Y = func(*x_sample) lies within [Y_low, Y_high] are kept,
        until N accepted outputs are collected (or attempts exhausted).

    Parameters
    ----------
    func : callable
        Function taking n_vars positional arguments -> scalar.
    x0, x_low, x_high : array-like
        Nominal and asymmetric 1σ limits (same shape, length n_vars).
        Must satisfy x_low[j] <= x0[j] <= x_high[j].
    N : int
        Number of *accepted* output samples desired.
    seed : int or None
        RNG seed.
    input_bounds : None, (L,U), or array-like shape (n_vars,2)
        Hard physical bounds per variable. If provided, input sampling rejects
        any draw outside its variable's [L_i, U_i].
    output_bounds : (Y_low, Y_high) or None
        If provided, triggers output truncation: only outputs within the interval
        are retained.
    batch_size : int or None
        Number of candidate draws generated per iteration. If None, picks
        a heuristic based on n_vars and N.
    max_input_resample_factor : int
        Safety cap on total candidate *input* draws relative to N when *only*
        input truncation is active (no output truncation). If exceeded before
        completing N samples, raises RuntimeError.
    max_output_resample_factor : int
        Safety cap on total candidate draws relative to N when output truncation
        is active. (Total attempted draws ≤ max_output_resample_factor * N.)
    return_ci : bool
        Also return 2.5% and 97.5% percentiles.
    return_samples : bool
        Append (S, y) of accepted samples to return tuple (S shape (N, n_vars)).

    Returns
    -------
    mean, sigma, p16, p84 [, ci2p5, ci97p5, input_accept_rate, output_accept_rate] [, S, y]
        sigma = (p84 - p16)/2.
        input_accept_rate  = accepted_input / attempted_input  (after input truncation)
        output_accept_rate = accepted_output / attempted_output (only if output_bounds)
    """
    rng = np.random.default_rng(seed)

    x0     = np.asarray(x0, dtype=float)
    x_low  = np.asarray(x_low, dtype=float)
    x_high = np.asarray(x_high, dtype=float)
    if not (x0.shape == x_low.shape == x_high.shape):
        raise ValueError("x0, x_low, x_high must match shapes.")
    n_vars = x0.size

    sig_left  = x0 - x_low
    sig_right = x_high - x0
    if np.any(sig_left < 0) or np.any(sig_right < 0):
        raise ValueError(f"Each x0 must satisfy x_low <= x0 <= x_high. x0: {x0}, x_low: {x_low}, x_high: {x_high}")

    # Normalize input bounds
    if input_bounds is not None:
        ib = np.asarray(input_bounds, dtype=float)
        if ib.ndim == 1 and ib.size == 2:
            input_bounds_arr = np.repeat(ib.reshape(1,2), n_vars, axis=0)
        elif ib.shape == (n_vars, 2):
            input_bounds_arr = ib
        else:
            raise ValueError("input_bounds must be None, (L,U) or shape (n_vars,2)")
        if np.any(input_bounds_arr[:,0] >= input_bounds_arr[:,1]):
            raise ValueError("Each input_bounds row must satisfy low < high.")
    else:
        input_bounds_arr = None

    # Output bounds
    if output_bounds is not None:
        # Normalize: allow [(low, high)] or (low, high)
        if isinstance(output_bounds, (list, tuple)) and len(output_bounds) == 1 and \
        isinstance(output_bounds[0], (list, tuple)) and len(output_bounds[0]) == 2:
            output_bounds = output_bounds[0]

        if (isinstance(output_bounds, (list, tuple)) and len(output_bounds) == 2
                and not isinstance(output_bounds[0], (list, tuple))):
            Y_low, Y_high = map(float, output_bounds)
        else:
            raise ValueError(
                "output_bounds must be (low, high) or a single-element list/tuple wrapping that."
            )
        if Y_low >= Y_high:
            raise ValueError("output_bounds must satisfy Y_low < Y_high.")
    else:
        Y_low = Y_high = None

    # Batch heuristic
    if batch_size is None:
        batch_size = min(max(1024, N // 10), 50_000)

    weights_left = sig_left / np.where(sig_left + sig_right == 0, 1, sig_left + sig_right)
    weights_left = np.where(sig_left + sig_right == 0, 0.5, weights_left)

    # Counters
    raw_input_draws = 0       # before input truncation
    survived_input  = 0       # after input truncation
    attempted_output = 0      # (same as survived_input)
    accepted_count  = 0       # after output truncation

    # Storage (list of batches)
    accepted_X_batches = []
    accepted_y_batches = []

    # Max draws safety cap
    if output_bounds is None:
        max_draws = max_input_resample_factor * N
    else:
        max_draws = max_output_resample_factor * N

    # Helper
    def draw_input_batch(size):
        nonlocal raw_input_draws, survived_input
        raw_input_draws += size

        U_mix = rng.random((size, n_vars))
        Z_left  = rng.normal(loc=x0, scale=sig_left,  size=(size, n_vars))
        Z_right = rng.normal(loc=x0, scale=sig_right, size=(size, n_vars))
        X = np.where(U_mix < weights_left, Z_left, Z_right)

        # Zero-uncertainty fixes
        zero_mask = (sig_left == 0) & (sig_right == 0)
        if zero_mask.any():
            X[:, zero_mask] = x0[zero_mask]

        if input_bounds_arr is not None:
            ok = np.ones(size, dtype=bool)
            for j in range(n_vars):
                L, U = input_bounds_arr[j]
                ok &= (X[:, j] >= L) & (X[:, j] <= U)
                if not ok.any():
                    break
            X = X[ok]
        survived_input += X.shape[0]
        return X

    while accepted_count < N:
        if attempted_output >= max_draws:
            raise RuntimeError(
                f"Exceeded maximum draws ({attempted_output}) with only "
                f"{accepted_count} accepted outputs (target {N}). "
                "Loosen bounds or raise max_*_resample_factor."
            )

        X_batch = draw_input_batch(batch_size)
        if X_batch.size == 0:
            continue

        # Evaluate outputs (vectorized attempt)
        try:
            y_batch = func(*[X_batch[:, j] for j in range(n_vars)])
            y_batch = np.asarray(y_batch, float)
            if y_batch.shape != (X_batch.shape[0],):
                raise TypeError
        except Exception:
            y_batch = np.fromiter((func(*row) for row in X_batch),
                                    dtype=float, count=X_batch.shape[0])

        attempted_output += X_batch.shape[0]

        # Output truncation
        if output_bounds is not None:
            keep = (y_batch >= Y_low) & (y_batch <= Y_high)
            if not keep.any():
                continue
            X_keep = X_batch[keep]
            y_keep = y_batch[keep]
        else:
            X_keep = X_batch
            y_keep = y_batch

        if X_keep.shape[0] == 0:
            continue

        needed = N - accepted_count
        if X_keep.shape[0] > needed:
            X_keep = X_keep[:needed]
            y_keep = y_keep[:needed]

        accepted_X_batches.append(X_keep)
        accepted_y_batches.append(y_keep)
        accepted_count += y_keep.shape[0]

    # Assemble final arrays
    S = np.vstack(accepted_X_batches)
    y = np.concatenate(accepted_y_batches)
    assert S.shape[0] == y.shape[0] == N

    # Acceptance statistics
    input_accept_rate  = survived_input / raw_input_draws if raw_input_draws else 1.0
    output_accept_rate = accepted_count / attempted_output if attempted_output else 1.0
    overall_accept_rate = accepted_count / raw_input_draws if raw_input_draws else 1.0

    # # truncate all y values to 10 significant figures
    # y = np.round(y, 10)

    # Summary stats
    p16, p84 = np.percentile(y, [16, 84])
    mean = y.mean()
    median = np.median(y)
    sigma = 0.5 * (p84 - p16)

    chosen_avg = median # mean

    # check if p16 and p84 are valid wrt mean : p16 < mean < p84
    if not (p16 <= chosen_avg <= p84):
        # if fractional difference between them is 1E-10, make them all equal to mean
        spread_1 = p84 - chosen_avg
        spread_2 = chosen_avg - p16
        center1 = 0.5*(p84 + chosen_avg)
        center2 = 0.5*(chosen_avg + p16)
        if spread_1 <= max(ABS_EPS, REL_EPS * abs(center1)):
            p84 = chosen_avg * (1 + REL_EPS)  # ensure p84 > mean
        elif spread_2 <= max(ABS_EPS, REL_EPS * abs(center2)):
            p16 = chosen_avg * (1 - REL_EPS)  # ensure p16 < mean
        else:
            raise ValueError(f"Invalid percentiles?: {p16}, {chosen_avg}, {p84}")
    elif np.allclose([p16, chosen_avg, p84], 0, atol=ABS_EPS): # near-zero check
        # print(f"Percentiles are valid wrt p16, chosen_avg, p84: {p16}, {chosen_avg}, {p84}")
        # print all parameters
        print(f"x0: {x0}, x_low: {x_low}, x_high: {x_high}")
        print(f"input_bounds: {input_bounds_arr}, output_bounds: {(Y_low, Y_high)}")
        print(f"raw_input_draws: {raw_input_draws}, survived_input: {survived_input}")
        print(f"attempted_output: {attempted_output}, accepted_count: {accepted_count}")
        print(f"input_accept_rate: {input_accept_rate}, output_accept_rate: {output_accept_rate}, overall_accept_rate: {overall_accept_rate}")
        print(f"y samples: {y[:10]}...{y[-10:]} (total {len(y)})") # show first 10 samples + last 10
        print("")
        raise ValueError(f"All percentiles are near-zero: {p16}, {mean}, {p84}. ")

    ret = [mean, sigma, p16, p84]
    if return_ci:
        p2, p97 = np.percentile(y, [2.5, 97.5])
        ret.extend([p2, p97])

    if return_rates:
        ret.extend([input_accept_rate, output_accept_rate, overall_accept_rate])

    if return_samples:
        ret.extend([S, y])

    return tuple(ret)


# # ---------------------------
# # Example usage
# if __name__ == "__main__":
#     # Toy function: product
#     def f(x, y):
#         return x * y

#     x0     = [2.0,  3.0]
#     x_low  = [1.8,  2.6]
#     x_high = [2.4,  3.5]

#     # Input bounds & output bounds
#     input_bounds  = [(1.5, 2.5), (2.0, 4.0)]
#     output_bounds = (3.0, 8.0)  # only keep x*y within [3,8]

#     mean, sigma, p16, p84, ci2, ci97, in_acc, out_acc = monte_carlo_error_asymmetric(
#         f, x0, x_low, x_high,
#         N=50_000,
#         seed=42,
#         input_bounds=input_bounds,
#         output_bounds=output_bounds,
#         return_ci=True
#     )

#     print(f"Mean={mean:.4f}  σ≈{sigma:.4f}  (16%={p16:.4f}, 84%={p84:.4f})")
#     print(f"95% CI: [{ci2:.4f}, {ci97:.4f}]")
#     print(f"Input accept rate ~ {in_acc:.3f}, Output accept rate ~ {out_acc:.3f}")































#!/usr/bin/env python3
import numpy as np

def monte_carlo_error_asymmetric_w_io_bounds_vectorized_outputs(
    func,
    x0,
    x_low,
    x_high,
    N=1000,
    seed=None,
    input_bounds=None,       # None, (L,U) or shape (n_vars,2)
    output_bounds=None,      # None, (low,high) or list/array shape (M,2)
    batch_size=None,
    max_input_resample_factor=10000000,
    max_output_resample_factor=20000000,
    return_ci=False,
    return_samples=False,
    return_rates=False,
    ABS_EPS = 1e-20,
    REL_EPS = 1e-10
):
    """
    Monte Carlo uncertainty propagation for potentially vector‐valued func.
    See docstring in prompt for details—now returns arrays of length M.
    """
    rng = np.random.default_rng(seed)
    x0     = np.asarray(x0,    float)
    x_low  = np.asarray(x_low,  float)
    x_high = np.asarray(x_high, float)
    if not (x0.shape == x_low.shape == x_high.shape):
        raise ValueError("x0, x_low, x_high must match shapes")
    n_vars = x0.size

    # asymmetric sigmas
    sig_left  = x0 - x_low
    sig_right = x_high - x0
    if np.any(sig_left < 0) or np.any(sig_right < 0):
        raise ValueError("Each x0 must satisfy x_low <= x0 <= x_high")

    # normalize input_bounds
    if input_bounds is not None:
        ib = np.asarray(input_bounds, float)
        if ib.ndim==1 and ib.size==2:
            input_bounds_arr = np.repeat(ib.reshape(1,2), n_vars, axis=0)
        elif ib.shape==(n_vars,2):
            input_bounds_arr = ib
        else:
            raise ValueError("input_bounds must be None, (L,U), or shape (n_vars,2)")
    else:
        input_bounds_arr = None

    # we'll defer parsing output_bounds until we know M
    raw_output_bounds = output_bounds
    Y_low = Y_high = None
    ob_parsed = False

    # batch heuristic
    if batch_size is None:
        batch_size = min(max(100, N//10), 50_000)
        # batch_size = min(max(1024, N//10), 50_000)

    # mixing weights
    weights_left = sig_left / np.where(sig_left+sig_right==0, 1, sig_left+sig_right)
    weights_left = np.where((sig_left+sig_right)==0, 0.5, weights_left)

    # counters & storage
    raw_input_draws = survived_input = attempted_output = accepted_count = 0
    accepted_X_batches = []
    accepted_y_batches = []

    # max draws cap
    max_draws = (max_output_resample_factor if raw_output_bounds is not None
                 else max_input_resample_factor) * N

    def draw_input_batch(sz):
        nonlocal raw_input_draws, survived_input
        raw_input_draws += sz
        U = rng.random((sz, n_vars))
        Zl = rng.normal(loc=x0, scale=sig_left,  size=(sz, n_vars))
        Zr = rng.normal(loc=x0, scale=sig_right, size=(sz, n_vars))
        X = np.where(U < weights_left, Zl, Zr)
        # zero-uncertainty
        zero_mask = (sig_left==0)&(sig_right==0)
        if zero_mask.any():
            X[:, zero_mask] = x0[zero_mask]
        if input_bounds_arr is not None:
            ok = np.ones(sz, bool)
            for j in range(n_vars):
                L,Uj = input_bounds_arr[j]
                ok &= (X[:,j]>=L)&(X[:,j]<=Uj)
                if not ok.any(): break
            X = X[ok]
        survived_input += X.shape[0]
        return X

    # main sampling loop
    while accepted_count < N:
        # print(f"T: {x0[-1]}, X: {x0[0]}, Draws: {raw_input_draws}, Accepted: {accepted_count}/{N}")
        if raw_input_draws >= max_draws:
            raise RuntimeError(f"Exceeded max draws ({raw_input_draws}) with only {accepted_count} accepts")

        Xb = draw_input_batch(batch_size)
        if Xb.size==0: 
            continue

        # evaluate vector func
        try:
            yb = func(*[Xb[:,j] for j in range(n_vars)])
            yb = np.asarray(yb, float)
        except Exception:
            # fallback to row‐by‐row
            yb = np.fromiter((func(*row) for row in Xb), float)

        # ensure shape (batch, M)
        if yb.ndim==1:
            yb = yb[:,None]
        M = yb.shape[1]

        # parse output_bounds once we know M
        if (not ob_parsed) and (raw_output_bounds is not None):
            ob = np.asarray(raw_output_bounds, float)
            if ob.ndim==1 and ob.size==2:
                Y_low  = np.full((M,), ob[0])
                Y_high = np.full((M,), ob[1])
            elif ob.ndim==2 and ob.shape==(M,2):
                Y_low  = ob[:,0]
                Y_high = ob[:,1]
            else:
                raise ValueError("output_bounds must be (low,high) or shape (M,2)")
            if np.any(Y_low >= Y_high):
                raise ValueError("Each output_bounds[i] must satisfy low < high")
            ob_parsed = True

        # count attempted outputs
        attempted_output += Xb.shape[0]

        # truncate on outputs if requested
        if Y_low is not None:
            keep = np.all((yb>=Y_low)&(yb<=Y_high), axis=1)
            if not keep.any():
                # print(f"Attempts: {raw_input_draws}")
                # print(f"Bad outputs in bounds {Y_low} to {Y_high} for batch of shape: {yb[~keep].shape}")
                # print(f"Bad outputs: {yb[~keep]}")
                # print(f"Bad inputs: {Xb[~keep]}")
                continue
            Xk = Xb[keep]
            yk = yb[keep]
        else:
            Xk, yk = Xb, yb

        # accept up to remaining
        nkeep = yk.shape[0]
        needed = N - accepted_count
        if nkeep > needed:
            Xk = Xk[:needed]
            yk = yk[:needed]
            nkeep = needed

        accepted_X_batches.append(Xk)
        accepted_y_batches.append(yk)
        accepted_count += nkeep

    # stack results: S=(N,n_vars), Y=(N,M)
    S = np.vstack(accepted_X_batches)
    Y = np.vstack(accepted_y_batches)
    assert S.shape[0]==Y.shape[0]==N

    # acceptance rates
    in_rate  = survived_input / raw_input_draws if raw_input_draws else 1.0
    out_rate = (accepted_count/attempted_output) if attempted_output else 1.0

    # compute per-dimension stats
    p16    = np.percentile(Y, 16, axis=0)
    p84    = np.percentile(Y, 84, axis=0)
    mean   = Y.mean(axis=0)
    median = np.median(Y, axis=0)
    sigma  = 0.5*(p84 - p16)

    chosen_avg = median  # or use mean

    # print(f"X: {x0[0]}, T: {x0[-1]}, Xw: {Y[0]}, accepted_count/raw_input_draws: {accepted_count/raw_input_draws:.3f}, raw_input_draws: {raw_input_draws}")

    # optional CI
    ret = [chosen_avg, sigma, p16, p84]
    if return_ci:
        p2, p97 = np.percentile(Y, [2.5,97.5], axis=0)
        ret.extend([p2, p97])
    if return_rates:
        ret.extend([in_rate, out_rate, accepted_count/raw_input_draws])
    if return_samples:
        ret.extend([S, Y])

    return tuple(ret)