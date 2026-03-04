
import math
import numpy as np

__all__ = [
    "shapiro_rudin_phase",
    "newman_phase",
    "multitone",
    "compute_crest_factor",
    "compute_amplitude_distribution",
    "compute_fourier_coeffs",
]


def _rudin_signs(N, x0=[1, 1]):
    assert len(x0) % 2 == 0, 'Length of x0 must be even'
    x = [_ for _ in x0]
    while len(x) < N:
        x += [y if i < len(x) // 2 else -y for i, y in enumerate(x)]
    return x


def shapiro_rudin_phase(k, N):
    k = np.asarray(k)
    assert np.all(k <= N)
    delta = np.array([0 if r == 1 else math.pi for r in _rudin_signs(N)])
    return delta[k - 1]


def newman_phase(k, N):
    k = np.asarray(k)
    assert np.all(k <= N)
    return math.pi * ((k - 1) ** 2) / N


def multitone(t, N, method='newman', N0=0, omega0=1.0):
    if method.lower() in ('newman', 'n'):
        delta_fun = newman_phase
    elif method.lower() in ('shapiro-rudin', 'sr'):
        delta_fun = shapiro_rudin_phase
    else:
        raise Exception("Accepted methods are 'Newman' or 'Shapiro-Rudin'")
    t = np.asarray(t, dtype=float)
    u = np.zeros_like(t)
    for k in range(1, N + 1):
        delta = delta_fun(k, N)
        u += np.cos((k + N0) * omega0 * t + delta)
    return np.sqrt(2 / N) * u


def compute_crest_factor(u, dt=1.0, dB=0):
    from scipy.integrate import trapezoid
    t = np.arange(u.size) * dt
    CF = np.max(np.abs(u)) / np.sqrt(trapezoid(u**2, t) / (t[-1] - t[0]))
    if dB > 0:
        return dB * np.log10(CF)
    return CF


def compute_amplitude_distribution(u, bins):
    x = np.abs(u)
    edges = np.histogram_bin_edges(x, bins)
    if edges[0] != 0:
        edges = np.concat(([0], edges))
    if edges[-1] != x.max():
        edges = np.concat((edges, [x.max()]))
    n = np.zeros_like(edges)
    n = np.array([(x > e).sum() for e in edges], dtype=float)
    return n / x.size, edges


def compute_fourier_coeffs(freq, t, x, phase=0.0, A=1.0):
    """
    Compute complex Fourier coefficients for one or multiple frequencies
    and phases.

    Parameters
    ----------
    freq : float or array_like
        Frequency or array of frequencies (Hz) at which the Fourier
        coefficients are evaluated.
    t : (N,) array_like
        Time vector corresponding to the signal samples. Must have the same
        length as the first dimension of `x`. Sampling need not be uniform.
    x : (N,) or (N, M) array_like
        Signal values sampled at times `t`. If 1D, a single time series
        of length `N`. If 2D, each column represents an independent signal
        with `N` time samples and `M` parallel series.
    phase : float or array_like
        Phase offset(s) in radians applied to the complex Fourier basis
        functions. Must be broadcastable with `freq`.
    A : float or array_like
        Amplitude(s) applied to the complex Fourier basis functions.
        Must be broadcastable with `freq`.

    Returns
    -------
    coeffs : complex or ndarray of complex
        Complex Fourier coefficient(s) of the form ``a + 1j * b``, where
        ``a`` and ``b`` are the cosine and sine projection components,
        respectively.

        If `freq` and/or `phase` are array-like, the output shape follows
        NumPy broadcasting rules. If `x` is 2D, coefficients are computed
        independently for each column, with projections performed along
        the time axis (axis 0).

    Notes
    -----
    The coefficients are obtained by projecting `x` onto complex
    exponentials (or equivalently sine and cosine basis functions) of
    frequency `freq` with phase shift `phase`.

    No assumption of uniform sampling is made; the time vector `t`
    may be irregularly spaced.
    """
    from scipy.integrate import trapezoid
    freq = np.atleast_1d(freq)
    phase = np.asarray(phase) + np.zeros_like(freq)
    A = np.asarray(A) + np.zeros_like(freq)
    if t.size == x.shape[0]:
        x = x.T
    T = t[-1] - t[0]
    # The `-` sign below is present in the definition of the Fourier
    # integral. Note that in Eq. 3 of [1], the `-` sign is missing.
    # [1] Bizzarri F, Brambilla A, Del Giudice D, Linaro D.
    # An Active-Perturbation Method to Estimate Online Inertia and
    # Damping in Electric Power Systems. In 2024 IEEE International
    # Symposium on Circuits and Systems (ISCAS) 2024 May 19 (pp. 1-5).
    return np.array([
        2.0 / T * (
            trapezoid(x * a * np.cos(2 * np.pi * f * t + phi), t) -
            1j * trapezoid(x * a * np.sin(2 * np.pi * f * t + phi), t)
        ) for f, phi, a in zip(freq, phase, A)
    ])
