
import math
import numpy as np

__all__ = [
    "RMS",
    "RMS_from_fourier_coeffs",
    "RMS_from_complex_fourier_coeffs",
    "shapiro_rudin_phase",
    "newman_phase",
    "compute_crest_factor",
    "compute_amplitude_distribution",
    "multitone",
    "multitone_boyd",
    "multitone_friese",
    "compute_multitone_pars",
    "optimize_phases",
    "compute_fourier_coeffs",
]


def RMS(x, t=None, dt=1.0):
    from scipy.integrate import trapezoid
    if t is None:
        return np.sqrt(trapezoid(np.abs(x)**2, dx=dt) / (dt * (x.size - 1)))
    return np.sqrt(trapezoid(np.abs(x)**2, t) / (t[-1] - t[0]))


def RMS_from_fourier_coeffs(a, b, a0=0):
    assert all(np.isreal(a) & np.isreal(b)), 'a and b must be real values'
    return np.sqrt((a0 / 2) ** 2 + 1 / 2 * (np.sum(a ** 2 + b ** 2)))


def RMS_from_complex_fourier_coeffs(coeffs, c0=0):
    return np.sqrt(c0 ** 2 + np.sum(0.5 * np.abs(coeffs) ** 2))


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


def multitone(t, A, omega, phases):
    t = np.asarray(t)
    A = np.asarray(A) + np.zeros_like(omega)
    x = np.zeros_like(t)
    for a, w, phi in zip(A, omega, phases):
        x += a * np.cos(w * t + phi)
    return x


def multitone_boyd(t, N, method='newman', N0=0, w0=1.0):
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
        u += np.cos((k + N0) * w0 * t + delta)
    return np.sqrt(2 / N) * u


def multitone_friese(t, N=None, phases=None, dw=1.0, w0=None, filename=None):
    if w0 is None:
        w0 = dw
    if phases is None:
        if filename is None:
            assert N is not None
            phases, _, _, _, _ = optimize_phases(dw, N, N_samples=5001, N_iters=2000, N_reps=10)
        else:
            phases = np.load(filename)['phases']
    m = np.exp(1j * phases)
    M = _compute_M(m, t, dw)
    return (M * np.exp(1j * w0 * t)).real


def compute_multitone_pars(N, method, w0=None, dw=None, N0=0, phase_method='newman', seed=None):
    method = method.lower()
    if method == 'boyd':
        k = 1 + np.arange(N)
        A = np.sqrt(2 / N) * np.ones(N)
        w = (k + N0) * w0
        phase_method = phase_method.lower()
        if phase_method in ('newman', 'n'):
            phi = newman_phase(k, N)
        elif phase_method in ('shapiro-rudin', 'sr'):
            phi = shapiro_rudin_phase(k, N)
        else:
            raise ValueError("Phase method must be one of 'newman' or 'shapiro-rudin'")
    elif method == 'friese':
        if w0 is None:
            w0 = dw
        k = np.arange(N)
        A = np.ones(N)
        w = w0 + k * dw
        phi, _, _, _, _ = optimize_phases(dw, N, N_samples=5001, N_iters=2000, N_reps=10, seed=seed)
    else:
        raise ValueError("Method must be one of 'boyd' or 'friese'")        
    return A, w, phi


def compute_crest_factor(u, dt=1.0, dB=0):
    CF = np.max(np.abs(u)) / RMS(u, dt=dt)
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


def _compute_M(m, t, dw):
    M = np.zeros_like(t, dtype=complex)
    for i, mi in enumerate(m):
        M += mi * np.exp(1j * i * dw * t)
    return M


def _compute_error(M, S):
    E = np.zeros_like(M)
    Ma = np.abs(M)
    idx = Ma >= S
    E[idx] = (Ma[idx] - S) * np.exp(1j * np.angle(M[idx]))
    return E


def optimize_phases(dw, N_tones, N_samples, N_iters, N_reps=1, S0=None, seed=None):
    from numpy.random import RandomState, SeedSequence, MT19937
    from scipy.fft import fft
    from tqdm import tqdm
    if S0 is None:
        S0 = 1.1 * np.sqrt(N_tones)
    if seed is None:
        rng = np.random
    elif isinstance(seed, int):
        rng = RandomState(MT19937(SeedSequence(seed)))
    else:
        rng = seed
    T = 2 * np.pi / dw
    t = np.linspace(0, T, N_samples)
    dt = t[1] - t[0]
    m = np.zeros((N_iters, N_tones), dtype=complex)
    S = np.zeros(N_iters)
    CF = np.zeros(N_iters)
    CF_min = None
    for i in tqdm(range(N_reps), ascii=True, ncols=60):
        m[0] = np.exp(1j * rng.uniform(0, 2 * np.pi, N_tones))
        S[0] = S0
        M = _compute_M(m[0], t, dw)
        CF[0] = compute_crest_factor(M, dt)
        for j in range(1, N_iters):
            e = _compute_error(M, S[j - 1])
            ef = fft(e)[:N_tones] / e.size
            m[j] = np.exp(1j * np.angle(m[j - 1] - ef))
            S[j] = min(CF[j - 1] * np.sqrt(N_tones) * 0.95, S[j - 1] * 1.001)
            M = _compute_M(m[j], t, dw)
            CF[j] = compute_crest_factor(M, dt)
        if CF_min is None or CF.min() < CF_min:
            CF_min = CF.min()
            S_opt = S
            CF_opt = CF.copy()
            m_opt = m[np.argmin(CF)].copy()
    M_opt = _compute_M(m_opt, t, dw)
    return np.angle(m_opt), CF_opt, S_opt, t, M_opt


def compute_fourier_coeffs(freq, t, x, phase=0.0, A=1.0, iter_fun=None):
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
    if iter_fun is None:
        return np.array([
            2.0 / T * (
                trapezoid(x * a * np.cos(2 * np.pi * f * t + phi), t) -
                1j * trapezoid(x * a * np.sin(2 * np.pi * f * t + phi), t)
            ) for f, phi, a in zip(freq, phase, A)
        ])
    M, N = freq.size, x.shape[0]
    coeffs = np.zeros((M, N), dtype=complex)
    for i in iter_fun(range(M)):
        coeffs[i] = 2.0 / T * (
            trapezoid(x * A[i] * np.cos(2 * np.pi * freq[i] * t + phase[i]), t) -
            1j * trapezoid(x * A[i] * np.sin(2 * np.pi * freq[i] * t + phase[i]), t)
        )
    return coeffs


if __name__ == '__main__':
    import os
    import sys

    progname = os.path.basename(sys.argv[0])
    def usage(exit_code=None):
        prefix = '       ' + ' ' * (len(progname) + 1)
        print(f'usage: {progname} [-h | --help] [-o | --outfile <file>] [-f | --force]')
        print(prefix + '[--N-samples <n>] [--N-iters <n>] [--N-reps <n>]')
        print(prefix + '<-F <base frequency>> <-N <number of tones>>')
        if exit_code is not None:
            sys.exit(exit_code)

    F0 = None
    N_tones = None
    outfile = None
    force = False
    N_samples = 5001
    N_iters = 2000
    N_reps = 10

    N_args = len(sys.argv)
    if N_args == 1:
        usage(0)

    i = 1
    while i < N_args:
        arg = sys.argv[i]
        if arg == '-F':
            i += 1
            F0 = float(sys.argv[i])
        elif arg == '-N':
            i += 1
            N_tones = int(sys.argv[i])
        elif arg in ('-h', '--help'):
            usage(0)
        elif arg in ('-f', '--force'):
            force = True
        elif arg in ('-o', '--outfile'):
            i += 1
            outfile = sys.argv[i]
        elif arg == '--N-samples':
            i += 1
            N_samples = int(sys.argv[i])
            assert N_samples > 0, 'number of samples must be positive'
        elif arg == '--N-iters':
            i += 1
            N_iters = int(sys.argv[i])
            assert N_iters > 0, 'number of iterations must be positive'
        elif arg == '--N-reps':
            i += 1
            N_reps = int(sys.argv[i])
            assert N_iters > 0, 'number of repetitions must be positive'
        elif arg[0] == '-':
            print(f'{progname}: unknown option `{arg}`.')
            sys.exit(1)
        else:
            print(f'{progname}: positional arguments not required.')
            sys.exit(2)
        i += 1

    if F0 is None:
        print(f'{progname}: you must specify the base frequency via the -F option')
        sys.exit(3)
    assert F0 > 0, 'Base frequency must be positive'

    if N_tones is None:
        print(f'{progname}: you must specify the number of tones via the -N option')
        sys.exit(4)
    assert N_tones > 0, 'Number of tones must be positive'

    if outfile is None:
        outfile = 'multitone_phases_F0_{:.4f}_Hz_N_tones_{}.npz'.format(F0, N_tones)
    elif len(os.path.splitext(outfile)[1]) < 2:
        outfile = os.path.splitext(outfile)[0] + '.npz'
    if os.path.isfile(outfile) and not force:
        print(f'{progname}: {outfile} exists: use -f to force overwrite.')
        sys.exit(5)

    T = 1 / F0
    dw = 2 * np.pi * F0
    print(f'F0 = {F0:g} Hz')
    print(f'T = {T:g} s')
    print(f'Number of tones: {N_tones}')
    print(f'Output file name: {outfile}')

    phases, CF, S, t, _ = optimize_phases(dw, N_tones, N_samples, N_iters, N_reps)
    np.savez_compressed(
        outfile,
        F0=F0,
        T=T,
        dw=dw,
        dt=t[1] - t[0],
        N_tones=N_tones,
        N_samples=N_samples,
        N_iters=N_iters,
        N_reps=N_reps,
        phases=phases,
        CF=CF,
        S=S,
    )
