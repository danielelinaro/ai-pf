
import math
import numpy as np

__all__ = [
    "multitone",
    "compute_crest_factor",
    "compute_amplitude_distribution",
    ]


def _rudin_signs(N, x0=[1, 1]):
    assert len(x0) % 2 == 0, 'Length of x0 must be even'
    x = [_ for _ in x0]
    while len(x) < N:
        x += [y if i < len(x) // 2 else -y for i, y in enumerate(x)]
    return x

_rudin_signs_list = _rudin_signs(1000)


def _shapiro_rudin_phase(k, N):
    assert k <= N
    if k <= len(_rudin_signs_list):
        return 0 if _rudin_signs_list[k - 1] == 1 else math.pi
    delta = [0 if r == 1 else math.pi for r in _rudin_signs(N)]        
    return delta[k - 1]


def _newman_phase(k, N):
    assert k <= N
    return math.pi * ((k - 1) ** 2) / N


def multitone(N, fs, tend, method='newman', N0=0):
    if method.lower() in ('newman', 'n'):
        delta_fun = _newman_phase
    elif method.lower() in ('shapiro-rudin', 'sr'):
        delta_fun = _shapiro_rudin_phase
    else:
        raise Exception("Accepted methods are 'Newman' or 'Shapiro-Rudin'")
    dt = 1 / fs
    t = np.r_[0 : tend + dt/2 : dt]
    u = np.zeros_like(t)
    for k in range(1, N + 1):
        delta = delta_fun(k, N)
        u += np.cos((k + N0) * t + delta)
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
