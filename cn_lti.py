"""
Crank-Nicolson integration of a linear time-invariant system

    x'(t) = A x(t) + B u(t),
    y (t) = C x(t) + D u(t),
    x(t0) = x0

Scheme (trapezoidal rule, 2nd order, A-stable):

    x_{k+1} - x_k = (h/2) [ A x_k + B u_k + A x_{k+1} + B u_{k+1} ]

    =>  x_{k+1} = ((I - h/2 A) ^ -1) (I + h/2 A) x_k +
                  h/2 ((I - h/2 A) ^ -1) B (u_k + u_{k+1})

Because the system is time invariant and the step h is constant, the matrix
(I - h/2 A) is the same at every step, so it is inverted once and reused.
"""

import numpy as np
import scipy.linalg as LA


__all__ = ["crank_nicolson_lti"]


def crank_nicolson_lti(A, B, u, x0, t0, tf, n_steps, C=None, D=None):
    """
    Integrate x' = A x + B u(t) from t0 to tf with n_steps Crank-Nicolson steps.
    If the matrixes C and D are passed, also compute y = C x + D u(t).

    Parameters
    ----------
    A : (n, n) array_like
        System matrix.
    B : (n, m) array_like
        Input matrix.
    u : callable or array_like
        Input signal. Either
          * a callable u(t) returning a scalar or an (m,) array, or
          * an array of shape (n_steps + 1, m) (or (n_steps + 1,) when m == 1)
            holding the input sampled at the grid points t_k = t0 + k*h.
    x0 : (n,) array_like
        Initial state.
    t0, tf : float
        Start and end times.
    n_steps : int
        Number of time steps (the grid has n_steps + 1 points).
    C : (l, n) array_like
    D : (l, m) array_like

    Returns
    -------
    t : (n_steps + 1,) ndarray
        Time grid.
    xy : (n_steps + 1, n + l) if C and D are present, (n_steps + 1, n) otherwise ndarray
        State (and optionally output) trajectory; xy[k] approximates xy(t[k]).
    """

    if (C is not None and D is None) or (C is None and D is not None):
        raise ValueError("If present, both C and D must be passed")

    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square")

    B = np.asarray(B, dtype=float)
    if B.ndim == 1:
        B = B.reshape(n, 1)
    if B.shape[0] != n:
        raise ValueError("B must have the same number of rows as A")
    m = B.shape[1]

    if C is not None:
        C = np.asarray(C, dtype=float)
        if C.ndim == 1:
            C = C.reshape(1, n)
        if C.shape[1] != n:
            raise ValueError("C must have the same number of columns as A")
        l = C.shape[0]

        D = np.asarray(D, dtype=float)
        if D.ndim == 1:
            D = D.reshape(l, 1)
        if D.shape[0] != l:
            raise ValueError("D must have the same number of rows as C")
    else:
        l = 0

    x0 = np.asarray(x0, dtype=float).reshape(-1)
    if x0.size != n:
        raise ValueError("x0 must have length n")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    h = (tf - t0) / n_steps
    t = t0 + h * np.arange(n_steps + 1)

    # Evaluate / validate the input on the grid.
    if callable(u):
        U = np.array([np.atleast_1d(u(tk)) for tk in t], dtype=float)
    else:
        U = np.asarray(u, dtype=float)
        if U.ndim == 1:
            U = U.reshape(-1, 1)
    if U.shape != (n_steps + 1, m):
        raise ValueError(
            f"u must have shape ({n_steps + 1}, {m}), got {U.shape}"
        )

    # Pre-compute the constant matrices; invert the left-hand side once.
    I = np.eye(n)
    Minv = LA.inv(I - h / 2 * A)
    P = Minv @ (I + h / 2 * A)
    Q = Minv @ B * h / 2
    S = U[:-1] + U[1:]
    xy = np.empty((n_steps + 1, n + l))
    xy[0, :n] = x0
    for k in range(n_steps):
        np.dot(P, xy[k, :n], out=xy[k + 1, :n])
        xy[k + 1, :n] += Q @ S[k]
    # Compute the algebraic part of the system
    if l > 0:
        xy[:, n:] = xy[:, :n] @ C.T + U @ D.T
    return t, xy


# ----------------------------------------------------------------------------
# Self-test: damped oscillator driven by a sinusoid, compared with the exact
# solution computed through an augmented matrix exponential (u treated as the
# output of an exosystem), and a convergence-order check.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    from scipy.linalg import expm

    # y'' + 2*zeta*w*y' + w^2*y = u,  state = [position, velocity]
    w, zeta = 2.0, 0.1
    A = np.array([[0.0, 1.0],
                  [-w**2, -2 * zeta * w]])
    B = np.array([[0.0],
                  [1.0]])
    y0 = np.array([1.0, 0.0])
    omega = 1.5
    u = lambda t: np.sin(omega * t)

    # Exact solution: augment with exosystem z = [sin(wt), cos(wt)], which is
    # itself an autonomous linear system z' = E z. Then [y; z]' = M [y; z].
    E = np.array([[0.0, omega],
                  [-omega, 0.0]])
    M = np.block([[A, B @ np.array([[1.0, 0.0]])],
                  [np.zeros((2, 2)), E]])
    z0 = np.array([0.0, 1.0])                # sin(0), cos(0)
    t0, tf = 0.0, 10.0
    exact = (expm(M * (tf - t0)) @ np.concatenate([y0, z0]))[:2]

    print(f"{'steps':>8} {'error at tf':>14} {'order':>8}")
    prev_err = None
    for N in (50, 100, 200, 400, 800):
        t, y = crank_nicolson_lti(A, B, u, y0, t0, tf, N)
        err = np.linalg.norm(y[-1] - exact)
        order = "" if prev_err is None else f"{np.log2(prev_err / err):.2f}"
        print(f"{N:>8} {err:>14.3e} {order:>8}")
        prev_err = err
