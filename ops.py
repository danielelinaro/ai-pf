
import os
import ctypes
import numpy as np

__all__ = ['compute_u_c', 'compute_fe_c', 'compute_fe_py']

lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), 'libops.so'))
lib.compute_u.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_size_t,
    ctypes.c_double,
    ctypes.c_double,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
]

lib.compute_fe.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_size_t,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
]


_copy_if_non_contiguous = lambda x: x if x.flags['C_CONTIGUOUS'] else x.copy('C')


def compute_u_c(ur, ui, ur_PF, ui_PF):
    n = ur.size
    u = np.empty(n, dtype=np.float64)
    assert u.flags['C_CONTIGUOUS']
    lib.compute_u(
        _copy_if_non_contiguous(ur),
        _copy_if_non_contiguous(ui),
        n,
        ur_PF,
        ui_PF,
        u,
    )
    return u


def compute_fe_c(t, ur, ui, omega_ref, ur_PF, ui_PF, f0):
    n = ur.size
    fe = np.empty(n)
    assert fe.flags['C_CONTIGUOUS']
    lib.compute_fe(
        _copy_if_non_contiguous(t),
        _copy_if_non_contiguous(ur),
        _copy_if_non_contiguous(ui),
        _copy_if_non_contiguous(omega_ref),
        n,
        ur_PF,
        ui_PF,
        f0,
        fe,
    )
    return fe


def compute_fe_py(t, ur, ui, f0, omega_ref):
    delta = np.atan2(ui, ur)
    omega = np.diff(delta) / np.diff(t)
    omega = np.concatenate([omega[:1], omega])
    omega /= 2 * np.pi * f0
    return omega + omega_ref


if __name__ == '__main__':
    import os
    import sys
    from time import time

    fname = 'ALHCDI_fe.npz'
    try:
        data = np.load(fname)
    except:
        print('{}: {}: no such file.'.format(os.path.basename(sys.argv[0]), fname))
        sys.exit(1)

    f0 = data['f0']
    ur_PF, ui_PF = data['ur_PF'], data['ui_PF']
    t = data['time']
    ur, ui = data['ur'], data['ui']
    omega_ref = data['omega_ref']
    print('# of samples:', t.size)

    ### Python
    t0 = time()
    fe_py = compute_fe_py(t, ur + ur_PF, ui + ui_PF, f0, omega_ref)
    print('Python elapsed time: {} sec'.format(time() - t0))

    ### C
    t0 = time()
    fe_c = compute_fe_c(t, ur, ui, omega_ref, ur_PF, ui_PF, f0)
    print('C elapsed time: {} sec'.format(time() - t0))
    
    assert np.allclose(fe_py, fe_c)

