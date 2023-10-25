# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:42:05 2023

@author: Daniele Linaro
"""

import os
import sys
import glob
import numpy as np
# from scipy.io import savemat
# import matplotlib.pyplot as plt
# import matplotlib
# import seaborn as sns
from tqdm import tqdm

progname = os.path.basename(sys.argv[0])

def usage(exit_code=None):
    print(f'usage: {progname} [-h | --help] [-m | --fmin <value>] [-M | --fmax <value>]')
    print( '       ' + ' ' * len(progname) + ' [-N | --n-steps <value>] [--save-mat]')
    print( '       ' + ' ' * len(progname) + ' [-o <value>] [--force] <files> <directory>')
    if exit_code is not None:
        sys.exit(exit_code)


if __name__ == '__main__':

    # default values    
    fmin,fmax = -6., 2.
    steps_per_decade = 100
    force = False
    save_mat = False
    outfile = None

    i = 1
    n_args = len(sys.argv)
    while i < n_args:
        arg = sys.argv[i]
        if arg in ('-h', '--help'):
            usage(0)
        elif arg in ('-m', '--fmin'):
            i += 1
            fmin = float(sys.argv[i])
        elif arg in ('-M', '--fmax'):
            i += 1
            fmax = float(sys.argv[i])
        elif arg in ('-N', '--n-steps'):
            i += 1
            steps_per_decade = int(sys.argv[i])
        elif arg == '-o':
            i += 1
            outfile= sys.argv[i]
        elif arg == '--save-mat':
            save_mat = True
        elif arg in ('-f', '--force'):
            force = True
        elif arg[0] == '-':
            print(f'{progname}: unknown option `{arg}`')
            sys.exit(1)
        else:
            break
        i += 1

    data_files = []
    for arg in sys.argv[i:]:
        if arg[0] == '-':
            print('Options are not allowed after file or directory names.')
            sys.exit(1)
        elif os.path.isfile(arg):
            data_files.append(arg)
        elif os.path.isdir(arg):
            for f in sorted(glob.glob(arg + os.path.sep + '*.npz')):
                data_files.append(f)
        else:
            print(f'{progname}: {arg}: no such file or directory.')
            sys.exit(1)

    if len(data_files) < 2:
        print('You must specifiy at least two files or a directory.')
        sys.exit(1)

    if outfile is None:
        outfile = 'TF_{}_{}_{}.npz'.format(fmin, fmax, steps_per_decade)

    if os.path.isfile(outfile) and not force:
        print(f'{progname}: {outfile}: file exists, use -f to overwrite')
        sys.exit(1)

    if fmin >= fmax:
        print('fmin must be < fmax')
        sys.exit(1)
    if steps_per_decade <= 0:
        print('number of steps per decade must be > 0')
        sys.exit(1)

    Nf = int(fmax - fmin) * steps_per_decade + 1
    F = np.logspace(fmin, fmax, Nf)    

    data = [np.load(f, allow_pickle=True) for f in data_files]
    SM_names = [d['gen_names'] for d in data]
    H = np.array([np.array([d['H'].item()[name] for name in names]) for
                  d,names in zip(data, SM_names)])
    S = np.array([np.array([d['S'].item()[name] for name in names]) for
                  d,names in zip(data,SM_names)])
    P,Q = [],[]
    for i,names in enumerate(SM_names):
        n_SMs = len(names)
        p,q = np.zeros(n_SMs), np.zeros(n_SMs)
        PF = data[i]['PF_without_slack'].item()
        for j,sm in enumerate(names):
            if sm in PF['SMs']:
                key = sm
            else:
                key = sm + '____GEN_____'
            p[j] = PF['SMs'][key]['P']
            q[j] = PF['SMs'][key]['Q']
        P.append(p)
        Q.append(q)

    A = np.array([d['A'] for d in data])
    N,Nv,_ = A.shape

    I = np.eye(Nv)
    M = np.zeros((Nf, Nv, Nv), dtype=complex)    
    # not all conditions have necessarily the same number of active synchronous generators
    TF = [np.zeros((Nf, Nv), dtype=complex) for _ in range(N)]
    for i in range(N):
        omega_col_idx = data[i]['omega_col_idx']
        b = np.zeros(Nv)
        b[omega_col_idx] = H[0] / H[i]
        for j in tqdm(range(Nf), ascii=True, ncols=70):
            M[j,:,:] = np.linalg.inv(-A[i] + 1j*2*np.pi*F[j]*I)
            TF[i][j,:] = M[j,:,:] @ b
        TF[i] = TF[i][:,omega_col_idx]
    mag = [20 * np.log10(np.abs(tf)) for tf in TF]
    phase = [np.angle(tf) for tf in TF]
    
    out = {'Htot': np.array([d['inertia'] for d in data]),
           'Etot': np.array([d['energy'] for d in data]),
           'Mtot': np.array([d['momentum'] for d in data]),
           'F': F, 'TF': TF, 'mag': mag, 'phase': phase,
           'SM_names': SM_names, 'H': H, 'S': S, 'P': P, 'Q': Q}
    np.savez_compressed(outfile, **out)

    if save_mat:
        from scipy.io import savemat
        mout = {}
        for k,v in out.items():
            if 'tot' in k:
                mout[k[0]] = v
            elif isinstance(v,list):
                for i in range(N):
                    mout[f'{k}_{i+1}'] = v[i]
            else:
                mout[k] = v
        savemat(os.path.splitext(outfile)[0] + '.mat', mout)