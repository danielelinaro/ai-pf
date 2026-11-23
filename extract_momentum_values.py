# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:57:10 2024

@author: Daniele Linaro
"""

import re
import sys
import json
import numpy as np
from pathlib import Path

if __name__ == '__main__':

    progname = Path(sys.argv[0]).name
    outfile = Path('HEM.npz')
    force = False
    i = 1
    n_args = len(sys.argv)
    while i < n_args:
        arg = sys.argv[i]
        if arg in ('-h', '--help'):
            print(f'usage: {progname} [-f | --force] [-o | --output <output_file_name>] base_data_dir')
            sys.exit(0)
        elif arg in ('-o', '--output'):
            i += 1
            outfile = Path(sys.argv[i])
        elif arg in ('-f', '--force'):
            force = True
        elif arg.startswith('-'):
            print(f"{progname}: unknown option '{arg}'.")
            sys.exit(0)
        else:
            break
        i += 1

    if i == n_args:
        print('You must provide the path to a directory.')
        sys.exit(1)

    if outfile.exists() and not force:
        print(f'{progname}: {outfile}: file exists, use -f to overwrite.')
        sys.exit(1)

    base_data_dir = Path(sys.argv[i])
    if not base_data_dir.is_dir():
        print(f'{progname}: {base_data_dir}: no such directory.')
        sys.exit(1)

    data_dirs = sorted(
        filter(
            lambda D: D.is_dir() and re.match('^[0-9]+$', D.name) is not None,
            base_data_dir.glob('*'),
        ),
    )
    n_dirs = len(data_dirs)
    print(f'Found {n_dirs} directories containing data.')

    H = np.nan + np.zeros(n_dirs)
    E = np.nan + np.zeros(n_dirs)
    M = np.nan + np.zeros(n_dirs)
    fn = 50. # [Hz] nominal frequency

    for i, d in enumerate(data_dirs):
        try:
            data_file = next(iter(d.glob('*_AC.npz')))
        except:
            for subd in d.iterdir():
                if subd.is_dir():
                    data_file = next(iter(subd.glob('*_AC.npz')))
        data = np.load(data_file, allow_pickle=True)
        Snom = data['Ssm'].item()
        h = data['Hsm'].item()
        h = np.array([h[k] for k in Snom])               # [s]
        Snom = np.array([Snom[k] * 1e-3 for k in Snom])  # [GW]
        E[i] = h @ Snom
        H[i] = E[i] / Snom.sum()
        M[i] = 2 * E[i] / fn

    print('Inertia in range {:g} - {:g} s.'.format(H.min(), H.max()))
    print('Momentum in range {:g} - {:g} GWs2.'.format(M.min(), M.max()))
    print('Energy in range {:g} - {:g} GWs.'.format(E.min(), E.max()))

    np.savez_compressed(outfile, H=H, E=E, M=M)
