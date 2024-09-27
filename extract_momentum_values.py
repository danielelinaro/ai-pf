# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:57:10 2024

@author: Daniele Linaro
"""

import os
import re
import sys
import json
import glob
import numpy as np

if __name__ == '__main__':
    
    base_data_dir = '/dati2tb/daniele/Research/ai-pf/data/Sardinia/SM_configs_from_data'
    data_dirs = sorted(filter(lambda s: os.path.isdir(s) and \
                              re.match('^[0-9]+', os.path.basename(s)),
                              glob.glob(os.path.join(base_data_dir, '*'))))
    n_dirs = len(data_dirs)
    print(f'Found {n_dirs} directories containing data.')

    H = -1 + np.zeros(n_dirs)
    E = -1 + np.zeros(n_dirs)
    M = -1 + np.zeros(n_dirs)
    fname = 'V2020_Rete_Sardegna_2021_06_03cr_AC_TF_-6.0_2.0_100.npz'
    
    for i,d in enumerate(data_dirs):
        data_file = os.path.join(d, fname)
        if not os.path.isfile(data_file):
            print(f'Directory {os.path.basename(d)} does not contain file {fname}.')
            continue
        data = np.load(data_file)
        S = data['S'] * 1e6 # [MW]
        num = data['H'] @ S
        den = S.sum()
        H[i] = num/den       # [s]
        E[i] = num*1e-9      # [GW s]
        M[i] = 2*num*1e-9/50 # [GW s2]

    np.savez_compressed('HEM.npz', H=H, E=E, M=M)
        
        
