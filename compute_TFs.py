# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:42:05 2023

@author: Daniele Linaro
"""

import os
import re
import sys
import numpy as np
# from scipy.io import savemat
# import matplotlib.pyplot as plt
# import matplotlib
# import seaborn as sns
from tqdm import tqdm

progname = os.path.basename(sys.argv[0])

def usage(exit_code=None):
    print(f'usage: {progname} [-h | --help] [-m | --fmin <value>] [-M | --fmax <value>]')
    prefix = '       ' + ' ' * (len(progname)+1)
    print(prefix + '[-N | --n-steps <value>] [--save-mat]')
    print(prefix + '[-o | --outfile <value>] [-f | --force] [--tau <value>] ')
    print(prefix + '<--P | --Q | --PQ> <--dP | --sigmaP value1<,value2,...>>')
    print(prefix + '<--dQ | --sigmaQ value1<,value2,...>> <-L | --loads load1<,load2,...>> file')
    if exit_code is not None:
        sys.exit(exit_code)


def save_to_mat(mat_file, data_dict=None, npz_data_file=None):
    from scipy.io import savemat
    if data_dict is not None:
        out = data_dict
    else:
        data = np.load(npz_data_file, allow_pickle=True)
        out = {key: data[key] for key in data.files}
    savemat(mat_file, out, long_field_names=True)


if __name__ == '__main__':

    # default values    
    fmin,fmax = -6., 2.
    steps_per_decade = 100
    force = False
    save_mat = False
    outdir, outfile = '', None
    load_names = None
    use_P_constraint, use_Q_constraint = False, False
    dP,dQ = [],[]
    sigmaP,sigmaQ = [],[]
    # time constant of the OU process
    tau = 20e-3

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
        elif arg in ('-L', '--loads'):
            i += 1
            load_names = sys.argv[i].split(',')
        elif arg == '--P':
            use_P_constraint = True
        elif arg == '--Q':
            use_Q_constraint = True
        elif arg == '--PQ':
            use_P_constraint = True
            use_Q_constraint = True
        elif arg == '--dP':
            i += 1
            dP = list(map(float, sys.argv[i].split(',')))
        elif arg == '--dQ':
            i += 1
            dP = list(map(float, sys.argv[i].split(',')))
        elif arg == '--sigmaP':
            i += 1
            sigmaP = list(map(float, sys.argv[i].split(',')))
        elif arg == '--sigmaQ':
            i += 1
            sigmaQ = list(map(float, sys.argv[i].split(',')))
        elif arg == '--tau':
            i += 1
            tau = float(sys.argv[i])
        elif arg in ('-o','--outfile'):
            i += 1
            outfile = sys.argv[i]
        elif arg == '--save-mat':
            save_mat = True
        elif arg in ('-f', '--force'):
            force = True
        elif arg[0] == '-':
            print(f'{progname}: unknown option `{arg}`.')
            sys.exit(1)
        else:
            break
        i += 1

    if i == n_args:
        print(f'{progname}: you must specify an input file')
        sys.exit(1)
    if i == n_args-1:
        data_file = sys.argv[i]
    else:
        print(f'{progname}: arguments after project name are not allowed')
        sys.exit(1)
        
    if not os.path.isfile(data_file):
        print(f'{progname}: {data_file}: no such file.')
        sys.exit(1)
    
    if not use_P_constraint and not use_Q_constraint:
        if save_mat:
            save_to_mat(os.path.join(outdir, os.path.splitext(outfile
                                                              if outfile is not None
                                                              else data_file)[0] + '.mat'),
                        npz_data_file=data_file)
            sys.exit(0)
        print(f'{progname}: at least one of --P and --Q must be specified.')
        sys.exit(1)
    
    if use_P_constraint:
        if len(dP) == 0 and len(sigmaP) == 0:
            print(f'{progname}: either --dP or --sigmaP must be specified with --P.')
            sys.exit(1)
        elif len(dP) > 0 and len(sigmaP) > 0:
            print(f'{progname}: only one of --dP and --sigmaP can be specified.')
            sys.exit(1)
    if use_Q_constraint:
        if len(dQ) == 0 and len(sigmaQ) == 0:
            print(f'{progname}: either --dQ or --sigmaQ must be specified with --Q.')
            sys.exit(1)
        elif len(dQ) > 0 and len(sigmaQ) > 0:
            print(f'{progname}: only one of --dP and --sigmaP can be specified.')
            sys.exit(1)

    if outfile is None:
        outdir = os.path.dirname(data_file)
        if outdir == '':
            outdir = '.'
        outfile = os.path.splitext(os.path.basename(data_file))[0] + \
            '_TF_{}_{}_{}'.format(fmin, fmax, steps_per_decade) + '.npz'
    if os.path.isfile(os.path.join(outdir, outfile)) and not force:
        print(f'{progname}: {os.path.join(outdir, outfile)}: file exists, use -f to overwrite.')
        sys.exit(1)

    if fmin >= fmax:
        print(f'{progname}: fmin must be < fmax.')
        sys.exit(1)
    if steps_per_decade <= 0:
        print(f'{progname}: number of steps per decade must be > 0.')
        sys.exit(1)

    if load_names is None:
        print(f'{progname}: you must specify the name of at least one load where the signal is injected.')
        sys.exit(1)

    N_freq = int(fmax - fmin) * steps_per_decade + 1
    F = np.logspace(fmin, fmax, N_freq)    

    data = np.load(data_file, allow_pickle=True)
    SM_names = [n for n in data['gen_names']]
    bus_names = [n for n in data['voltages'].item().keys()]
    H = np.array([data['H'].item()[name] for name in SM_names])
    S = np.array([data['S'].item()[name] for name in SM_names])
    PF = data['PF_without_slack'].item()
    n_SMs = len(SM_names)
    P,Q = np.zeros(n_SMs), np.zeros(n_SMs)
    for i,name in enumerate(SM_names):
        if name in PF['SMs']:
            key = name
        else:
            key = name + '____GEN_____'
        P[i] = PF['SMs'][key]['P']
        Q[i] = PF['SMs'][key]['Q']

    J,A = data['J'], data['A']
    vars_idx = data['vars_idx'].item()
    state_vars = data['state_vars'].item()
    N_vars = J.shape[0]
    N_state_vars = np.sum([len(v) for v in state_vars.values()])
    N_algebraic_vars = N_vars - N_state_vars
    Jfx = J[:N_state_vars, :N_state_vars]
    Jfy = J[:N_state_vars, N_state_vars:]
    Jgx = J[N_state_vars:, :N_state_vars]
    Jgy = J[N_state_vars:, N_state_vars:]
    Jgy_inv = np.linalg.inv(Jgy)
    Atmp = Jfx - Jfy @ Jgy_inv @ Jgx
    assert np.all(np.abs(A-Atmp) < 1e-8)

    load_buses = data['load_buses'].item()
    all_load_names = []
    all_dP,all_dQ,all_sigmaP,all_sigmaQ = [],[],[],[]
    def try_append(dst, src, i):
        try: dst.append(src[i])
        except: pass
    for i,load_name in enumerate(load_names):
        if '*' in load_name:
            for load in load_buses.keys():
                if re.match(load_name, load):
                    all_load_names.append(load)
                    try_append(all_dP, dP, i)
                    try_append(all_dQ, dQ, i)
                    try_append(all_sigmaP, sigmaP, i)
                    try_append(all_sigmaQ, sigmaQ, i)
        elif load_name not in load_buses:
            print(f'{progname}: cannot find load `{load_name}`.')
            sys.exit(0)
        else:
            all_load_names.append(load_name)
            try_append(all_dP, dP, i)
            try_append(all_dQ, dQ, i)
            try_append(all_sigmaP, sigmaP, i)
            try_append(all_sigmaQ, sigmaQ, i)
    load_names = all_load_names
    dP,dQ,sigmaP,sigmaQ = all_dP,all_dQ,all_sigmaP,all_sigmaQ
    def fix_len(lst1, lst2):
        if len(lst1) == 1 and len(lst2) > 1:
            return lst1*len(lst2)
        return lst1
    dP = fix_len(dP, load_names)
    dQ = fix_len(dQ, load_names)
    sigmaP = fix_len(sigmaP, load_names)
    sigmaQ = fix_len(sigmaQ, load_names)

    PF_loads = data['PF_without_slack'].item()['loads']
    idx = []
    c,alpha = [], []
    for i,load_name in enumerate(load_names):
        keys = []
        bus_name = load_buses[load_name]
        if bus_name not in vars_idx:
            # we have to look through the equivalent terms of bus_name
            bus_equiv_terms = data['bus_equiv_terms'].item()
            for equiv_term_name in bus_equiv_terms[bus_name]:
                if equiv_term_name in vars_idx:
                    print('Load {} is connected to bus {}, which is not among the '.
                          format(load_name, bus_name) + 
                          'buses whose ur and ui variables are in the Jacobian, but {} is.'.
                          format(equiv_term_name))
                    bus_name = equiv_term_name
                    break
        if use_P_constraint:
            # real part of voltage
            idx.append(vars_idx[bus_name]['ur'])
            keys.append('P')
        if use_Q_constraint:
            # imaginary part of voltage
            idx.append(vars_idx[bus_name]['ui'])
            keys.append('Q')
        for key in keys:
            mean = PF_loads[load_name][key]
            if key == 'P':
                if len(dP) > 0:
                    stddev = dP[i] * abs(mean)
                else:
                    stddev = sigmaP[i]
            else:
                if len(dQ) > 0:
                    stddev = dQ[i] * abs(mean)
                else:
                    stddev = sigmaQ[i]
            c.append(stddev*np.sqrt(2/tau))
            alpha.append(1/tau)

    idx = np.array(idx) - N_state_vars
    c,alpha = np.array(c), np.array(alpha)
    B = -Jfy @ Jgy_inv
    C = -Jgy_inv @ Jgx
    
    N_inputs = c.size
    I = np.eye(N_state_vars)
    M = np.zeros((N_freq, N_state_vars, N_state_vars), dtype=complex)
    TF = np.zeros((N_inputs, N_freq, N_state_vars+N_algebraic_vars), dtype=complex)

    for i in tqdm(range(N_freq), ascii=True, ncols=70):
        M[i,:,:] = np.linalg.inv(-A + 1j*2*np.pi*F[i]*I)
        MxB = M[i,:,:] @ B
        PSD = np.sqrt((c/alpha)**2 / (1 + (2*np.pi*F[i]/alpha)**2))
        for j,psd in enumerate(PSD):
            v = np.zeros(N_algebraic_vars, dtype=float)
            v[idx[j]] = psd
            tmp = MxB @ v
            TF[j,i,:N_state_vars] = tmp
            TF[j,i,N_state_vars:] = (C @ tmp - Jgy_inv @ v)
    TF[TF==0] = 1e-20 * (1+1j)
    vars_idx = data['vars_idx'].item()
    var_names,idx = [],[]
    for k1,D in vars_idx.items():
        for k2,v in D.items():
            var_names.append(k1 + '.' + k2)
            idx.append(v)
    var_names = [var_names[i] for i in np.argsort(idx)]
    
    Htot = data['inertia']
    Etot = data['energy']
    Mtot = data['momentum']
    out = {'A': A, 'F': F, 'TF': TF,
           'var_names': var_names, 'SM_names': SM_names, 'bus_names': bus_names,
           'Htot': Htot, 'Etot': Etot, 'Mtot': Mtot, 'H': H, 'S': S, 'P': P, 'Q': Q,
           'PF': data['PF_without_slack'], 'bus_equiv_terms': data['bus_equiv_terms']}
    np.savez_compressed(os.path.join(outdir, outfile), **out)

    if save_mat:
        save_to_mat(os.path.join(outdir, os.path.splitext(outfile)[0] + '.mat'),
                    data_dict=out)
