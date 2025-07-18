# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:42:05 2023

@author: Daniele Linaro
"""

import os
import re
import sys
import numpy as np
from tqdm import tqdm

progname = os.path.basename(sys.argv[0])

def usage(exit_code=None):
    print(f'usage: {progname} [-h | --help] [-m | --fmin <value>] [-M | --fmax <value>]')
    prefix = '       ' + ' ' * (len(progname)+1)
    print(prefix + '[-N | --n-steps <value>] [--save-mat] [--F0 <value>] [--use-numpy-inv]')
    print(prefix + '[-o | --outfile <value>] [-f | --force] [--tau <value>]')
    print(prefix + '[--ref-sm <name>] [--no-add-TF] <--P | --Q | --PQ>')
    print(prefix + '<--dP | --sigmaP value1<,value2,...>> <--dQ | --sigmaQ value1<,value2,...>>')
    print(prefix + '<-L | --loads load1<,load2,...> | filename>')
    print(prefix + '<-V | --vars-to-save var1<,var2,...> | filename> AC_data_file')
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
    from time import time as TIME
    tstart = TIME()

    # default values    
    fmin,fmax = -6., 2.
    steps_per_decade = 100
    force = False
    save_mat = False
    outdir, outfile = '', None
    load_names = None
    vars_to_save = None
    use_P_constraint, use_Q_constraint = False, False
    dP,dQ = [],[]
    sigmaP,sigmaQ = [],[]
    # time constant of the OU process
    tau = 20e-3
    F0 = 50.
    ref_SM_name = 'CODCTI0201GGR1____GEN_____'
    compute_additional_TFs = True
    use_numpy_inv = False

    i = 1
    N_args = len(sys.argv)
    while i < N_args:
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
            v = sys.argv[i]
            if os.path.isfile(v):
                if os.path.splitext(v)[1] == '.json':
                    load_names = json.load(open(v,'r'))['load_names']
                else:
                    with open(v,'r') as fid:
                        load_names = [l.strip() for l in fid]
            else:
                load_names = v.split(',')
        elif arg in ('-V', '--vars-to-save'):
            i += 1
            v = sys.argv[i]
            if os.path.isfile(v):
                if os.path.splitext(v)[1] == '.json':
                    vars_to_save = json.load(open(v,'r'))['var_names']
                else:
                    with open(v,'r') as fid:
                        vars_to_save = [l.strip() for l in fid]
            else:
                vars_to_save = v.split(',')
        elif arg == '--no-add-TF':
            compute_additional_TFs = False
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
            dQ = list(map(float, sys.argv[i].split(',')))
        elif arg == '--sigmaP':
            i += 1
            sigmaP = list(map(float, sys.argv[i].split(',')))
        elif arg == '--sigmaQ':
            i += 1
            sigmaQ = list(map(float, sys.argv[i].split(',')))
        elif arg == '--F0':
            i += 1
            F0 = float(sys.argv[i])
        elif arg == '--tau':
            i += 1
            tau = float(sys.argv[i])
        elif arg == '--ref-sm':
            i += 1
            ref_SM_name = sys.argv[i]
        elif arg in ('-o','--outfile'):
            i += 1
            outfile = sys.argv[i]
        elif arg == '--save-mat':
            save_mat = True
        elif arg in ('-f', '--force'):
            force = True
        elif arg == '--use-numpy-inv':
            use_numpy_inv = True
        elif arg[0] == '-':
            print(f'{progname}: unknown option `{arg}`.')
            sys.exit(1)
        else:
            break
        i += 1

    if i == N_args:
        print(f'{progname}: you must specify an input file')
        sys.exit(1)
    if i == N_args-1:
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

    if F0 <= 0:
        print(f'{progname}: F0 must be > 0.')
        sys.exit(1)

    if load_names is None:
        print(f'{progname}: you must specify the name of at least one load where the signal is injected.')
        sys.exit(1)

    if use_numpy_inv:
        from numpy.linalg import inv
    else:
        from scipy.linalg import inv

    import platform
    use_at_matmul = not 'arm64' in platform.platform()

    N_freq = int(fmax - fmin) * steps_per_decade + 1
    F = np.logspace(fmin, fmax, N_freq)    

    data = np.load(data_file, allow_pickle=True)
    SM_names = [n for n in data['gen_names']]
    if ref_SM_name is not None and ref_SM_name not in SM_names:
        print(f'{progname}: {ref_SM_name} is not among the available synchronous machines.')
    static_gen_names = [n for n in data['static_gen_names']] if 'static_gen_names' in data else None
    bus_names = [n for n in data['voltages'].item().keys()]
    H = np.array([data['H'].item()[name] for name in SM_names])
    S = np.array([data['S'].item()[name] for name in SM_names])
    PF = data['PF_without_slack'].item()
    N_SMs = len(SM_names)
    P,Q = np.zeros(N_SMs), np.zeros(N_SMs)
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
    Jgy_inv = inv(Jgy)
    Atmp = Jfx - (Jfy @ Jgy_inv @ Jgx if use_at_matmul else np.dot(np.dot(Jfy, Jgy_inv), Jgx))
    assert np.allclose(A, Atmp)

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
    dP, dQ, sigmaP, sigmaQ = all_dP, all_dQ, all_sigmaP, all_sigmaQ
    def fix_len(lst1, lst2):
        if len(lst1) == 1 and len(lst2) > 1:
            return lst1*len(lst2)
        return lst1
    dP = fix_len(dP, load_names)
    dQ = fix_len(dQ, load_names)
    sigmaP = fix_len(sigmaP, load_names)
    sigmaQ = fix_len(sigmaQ, load_names)

    PF_loads = PF['loads']
    idx = []
    mu, c, alpha = [], [], []
    full_element_names = list(vars_idx.keys())
    element_names = list(map(lambda s: s.split('.')[0].split('-')[-1], full_element_names))
    bus_equiv_terms = data['bus_equiv_terms'].item()
    for i,load_name in enumerate(load_names):
        found = False
        bus_name = load_buses[load_name]
        if bus_name in element_names:
            full_bus_name = full_element_names[element_names.index(bus_name)]
            found = True
            flag = ''
        else:
            for equiv_term in bus_equiv_terms[bus_name]:
                if equiv_term in element_names:
                    full_bus_name = full_element_names[element_names.index(equiv_term)]
                    found = True
                    flag = '*'
                    break
        if not found:
            print(f"Variable index for '{bus_name}' not found.")
            continue
        print(f'[{i+1:2d}] {bus_name} -> {full_bus_name} {flag}')
        keys = []
        if use_P_constraint:
            # real part of voltage
            assert len(vars_idx[full_bus_name]['ur']) == 1
            idx.append(vars_idx[full_bus_name]['ur'][0])
            keys.append('P')
        if use_Q_constraint:
            # imaginary part of voltage
            assert len(vars_idx[full_bus_name]['ui']) == 1
            idx.append(vars_idx[full_bus_name]['ui'][0])
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
            mu.append(mean)
            c.append(stddev*np.sqrt(2/tau))
            alpha.append(1/tau)

    idx = np.array(idx) - N_state_vars
    mu, c, alpha = np.array(mu), np.array(c), np.array(alpha)
    B = - (Jfy @ Jgy_inv if use_at_matmul else np.dot(Jfy, Jgy_inv))
    C = - (Jgy_inv @ Jgx if use_at_matmul else np.dot(Jgy_inv, Jgx))
    
    N_inputs = c.size
    I = np.eye(N_state_vars)
    M = np.zeros((N_freq, N_state_vars, N_state_vars), dtype=complex)
    # the transfer functions are complex numbers
    TF  = np.zeros((N_freq, N_inputs, N_state_vars + N_algebraic_vars), dtype=complex)
    # the absolute value of the spectra of the outputs are real numbers:
    # we will take the abs at the end of the function
    OUT = np.zeros_like(TF)

    for i in tqdm(range(N_freq), ascii=True, ncols=70):
        M[i, :, :] = inv(-A + 1j * 2 * np.pi * F[i] * I)
        MxB = M[i, :, :] @ B if use_at_matmul else np.dot(M[i, :, :], B)
        PSD = np.sqrt((c / alpha)**2 / (1 + (2 * np.pi * F[i] / alpha)**2))
        for j, psd in enumerate(PSD):
            v = np.zeros(N_algebraic_vars, dtype=float)
            v[idx[j]] = 1
            tmp = MxB @ v if use_at_matmul else np.dot(MxB, v)
            TF[i, j, :N_state_vars] = tmp
            TF[i, j, N_state_vars:] = C @ tmp - Jgy_inv @ v if use_at_matmul else np.dot(C, tmp) - np.dot(Jgy_inv, v)
            v[idx[j]] = psd
            tmp = MxB @ v
            OUT[i, j, :N_state_vars] = tmp
            OUT[i, j, N_state_vars:] = C @ tmp - Jgy_inv @ v if use_at_matmul else np.dot(C, tmp) - np.dot(Jgy_inv, v)
    TF[TF == 0] = 1e-20 * (1+1j)
    OUT[OUT == 0] = 1e-20 * (1+1j)
    var_names, idx = [], []
    for k1,D in vars_idx.items():
        for k2,V in D.items():
            k = k1 + '.' + k2
            for v in V:
                var_names.append(k)
                idx.append(v)
    var_names = [var_names[i] for i in np.argsort(idx)]

    if compute_additional_TFs:
        if ref_SM_name is None:
            print('Please select the synchronous machine to be used as a reference:')
            for i,SM_name in enumerate(SM_names):
                print('[{:2d}] {}'.format(i+1, SM_name))
            while True:
                try:
                    idx = int(input(f'Enter a number between 1 and {N_SMs}: '))
                except ValueError:
                    continue
                if idx > 0 and idx <= N_SMs:
                    break
            ref_SM_name = SM_names[idx-1]
        print(f'Will use "{ref_SM_name}" as reference.')
        full_var_name = full_element_names[element_names.index(ref_SM_name)] + '.speed'
        ref_SM_idx = var_names.index(full_var_name)
        N_buses = len(bus_names)
        TF2  = np.zeros(( TF.shape[0],  TF.shape[1], N_buses), dtype=complex)
        OUT2 = np.zeros((OUT.shape[0], OUT.shape[1], N_buses), dtype=complex)

        def do_calc(X, coeffs, F, F0, ref):
            ret = np.zeros(X.shape[:2], dtype=complex)
            N_TF = X.shape[1]
            for j in range(N_TF):
                ret[:, j] = coeffs[0] * X[:, j, 0] + coeffs[1] * X[:, j, 1]
                ret[:, j] *= 1j * 2 * np.pi * F # Δω = jωΔθ
                ret[:, j] /= 2 * np.pi * F0 # !!! scaling factor !!!
                ret[:, j] += ref[:, j]
            return ret

        for i in tqdm(range(N_buses), ascii=True, ncols=70):
            # name = bus_names[i]
            # idx = var_names.index(name+'.ur'), var_names.index(name+'.ui')
            ### FIX THIS IN THE POWER FLOW LABELS
            name = bus_names[i].split('-')[-1].split('.')[0]
            if name in PF['buses']:
                idx = np.array([var_names.index(bus_names[i] + '.ur'),
                                var_names.index(bus_names[i] + '.ui')])
                ur, ui = PF['buses'][name]['ur'], PF['buses'][name]['ui']
                if ur != 0:
                    coeffs = -ui / ur**2 / (1 + (ui / ur)**2), 1 / (ur * (1 + (ui / ur)**2))
                    ##### is it TF or OUT in the following line?!? #####
                    TF2[:,:,i]  = do_calc( TF[:,:,idx], coeffs, F, F0,  TF[:,:,ref_SM_idx])
                    OUT2[:,:,i] = do_calc(OUT[:,:,idx], coeffs, F, F0, OUT[:,:,ref_SM_idx])

        var_names += [name + '.fe' for name in bus_names]
        TF  = np.concatenate((TF, TF2), axis=-1)
        OUT = np.concatenate((OUT, OUT2), axis=-1)
        assert(len(var_names) == TF.shape[2])

    if vars_to_save is not None:
        idx = []
        for var_to_save in vars_to_save:
            idx.append(np.where([re.search(var_to_save, var_name) is not None for var_name in var_names])[0])
        idx = np.sort(np.concatenate(idx))
        print(f'Will save only {len(idx)} out of {len(var_names)} variables.')
        var_names = [var_names[i] for i in idx]
        TF = TF[:,:,idx]
        OUT = OUT[:,:,idx]

    Htot = data['inertia']
    Etot = data['energy']
    Mtot = data['momentum']
    out = {'A': A, 'F': F, 'TF': TF, 'OUT': OUT, 'var_names': var_names, 'SM_names': SM_names,
           'static_gen_names': static_gen_names, 'bus_names': bus_names, 'load_names': load_names,
           'Htot': Htot, 'Etot': Etot, 'Mtot': Mtot, 'H': H, 'S': S, 'P': P, 'Q': Q,
           'PF': data['PF_without_slack'], 'bus_equiv_terms': data['bus_equiv_terms'],
           'mu': mu, 'c': c, 'alpha': alpha, 'use_P_constraint': use_P_constraint,
           'use_Q_constraint': use_Q_constraint, 'dP': dP, 'dQ': dQ, 'sigmaP': sigmaP,
           'sigmaQ': sigmaQ, 'ref_SM_name': ref_SM_name, 'data_file': data_file,
           'with_additional_TFs': compute_additional_TFs}
    np.savez_compressed(os.path.join(outdir, outfile), **out)

    if save_mat:
        save_to_mat(os.path.join(outdir, os.path.splitext(outfile)[0]+'.mat'), data_dict=out)
        
    tend = TIME()
    print('Elapsed time: {:.3f} sec.'.format(tend-tstart))
