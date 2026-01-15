# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:42:05 2023

@author: Daniele Linaro
"""

import os
import re
import sys
import json
import numpy as np
from tqdm import tqdm

progname = os.path.basename(sys.argv[0])

def usage(exit_code=None):
    print(f'usage: {progname} [-h | --help] [-m | --fmin <value>] [-M | --fmax <value>]')
    prefix = '       ' + ' ' * (len(progname)+1)
    print(prefix + '[-N | --n-steps <value>] [--F0 <value>] [--use-numpy-inv]')
    print(prefix + '[-o | --outfile <value>] [-f | --force] [--tau <value>]')
    print(prefix + '[--ref-sm <name>] [--no-add-TF] <--dP | --sigmaP value1<,value2,...>>')
    print(prefix + '[--S-base <value>] [--load-exp <0, 1, 2 for const P, I, Z respectively>]')
    print(prefix + '<-L | --loads load1<,load2,...> | filename>')
    print(prefix + '<-I | --inputs filename>')
    print(prefix + '<-V | --vars-to-save var1<,var2,...> | filename>')
    print(prefix + '[--save-mat] [-v | --verbose] AC_data_file')
    if exit_code is not None:
        sys.exit(exit_code)


if __name__ == '__main__':
    from time import time as TIME
    tstart = TIME()

    # default values    
    fmin,fmax = -6., 2.
    steps_per_decade = 100
    force = False
    outdir, outfile = '', None
    input_loads = None
    input_keys = None
    vars_to_save = None
    dP = []
    sigmaP = []
    # time constant of the OU process
    tau = 20e-3
    F0 = 50.
    #ref_SM_name = 'CODCTI0201GGR1____GEN_____'
    ref_SM_name = None
    compute_additional_TFs = True
    use_numpy_inv = False
    save_mat = False
    verbose = False
    S_base = 1 # [MVA]
    load_exp = 0 # constant P: 0, constant I: 1, constant Z: 2

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
                    input_loads = json.load(open(v, 'r'))['input_loads']
                else:
                    with open(v, 'r') as fid:
                        input_loads = [l.strip() for l in fid]
            else:
                input_loads = v.split(',')
        elif arg in ('-I', '--inputs'):
            i += 1
            fname = sys.argv[i]
            if not os.path.isfile(fname):
                print(f'{fname}: no such file.')
                sys.exit(1)
            with open(fname, 'r') as fid:
                input_keys = []
                input_var_names = []
                for line in fid:
                    try:
                        device_name, device_type, var_name, block_name = line.strip().split(',')
                        input_keys.append('{}{}.{}'.format(block_name + '-' if len(block_name) else '', device_name, device_type))
                        input_var_names.append(var_name)
                    except:
                        pass
        elif arg in ('-V', '--vars-to-save'):
            i += 1
            v = sys.argv[i]
            if os.path.isfile(v):
                if os.path.splitext(v)[1] == '.json':
                    vars_to_save = json.load(open(v,'r'))['var_names']
                else:
                    with open(v,'r') as fid:
                        vars_to_save = [l.strip() for l in fid if l.strip()[0] != '#']
            else:
                vars_to_save = v.split(',')
        elif arg == '--no-add-TF':
            compute_additional_TFs = False
        elif arg == '--dP':
            i += 1
            dP = list(map(float, sys.argv[i].split(',')))
        elif arg == '--sigmaP':
            i += 1
            sigmaP = list(map(float, sys.argv[i].split(',')))
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
        elif arg in ('-f', '--force'):
            force = True
        elif arg in ('-v', '--verbose'):
            verbose = True
        elif arg == '--use-numpy-inv':
            use_numpy_inv = True
        elif arg == '--save-mat':
            try:
                from scipy.io import savemat
                save_mat = True
            except:
                raise Warning('scipy not available: will not save MAT file')
        elif arg == '--S-base':
            i += 1
            S_base = float(sys.argv[i])
        elif arg == '--load-exp':
            i += 1
            load_exp = int(sys.argv[i])
        elif arg[0] == '-':
            print(f'{progname}: unknown option `{arg}`.')
            sys.exit(1)
        else:
            break
        i += 1

    if i == N_args:
        print(f'{progname}: you must specify an input file')
        sys.exit(1)
    if i == N_args - 1:
        data_file = sys.argv[i]
    else:
        print(f'{progname}: arguments after project name are not allowed')
        sys.exit(1)

    if not os.path.isfile(data_file):
        print(f'{progname}: {data_file}: no such file.')
        sys.exit(1)
    
    if len(dP) == 0 and len(sigmaP) == 0:
        print(f'{progname}: either --dP or --sigmaP must be specified with --P.')
        sys.exit(1)
    elif len(dP) > 0 and len(sigmaP) > 0:
        print(f'{progname}: only one of --dP and --sigmaP can be specified.')
        sys.exit(1)

    if outfile is None:
        outdir = os.path.dirname(data_file)
        outfile = os.path.splitext(os.path.basename(data_file))[0] + \
            '_TF_{}_{}_{}'.format(fmin, fmax, steps_per_decade) + '.npz'
    else:
        outdir = os.path.dirname(outfile)
        outfile = os.path.basename(outfile)
        if outfile[-4:] != '.npz':
            outfile += '.npz'
    if outdir == '':
        outdir = '.'
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

    if input_loads is None and input_keys is None:
        print(f'{progname}: you must specify the name of at least one load or input where the signal is injected.')
        sys.exit(1)

    if S_base <= 0:
        print(f'{progname}: Sbase must be > 0.')
        sys.exit(1)

    if load_exp not in (0, 1, 2):
        print(f'{progname}: load exponent must be 0, 1, or 2 for constant P, I, and Z, respectively.')
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
    try:
        # current version of data file
        H_SM = np.array([data['Hsm'].item()[name] for name in SM_names])
        S_SM = np.array([data['Ssm'].item()[name] for name in SM_names])
    except:
        # old version of data file
        H_SM = np.array([data['H'].item()[name] for name in SM_names])
        S_SM = np.array([data['S'].item()[name] for name in SM_names])
    PF = data['PF_without_slack'].item()
    N_SMs = len(SM_names)
    P_SM = np.zeros(N_SMs)
    Q_SM = np.zeros(N_SMs)
    for i, name in enumerate(SM_names):
        if name in PF['SMs']:
            key = name
        else:
            key = name + '____GEN_____'
        P_SM[i] = PF['SMs'][key]['P']
        Q_SM[i] = PF['SMs'][key]['Q']

    J, Amat = data['J'], data['A']
    vars_idx = data['vars_idx'].item()
    state_vars = data['state_vars'].item()
    N_vars = J.shape[0]
    N_state_vars = np.sum([len(v) for v in state_vars.values()])
    N_algebraic_vars = N_vars - N_state_vars
    Jfx = J[:N_state_vars, :N_state_vars]
    Jfy = J[:N_state_vars, N_state_vars:]
    Jgx = J[N_state_vars:, :N_state_vars]
    Jgy = J[N_state_vars:, N_state_vars:]
    print('Shape of Jfx: {}'.format(Jfx.shape))
    print('Shape of Jfy: {}'.format(Jfy.shape))
    print('Shape of Jgx: {}'.format(Jgx.shape))
    print('Shape of Jgy: {}'.format(Jgy.shape))
    Jgy_inv = inv(Jgy)
    A = Jfx - (Jfy @ Jgy_inv @ Jgx if use_at_matmul else np.dot(np.dot(Jfy, Jgy_inv), Jgx))
    assert np.allclose(A, Amat), 'Error in the computation of the matrix A'
    eig, _ = np.linalg.eig(A)
    if any(eig.real > 0):
        print('>>> Some eigenvalues are positive (max value: {:.3e}): results may be inaccurate! <<<'.\
              format(max(eig.real)))
    B = - (Jfy @ Jgy_inv if use_at_matmul else np.dot(Jfy, Jgy_inv))
    C = - (Jgy_inv @ Jgx if use_at_matmul else np.dot(Jgy_inv, Jgx))
    D = - Jgy_inv
    print('Shape of A: {}'.format(A.shape))
    print('Shape of B: {}'.format(B.shape))
    print('Shape of C: {}'.format(C.shape))
    print('Shape of D: {}'.format(D.shape))

    def _try_append(dst, src, i):
        try: dst.append(src[i])
        except: pass

    def _fix_len(lst1, lst2):
        if len(lst1) == 1 and len(lst2) > 1:
            return lst1*len(lst2)
        return lst1

    full_element_names = list(vars_idx.keys())
    grid_name = full_element_names[0].split('-')[0]
    print(f'Grid name: {grid_name}.')
    element_names = list(map(lambda s: s.split('.')[0].split('-')[-1], full_element_names))
    bus_equiv_terms = data['bus_equiv_terms'].item()

    input_rows = {}
    injection_coeffs = {}
    if input_loads is not None:
        load_buses = data['load_buses'].item()
        all_load_names = []
        all_dP, all_sigmaP = [], []
        for i, input_load in enumerate(input_loads):
            if '*' in input_load:
                for load in load_buses.keys():
                    if re.match(input_load, load):
                        all_load_names.append(load)
                        _try_append(all_dP, dP, i)
                        _try_append(all_sigmaP, sigmaP, i)
            elif input_load not in load_buses:
                print(f'{progname}: cannot find load `{input_load}`.')
                sys.exit(0)
            else:
                all_load_names.append(input_load)
                _try_append(all_dP, dP, i)
                _try_append(all_sigmaP, sigmaP, i)
        input_loads = all_load_names
        dP, sigmaP = all_dP, all_sigmaP
        dP = _fix_len(dP, input_loads)
        sigmaP = _fix_len(sigmaP, input_loads)

        input_rows.update({ld: np.zeros(2, dtype=int) for ld in input_loads})
        injection_coeffs.update({ld: np.zeros(2, dtype=float) for ld in input_loads})
        PF_loads = PF['loads']
        mu, c, alpha = {}, {}, {}
        for i, input_load in enumerate(input_loads):
            found = False
            bus_name = load_buses[input_load]
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
            if verbose:
                print(f'[{i+1:2d}] {bus_name} -> {full_bus_name} {flag}')

            P0_pu = PF_loads[input_load]['P'] / S_base
            print("P of load '{}': {:g} MW ({:g} p.u.).".format(input_load, P0_pu * S_base, P0_pu))
            ur, ui = PF['buses'][bus_name]['ur'], PF['buses'][bus_name]['ui']
            u_abs = np.abs(ur + 1j * ui)
            u_abs2 = u_abs ** 2
            coeff = P0_pu / u_abs2 * (u_abs ** load_exp)
            key = f'{grid_name}-{input_load}.ElmLod'
            for j, suffix in enumerate('ri'):
                if key not in vars_idx:
                    keys = [key for key in vars_idx if input_load in key]
                    assert len(keys) == 1
                    key = keys[0]
                cols = vars_idx[key]['i' + suffix]
                assert len(cols) == 1
                col = cols[0]
                input_rows[input_load][j] = int(np.argmin(np.abs(J[:, col] - (-1))))
                injection_coeffs[input_load][j] = PF['buses'][bus_name][f'u{suffix}'] * coeff
                if verbose:
                    print("Variable 'i{}' of object '{}' is at column {}: equation #{}.".\
                          format(suffix, input_load, col + 1, input_rows[input_load][j] + 1))

            mean = PF_loads[input_load]['P']
            if len(dP) > 0:
                stddev = dP[i] * abs(mean)
            else:
                stddev = sigmaP[i]
            mu[input_load] = mean
            c[input_load] = stddev * np.sqrt(2. / tau)
            alpha[input_load] = 1. / tau

    if input_keys is not None:
        for inp_key, var_name in zip(input_keys, input_var_names):
            key = inp_key + '.' + var_name
            input_rows[key] = vars_idx[grid_name + '-' + inp_key][var_name]
            injection_coeffs[key] = 1.

    N_inputs = len(input_rows)

    I = np.eye(N_state_vars)
    # the transfer functions are complex numbers
    TF = np.zeros((N_freq, N_inputs, N_state_vars + N_algebraic_vars), dtype=complex)
    # the absolute value of the spectra of the outputs are real numbers:
    # we will take the abs at the end of the function
    OUT = np.zeros_like(TF)

    for i in tqdm(range(N_freq), ascii=True, ncols=70):
        M = 1j * 2 * np.pi * F[i] * I - A # sI - A
        MinvxB = np.dot(inv(M), B)        # (sI - A)^-1 x B
        for j, (key, rows) in enumerate(input_rows.items()):
            v = np.zeros(N_algebraic_vars)
            v[rows - N_state_vars] = 1.
            v[rows - N_state_vars] *= injection_coeffs[key]
            TF[i, j, :N_state_vars] = MinvxB @ v if use_at_matmul else np.dot(MinvxB, v)
            TF[i, j, N_state_vars:] = ((C @ MinvxB) + D) @ v if use_at_matmul else np.dot(np.dot(C, MinvxB) + D, v)
            if key in alpha:
                psd = np.sqrt((c[key] / alpha[key])**2 / (1 + (2 * np.pi * F[i] / alpha[key])**2))
                v[rows - N_state_vars] = psd
                v[rows - N_state_vars] *= injection_coeffs[key]
                OUT[i, j, :N_state_vars] = MinvxB @ v if use_at_matmul else np.dot(MinvxB, v)
                OUT[i, j, N_state_vars:] = ((C @ MinvxB) + D) @ v if use_at_matmul else np.dot(np.dot(C, MinvxB) + D, v)

    var_names, idx = [], []
    for k1, D in vars_idx.items():
        for k2, V in D.items():
            k = k1 + '.' + k2
            for v in V:
                var_names.append(k)
                idx.append(v)
    var_names = [var_names[i] for i in np.argsort(idx)]

    TF[TF == 0] = 1e-20 * (1 + 1j)
    OUT[OUT == 0] = 1e-20 * (1 + 1j)

    if compute_additional_TFs:
        if ref_SM_name is None:
            print('Please select the synchronous machine to be used as a reference:')
            for i, SM_name in enumerate(SM_names):
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
                    TF2[:, :, i]  = do_calc( TF[:, :, idx], coeffs, F, F0,  TF[:, :, ref_SM_idx])
                    OUT2[:, :, i] = do_calc(OUT[:, :, idx], coeffs, F, F0, OUT[:, :, ref_SM_idx])

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

    Htot_SM = data['inertia']
    Etot_SM = data['energy']
    Mtot_SM = data['momentum']
    out = {'A': A, 'F': F, 'TF': TF, 'OUT': OUT, 'var_names': var_names, 'SM_names': SM_names,
           'static_gen_names': static_gen_names, 'bus_names': bus_names, 'input_loads': input_loads,
           'Htot_SM': Htot_SM, 'Etot_SM': Etot_SM, 'Mtot_SM': Mtot_SM,
           'H_SM': H_SM, 'S_SM': S_SM, 'P_SM': P_SM, 'Q_SM': Q_SM,
           'S_SG': data['Ssg'].item(), 'DSL_params': data['DSL_params'].item(),
           'PF': data['PF_without_slack'], 'bus_equiv_terms': data['bus_equiv_terms'],
           'mu': mu, 'c': c, 'alpha': alpha, 'dP': dP, 'sigmaP': sigmaP, 'ref_SM_name': ref_SM_name,
           'data_file': data_file, 'with_additional_TFs': compute_additional_TFs}
    np.savez_compressed(os.path.join(outdir, outfile), **out)
    if save_mat:
        savemat(os.path.join(outdir, os.path.splitext(outfile)[0] + '.mat'), out, long_field_names=True)

    tend = TIME()
    print('Elapsed time: {:.3f} sec.'.format(tend - tstart))
