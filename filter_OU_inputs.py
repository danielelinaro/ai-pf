# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 09:25:05 2024

@author: Daniele Linaro
"""

import os
import re
import sys
import json
import warnings
from pathlib import Path
from time import time as TIME
import numpy as np
from scipy.signal import welch
from numpy.random import RandomState, SeedSequence, MT19937
from pfcommon import OU_2, combine_output_spectra
import tables
from tqdm import tqdm
iter_fun = lambda it: tqdm(it, ascii=True, ncols=70)


__all__ = ['run_vf', 'fit_TF', 'filter_inputs', 'run_welch']


def run_vf(X, F, n_poles, n_iter=3, weights=None, poles_guess=None, do_plot=False):
    """
    Runs the vector fitting algorithm for a given number of poles
    """

    Y = X.astype(np.complex128)
    if weights is None:
        weights = np.ones(F.size, dtype=np.float64)
    else:
        assert weights.size == F.size
        weights = weights.astype(np.float64)

    F0,F1 = np.log10(F[[0,-1]])
    s = (2j*np.pi*F).astype(np.complex128)

    import vectfit3 as vf
    opts = vf.opts.copy()
    opts['asymp'] = 2
    opts['skip_res'] = True  # skip residue computation
    opts['spy2'] = False     # do not plot the results

    # initial guess for pole positions
    if poles_guess is not None:
        poles = poles_guess
    else:
        poles = -2 * np.pi * np.logspace(F0, F1, n_poles, dtype=np.complex128)
    for i in range(n_iter):
        if i == n_iter-1:
            opts['skip_res'] = False
            opts['spy2'] = do_plot
        SER, poles, rmserr, fit = vf.vectfit(Y, s, poles, weights, opts)
    return SER, poles, rmserr, fit


def fit_TF(F, TF, max_N_poles=50, iter_fun=None):
    from scipy.signal import lti
    if iter_fun is None:
        iter_fun = lambda i: i
    N_samples, N_inputs, N_vars = TF.shape
    fit = np.zeros((N_vars, N_inputs, N_samples), dtype=complex)
    systems = [[] for _ in range(N_vars)]
    for i in iter_fun(range(N_vars)):
        for j in range(N_inputs):
            tf = TF[:, j, i]
            thresh = 10 ** (np.floor(np.log10(np.abs(tf).mean())) - 3)
            for n in range(max_N_poles):
                SER, _, RMSE, fit[i, j, :] = run_vf(tf, F, n + 1)
                if abs(RMSE) <= thresh:
                    break
            systems[i].append(lti(SER['A'], SER['B'], SER['C'], SER['D']))
    return systems, fit


def filter_inputs(systems, U, t, iter_fun=None):
    from scipy.signal import lsim
    if iter_fun is None:
        iter_fun = lambda i: i
    N_vars = len(systems)
    N_samples = len(t)
    Y = np.zeros((N_vars, N_samples))
    for i in iter_fun(range(N_vars)):
        y_all = []
        for S, u in zip(systems[i], U):
            _, y, _ = lsim(S, u, t)
            assert y.imag.max() < 1e-6
            y_all.append(y.real)
        Y[i, :] = np.sum(y_all, axis=0)
    return Y


def run_welch(x, dt, window, onesided=True):
    """
    Computes the PSD of a signal using Welch's method
    """
    from scipy.signal import welch
    freq, P = welch(
        x,
        1/dt,
        window='hamming',
        nperseg=window,
        noverlap=window/2,
        return_onesided=onesided,
        scaling='density',
    )
    if onesided:
        P /= 2
    else:
        Nf = freq.size
        freq = freq[:Nf//2]
        P = P[:Nf//2]
    return freq, P, np.sqrt(P)


def usage(progname, exit_code=None):
    print(f'usage: {progname} [-h | --help] [--data-file <fname>] [--tau <value>]')
    prefix = '       ' + ' ' * (len(progname)+1)
    print(prefix + '[-o | --outfile <value>] [-s | --suffix <value>]')
    print(prefix + '[-f | --force] [--dB <10|20>] [--tend <value>] config_file')
    if exit_code is not None:
        sys.exit(exit_code)


if __name__ == '__main__':

    tmp = Path(sys.argv[0])
    progdir, progname = tmp.parent, tmp.name

    # default values
    data_file = None
    force = False
    outfile = None
    suffix = 'noisy_traces'
    tend = None
    dB = 20

    i = 1
    N_args = len(sys.argv)
    while i < N_args:
        arg = sys.argv[i]
        if arg in ('-h', '--help'):
            usage(progname, 0)
        elif arg == '--data-file':
            i += 1
            data_file = Path(sys.argv[i])
        elif arg in ('-o','--outfile'):
            i += 1
            outfile = Path(sys.argv[i])
        elif arg in ('-s','--suffix'):
            i += 1
            suffix = re.sub('^_*', '', sys.argv[i])
        elif arg == '--tend':
            i += 1
            tend = float(sys.argv[i])
            if tend <= 0:
                print('tend must be > 0')
                sys.exit(0)
        elif arg in ('-f', '--force'):
            force = True
        elif arg.lower() == '--db':
            i += 1
            dB = int(sys.argv[i])
            if dB not in (10, 20):
                print(f'{progname}: dB value must be either 10 or 20.')
                sys.exit(1)
        elif arg[0] == '-':
            print(f'{progname}: unknown option `{arg}`.')
            sys.exit(1)
        else:
            break
        i += 1

    if i == N_args:
        print(f'{progname}: you must specify a configuration file')
        sys.exit(1)
    if i == N_args - 1:
        config_file = Path(sys.argv[i])
    else:
        print(f'{progname}: arguments after configuration file name are not allowed')
        sys.exit(1)

    if outfile is not None and suffix is not None:
        print('{progname}: output file name and suffix cannot be specified at the same time.')
        sys.exit(1)
        
    if not config_file.exists():
        print(f'{progname}: {config_file}: no such file.')
        sys.exit(1)

    with open(config_file) as fid:
        config = json.load(fid)
    if data_file is None:
        data_file = Path(config['data_file'])

    if not data_file.exists():
        print(f'{progname}: {data_file}: no such file.')
        sys.exit(1)

    if outfile is None:
        outfile = data_file.parent / (data_file.stem + f'_{suffix}.h5')
    if outfile.exists() and not force:
        print(f'{progname}: {outfile}: file exists, use -f to overwrite.')
        sys.exit(1)

    SM_info = {}
    for f in progdir.glob('*_SM_info.json'):
        print(f"Loading SM information from file '{f}'.")
        with open(f) as fid:
            SM_info.update(json.load(fid))

    data = np.load(data_file, allow_pickle=True)
    PF = data['PF'].item()
    # all the variable names in the power network
    all_var_names = data['var_names']
    # the variable names to simulate
    vars_to_save = config['vars_to_save']
    vars_to_save_file = None
    if isinstance(vars_to_save, str):
        # let's try to load a JSON file containing the variable names
        vars_to_save_file = Path(vars_to_save)
        if not vars_to_save_file.exists() and not vars_to_save_file.is_absolute():
            vars_to_save_file = config_file.parent / vars_to_save_file
        if vars_to_save_file.exists():
            with open(vars_to_save_file) as fid:
                vars_to_save = json.load(fid)['vars_to_save']
        else:
            print(f'Could not find a file named {vars_to_save}, will interpret that as a regular expression')
    elif not isinstance(vars_to_save, list):
        print(f"{progname}: the entry 'vars_to_save' in configuration file must be either a list or a string.")
        sys.exit(1)

    var_names, vars_idx = [], []
    for pattern in vars_to_save:
        matches_any = False
        for i, name in enumerate(all_var_names):
            if re.match(pattern, name) is not None:
                matches_any = True
                if i not in vars_idx:
                    var_names.append(name)
                    vars_idx.append(i)
                else:
                    print(f"Variable '{name}' matches more than one variable pattern.")
        if not matches_any:
            warnings.warn(f"Variable pattern '{pattern}' does not match any variable name.")

    N_vars = len(var_names)
    vars_idx = np.array(vars_idx)

    # simulation parameters
    if tend is None:
        tend = config['tend']
    try:
        dt = config['dt']
        srate = 1 / dt
    except:
        srate = config['sampling_rate']
        dt = 1 / srate

    N_samples = int(np.round(tend / dt)) + 1
    time = np.arange(N_samples) * dt
    assert time[-1] == tend

    if 'seed' in config:
        seed = config['seed']
    else:
        try:
            with open('/dev/urandom', 'rb') as fid:
                seed = int.from_bytes(fid.read(4), 'little') % 10000000
        except:
            seed = int(TIME()) % 10000000
    print(f'Seed: {seed}')
    rs = RandomState(MT19937(SeedSequence(seed)))

    all_input_names = data['input_loads'].tolist()
    if all_input_names is not None:
        # these are the loads for which individual TFs were computed by compute_spectra.py:
        # they are NOT necessarily all the loads that are present in the power network
        # these are the loads that should be stochastic in this simulation: this must be
        # a list of names
        input_names = config['input_loads']
        N_inputs = len(input_names)
        inputs_idx = np.array([all_input_names.index(name) for name in input_names])
        # mean has to be zero because we are simulating small signal fluctuations around the mean
        mu = np.zeros(N_inputs)
        c = np.array([data['c'].item()[name] for name in input_names])
        alpha = np.array([data['alpha'].item()[name] for name in input_names])
        OU_seeds = rs.randint(0, 100000, size=N_inputs)
    else:
        # these are the inputs for which individual TFs were computed by compute_spectra.py"
        # they do NOT need to be loads and are NOT necessarily all the loads in the system
        all_input_names = data['input_names'].tolist()
        if all_input_names is None:
            print("Both 'input_loads' and 'input_names' are empty in data file '{data_file}'.")
            sys.exit(1)
        # these are the loads that should be stochastic in this simulation: this must be
        # a list of dictionaries, with each dictionary containing the parameters of the
        # corresponding OU process
        inputs = config['input_loads']
        input_names = [inp['name'] for inp in inputs]
        N_inputs = len(input_names)
        inputs_idx = np.array([all_input_names.index(name) for name in input_names])
        mu = np.zeros(N_inputs)
        c, alpha = np.nan + np.zeros(N_inputs), np.nan + np.zeros(N_inputs)
        OU_seeds = rs.randint(0, 100000, size=N_inputs)
        for i, inp in enumerate(inputs):
            if 'mu' in inp:
                mu[i] = inp['mu']
            elif 'mean' in inp:
                mu[i] = inp['mean']
            if mu[i] != 0:
                warnings.warn(f'Setting the mean of OU input #{i} to {mu[i]} (which is != 0).')
            try:
                alpha[i] = inp['alpha']
            except:
                alpha[i] = 1 / inp['tau']
            if 'c' in inp:
                c[i] = inp['c']
            elif 'stddev' in inp:
                c[i] = inp['stddev'] * np.sqrt(2 * alpha[i])
            elif 'stddev_percent' in inp:
                name = re.sub('^[A-Z]+_', '', input_names[i])
                keys = [_ for _ in PF['loads'] if name in _]
                if len(keys) == 0:
                    print(f"No load matches '{name}' at power flow.")
                    sys.exit(1)
                if len(keys) > 1:
                    print(f"More than one load matches '{name}' at power flow.")
                    sys.exit(1)
                sigma = inp['stddev_percent'] / 100 * PF['loads'][keys[0]]['P']
                print("Standard deviation for input '{}': {:.2f} MW".format(inp['name'], sigma))
                c[i] = sigma * np.sqrt(2 * alpha[i])
            if 'seed' in inp:
                OU_seeds[i] = inp['seed']

    print('Building the OU stimuli...')
    U = np.zeros((N_inputs, N_samples))
    for i in iter_fun(range(N_inputs)):
        U[i] = OU_2(dt, alpha[i], mu[i], c[i], N_samples, random_state=OU_seeds[i])
            
    print('(Vector) fitting the TFs...')
    J, K = np.ix_(inputs_idx, vars_idx)
    F = data['F']
    TF = data['TF'][:, J, K]
    systems, fit = fit_TF(F, TF, iter_fun=iter_fun)

    print('Computing the output time series...')
    Y = filter_inputs(systems, U, time, iter_fun=iter_fun)

    print('Computing the power spectral densities of the outputs...')
    df = max(1e-6, 5 / tend)
    nperseg = int(srate / df)
    kwargs = {
        'fs': srate,
        'window': 'hamming',
        'nperseg': nperseg,
        'noverlap': nperseg // 2,
        'scaling': 'density',
        'return_onesided': True,
        'detrend': None,
    }
    freq, P_Y = welch(Y - Y.mean(), **kwargs)
    _, P_U = welch(U - U.mean(), **kwargs)
    abs_Y, abs_U = np.sqrt(P_Y), np.sqrt(P_U)

    F0 = 50.
    if 'OUT' in data:
        print('Combining the output spectra...')
        OUT = data['OUT']
        var_types = [os.path.splitext(name)[1][1:] for name in vars_to_save]
        OUT_multi = combine_output_spectra(
            OUT,
            input_names,
            vars_to_save,
            all_input_names,
            all_var_names,
            var_types,
            F,
            PF,
            data['bus_equiv_terms'].item(),
            ref_freq=F0,
            ref_SM_name=data['ref_SM_name'].item(),
        )
    else:
        OUT_multi = None
    
    print('Plotting the results...')
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    fontsize = 9
    lw = 0.75
    matplotlib.rc('font', **{'family': 'Arial', 'size': fontsize})
    matplotlib.rc('axes', **{'linewidth': 0.75, 'labelsize': fontsize})
    matplotlib.rc('xtick', **{'labelsize': fontsize})
    matplotlib.rc('ytick', **{'labelsize': fontsize})
    matplotlib.rc('xtick.major', **{'width': lw, 'size':3})
    matplotlib.rc('ytick.major', **{'width': lw, 'size':3})
    matplotlib.rc('ytick.minor', **{'width': lw, 'size':1.5})

    use_dBs = True
    if use_dBs:
        abs_Y = dB * np.log10(abs_Y)
        abs_U = dB * np.log10(abs_U)
        ylbl = r'|Y(j$\omega$)| [dB{}]'.format(dB)
    else:
        ylbl = r'|Y(j$\omega$)|'
    black = .1 + np.zeros(3)
    gray = .5 + np.zeros(3)
    width, height = 2.5, 2
    rows, cols = 2, N_vars + 1
    fig, ax = plt.subplots(
        rows,
        cols,
        figsize=(cols * width, rows * height),
        squeeze=False,
        sharex=True,
    )
    a = ax[1, 0]
    ax[0, 0].set_visible(False)
    for i in range(N_inputs):
        a.plot(freq, abs_U[i], color=gray, lw=0.75)
        try:
            P_U_theor = 2 * (c[i] / alpha[i]) ** 2 / (1 + (2 * np.pi * F / alpha[i]) ** 2)
            abs_U_theor = np.sqrt(P_U_theor)
            if use_dBs:
                abs_U_theor = dB * np.log10(abs_U_theor)
            a.plot(F, abs_U_theor, color='tab:red', lw=1)
        except:
            pass
    a.set_xlabel('Frequency [Hz]')
    a.set_ylabel(ylbl)
    a.set_title('Inputs')

    for i in range(N_vars):
        for j in range(N_inputs):
            tf = np.abs(TF[:, j, i])
            tf_fit = np.abs(fit[i, j, :])
            if use_dBs:
                tf, tf_fit = dB * np.log10(tf), dB * np.log10(tf_fit)
            ax[0, i + 1].plot(F, tf, color=black, lw=2)
            ax[0, i + 1].plot(F, tf_fit, color='tab:red', lw=0.75)

        ax[1, i + 1].plot(freq, abs_Y[i], color=gray, lw=0.75)
        if OUT_multi is not None:
            out = np.abs(OUT_multi[i])
            if use_dBs:
                out = dB * np.log10(out)
            ax[1, i + 1].plot(F, out, color='tab:red', lw=1)

        for a in ax[:, i + 1]:
            a.set_xscale('log')

        title = var_names[i].split('-')[-1].split('.')[0].split('__')[0]
        title += '.' + '.'.join(var_names[i].split('.')[-2:])
        ax[0, i + 1].set_title(title, fontsize=8)

    for a in ax[-1, 1:]:
        a.set_xlabel('Frequency [Hz]')
    for a in ax[:, 1]:
        a.set_ylabel(ylbl)
    sns.despine()
    fig.tight_layout()
    plt.savefig(outfile.parent / (outfile.stem + '.pdf'))

    block_dur = config['block_dur']
    N_samples_per_block = round(block_dur / dt)
    N_blocks = N_samples // N_samples_per_block

    skip_keys = 'Ptot', 'Qtot'
    inertia = [SM_info[k]['h'] for k in PF['SMs'] if k not in skip_keys]
    S = [SM_info[k]['S'] for k in PF['SMs'] if k not in skip_keys]
    generator_IDs = [k for k in PF['SMs'] if k not in skip_keys]
    N_generators = len(generator_IDs)

    sys.stdout.write('Saving data to {}... '.format(outfile.name))
    sys.stdout.flush()
    t0 = TIME()
    compression_filter = tables.Filters(complib='zlib', complevel=5)
    fid = tables.open_file(outfile, 'w', filters=compression_filter)

    maxlen = lambda arr: max(len(_) for _ in arr)
    var_len = 2 ** np.ceil(np.log2(max(maxlen(var_names), maxlen(input_names)))).astype(int)
    class Parameters (tables.IsDescription):
        F0             = tables.Float64Col()
        srate          = tables.Float64Col()
        tend           = tables.Float64Col()
        alpha          = tables.Float64Col(shape=(N_inputs,))
        mu             = tables.Float64Col(shape=(N_inputs,))
        c              = tables.Float64Col(shape=(N_inputs,))
        inertia        = tables.Float64Col(shape=(N_generators,))
        S              = tables.Float64Col(shape=(N_generators,))
        seed           = tables.Int64Col()
        OU_seeds       = tables.Int64Col(shape=(N_inputs,))
        generator_IDs  = tables.StringCol(128, shape=(N_generators,))
        var_names_file = tables.StringCol(128, shape=(1,))
        var_names      = tables.StringCol(var_len, shape=(N_vars,))
        load_names     = tables.StringCol(var_len, shape=(N_inputs,))

    tbl = fid.create_table(fid.root, 'parameters', Parameters, 'parameters')
    params = tbl.row
    params['F0']             = 50.
    params['srate']          = srate
    params['alpha']          = alpha
    params['mu']             = mu
    params['c']              = c
    params['tend']           = tend
    params['inertia']        = inertia
    params['S']              = S
    params['seed']           = seed
    params['OU_seeds']       = OU_seeds
    params['generator_IDs']  = generator_IDs
    params['var_names_file'] = [vars_to_save_file] if vars_to_save_file is not None else ['N/A']
    params['var_names']      = var_names
    params['load_names']     = input_names
    params.append()
    tbl.flush()

    fid.create_array(
        fid.root,
        'time',
        np.arange(N_samples_per_block) * dt,
    )

    for var_name, y in zip(var_names, Y):
        fid.create_array(
            fid.root,
            var_name.replace('-','_').replace('.','_'),
            y[:N_blocks * N_samples_per_block].reshape(-1, N_samples_per_block),
        )

    fid.close()

    print('done in {:.1f} sec.'.format(TIME() - t0))
