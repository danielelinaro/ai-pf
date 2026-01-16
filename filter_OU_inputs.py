# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 09:25:05 2024

@author: Daniele Linaro
"""

import os
import re
import sys
import json
import numpy as np
from numpy.random import RandomState, SeedSequence, MT19937
from scipy.signal import lti, lsim, welch
from pfcommon import OU_2, combine_output_spectra
import tables
from tqdm import tqdm
iter_fun = lambda it: tqdm(it, ascii=True, ncols=70)


__all__ = ['run_vf', 'run_welch']


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
        poles = -2*np.pi*np.logspace(F0, F1, n_poles, dtype=np.complex128)
    for i in range(n_iter):
        if i == n_iter-1:
            opts['skip_res'] = False
            opts['spy2'] = do_plot
        SER,poles,rmserr,fit = vf.vectfit(Y, s, poles, weights, opts)
    return SER,poles,rmserr,fit


def run_welch(x, dt, window, onesided):
    """
    Computes the PSD of a signal using Welch's method
    """
    
    freq,P = welch(x, 1/dt, window='hamming',
                   nperseg=window, noverlap=window/2,
                   return_onesided=onesided, scaling='density')
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

    progname = os.path.basename(sys.argv[0])

    # default values
    data_file = None

    force = False
    outdir, outfile = '', None
    suffix = None
    tend = None
    dB = 10

    i = 1
    N_args = len(sys.argv)
    while i < N_args:
        arg = sys.argv[i]
        if arg in ('-h', '--help'):
            usage(progname, 0)
        elif arg == '--data-file':
            i += 1
            data_file = sys.argv[i]
        elif arg in ('-o','--outfile'):
            i += 1
            outfile = sys.argv[i]
        elif arg in ('-s','--suffix'):
            i += 1
            suffix = sys.argv[i]
            if suffix[0] == '_':
                suffix = suffix[1:]
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
            if dB not in (10,20):
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
    if i == N_args-1:
        config_file = sys.argv[i]
    else:
        print(f'{progname}: arguments after configuration file name are not allowed')
        sys.exit(1)

    if outfile is not None and suffix is not None:
        print('Output file name and suffix cannot be specified at the same time.')
        sys.exit(1)
        
    if not os.path.isfile(config_file):
        print(f'{progname}: {config_file}: no such file.')
        sys.exit(1)

    with open(config_file) as fid:
        config = json.load(fid)
    if data_file is None:
        data_file = config['data_file']

    if not os.path.isfile(data_file):
        print(f'{progname}: {data_file}: no such file.')
        sys.exit(1)

    if outfile is None:
        outdir = os.path.dirname(data_file)
        if outdir == '':
            outdir = '.'
        outfile = os.path.splitext(os.path.basename(data_file))[0]
        if suffix is None:
            outfile += '_noisy_traces.h5'
        else:
            outfile += '_' + suffix + '.h5'
    if os.path.isfile(os.path.join(outdir, outfile)) and not force:
        print(f'{progname}: {os.path.join(outdir, outfile)}: file exists, use -f to overwrite.')
        sys.exit(1)

    from glob import glob
    SM_info = {}
    for f in glob('*_SM_info.json'):
        with open(f) as fid:
            SM_info.update(json.load(fid))

    data = np.load(data_file, allow_pickle=True)
    F = data['F']
    # all the variable names in the power network
    all_var_names = data['var_names']
    # the variable names to simulate
    var_names = config['var_names']
    if isinstance(var_names, str):
        # config['var_names'] is the name of a JSON file that contains the variable names
        if os.path.isabs(var_names):
            var_names_file = var_names
        else:
            var_names_file = os.path.join(os.path.dirname(config_file), var_names)
        with open(var_names_file) as fid:
            var_names = json.load(fid)['var_names']
    else:
        var_names_file = None
    vars_idx = []
    for name in var_names:
        idx, = np.where(all_var_names == name)
        assert len(idx) == 1, f'{name}: no such variable.'
        vars_idx.append(idx[0])

    vars_idx = np.array(vars_idx)
    N_vars = len(var_names)

    # these are the loads that should be stochastic in this simulation
    load_names = config['input_loads']
    N_loads = len(load_names)
    # these are the loads for which individual TFs were computed by compute_spectra.py:
    # they are NOT necessarily all the loads that are present in the power network
    all_load_names = data['input_loads'].tolist()
    loads_idx = np.array([all_load_names.index(load_name) for load_name in load_names])
    # mean has to be zero because we are simulating small signal fluctuations around the mean
    mu = np.zeros(N_loads)
    c = np.array([data['c'].item()[name] for name in load_names])
    alpha = np.array([data['alpha'].item()[name] for name in load_names])
    if tend is None:
        tend = config['tend']
    srate = config['sampling_rate']
    dt = 1/srate
    time = np.r_[0 : tend+dt/2 : dt]
    N_samples = time.size
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
    OU_seeds = rs.randint(0, 100000, size=N_loads)
    rs = [RandomState(MT19937(SeedSequence(seed))) for seed in OU_seeds]
    print('Building the OU stimuli...')
    U = np.zeros((N_loads, N_samples))
    for i in iter_fun(range(N_loads)):
        U[i,:] = OU_2(dt, alpha[i], mu[i], c[i], N_samples, random_state=rs[i])

    print('(Vector) fitting the TFs...')
    # data['TF'][:, loads_idx, vars_idx] does not return what you would expect...
    # we need to do this:
    J, K = np.meshgrid(loads_idx, vars_idx, indexing='ij')
    TF = data['TF'][:, J, K]
    shp = N_vars, N_loads
    N_poles = np.zeros(shp, dtype=int)
    rms_err = np.zeros(shp)
    rms_thresh = np.zeros(shp)
    fit = np.zeros((N_vars, N_loads, F.size), dtype=complex)
    systems = [[] for _ in range(N_vars)]
    max_N_poles = 50
    for i in iter_fun(range(N_vars)):
        for j in range(N_loads):
            tf = TF[:, j, i]
            rms_thresh[i, j] = 10 ** (np.floor(np.log10(np.abs(tf).mean())) - 3)
            for n in range(max_N_poles):
                SER, _, rms_err[i, j], fit[i, j, :] = run_vf(tf, F, n + 1)
                if abs(rms_err[i, j]) < rms_thresh[i, j]:
                    break
            N_poles[i, j] = n + 1
            systems[i].append(lti(SER['A'], SER['B'], SER['C'], SER['D']))

    print('Computing the output time series...')
    Y = np.zeros((N_vars, N_samples))
    for i in iter_fun(range(N_vars)):
        y_all = []
        for S, u in zip(systems[i], U):
            _, y, _ = lsim(S, u, time)
            assert y.imag.max() < 1e-6
            y_all.append(y.real)
        Y[i, :] = np.sum(y_all, axis=0)

    print('Computing the power spectral densities of the outputs...')
    window = min(int(200 / dt), N_samples // 2)
    onesided = True
    freq, P_Y, abs_Y = run_welch(Y, dt, window, onesided)
    _, P_U, abs_U = run_welch(U, dt, window, onesided)

    print('Combining the output spectra...')
    OUT = data['OUT']
    PF = data['PF'].item()
    F0 = 50.
    var_types = [os.path.splitext(name)[1][1:] for name in var_names]
    OUT_multi =  combine_output_spectra(OUT, load_names, var_names, all_load_names,
                                        all_var_names, var_types, F, PF,
                                        data['bus_equiv_terms'].item(), ref_freq=F0,
                                        ref_SM_name=data['ref_SM_name'].item())
    
    print('Plotting the results...')
    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
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
        abs_Y = dB*np.log10(abs_Y)
        abs_U = dB*np.log10(abs_U)
        ylbl = r'|Y(j$\omega$)| [dB{}]'.format(dB)
    else:
        ylbl = r'|Y(j$\omega$)|'
    black = .1 + np.zeros(3)
    gray = .5 + np.zeros(3)
    width,height = 2.5,2
    rows,cols = 2,N_vars+1
    fig,ax = plt.subplots(rows, cols, figsize=(cols*width, rows*height),
                          squeeze=False, sharex=True)
    a = ax[1,0]
    ax[0,0].set_visible(False)
    for i in range(N_loads):
        P_U_theor = (c[i]/alpha[i])**2 / (1 + (2*np.pi*F/alpha[i])**2)
        abs_U_theor = np.sqrt(P_U_theor)
        if use_dBs:
            abs_U_theor = dB*np.log10(abs_U_theor)
        a.plot(freq, abs_U[i,:], color=gray, lw=0.75)
        a.plot(F, abs_U_theor, color='tab:red', lw=1)
    a.set_xlabel('Frequency [Hz]')
    a.set_ylabel(ylbl)
    a.set_title('Inputs')

    for i in range(N_vars):
        for j in range(N_loads):
            tf = np.abs(TF[:,j,i])
            tf_fit = np.abs(fit[i,j,:])
            if use_dBs:
                tf,tf_fit = dB*np.log10(tf),dB*np.log10(tf_fit)
            ax[0,i+1].plot(F, tf, color=black, lw=2)
            ax[0,i+1].plot(F, tf_fit, color='tab:red', lw=0.75)

        out = np.abs(OUT_multi[i,:])
        if use_dBs:
            out = dB*np.log10(out)
        ax[1,i+1].plot(freq, abs_Y[i,:], color=gray, lw=0.75)
        ax[1,i+1].plot(F, out, color='tab:red', lw=1)

        for a in ax[:,i+1]:
            a.set_xscale('log')

        title = var_names[i].split('-')[-1].split('.')[0].split('__')[0]
        title += '.' + '.'.join(var_names[i].split('.')[-2:])
        ax[0,i+1].set_title(title, fontsize=8)

    for a in ax[-1,1:]:
        a.set_xlabel('Frequency [Hz]')
    for a in ax[:,1]:
        a.set_ylabel(ylbl)
    sns.despine()
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, os.path.splitext(outfile)[0]+'.pdf'))

    block_dur = config['block_dur']
    N_samples_per_block = int(block_dur / dt)
    N_blocks = N_samples // N_samples_per_block
    
    inertia = [SM_info[k]['h'] for k in PF['SMs'] if k in SM_info]
    S = [SM_info[k]['S'] for k in PF['SMs'] if k in SM_info]
    generator_IDs = [k for k in PF['SMs'] if k not in ('Ptot','Qtot')]
    N_generators = len(generator_IDs)

    from time import time as TIME
    sys.stdout.write('Saving data to {}... '.format(os.path.join(outdir,outfile)))
    sys.stdout.flush()
    t0 = TIME()
    compression_filter = tables.Filters(complib='zlib', complevel=5)
    fid = tables.open_file(os.path.join(outdir, outfile), 'w', filters=compression_filter)

    from itertools import chain
    var_len = 2**int(np.ceil(np.log2(max([len(v) for v in chain(var_names,load_names)]))))
    class Parameters (tables.IsDescription):
        F0             = tables.Float64Col()
        srate          = tables.Float64Col()
        tend           = tables.Float64Col()
        alpha          = tables.Float64Col(shape=(N_loads,))
        mu             = tables.Float64Col(shape=(N_loads,))
        c              = tables.Float64Col(shape=(N_loads,))
        inertia        = tables.Float64Col(shape=(N_generators,))
        S              = tables.Float64Col(shape=(N_generators,))
        seed           = tables.Int64Col()
        OU_seeds       = tables.Int64Col(shape=(N_loads,))
        generator_IDs  = tables.StringCol(128, shape=(N_generators,))
        var_names_file = tables.StringCol(128, shape=(1,))
        var_names      = tables.StringCol(var_len, shape=(N_vars,))
        load_names     = tables.StringCol(var_len, shape=(N_loads,))

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
    params['var_names_file'] = [var_names_file] if var_names_file is None else ['N/A']
    params['var_names']      = var_names
    params['load_names']     = load_names
    params.append()
    tbl.flush()

    fid.create_array(fid.root, 'time', np.arange(N_samples_per_block)*dt)
    for var_name,y in zip(var_names,Y):
        fid.create_array(fid.root,
                         var_name.replace('-','_').replace('.','_'),
                         np.reshape(y[:N_blocks*N_samples_per_block],
                                    [N_blocks, N_samples_per_block]))


    fid.close()

    t1 = TIME()
    print(f'done in {t1-t0:.1f} sec.')
