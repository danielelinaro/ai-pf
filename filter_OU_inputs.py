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
from scipy.signal import lti, lsim, welch
from pfcommon import OU_2
import tables

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

import seaborn as sns

progname = os.path.basename(sys.argv[0])


def run_vf(X, F, n_poles, n_iter=4, weights=None, do_plot=False):
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


def usage(exit_code=None):
    print(f'usage: {progname} [-h | --help] [--data-file <fname>] [--tau <value>]')
    prefix = '       ' + ' ' * (len(progname)+1)
    print(prefix + '[-o | --outfile <value>] [-f | --force] [--dB <10|20>] config_file')
    if exit_code is not None:
        sys.exit(exit_code)


SM_info_fname = 'V2020_Rete_Sardegna_2021_06_03cr_SM_info.json'
SM_info = json.load(open(SM_info_fname, 'r'))

if __name__ == '__main__':

    # default values
    data_file = None

    force = False
    outdir, outfile = '', None
    dB = 10

    i = 1
    N_args = len(sys.argv)
    while i < N_args:
        arg = sys.argv[i]
        if arg in ('-h', '--help'):
            usage(0)
        elif arg == '--data-file':
            i += 1
            data_file = sys.argv[i]
        elif arg in ('-o','--outfile'):
            i += 1
            outfile = sys.argv[i]
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
        
    if not os.path.isfile(config_file):
        print(f'{progname}: {config_file}: no such file.')
        sys.exit(1)

    config = json.load(open(config_file, 'r'))
    if data_file is None:
        data_file = config['data_file']

    if not os.path.isfile(data_file):
        print(f'{progname}: {data_file}: no such file.')
        sys.exit(1)

    if outfile is None:
        outdir = os.path.dirname(data_file)
        if outdir == '':
            outdir = '.'
        outfile = os.path.splitext(os.path.basename(data_file))[0] + '_noisy_traces.h5'
    if os.path.isfile(os.path.join(outdir, outfile)) and not force:
        print(f'{progname}: {os.path.join(outdir, outfile)}: file exists, use -f to overwrite.')
        sys.exit(1)

    data = np.load(data_file, allow_pickle=True)
    F = data['F']
    # all the variable names in the power network
    all_var_names = data['var_names']
    # the variable names to simulate
    vars_to_sim = config['var_names']
    if isinstance(vars_to_sim, str):
        # config['var_names'] is the name of a JSON file that contains the variable names
        if os.path.isabs(vars_to_sim):
            var_names_file = vars_to_sim
        else:
            var_names_file = os.path.join(os.path.dirname(config_file), vars_to_sim)
        vars_to_sim = json.load(open(var_names_file))['var_names']
    else:
        var_names_file = None
    vars_idx = []
    for name in vars_to_sim:
        idx, = np.where(all_var_names == name)
        if len(idx) != 1:
            print(f'{name}: no such variable.')
            import ipdb
            ipdb.set_trace()
        vars_idx.append(idx[0])

    vars_idx = np.array(vars_idx)
    N_vars = len(vars_to_sim)

    load_names = config['load_names']
    N_loads = len(load_names)
    if N_loads > 1:
        raise NotImplementedError('not implemented yet')

    mu,c,alpha = data['mu'],data['c'],data['alpha']
    tend,srate = config['tend'],config['sampling_rate']
    dt = 1/srate
    time = np.r_[0 : tend+dt/2 : dt]
    N_samples = time.size
    U = np.zeros((N_loads, N_samples))
    for i in range(N_loads):
        U[i,:] = OU_2(dt, alpha[i], mu[i], c[i], N_samples)
    
    TFs = data['TF'][:,:,vars_idx]
    OUT_PSDs = data['OUT'][:,:,vars_idx]

    systems = [[] for _ in range(N_loads)]
    N_poles = 20
    Y = np.zeros((N_loads, N_vars, N_samples))

    use_dBs = True

    window = 200/dt
    onesided = True
    width,height = 2.5,1.5
    rows,cols = 2*N_loads,N_vars
    fig,ax = plt.subplots(rows, cols, figsize=(cols*width, rows*height),
                          squeeze=False, sharex=True)
    for i in range(N_loads):
        P_U_theor = (c[i]/alpha[i])**2 / (1 + (2*np.pi*F/alpha[i])**2)
        abs_U_theor = np.sqrt(P_U_theor)
        P_U_theor_db = dB*np.log10(P_U_theor)

        for j in range(N_vars):
            tf = TFs[i,:,j]
            SER,poles,rmserr,fit = run_vf(tf, F, N_poles)
            S = lti(SER['A'], SER['B'], SER['C'], SER['D'])
            _,y,_ = lsim(S, U[i,:], time)
            assert y.imag.max() < 1e-10
            Y[i,j,:] = y.real
            systems[i].append(S)

            freq,P_Y,abs_Y = run_welch(Y[i,j,:], dt, window, onesided)
            
            ax[i,j].plot(F, dB*np.log10(np.abs(tf)), color=.1+np.zeros(3),
                         lw=2, label=load_names[i].split('__')[0])
            ax[i,j].plot(F, dB*np.log10(np.abs(np.squeeze(fit))),
                         color=[1,0,0], lw=1, label='Fit')

            I = i + N_loads
            y = dB*np.log10(abs_Y) if use_dBs else abs_Y
            TFxU = dB*np.log10(np.abs(tf)*abs_U_theor) if use_dBs else np.abs(tf)*abs_U_theor
            out = dB*np.log10(np.abs(OUT_PSDs[i,:,j])) if use_dBs else np.abs(OUT_PSDs[i,:,j])
            ax[I,j].plot(freq, y, color=[1,0.5,0], lw=0.75, label='Y')
            ax[I,j].plot(F, TFxU, color=[.9,0,.9], lw=2, label='TFxIN')
            ax[I,j].plot(F, out, color=[0,.9,0], lw=1, label='OUT')

            title = vars_to_sim[j].split('-')[-1].split('.')[0].split('__')[0]
            title += '.' + '.'.join(vars_to_sim[j].split('.')[-2:])
            ax[i,j].set_title(title, fontsize=8)
            ax[i,j].set_xscale('log')
            ax[I,j].set_xscale('log')

    lgnd_kwargs = {'loc':'best', 'fontsize':8, 'frameon':False}
    ax[0,0].legend(**lgnd_kwargs)
    ax[N_loads,0].legend(**lgnd_kwargs)
    for a in ax[-1,:]:
        a.set_xlabel('Frequency [Hz]')
    for a in ax[:,0]:
        if use_dBs:
            a.set_ylabel(r'|Y(j$\omega$)| [dB{}]'.format(dB))
        else:
            a.set_ylabel(r'|Y(j$\omega$)|')
    sns.despine()
    fig.tight_layout()
    plt.savefig(os.path.join(outdir, os.path.splitext(outfile)[0]+'.pdf'))

    block_dur = config['block_dur']
    N_samples_per_block = int(block_dur / dt)
    N_blocks = N_samples // N_samples_per_block
    
    PF = data['PF'].item()
    inertia = [SM_info[k]['h'] for k in PF['SMs'] if k in SM_info]
    S = [SM_info[k]['S'] for k in PF['SMs'] if k in SM_info]
    generator_IDs = [k for k in PF['SMs'] if k not in ('Ptot','Qtot')]
    N_generators = len(generator_IDs)
    
    compression_filter = tables.Filters(complib='zlib', complevel=5)
    fid = tables.open_file(os.path.join(outdir, outfile), 'w', filters=compression_filter)

    var_len = 2**int(np.ceil(np.log2(max([len(v) for v in vars_to_sim]))))
    N_vars_to_sim = len(vars_to_sim)
    class Parameters (tables.IsDescription):
        F0             = tables.Float64Col()
        srate          = tables.Float64Col()
        tend           = tables.Float64Col()
        alpha          = tables.Float64Col(shape=(N_loads,))
        mu             = tables.Float64Col(shape=(N_loads,))
        c              = tables.Float64Col(shape=(N_loads,))
        inertia        = tables.Float64Col(shape=(N_generators,))
        S              = tables.Float64Col(shape=(N_generators,))
        generator_IDs  = tables.StringCol(128, shape=(N_generators,))
        var_names_file = tables.StringCol(128, shape=(1,))
        vars_to_sim    = tables.StringCol(var_len, shape=(N_vars_to_sim,))

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
    params['generator_IDs']  = generator_IDs
    params['var_names_file'] = [var_names_file] if var_names_file is None else ['N/A']
    params['vars_to_sim']    = vars_to_sim
    params.append()
    tbl.flush()

    fid.create_array(fid.root, 'time', np.arange(N_samples_per_block)*dt)
    for var_name,y in zip(vars_to_sim,Y[0,:,:]):
        fid.create_array(fid.root,
                         var_name.replace('-','_').replace('.','_'),
                         np.reshape(y[:N_blocks*N_samples_per_block],
                                    [N_blocks, N_samples_per_block]))


    fid.close()
    
