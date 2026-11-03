
import os
import re
import sys
import numpy as np
from tqdm import tqdm
from multitone import compute_fourier_coeffs, compute_multitone_pars

progname = os.path.basename(sys.argv[0])

HELP_MESSAGE = """\
usage: {} [-h | --help] [--f0 <freq>] [--N-tones <num>] [--variable-timestep] [--dt <val>]
       {} [--N-periods <num>] [--tran <num[T]>] [-f | --force] [--save-mat] data_file(s)

Compute the Fourier coefficients of one or more time-domain data files.

positional arguments:
  data_file(s)          One or more files containing the time-domain data to
                        analyze. Each file is processed independently.

options:
  -h, --help            Print this help message and exit.

  --f0 <freq>           Fundamental frequency of the signal, in Hz.

  --N-tones <num>       Number of tones (harmonics of f0) to compute.

  --variable-timestep   Indicate that the data was sampled with a non-uniform
                        time step. No interpolation is performed

  --dt <val>            Time step, in seconds, of the uniform grid used for
                        resampling. Only meaningful when --variable-timestep
                        is not specified. If not passed it is inferred from
                        the data file.

  --N-periods <num>     Number of periods of the fundamental to include in
                        the analysis, counted from the end of the data.
                        [default: 1]

  --tran <num[T]>       Initial transient to discard before analysis. A plain
                        number is interpreted as seconds; a number followed by
                        'T' is interpreted as periods of the fundamental
                        (e.g. --tran 5T discards the first 5 periods).

  -f, --force           Overwrite existing output files without prompting.

  --save-mat            In addition to the default output, save the computed
                        coefficients to a MATLAB-compatible .mat file.

examples:
  python {} --dt 0.005 --N-tones 50 --f0 0.01 --N-periods 4 file.npz
  python {} --dt 0.005 --N-tones 50 --f0 0.01 --tran 1T file1.npz file2.npz
""".format(progname, ' ' * len(progname), progname, progname, progname)

def usage(exit_code=None):
    print(HELP_MESSAGE)
    if exit_code is not None:
        sys.exit(exit_code)
        
if __name__ == '__main__':

    timestep = None
    f0 = None
    N_tones = None
    N_periods_to_analyze = None
    transient_to_skip = None
    force = False
    save_mat = False
    fixed_timestep = True

    i = 1
    N_args = len(sys.argv)
    while i < N_args:
        arg = sys.argv[i]
        if arg in ('-h', '--help'):
            usage(0)
        elif arg == '--dt':
            i += 1
            timestep = float(sys.argv[i])
            if timestep <= 0:
                print(f'{progname}: time step must be > 0.')
                sys.exit(0)
        elif arg == '--variable-timestep':
            fixed_timestep = False
        elif arg == '--f0':
            i += 1
            f0 = float(sys.argv[i])
            if f0 <= 0:
                print(f'{progname}: f0 must be > 0.')
                sys.exit(1)
        elif arg == '--N-tones':
            i += 1
            N_tones = int(sys.argv[i])
            if N_tones <= 0:
                print(f'{progname}: number of tones must be > 0.')
                sys.exit(1)
        elif arg == '--N-periods':
            i += 1
            N_periods_to_analyze = int(sys.argv[i])
            if N_periods_to_analyze < 1:
                print(f'{progname}: number of periods must be >= 1.')
                sys.exit(1)
        elif arg == '--tran':
            i += 1
            if sys.argv[i][-1] == 'T':
                transient_to_skip = int(sys.argv[i][:-1])
                if transient_to_skip <= 0:
                    print(f'{progname}: transient duration must be > 0.')
                    sys.exit(1)
                # make it negative so we know that we must multiply by T0
                transient_to_skip *= -1
            else:
                transient_to_skip = float(sys.argv[i])
                if transient_to_skip <= 0:
                    print(f'{progname}: transient duration must be > 0.')
                    sys.exit(1)
        elif arg in ('-f', '--force'):
            force = True
        elif arg == '--save-mat':
            try:
                from scipy.io import savemat
                save_mat = True
            except:
                raise Warning('scipy not available: will not save MAT file')
        elif arg[0] == '-':
            print(f'{progname}: unknown option `{arg}`.')
            sys.exit(1)
        else:
            break
        i += 1

    if i == N_args:
        print(f'{progname}: you must specify at least one input file')
        sys.exit(1)


    if timestep is not None and not fixed_timestep:
        print(f'{progname}: you cannot pass both --dt and --variable-timestep.')
        sys.exit(1)

    if f0 is None or N_tones is None:
        print(f'{progname}: you must specify both f0 and N_tones.')
        sys.exit(1)

    data_files = sys.argv[i:]

    if N_periods_to_analyze is None and transient_to_skip is None:
        N_periods_to_analyze = 1

    T0 = 1.0 / f0
    freqs = (np.arange(N_tones) + 1) * f0
    if transient_to_skip is not None and transient_to_skip < 0:
        transient_to_skip = abs(transient_to_skip) * T0

    iter_fun = lambda n: tqdm(n, ascii=True, ncols=60)
    devices_to_consider = 'gen', 'bus', 'genstat'
    PF_device_names = {'gen': 'ElmSym', 'bus': 'ElmTerm', 'genstat': 'ElmGenStat'}
    for infile in data_files:
        print(f'Processing {infile}...')
        outfile = os.path.splitext(infile)[0] + '_fourier_coeffs.npz'
        if os.path.isfile(outfile) and not force:
            print(f'{outfile} exists: use -f to force overwrite.')
            continue
        blob = np.load(infile, allow_pickle=True)
        device_names = blob['device_names'].item()

        config = blob['config'].item()
        if 'inputs_from_DSL' in config:
            assert len(config['inputs_from_DSL']) == 1, "There are multiple inputs..."
            algo = config['inputs_from_DSL'][0]['algorithm']
            if algo == 'friese':
                seed = config['inputs_from_DSL'][0]['seed']
        elif 'inputs_from_file' in config:
            algo = None
            for inp in config['inputs_from_file']:
                if 'multitone' in inp['waveform']:
                    M = re.search("algorithm='[a-z]+'", inp['waveform'])
                    algo = M.group().split('=')[1][1:-1]
                    if algo == 'friese':
                        M = re.search('seed=[0-9]+', inp['waveform'])
                        seed = int(M.group().split('=')[1])
                    break
            if algo is None:
                raise Exception('Cannot find a multitone input in the configuration file')
        else:
            raise Exception('Cannot find the input in the configuration file')
        if algo == 'friese':
            w0 = dw = 2 * np.pi * f0
            A, _, phases = compute_multitone_pars(N_tones, algo, dw=dw, w0=w0, seed=seed)
        elif algo == 'boyd':
            A, _, phases = compute_multitone_pars(N_tones, algo, phases='newman', dw=2 * np.pi * f0)
        else:
            raise Exception(f"Unknown crest shaping algorithm '{algo}'")

        data = blob['data'].item()
        time = blob['time'].astype(float)
        dt, counts = np.unique(np.diff(time), return_counts=True)
        dt = dt[np.argmax(counts)]
        if timestep is not None:
            assert abs(timestep - dt) / timestep < 1e-6, 'Timestep in file does not match the one pased'
            dt = timestep
        else:
            assert dt > 1e-6, 'Timestep is too small'
        if fixed_timestep:
            samples_per_period = int(np.round(T0 / dt))
            assert abs(T0 - samples_per_period * dt) / T0 < 1e-6, 'Problems with timestep'
        print(f'Timestep: {dt:g} s.')
        while True:
            if time.size > 1 and time[-1] - time[-2] < dt / 10:
                time = time[:-1]
                print('Removing last sample from time vector.')
            else:
                break
        N_samples = time.size
        for dev_type in data:
            for var_name in data[dev_type]:
                data[dev_type][var_name] = data[dev_type][var_name][:N_samples]

        if N_periods_to_analyze is None:
            N_periods = (time[-1] - transient_to_skip) / T0
            # be a little flexible
            if N_periods % 1 > 0.999:
                N_periods = np.ceil(N_periods)
                print(f'Rounding number of periods to analyze to {N_periods:g}.')
            N_periods = int(N_periods)
        else:
            N_periods = N_periods_to_analyze

        assert N_periods > 0, 'Not enough data in recording: reduce transient duration.'

        if transient_to_skip is not None:
            tran = transient_to_skip
        elif fixed_timestep:
            start = N_samples - (N_periods * samples_per_period + 1)
            assert start >= 0, 'Not enough data in recording: reduce the number of periods'
            tran = time[start]
        else:
            tran = time[-1] - (N_periods * T0 + dt / 2)
            assert tran >= 0, 'Not enough data in recording: reduce the number of periods'

        print(f'Simulation duration: {time[-1]:g} s.')
        if transient_to_skip is not None:
            print('Discarding the first {:g} s (i.e., {:g} period{}).'.format(
                tran, tran / T0, 's' if tran / T0 > 1 else ''
            ))
        else:
            print('Considering the last {} period{} (i.e., {} s).'.format(
                N_periods, 's' if N_periods > 1 else '', N_periods * T0
            ))

        if fixed_timestep:
            start = np.where(time > tran)[0][0]
            stop = start + N_periods * samples_per_period + 1
            if stop > N_samples:
                start, stop = N_samples - (stop - start), N_samples
            assert start >= 0, 'Not enough data in recording: reduce the number of periods'
            idx = np.arange(start, stop)
            assert np.allclose(np.diff(time[idx]), dt, rtol=1e-6), \
                'Timestep not constant in analysis window: use --variable-timestep'
        else:
            idx, = np.where((time > tran) & (time <= tran + N_periods * T0 + dt / 2))
            assert idx.size and time[idx[-1]] - time[idx[0]] >= N_periods * T0 - dt, \
                'Not enough data in recording: reduce transient duration.'

        if time[idx[0]] < T0:
            print('Warning: discarding less than a full period.')

        coeffs = []
        var_names = []
        for dev_type in data:
            if dev_type not in devices_to_consider:
                continue
            for var_name, X in data[dev_type].items():
                coeffs.append(
                    compute_fourier_coeffs(
                        freqs,
                        time[idx],
                        X[idx].T,
                        phases,
                        A,
                        iter_fun,
                    )
                )
                vn = var_name.split(':')[1]
                vn = 'speed' if vn == 'xspeed' else vn
                var_names += ['{}.{}.{}'.format(n, PF_device_names[dev_type], vn) for n in device_names[dev_type]]
        Hsm = blob['Hsm'].item()
        Ssm = blob['Ssm'].item()
        SM_names = list(Hsm)
        Ssg = blob['Ssg'].item()
        SG_names = list(Ssg)
        out = {
            'DSL_params': blob['DSL_params'].item(),
            'H_SM': np.array([Hsm[n] for n in SM_names]),
            'S_SM': np.array([Ssm[n] for n in SM_names]),
            'S_SG': Ssg,
            'SM_names': SM_names,
            'static_gen_names': SG_names,
            'Etot_SM': blob['energy'].item(),
            'F': freqs,
            'TF': np.hstack(coeffs)[:, np.newaxis, :],
            'var_names': var_names,
        }
        for key in ('inputs', 'inputs_from_DSL', 'inputs_from_file'):
            if key in config:
                out['input_names'] = [inp['name'] for inp in config[key]]
                break
        if 'input_names' not in out:
            print('Could not figure out the input names.')
        np.savez_compressed(outfile, **out)
        if save_mat:
            from pathlib import Path
            for k in out:
                if out[k] is None:
                    out[k] = []
                elif isinstance(out[k], Path):
                    out[k] = str(out[k])
            savemat(os.path.splitext(outfile)[0] + '.mat', out, long_field_names=True)
