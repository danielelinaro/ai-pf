
import os
import sys
import numpy as np
from tqdm import tqdm
from multitone import compute_fourier_coeffs, newman_phase

progname = os.path.basename(sys.argv[0])

def usage(exit_code=None):
    print(f'usage: {progname} [-h | --help] [--f0 <freq>] [--N-tones <num>]')
    prefix = '       ' + ' ' * (len(progname) + 1)
    print(prefix + '[--N-periods <num>] [-f | --force] [--save-mat] data_file(s)')
    if exit_code is not None:
        sys.exit(exit_code)
        
if __name__ == '__main__':

    f0 = None
    N_tones = None
    N_periods = 1
    force = False
    save_mat = False
    
    i = 1
    N_args = len(sys.argv)
    while i < N_args:
        arg = sys.argv[i]
        if arg in ('-h', '--help'):
            usage(0)
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
            N_periods = int(sys.argv[i])
            if N_periods < 1:
                print(f'{progname}: number of periods must be >= 1.')
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

    data_files = sys.argv[i:]

    if f0 is None or N_tones is None:
        print('{progname}: you must specify both f0 and N_tones.')
        sys.exit(1)

    T0 = 1.0 / f0
    freqs = np.array([(i + 1) * f0 for i in range(N_tones)])
    phases = newman_phase(np.arange(N_tones) + 1, N_tones)
    A = np.sqrt(N_tones / 2)

    iter_fun = lambda n: tqdm(n, ascii=True, ncols=60)
    devices_to_consider = 'gen', 'bus'
    PF_device_names = {'gen': 'ElmSym', 'bus': 'ElmTerm'}
    for infile in data_files:
        print(f'Processing {infile}...')
        outfile = os.path.splitext(infile)[0] + '_fourier_coeffs.npz'
        if os.path.isfile(outfile) and not force:
            print(f'{outfile} exists: use -f to force overwrite.')
            continue
        blob = np.load(infile, allow_pickle=True)
        device_names = blob['device_names'].item()
        data = blob['data'].item()
        time = blob['time'].astype(float)
        N_samples = time.size
        dt = time[1] - time[0]
        while True:
            if time[-1] - time[-2] < dt / 10:
                time = time[:-1]
                print('Removing last sample from time vector.')
            else:
                break
        ttran = time[-1] - N_periods * T0 - dt / 2
        idx, = np.where(time > ttran)
        print(f'Simulation duration: {time[-1]:g} sec.')
        print(f'Discarding the first {ttran:g} sec.')
        if ttran < T0:
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
                        time[idx] - time[idx[0]],
                        X[idx].T,
                        phases,
                        A,
                        iter_fun,
                    )
                )
                vn = var_name.split(':')[1]
                if vn == 'xspeed':
                    vn = 'speed'
                var_names += ['{}.{}.{}'.format(n, PF_device_names[dev_type], vn) for n in device_names[dev_type]]
        Hsm = blob['Hsm'].item()
        Ssm = blob['Ssm'].item()
        SM_names = list(Hsm)
        Ssg = blob['Ssg'].item()
        SG_names = list(Ssg)
        config = blob['config'].item()
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
            'input_names': [inp['name'] for inp in config['inputs']],
        }
        np.savez_compressed(outfile, **out)
        if save_mat:
            from pathlib import Path
            for k in out:
                if out[k] is None:
                    out[k] = []
                elif isinstance(out[k], Path):
                    out[k] = str(out[k])
            savemat(os.path.splitext(outfile)[0] + '.mat', out, long_field_names=True)
