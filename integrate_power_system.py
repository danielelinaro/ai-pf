
import re
import sys
import json
import math
import time
from pathlib import Path
import numpy as np
import scipy.linalg as LA

import multitone
from pfcommon import parse_sparse_matrix_file, parse_Amat_vars_file, parse_Jacobian_vars_file, OU, OU_2
from cn_lti import crank_nicolson_lti

def usage(progname, exit_code=0):
    """Print the help message and exit with the given exit code."""
    msg = f"""\
Usage: {progname} [OPTIONS] CONFIG_FILE

Simulate a linearized power system (LTI model) with the Crank-Nicolson method.
The state matrix A and the input/output matrices B, C, D are computed from the
Jacobian and the A matrix found in the data directory, the inputs defined in
CONFIG_FILE are applied, and the requested variables are saved.

Arguments:
  CONFIG_FILE            JSON configuration file (required, must be the last
                         argument). It must contain the following keys:
                           data_dir     directory with the system matrices
                                        (can be overridden by --data-dir)
                           tend         end time of the simulation
                           dt           integration time step
                           inputs       list of inputs, each with the keys
                                        'dsl_name', 'var_name' and 'waveform',
                                        and optionally 'dev_name',
                                        'substation' and 'site'
                           vars_to_save mapping from a regular expression
                                        (matching object names) to the list
                                        of variable names to save

Options:
  -h, --help             Show this help message and exit.
  --data-dir DIR         Directory containing the files 'Jacobian.mtl',
                         'Amat.mtl' and 'VariableToIdx_Jacobian.txt'.
                         Overrides 'data_dir' in the configuration file.
  -o, --outfile FILE     Output file where the results are saved.
                         Default: lti_simulation.npz
  --tend T               End time of the simulation (must be > 0).
  -f, --force            Overwrite the output file if it already exists.

Examples:
  {progname} config.json
  {progname} --data-dir ./data -o results.npz --force config.json
"""
    stream = sys.stdout if exit_code == 0 else sys.stderr
    stream.write(msg)
    sys.exit(exit_code)


if __name__ == '__main__':
    progname = Path(sys.argv[0]).name

    # default values
    data_dir = None
    force = False
    outfile = Path('lti_simulation.npz')
    t_end = None

    i = 1
    N_args = len(sys.argv)
    while i < N_args:
        arg = sys.argv[i]
        if arg in ('-h', '--help'):
            usage(progname, 0)
        elif arg == '--data-dir':
            i += 1
            data_dir = Path(sys.argv[i])
        elif arg in ('-o','--outfile'):
            i += 1
            outfile = Path(sys.argv[i])
        elif arg == '--tend':
            i += 1
            t_end = float(sys.argv[i])
            if t_end <= 0:
                print(f'{progname}: tend must be > 0', file=sys.stderr)
                sys.exit(1)
        elif arg in ('-f', '--force'):
            force = True
        elif arg.startswith('-'):
            print(f'{progname}: unknown option `{arg}`.', file=sys.stderr)
            sys.exit(1)
        else:
            break
        i += 1

    if outfile.exists() and not force:
        print(f"{progname}: output file '{outfile}' exists: use -f to overwrite.", file=sys.stderr)
        sys.exit(1)

    if not outfile.parent.exists():
        print(f"{progname}: output directory '{outfile.parent}' does not exist", file=sys.stderr)
        sys.exit(1)

    if i == N_args:
        print(f'{progname}: you must specify a configuration file', file=sys.stderr)
        sys.exit(1)
    if i == N_args-1:
        config_file = Path(sys.argv[i])
    else:
        print(f'{progname}: arguments after configuration file name are not allowed', file=sys.stderr)
        sys.exit(1)

    if not config_file.exists():
        print(f'{progname}: {config_file}: no such file.', file=sys.stderr)
        sys.exit(1)

    with open(config_file, 'r') as fid:
        config = json.load(fid)

    if data_dir is None:
        data_dir = Path(config['data_dir'])
    else:
        config['data_dir'] = str(data_dir.absolute())
    if not data_dir.exists():
        print(f'{progname}: {data_dir}: no such directory', file=sys.stderr)
        sys.exit(1)

    if 'inputs' not in config or len(config['inputs']) == 0:
        print(f'{progname}: no inputs present in configuration file.', file=sys.stderr)
        sys.exit(1)

    J_file = data_dir / 'Jacobian.mtl'
    Amat_file = data_dir / 'Amat.mtl'
    J_vars_file = data_dir / 'VariableToIdx_Jacobian.txt'
    for f in (J_file, Amat_file, J_vars_file):
        if not f.exists():
            print(f'{progname}: {f}: no such file.', file=sys.stderr)
            sys.exit(1)
    J = parse_sparse_matrix_file(J_file)
    Amat = parse_sparse_matrix_file(Amat_file)
    N_vars = J.shape[0]
    N_state_vars = Amat.shape[0]
    N_alg_vars = N_vars - N_state_vars
    Jfx = J[:N_state_vars, :N_state_vars]
    Jfy = J[:N_state_vars, N_state_vars:]
    Jgx = J[N_state_vars:, :N_state_vars]
    Jgy = J[N_state_vars:, N_state_vars:]
    Jgy_inv = LA.inv(Jgy)
    A = Jfx - Jfy @ Jgy_inv @ Jgx
    if not np.allclose(A, Amat):
        print(f'{progname}: error in the computation of the matrix A', file=sys.stderr)
        sys.exit(1)
    eig, _ = LA.eig(A)
    eig_max = eig.real.max()
    if eig_max > 1e-6:
        print(f'Some eigenvalues of A are positive (max value: {eig_max:.3e}).')
    B = - Jfy @ Jgy_inv
    C = - Jgy_inv @ Jgx
    D = - Jgy_inv

    # variable names and corresponding indexes in the Jacobian matrix
    vars_idx, _, _, _, _ = parse_Jacobian_vars_file(J_vars_file)

    # simulation parameters and time vector (needed later to build the inputs)
    if t_end is None:
        t_end = float(config['tend'])
    if t_end <= 0:
        print(f'{progname}: t_end must be > 0.', file=sys.stderr)
        sys.exit(1)
    dt = config['dt']
    if dt <= 0:
        print(f'{progname}: dt must be > 0.', file=sys.stderr)
        sys.exit(1)
    N_steps = round(t_end / dt)
    t = np.arange(N_steps + 1) * dt
    config['tend'] = t[-1]

    # find the indexes of the inputs and build the corresponding waveforms
    U = []
    input_idx = []
    for inp in config['inputs']:
        pattern = inp['dsl_name']
        for part in ('dev_name', 'substation', 'site'):
            if part in inp:
                pattern = inp[part] + '-' + pattern
        var_name = inp['var_name']
        keys = [k1 for k1 in vars_idx if pattern in k1 and var_name in vars_idx[k1]]
        if len(keys) == 0:
            print(f"{progname}: no variable names match pattern: '{pattern}'.", file=sys.stderr)
            sys.exit(1)
        elif len(keys) > 1:
            print(f"{progname}: more than one variable matches the pattern '{pattern}'.", file=sys.stderr)
            sys.exit(1)
        key = keys[0]
        idx = vars_idx[key][var_name]
        if len(idx) != 1:
            raise NotImplementedError("only one index handled so far")
        idx = idx[0] - N_state_vars
        if idx < 0:
            print(f'{progname}: only algebraic variables can be used as inputs.', file=sys.stderr)
            sys.exit(1)
        if idx in input_idx:
            print(f'{progname}: duplicate inputs are not supported.', file=sys.stderr)
            sys.exit(1)
        input_idx.append(idx)
        fun = eval(inp['waveform'])
        print("Evaluating input '{}'.".format(inp['waveform']))
        u = np.asarray(fun(t), dtype=float)
        if u.ndim == 0:
            # fun returns a scalar
            u = np.broadcast_to(u, t.shape)
        if u.ndim == 1:
            # make sure that u has shape (N_steps + 1, 1)
            u = u.reshape(-1, 1)
        U.append(u)
        
    U = np.hstack(U)
    input_idx = np.asarray(input_idx)

    print('Shape of A:', A.shape)
    print('Shape of B:', B.shape)
    print('Shape of C:', C.shape)
    print('Shape of D:', D.shape)
    print('Shape of U:', U.shape)
    print('Number of state variables:', N_state_vars)
    print('Number of algebraic variables:', N_alg_vars)

    # find the indexes of the variables to save
    vars_to_save = []
    vars_to_save_idx = []
    for pattern, var_names in config['vars_to_save'].items():
        for key in vars_idx:
            if re.match(pattern, key) is not None:
                for var_name in var_names:
                    if var_name in vars_idx[key]:
                        idx = vars_idx[key][var_name]
                        if len(idx) != 1:
                            raise NotImplementedError('not implemented yet')
                        vars_to_save.append(key + '.' + var_name)
                        vars_to_save_idx.append(idx[0])
                    else:
                        print(f"Object '{key}' does not contain variable '{var_name}'.")
    vars_to_save_idx = np.asarray(vars_to_save_idx, dtype=int)
    # sorting is necessary in order for the variable names to match
    # the columns of the matrix x below, since we take subsets of
    # state and algebraic variables separately
    idx = np.argsort(vars_to_save_idx)
    vars_to_save = [vars_to_save[i] for i in idx]
    vars_to_save_idx = vars_to_save_idx[idx]
    print(f'Will save {vars_to_save_idx.size} variables.')

    # integrate the system
    x0 = np.zeros(N_state_vars)
    sys.stdout.write('Integrating the system... ')
    sys.stdout.flush()
    t0 = time.time()
    _, x = crank_nicolson_lti(A, B[:, input_idx], U, x0, t[0], t[-1], N_steps)
    # compute only the algebraic variables that are to be saved
    state_vars_to_save_idx = vars_to_save_idx[vars_to_save_idx < N_state_vars]
    algebraic_vars_to_save_idx = vars_to_save_idx[vars_to_save_idx >= N_state_vars] - N_state_vars
    x = np.hstack((
        x[:, state_vars_to_save_idx],
        x @ C[algebraic_vars_to_save_idx].T + U @ D[np.ix_(algebraic_vars_to_save_idx, input_idx)].T
    ))
    t1 = time.time()
    sys.stdout.write(f'done in {t1 - t0:.3f} sec.\n')

    # save everything
    data = {
        'config': config,
        'time': t,
        'xy': x,
        'var_names': vars_to_save,
        'var_indices': vars_to_save_idx,
        'N_state_vars': N_state_vars,
        'N_algebraic_vars': N_alg_vars,
        'input_indices': input_idx + N_state_vars,
        'U': U,
        'A_eigs': eig,
    }
    sys.stdout.write(f'Saving data to {outfile}... ')
    sys.stdout.flush()
    t0 = time.time()
    np.savez_compressed(outfile, **data)
    t1 = time.time()
    sys.stdout.write(f'done in {t1 - t0:.3f} sec.\n')

