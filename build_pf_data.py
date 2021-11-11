
import os
import sys
import json
import time
import argparse as arg
import numpy as np
import tables

from run_pf_simulation import run_sim

progname = os.path.basename(sys.argv[0])

if __name__ == '__main__':

    parser = arg.ArgumentParser(description = 'Build data for inertia estimation with deep neural networks', \
                                formatter_class = arg.ArgumentDefaultsHelpFormatter, \
                                prog = progname)
    parser.add_argument('config_file', type=str, action='store', help='configuration file')
    parser.add_argument('-d', '--dur', default=None, type=float, help='simulation duration in seconds')
    parser.add_argument('-n', '--n-trials',  default=None,  type=int, help='number of trials')
    parser.add_argument('-s', '--suffix',  default='',  type=str, help='suffix to add to the output files')
    parser.add_argument('-o', '--output-dir',  default=None,  type=str, help='output directory')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')
    args = parser.parse_args(args=sys.argv[1:])
    
    verbose = args.verbose
    
    config_file = args.config_file
    if not os.path.isfile(config_file):
        print(f'{progname}: {config_file}: no such file.')
        sys.exit(1)
    in_config = json.load(open(config_file, 'r'))
    generator_IDs = sorted(list(in_config['inertia'].keys()))

    out_config = in_config.copy()
    for key in 'inertia', 'inertia_mode', 'dur', 'Ntrials':
        out_config.pop(key)
    
    # OU parameters
    alpha = in_config['OU']['alpha']
    mu = in_config['OU']['mu']
    c = in_config['OU']['c']

    # simulation parameters
    frand = in_config['frand']  # [Hz] sampling rate of the random signal
    if args.dur is not None:
        out_config['tstop'] = [args.dur]     # [s]  simulation duration
    else:
        out_config['tstop'] = [in_config['dur']]

    # the name of the stochastic load
    stochastic_load_name = in_config['random_load_name']

    # inertia values
    try:
        inertia_mode = in_config['inertia_mode']
    except:
        inertia_mode = 'combinatorial'
    if inertia_mode == 'combinatorial':
        inertia = in_config['inertia']
        inertia_values = []
        for gen_id in generator_IDs:
            inertia_values.append(inertia[gen_id])
        H = np.meshgrid(*inertia_values)
        inertia_values = {}
        for i,gen_id in enumerate(generator_IDs):
            inertia_values[gen_id] = H[i].flatten()
        N_inertia = inertia_values[generator_IDs[0]].size
    elif inertia_mode == 'sequential':
        inertia_values = in_config['inertia'].copy()
        N_inertia = 1
        for v in inertia_values.values():
            if len(v) > N_inertia:
                N_inertia = len(v)
        for k in inertia_values:
            N_values = len(inertia_values[k])
            if N_values == 1:
                inertia_values[k] = inertia_values[k][0] + np.zeros(N_inertia)
            elif N_values != N_inertia:
                raise Exception(f'The number of inertia values for generator "{k}" does not match the other generators')
    else:
        print(f'Unknown value for inertia_mode: "{inertia_mode}".')
        print('Accepted values are: "combinatorial" and "sequential".')
        sys.exit(1)

    # how many trials per inertia value
    if args.n_trials is not None:
        N_trials = args.n_trials
    else:
        N_trials = in_config['Ntrials']

    if args.output_dir is None:
        output_dir = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    else:
        output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    suffix = args.suffix
    if suffix != '':
        if suffix[0] != '_':
            suffix = '_' + suffix
        if suffix[-1] == '_':
            suffix = suffix[:-1]

    for i in range(N_inertia):

        out_config['inertia'] = {gen_ID: [inertia_values[gen_ID][i]] for gen_ID in generator_IDs}

        out_file = ''
        # change generators' inertia values
        msg = 'Setting inertia of generator "{}" to {:5.3f} s.'
        msg_len = len(msg) - 9 + 5 + np.max([len(gen_ID) for gen_ID in generator_IDs])
        banner = ' Inertia Value {:02d}/{:02d} '
        banner_len = len(banner) - 8
        symbol = '='
        symbols_len = (msg_len - banner_len) // 2
        if symbols_len * 2 + banner_len < msg_len:
            banner = symbol + banner
            banner_len += 1
        print(symbol * symbols_len + banner.format(i+1, N_inertia) + symbol * symbols_len)
        for gen_ID in generator_IDs:
            print(msg.format(gen_ID, inertia_values[gen_ID][i]))
            out_file += f'_{inertia_values[gen_ID][i]:.3f}'
        out_file = '{}/inertia{}{}.h5'.format(output_dir, out_file, suffix)

        goto_next_iter = False
        append, force = False, False
        completed_trials = 0
        if os.path.isfile(out_file):
            fid = tables.open_file(out_file, 'r')
            try:
                params = fid.root.parameters.read()
                # make sure that the simulation is indeed the same
                if params['alpha'][0][0] != alpha or \
                    params['mu'][0][0] != mu or \
                    params['c'][0][0] != c or \
                    params['frand'][0] != frand or \
                    params['rnd_load_names'][0][0].decode('utf-8') != stochastic_load_name:
                        goto_next_iter = True
                else:
                    for child in fid.root:
                        if len(child.shape) > 1:
                            completed_trials = child.shape[0]
                            break
                    if completed_trials == N_trials:
                        goto_next_iter = True
            except:
                goto_next_iter = True
                import pdb
                pdb.set_trace()
            fid.close()

            if completed_trials == 0:
                append, force = False, True
            else:
                append, force = True, False

        if goto_next_iter:
            continue

        if not verbose:
            for j in range(completed_trials):
                sys.stdout.write('{}'.format((j+1)%10))
                if (j+1) % 50 == 0:
                    sys.stdout.write('\n')

        for j in range(completed_trials, N_trials):

            if not verbose:
                sys.stdout.write('{}'.format((j+1)%10))
                sys.stdout.flush()

            ### write configuration file
            out_config_file = output_dir + '/config' + suffix + '.json'
            json.dump(out_config, open(out_config_file, 'w'), indent=4)
            ### call run_sim
            run_sim(out_config_file, out_file, append, force, verbose)

            if not verbose and (j+1) % 50 == 0:
                sys.stdout.write('\n')
                
            append, force = True, False
            
        if not verbose and (j+1) % 50 != 0:
            sys.stdout.write('\n')

