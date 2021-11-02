
import os
import sys
import json
import glob
import time
import shutil
import argparse as arg
import numpy as np
from numpy.random import RandomState, SeedSequence, MT19937
import tables

powerfactory_path = r'C:\Program Files\DIgSILENT\PowerFactory 2020 SP4\Python\3.8'
if powerfactory_path not in sys.path:
    sys.path.append(powerfactory_path)
import powerfactory as pf

from pfcommon import *

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
    
    app = pf.GetApplication()
    if app is None:
        raise Exception('Cannot get PowerFactory application')
    if verbose: print('Successfully obtained PowerFactory application.')

    config_file = args.config_file
    if not os.path.isfile(config_file):
        print(f'{progname}: {config_file}: no such file.')
        sys.exit(1)
    config = json.load(open(config_file, 'r'))
    
    project_name = config['project_name']
    err = app.ActivateProject(project_name)
    if err:
        raise Exception(f'Cannot activate project {project_name}')
    print(f'Successfully activated project "{project_name}".')
    
    grid = app.GetCalcRelevantObjects('*.ElmNet')[0]
    nominal_frequency = grid.frnom
    generators = sort_objects_by_name(app.GetCalcRelevantObjects('*.ElmSym'))
    lines = sort_objects_by_name(app.GetCalcRelevantObjects('*.ElmLne'))
    buses = sort_objects_by_name(app.GetCalcRelevantObjects('*.ElmTerm'))
    loads = sort_objects_by_name(app.GetCalcRelevantObjects('*.ElmLod'))
    generator_IDs = [gen.loc_name for gen in generators]
    bus_IDs = [bus.loc_name for bus in buses]
    line_IDs = [line.loc_name for line in lines]
    load_IDs = [load.loc_name for load in loads]
    N_generators, N_lines, N_buses, N_loads = len(generators), len(lines), len(buses), len(loads)
    print(f'The nominal frequency of the system is {nominal_frequency:g} Hz.')
    print(f'There are {N_generators} generators.')
    print(f'There are {N_lines} lines.')
    print(f'There are {N_buses} buses.')
    print(f'There are {N_loads} loads.')

    Vrating = {'buses': {}, 'lines': {}}
    Prating = {'loads': {'P': {}, 'Q': {}}}
    for bus in buses:
        Vrating['buses'][bus.loc_name] = bus.uknom
    for line in lines:
        Vrating['lines'][line.loc_name] = line.typ_id.uline
    for load in loads:
        Prating['loads']['P'][load.loc_name] = load.plini
        Prating['loads']['Q'][load.loc_name] = load.qlini

    study_project_folder = app.GetProjectFolder('study')
    if study_project_folder is None:
        raise Exception('No folder named "study" present')
    if verbose: print('Successfully obtained folder "study".')

    ### activate the study case corresponding to the transient analysis
    study_case_name = config['study_case_name']
    study_case = study_project_folder.GetContents(study_case_name)[0]
    err = study_case.Activate() # don't know why this returns 1
    # if err:
    #     raise Exception(f'Cannot activate study case {study_case_name')
    print(f'Successfully activated study case "{study_case_name}".')
    
    ### tell PowerFactory which variables should be saved to its internal file
    elements_map = {'generators': '*.ElmSym', 'loads': '*.ElmLod',
                    'buses': '*.ElmTerm', 'lines': '*.ElmLne'}
    monitored_variables = {}
    for k,v in elements_map.items():
        try:
            var_names = []
            for req in config['vars_map'][k]:
                for var_in in req['vars_in']:
                    if var_in not in var_names:
                        var_names.append(var_in)
            monitored_variables[v] = var_names
        except:
            pass

    # the results of the transient simulation will be stored in this variable
    res = app.GetFromStudyCase('*.ElmRes')
    for elements,var_names in monitored_variables.items():
        for element in app.GetCalcRelevantObjects(elements):
            for var_name in var_names:
                res.AddVariable(element, var_name)

    ### find the load that should be stochastic
    stochastic_load_name = config['random_load_name']
    found = False
    for load in loads:
        if load.loc_name == stochastic_load_name:
            stochastic_load = load
            found = True
            print(f'Found load named {stochastic_load_name}.')
            break
    if not found:
        raise Exception(f'Cannot find load named {stochastic_load_name}')
    
    composite_model_name = 'Stochastic Load'
    found = False
    for composite_model in app.GetCalcRelevantObjects('*.ElmComp'):
        if composite_model.loc_name == composite_model_name:
            stochastic_load_model = composite_model
            found = True
            if verbose: print(f'Found composite model named {composite_model_name}.')
            break
    if not found:
        raise Exception(f'Cannot find composite model named {composite_model_name}')
    
    for slot,net_element in zip(stochastic_load_model.pblk, stochastic_load_model.pelm):
        if slot.loc_name == 'load slot':
            net_element = stochastic_load
            if verbose: print(f'Set {stochastic_load_name} as stochastic load.')
    
    stochastic_load_filename = app.GetCalcRelevantObjects('*.ElmFile')[0].f_name
    print(f'The stochastic load file is {stochastic_load_filename}.')

    # OU parameters
    alpha = config['OU']['alpha']
    mu = config['OU']['mu']
    c = config['OU']['c']

    # simulation parameters
    frand = config['frand']  # [Hz] sampling rate of the random signal
    if args.dur is not None:
        tstop = args.dur     # [s]  simulation duration
    else:
        tstop = config['dur']
    try:
        decimation = config['decimation']
    except:
        decimation = 1

    # inertia values
    try:
        inertia_mode = config['inertia_mode']
    except:
        inertia_mode = 'combinatorial'
    if inertia_mode == 'combinatorial':
        inertia = config['inertia']
        inertia_values = []
        for gen_id in generator_IDs:
            inertia_values.append(inertia[gen_id])
        H = np.meshgrid(*inertia_values)
        inertia_values = {}
        for i,gen_id in enumerate(generator_IDs):
            inertia_values[gen_id] = H[i].flatten()
        N_inertia = inertia_values[generator_IDs[0]].size
    elif inertia_mode == 'sequential':
        inertia_values = config['inertia'].copy()
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
        N_trials = config['Ntrials']

    N_random_loads = 1
    # seeds for the random number generators
    seed = int(time.time())
    rs = RandomState(MT19937(SeedSequence(seed)))
    # these are not HW random seeds, but we use this variable name so that
    # it matches the one in the original script that works on UNIX platforms
    hw_seeds = [rs.randint(0, 1000000) for _ in range(N_random_loads)]
    random_states = [RandomState(MT19937(SeedSequence(seed))) for seed in hw_seeds]
    seeds = [rs.randint(low=0, high=1000000, size=(N_inertia, N_trials)) for rs in random_states]

    dt = 1 / frand
    t = dt + np.r_[0 : tstop + dt/2 : dt]
    N_samples = t.size

    if args.output_dir is None:
        output_dir = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    else:
        output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if not os.path.isfile(output_dir + '/' + os.path.basename(config_file)):
        config['dur'] = tstop
        config['Ntrials'] = N_trials
        json.dump(config, open(output_dir + '/' + os.path.basename(config_file), 'w'), indent=4)

    suffix = args.suffix
    if suffix != '':
        if suffix[0] != '_':
            suffix = '_' + suffix
        if suffix[-1] == '_':
            suffix = suffix[:-1]

    compression_filter = tables.Filters(complib='zlib', complevel=5)
    atom = tables.Float64Atom()

    class Parameters (BaseParameters):
        hw_seeds       = tables.Int64Col(shape=(N_random_loads,))
        seeds          = tables.Int64Col(shape=(N_random_loads,N_trials))
        count          = tables.Int64Col()
        decimation     = tables.Int64Col()
        generator_IDs  = tables.StringCol(32, shape=(N_generators,))
        bus_IDs        = tables.StringCol(32, shape=(N_buses,))
        line_IDs       = tables.StringCol(32, shape=(N_lines,))
        load_IDs       = tables.StringCol(32, shape=(N_loads,))
        V_rating_buses = tables.Float64Col(shape=(N_buses,))
        V_rating_lines = tables.Float64Col(shape=(N_lines,))
        P_rating_loads = tables.Float64Col(shape=(N_loads,))
        Q_rating_loads = tables.Float64Col(shape=(N_loads,))
        rnd_load_names = tables.StringCol(32, shape=(N_random_loads,))
        rng_seeds      = tables.Int64Col(shape=(N_random_loads,))
        inertia        = tables.Float64Col(shape=(N_generators,))
        alpha          = tables.Float64Col(shape=(N_random_loads,))
        mu             = tables.Float64Col(shape=(N_random_loads,))
        c              = tables.Float64Col(shape=(N_random_loads,))

    generator_types = {gen.loc_name: gen.typ_id for gen in generators}

    # the time series describing the stochastic load: tPQ[:,1] is set inside
    # the for loops
    P0 = stochastic_load.plini
    Q0 = stochastic_load.qlini
    tPQ = np.zeros((N_samples,3))
    tPQ[:,0] = t
    tPQ[:,2] = Q0

    N_samples_decimated = len(range(N_samples)[::decimation])
    elements_map = {'generators': generators, 'buses': buses,
                    'loads': loads, 'lines': lines}
    vars_map = config['vars_map']
    random_load_buses = [int(stochastic_load_name.split(' ')[1])]

    for i in range(N_inertia):

        out_file = ''
        # change generators' inertia values
        msg = 'Setting inertia of generator "{}" to {:5.3f} s.'
        msg_len = len(msg) - 9 + 5 + np.max([len(gen.loc_name) for gen in generators])
        banner = ' Inertia Value {:02d}/{:02d} '
        banner_len = len(banner) - 8
        symbol = '='
        symbols_len = (msg_len - banner_len) // 2
        if symbols_len * 2 + banner_len < msg_len:
            banner = symbol + banner
            banner_len += 1
        print(symbol * symbols_len + banner.format(i+1, N_inertia) + symbol * symbols_len)
        for generator in generators:
            name = generator.loc_name
            generator_types[name].h = inertia_values[name][i]
            print(msg.format(name, inertia_values[name][i]))
            out_file += f'_{inertia_values[name][i]:.3f}'
        out_file = '{}/inertia{}{}.h5'.format(output_dir, out_file, suffix)

        goto_next_iter = False
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
                    params['decimation'][0] != decimation or \
                    params['rnd_load_names'][0][0].decode('utf-8') != stochastic_load_name:
                        goto_next_iter = True
                else:
                    for child in fid.root:
                        if len(child.shape) > 1:
                            completed_trials = child.shape[0]
                            break
                    if completed_trials == N_trials:
                        goto_next_iter = True
                    else:
                        # forget about this...
                        #hw_seeds = [s for s in params['hw_seeds'][0]]
                        # ... these are the only seeds that matter
                        for j in range(N_random_loads):
                            seeds[j][i,:] = params['seeds'][0][j,:]
            except:
                goto_next_iter = True
            fid.close()

        if goto_next_iter:
            continue
        
        if completed_trials == 0:
            ### write to file all the data and parameters that we already have
            fid = tables.open_file(out_file, 'w', filters=compression_filter)
            tbl = fid.create_table(fid.root, 'parameters', Parameters, 'parameters')
            params = tbl.row
            params['hw_seeds']       = hw_seeds
            params['seeds']          = np.array([s[i,:] for s in seeds])
            params['count']          = i
            params['alpha']          = alpha
            params['mu']             = mu
            params['c']              = c
            params['frand']          = frand
            params['F0']             = nominal_frequency
            params['decimation']     = decimation
            params['rnd_load_names'] = [stochastic_load_name]
            params['generator_IDs']  = generator_IDs
            params['bus_IDs']        = bus_IDs
            params['line_IDs']       = line_IDs
            params['load_IDs']       = load_IDs
            params['V_rating_buses'] = [Vrating['buses'][ID] for ID in bus_IDs]
            params['V_rating_lines'] = [Vrating['lines'][ID] for ID in line_IDs]
            params['P_rating_loads'] = [Prating['loads']['P'][ID] for ID in load_IDs]
            params['Q_rating_loads'] = [Prating['loads']['Q'][ID] for ID in load_IDs]
            params['inertia']        = [inertia_values[gen_id][i] for gen_id in generator_IDs]
            params.append()
            tbl.flush()

            for key in vars_map:
                if key == 'time':
                    continue
                if key not in elements_map:
                    raise Exception(f'Unknown element name "{key}"')
                for req in vars_map[key]:
                    for var_out in req['vars_out']:
                        fid.create_earray(fid.root, var_out, atom,
                                          (0, N_samples_decimated))
            for bus in random_load_buses:
                fid.create_earray(fid.root, f'noise_bus_{bus}', atom, (0, N_samples_decimated))
            # close the file so that other programs can read it
            fid.close()

        correct_voltages = False
        if 'correct_Vd_Vq' in config and config['correct_Vd_Vq']:
            try:
                for delta_ref_entry in vars_map['generators']:
                    vars_in = delta_ref_entry['vars_in']
                    if 'c:fi' in vars_in:
                        vars_out = delta_ref_entry['vars_out']
                        delta_ref_var_out = vars_out[vars_in.index('c:fi')]
                        correct_voltages = True
                        if verbose: print(f'{delta_ref_entry["name"]} is the reference generator.')
                        break
            except:
                pass
            if not correct_voltages:
                print('Cannot correct Vd and Vq because no generator delta is specified as reference.')
            
        if not verbose:
            for j in range(completed_trials):
                sys.stdout.write('{}'.format((j+1)%10))
                if (j+1) % 50 == 0:
                    sys.stdout.write('\n')

        for j in range(completed_trials, N_trials):

            if not verbose:
                sys.stdout.write('{}'.format((j+1)%10))
                sys.stdout.flush()

            # generate the dynamics of the stochastic load and save it to file
            rs = RandomState(MT19937(SeedSequence(seeds[0][i,j])))
            ou = OU_2(dt, alpha, mu, c, N_samples, rs)
            tPQ[:,1] = P0 + ou
            with open(stochastic_load_filename, 'w') as fid:
                fid.write('2\n\n')
                for row in tPQ:
                    fid.write(f'{row[0]:.6f}\t{row[1]:.2f}\t{row[2]:.2f}\n\n')

            # run a transient analysis
            ### compute the initial condition of the simulation
            inc = app.GetFromStudyCase('ComInc')
            inc.iopt_sim = 'rms'
            inc.iopt_coiref = 2
            inc.tstart = 0
            inc.dtgrd = dt * 1e3
            err = inc.Execute()
            if err:
                raise Exception('Cannot compute initial condition')
            if verbose: print('Successfully computed initial condition.')
        
            ### run the transient simulation
            sim = app.GetFromStudyCase('ComSim')

            if verbose:
                sys.stdout.write(f'Running simulation until t = {tstop} s... ')
                sys.stdout.flush()
            sim.tstop = tstop
            err = sim.Execute()
            if err:
                fid.close()
                os.remove(out_file)
                raise Exception('Error while running transient simulation')
            if verbose:
                sys.stdout.write('done.\n')
    
            res.Load()

            fid = tables.open_file(out_file, 'a')

            for bus in random_load_buses:
                fid.root[f'noise_bus_{bus}'].append(tPQ[np.newaxis,::decimation,1])

            if j == 0:
                time = get_simulation_time(res, decimation=decimation)
                fid.create_array(fid.root, config['vars_map']['time'], time, atom=atom)

            if correct_voltages:
                elem = find_element_by_name(generators, delta_ref_entry['name'])
                if elem is not None:
                    # delta_ref is measured in degrees
                    delta_ref = get_simulation_variables(res, 'c:fi', [elem],
                                                         decimation=decimation)
                    fid.root[delta_ref_var_out].append(delta_ref[np.newaxis,:])
                for bus_entry in vars_map['buses']:
                    if 'm:ur' in bus_entry['vars_in'] and 'm:ui' in bus_entry['vars_in']:
                        bus = find_element_by_name(buses, bus_entry['name'])
                        if bus is not None:
                            Vd = get_simulation_variables(res, 'm:ur',
                                                          elements=[bus],
                                                          decimation=decimation)
                            Vq = get_simulation_variables(res, 'm:ui',
                                                          elements=[bus],
                                                          decimation=decimation)
                            Vd, Vq = correct_Vd_Vq(Vd, Vq, delta_ref)
                            if config['use_physical_units']:
                                Vd *= Vrating['buses'][bus_entry['name']]
                                Vq *= Vrating['buses'][bus_entry['name']]
                            Vd_var_out = bus_entry['vars_out'][bus_entry['vars_in'].index('m:ur')]
                            Vq_var_out = bus_entry['vars_out'][bus_entry['vars_in'].index('m:ui')]
                            fid.root[Vd_var_out].append(Vd[np.newaxis,:])
                            fid.root[Vq_var_out].append(Vq[np.newaxis,:])

            for key in vars_map:
                if key == 'time':
                    continue
                try:
                    elements = elements_map[key]
                except:
                    print(f'Unknown element name "{key}".')
                    continue
                
                for req in config['vars_map'][key]:
                    elem = find_element_by_name(elements, req['name'])
                    if elem is None:
                        print(f'Cannot find an element named {req["name"]} among the elements of type "{key}".')
                        continue
                    for var_in,var_out in zip(req['vars_in'], req['vars_out']):
                        if correct_voltages and (
                                (req == delta_ref_entry and var_in == 'c:fi') or
                                var_in in ('m:ur','m:ui')):
                            # we have already saved these variables
                            continue
                        if verbose:
                            sys.stdout.write(f'Reading {var_in} from {req["name"]}... ')
                            sys.stdout.flush()
                        x = get_simulation_variables(res, var_in, elements=[elem], decimation=decimation)
                        if config['use_physical_units']:
                            if is_voltage(var_in):
                                x *= Vrating[key][req['name']]
                            elif is_frequency(var_in):
                                x *= nominal_frequency
                        fid.root[var_out].append(x[np.newaxis,:])
                        if verbose: sys.stdout.write('done.\n')

            fid.close()

            if not verbose and (j+1) % 50 == 0:
                sys.stdout.write('\n')
                
        if not verbose and (j+1) % 50 != 0:
            sys.stdout.write('\n')

