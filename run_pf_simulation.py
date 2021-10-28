
import os
import sys
import json
import time
import argparse as arg
import tables
import numpy as np
from numpy.random import RandomState, SeedSequence, MT19937

powerfactory_path = r'C:\Program Files\DIgSILENT\PowerFactory 2020 SP4\Python\3.8'
if powerfactory_path not in sys.path:
    sys.path.append(powerfactory_path)
import powerfactory as pf

from pfcommon import * # sort_objects_by_name, OU_2, BaseParameters, get_simulation_time

progname = os.path.basename(sys.argv[0])

if __name__ == '__main__':
    parser = arg.ArgumentParser(description = 'Simulate the IEEE14 network at a fixed value of inertia', \
                                formatter_class = arg.ArgumentDefaultsHelpFormatter, \
                                prog = progname)
    parser.add_argument('config_file', type=str, action='store', help='PAN netlist')
    parser.add_argument('-o', '--output',  default=None, type=str, help='output file name')
    parser.add_argument('-f', '--force', action='store_true', help='force overwrite of output file')
    args = parser.parse_args(args=sys.argv[1:])

    app = pf.GetApplication()
    if app is None:
        raise Exception('Cannot get PowerFactory application')
    print('Successfully obtained PowerFactory application.')

    config_file = args.config_file
    if not os.path.isfile(config_file):
        print('{}: {}: no such file.'.format(progname, config_file))
        sys.exit(1)
    config = json.load(open(config_file, 'r'))

    project_name = config['project_name']
    err = app.ActivateProject(project_name)
    if err:
        raise Exception(f'Cannot activate project {project_name}')
    print(f'Successfully activated project {project_name}.')
    
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
    print('Successfully obtained folder "study".')
    
    N_blocks = len(config['tstop'])
    inertia_values = []
    for gen_id in generator_IDs:
        if len(config['inertia'][gen_id]) == 1:
            inertia_values.append([config['inertia'][gen_id][0] for _ in range(N_blocks)])
        elif len(config['inertia'][gen_id]) == N_blocks:
            inertia_values.append(config['inertia'][gen_id])
        else:
            raise Exception(f'Wrong number of inertia values for generator {gen_id}')
    inertia_values = np.array(inertia_values)

    ### activate the study case corresponding to the transient analysis
    study_case_name = config['study_case_name']
    study_case = study_project_folder.GetContents(study_case_name)[0]
    err = study_case.Activate() # don't know why this returns 1
    # if err:
    #     raise Exception(f'Cannot activate study case {study_case_name')
    print(f'Successfully activated study case {study_case_name}.')
    
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
            print(f'Found composite model named {composite_model_name}.')
            break
    if not found:
        raise Exception(f'Cannot find composite model named {composite_model_name}')
    
    for slot,net_element in zip(stochastic_load_model.pblk, stochastic_load_model.pelm):
        if slot.loc_name == 'load slot':
            net_element = stochastic_load
            print(f'Set {stochastic_load_name} as stochastic load.')
    
    stochastic_load_filename = app.GetCalcRelevantObjects('*.ElmFile')[0].f_name
    print(f'The stochastic load file is {stochastic_load_filename}.')

    try:
        rng_seed = config['seed']
    except:
        rs = RandomState(MT19937(SeedSequence(int(time.time()))))
        rng_seed = rs.randint(0, 1000000)
    rnd_state = RandomState(MT19937(SeedSequence(rng_seed)))

    # OU parameters
    alpha = config['OU']['alpha']
    mu = config['OU']['mu']
    c = config['OU']['c']

    # simulation parameters
    frand = config['frand']        # [Hz] sampling rate of the random signal
    try:
        decimation = config['decimation']
    except:
        decimation = 1
    tstop = config['tstop'][-1]    # [s]  total simulation duration
    dt = 1 / frand
    t = dt + np.r_[0 : tstop + dt/2 : dt]
    N_samples = t.size

    # generate the dynamics of the stochastic load and save it to file
    ou = OU_2(dt, alpha, mu, c, N_samples, rnd_state)
    P0 = stochastic_load.plini
    Q0 = stochastic_load.qlini
    tPQ = np.zeros((N_samples,3))
    tPQ[:,0] = t
    tPQ[:,1] = P0 + ou
    tPQ[:,2] = Q0
    with open(stochastic_load_filename, 'w') as fid:
        fid.write('2\n\n')
        for row in tPQ:
            fid.write(f'{row[0]:.6f}\t{row[1]:.2f}\t{row[2]:.2f}\n\n')

    N_random_loads = 1

    class Parameters (BaseParameters):
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
        inertia        = tables.Float64Col(shape=(N_generators,N_blocks))
        alpha          = tables.Float64Col(shape=(N_random_loads,))
        mu             = tables.Float64Col(shape=(N_random_loads,))
        c              = tables.Float64Col(shape=(N_random_loads,))
        tstop          = tables.Float64Col(shape=(N_blocks,))

    if args.output is None:
        #import subprocess
        #name_max = int(subprocess.check_output('getconf NAME_MAX /', shell=True))
        # TODO: fix this
        name_max = 256
        output_file = project_name.split('\\')[-1].replace(' ', '_').replace('/','_') + '_' + \
            '_'.join(['-'.join(map(lambda h: f'{h:.3f}', H)) for H in inertia_values])
        if len(output_file) > name_max:
            output_file = project_name.split('\\')[-1].replace(' ', '_').replace('/','_') + '_' + \
                '_'.join(['-'.join(map(lambda h: f'{h:.3f}', np.unique(H))) for H in inertia_values])
        output_file += '.h5'
    else:
        output_file = args.output
    
    if os.path.isfile(output_file) and not args.force:
        print('{}: {}: file exists: use -f to overwrite.'.format(progname, output_file))
        sys.exit(2)

    fid = tables.open_file(output_file, 'w', filters=tables.Filters(complib='zlib', complevel=5))
    tbl = fid.create_table(fid.root, 'parameters', Parameters, 'parameters')
    params = tbl.row
    params['rng_seeds']      = [rng_seed]
    params['tstop']          = config['tstop']
    params['alpha']          = [alpha]
    params['mu']             = [mu]
    params['c']              = [c]
    params['frand']          = frand
    params['F0']             = nominal_frequency
    params['inertia']        = inertia_values
    params['generator_IDs']  = generator_IDs
    params['bus_IDs']        = bus_IDs
    params['line_IDs']       = line_IDs
    params['load_IDs']       = load_IDs
    params['V_rating_buses'] = [Vrating['buses'][ID] for ID in bus_IDs]
    params['V_rating_lines'] = [Vrating['lines'][ID] for ID in line_IDs]
    params['P_rating_loads'] = [Prating['loads']['P'][ID] for ID in load_IDs]
    params['Q_rating_loads'] = [Prating['loads']['Q'][ID] for ID in load_IDs]
    params['rnd_load_names'] = [stochastic_load_name]
    params.append()
    tbl.flush()
    
    atom = tables.Float64Atom()

    if 'save_OU' in config and config['save_OU']:
        fid.create_array(fid.root, 'OU', np.array(ou[::decimation]), atom=atom)

    ### compute the initial condition of the simulation
    inc = app.GetFromStudyCase('ComInc')
    inc.iopt_sim = 'rms'
    inc.iopt_coiref = 2
    inc.tstart = 0
    inc.dtgrd = dt * 1e3
    err = inc.Execute()
    if err:
        raise Exception('Cannot compute initial condition')
    print('Successfully computed initial condition.')

    ### run the transient simulation
    sim = app.GetFromStudyCase('ComSim')
    generator_types = {gen.loc_name: gen.GetAttribute('typ_id') for gen in generators}
    
    for i, tstop in enumerate(config['tstop']):

        for generator in generators:
            name = generator.loc_name
            j = generator_IDs.index(name)
            generator_types[name].h = inertia_values[j,i]
            print(f'Setting inertia of generator {name} to {inertia_values[j,i]:g} s.')

        sys.stdout.write(f'Running simulation until t = {tstop} s... ')
        sys.stdout.flush()
        sim.tstop = tstop
        err = sim.Execute()
        if err:
            fid.close()
            os.remove(output_file)
            raise Exception('Error while running transient simulation')
        sys.stdout.write('done.\n')

    res.Load()

    ### save the simulation data to file
    vars_map = config['vars_map']
    sys.stdout.write('Reading time... ')
    sys.stdout.flush()
    time = get_simulation_time(res, decimation=decimation)
    fid.create_array(fid.root, vars_map['time'], time, atom=atom)
    sys.stdout.write('done.\n')

    correct_voltages = False
    if 'correct_Vd_Vq' in config and config['correct_Vd_Vq']:
        try:
            for delta_ref_entry in vars_map['generators']:
                vars_in = delta_ref_entry['vars_in']
                if 'c:fi' in vars_in:
                    elem = find_element_by_name(generators, delta_ref_entry['name'])
                    delta_ref = get_simulation_variables(res, 'c:fi', [elem],
                                                         decimation=decimation)
                    vars_out = delta_ref_entry['vars_out']
                    delta_ref_var_out = vars_out[vars_in.index('c:fi')]
                    fid.create_array(fid.root, delta_ref_var_out, delta_ref, atom=atom)
                    correct_voltages = True
                    print(f'{delta_ref_entry["name"]} is the reference generator.')
                    break
        except:
            pass
        if not correct_voltages:
            print('Cannot correct Vd and Vq because no generator delta is specified as reference.')

    elements_map = {'generators': generators, 'buses': buses, 'loads': loads, 'lines': lines}
    for key in config['vars_map']:
        if key == 'time':
            continue
        try:
            elements = elements_map[key]
        except:
            print(f'Unknown element name "{key}".')
            continue
        
        for req in config['vars_map'][key]:
            found = False
            Vd, Vq = None, None
            for elem in elements:
                if elem.loc_name == req['name']:
                    found = True
                    break
            if not found:
                print(f'Cannot find an element named {req["name"]} among the elements of type "{key}".')
                continue
            for var_in,var_out in zip(req['vars_in'], req['vars_out']):
                if correct_voltages and var_in == 'c:fi' and req['name'] == delta_ref_entry['name']:
                    continue
                sys.stdout.write(f'Reading {var_in} from {req["name"]}... ')
                sys.stdout.flush()
                x = get_simulation_variables(res, var_in, elements=[elem], decimation=decimation)
                if correct_voltages and var_in in ('m:ur', 'm:ui'):
                    if var_in == 'm:ur':
                        Vd = x
                        Vd_var_out = var_out
                    elif var_in == 'm:ui':
                        Vq = x
                        Vq_var_out = var_out
                    if Vd is not None and Vq is not None:
                        Vd, Vq = correct_Vd_Vq(Vd, Vq, delta_ref)
                        if config['use_physical_units']:
                            Vd *= Vrating[key][req['name']]
                            Vq *= Vrating[key][req['name']]
                        fid.create_array(fid.root, Vd_var_out, Vd, atom=atom)
                        fid.create_array(fid.root, Vq_var_out, Vq, atom=atom)
                        Vd, Vq = None, None
                else:
                    if config['use_physical_units']:
                        if is_voltage(var_in):
                            x *= Vrating[key][req['name']]
                        elif is_frequency(var_in):
                            x *= nominal_frequency
                    fid.create_array(fid.root, var_out, x, atom=atom)
                sys.stdout.write('done.\n')

    fid.close()

