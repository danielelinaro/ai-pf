# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:05:58 2023

@author: Daniele Linaro
"""

import os
import sys
import numpy as np
from tqdm import tqdm

powerfactory_path = r'C:\Program Files\DIgSILENT\PowerFactory 2020 SP4\Python\3.8'
if powerfactory_path not in sys.path:
    sys.path.append(powerfactory_path)
import powerfactory as pf

from pfcommon import get_simulation_time, get_simulation_variables


def run_sim(F, dt, n_periods, ttran, load_file):    
    T = 1/F
    tstop = ttran + n_periods * T
    dt = min(dt, T/100)
    t = np.r_[0 : tstop : dt]
    n_samples = t.size
    tPQ = np.zeros((n_samples, 3))
    P, Q = 125., 50.
    tPQ[:,0] = t
    tPQ[:,1] = P + P/100 * np.sin(2*np.pi*F*t)
    tPQ[:,2] = Q
    with open(load_file, 'w') as fid:
        fid.write('2\n\n')
        for row in tPQ:
            fid.write(f'{row[0]:.6f}\t{row[1]:.2f}\t{row[2]:.2f}\n\n')
    
    ### compute the initial condition of the simulation
    inc = app.GetFromStudyCase('ComInc')
    inc.iopt_sim = 'rms'
    inc.iopt_coiref = 2
    inc.tstart = 0
    inc.dtgrd = dt
    err = inc.Execute()
    if err:
        raise Exception('Cannot compute initial condition')
    
    ### run the transient simulation
    sim = app.GetFromStudyCase('ComSim')
    sim.tstop = ttran
    err = sim.Execute()
    if err:
        raise Exception('Cannot run transient simulation')
    
    ### tell PowerFactory which variables should be saved to its internal file
    # speed, mechanical torque, electrical torque, terminal voltage, electrical power
    var_names = 's:xspeed', #'s:xme', 's:xmt', 's:ut', 's:pgt'
    res = app.GetFromStudyCase('*.ElmRes')
    gens = app.GetCalcRelevantObjects('*.ElmSym')
    for gen in gens:
        for var_name in var_names:
            res.AddVariable(gen, var_name)
    
    ### run the transient simulation
    sim.tstop = tstop
    err = sim.Execute()
    if err:
        raise Exception('Cannot run transient simulation')

    ### get the data
    res.Load()
    time = get_simulation_time(res)
    data = {}
    for var_name in var_names:
        data[var_name] = get_simulation_variables(res, var_name, elements=gens)
        
    return time,data


progname = os.path.basename(sys.argv[0])

def usage(exit_code=None):
    _spaces_ = ' ' * len(progname)
    print(f'usage: {progname} [--f-start <F>] [--f-stop <F>] [--steps-per-decade <num>]')
    print(f'       {_spaces_} [--dt <num>] [--ttran <num>] [--n-periods <num>]')
    print(f'       {_spaces_} [-o | --outfile <filename>] [-f | --force] [project name]')
    print('')
    print('The frequency in the options `--f-start` and `--f-stop` must be a log-10 value.')
    print('If not specified, project name is "WSCC_AC".')
    if exit_code is not None:
        sys.exit(exit_code)


if __name__ == '__main__':
    
    start, stop, steps_per_decade = -3, 1, 10
    dt = 10e-3
    n_periods = 10
    ttran = 200.
    force, outfile = False, "AC.npz"

    i = 1
    n_args = len(sys.argv)
    while i < n_args:
        arg = sys.argv[i]
        if arg in ('-h','--help'):
            usage(0)
        elif arg == '--f-start':
            i += 1
            start = float(sys.argv[i])
        elif arg == '--f-stop':
            i += 1
            stop = float(sys.argv[i])
        elif arg == '--steps-per-decade':
            i += 1
            steps_per_decade = int(sys.argv[i])
            if steps_per_decade <= 0:
                print(f'{progname}: number of steps per decade must be > 0')
                sys.exit(1)
        elif arg == '--dt':
            i += 1
            dt = float(sys.argv[i])
            if dt <= 0:
                print(f'{progname}: dt must be > 0')
                sys.exit(1)
        elif arg == '--ttran':
            i += 1
            ttran = float(sys.argv[i])
            if ttran < 0:
                print(f'{progname}: transient duration  must be >= 0')
                sys.exit(1)
        elif arg == '--n-periods':
            i += 1
            n_periods = int(sys.argv[i])
            if n_periods <= 0:
                print(f'{progname}: number of periods must be > 0')
                sys.exit(1)
        elif arg in ('-f', '--force'):
            force = True
        elif arg in ('-o', '--outfile'):
            i += 1
            outfile = sys.argv[i]
        elif arg[0] == '-':
            print(f'{progname}: unknown option `{arg}`')
            sys.exit(1)
        else:
            break
        i += 1
    
    if i < n_args:
        project_name = '\\Terna_Inerzia\\' + sys.argv[i]
    else:
        project_name = '\\Terna_Inerzia\\WSCC_AC'

    if os.path.isfile(outfile) and not force:
        print(f'{progname}: output file `{outfile}` exists: use -f to overwrite.')
        sys.exit(1)

    ### Get the PowerFactory application
    app = pf.GetApplication()
    if app is None:
        print('Cannot get PowerFactory application.')
        sys.exit(2)
    print('Got PowerFactory application.')
    
    ### Activate the project
    err = app.ActivateProject(project_name)
    if err:
        print(f'Cannot activate project {project_name}.')
        sys.exit(3)
    print(f'Activated project "{project_name}".')
    
    ### Get the active project
    project = app.GetActiveProject()
    if project is None:
        print('Cannot get active project.')
        sys.exit(3)
    print('Got active project.')
    
    ### Get some info over the network
    generators = app.GetCalcRelevantObjects('*.ElmSym')
    lines = app.GetCalcRelevantObjects('*.ElmLne')
    buses = app.GetCalcRelevantObjects('*.ElmTerm')
    loads = app.GetCalcRelevantObjects('*.ElmLod')
    transformers = app.GetCalcRelevantObjects('*.ElmTr2')
    n_generators, n_lines, n_buses = len(generators), len(lines), len(buses)
    n_loads, n_transformers = len(loads), len(transformers)
    print(f'There are {n_generators} generators.')
    print(f'There are {n_lines} lines.')
    print(f'There are {n_buses} buses.')
    print(f'There are {n_loads} loads.')
    print(f'There are {n_transformers} transformers.')
    
    ### Get the correct study case
    # study_cases_proj_folder = app.GetProjectFolder('study')
    # if study_cases_proj_folder is None:
    #     print('Cannot get the study cases project folder.')
    #     sys.exit(4)
    # print('Got study cases project folder.')
    # study_case_name = '10- Sinusoidal Load'
    # study_case = study_cases_proj_folder.GetContents(study_case_name)[0]
    # err = study_case.Activate()
    # if err:
    #     print(f'Cannot activate study case {study_case_name}.')
    #     sys.exit(5)
    # print(f'Activated study case "{study_case_name}".')
    
    ### Find the load that will vary sinusoidally
    comp_models = app.GetCalcRelevantObjects('*.ElmComp')
    sin_load = None
    for mod in comp_models:
        if mod.loc_name == 'Sinusoidal Load':
            sin_load = mod
            break
    if sin_load == None:
        print('Cannot find sinusoidal load')
        sys.exit(6)
    meas_file = sin_load.pelm[0]
    meas_filepath = meas_file.f_name
    print(f'Measurement file: {meas_filepath}.')
    
    time,speed = [], []

    n = int((stop - start) * steps_per_decade)
    F = np.logspace(start, stop, n)
    
    for f in tqdm(F, ascii=True, ncols=100):
        t,data = run_sim(f, dt, n_periods, ttran, meas_filepath)
        time.append(t)
        speed.append(data['s:xspeed'])
        
    n_F = F.size
    mu = 1.
    gamma_c = np.zeros((n_F, n_generators))
    gamma_s = np.zeros((n_F, n_generators))
    for i,(f,t,spd) in enumerate(zip(F, time, speed)):
        dt = t[1] - t[0]
        idx = t > t[-1] - mu / f
        for j in range(n_generators):
            gamma_c[i,j] = f/mu*dt*np.cos(2*np.pi*f*t[np.newaxis,idx]) @ (spd[idx,j]-1)
            gamma_s[i,j] = f/mu*dt*np.sin(2*np.pi*f*t[np.newaxis,idx]) @ (spd[idx,j]-1)
            
    data = {'F': F, 'n_periods': n_periods, 'ttran': ttran, 'time': time,
            'speed': speed, 'mu': mu, 'gamma_c': gamma_c, 'gamma_s': gamma_s}

    np.savez_compressed(outfile, **data)
