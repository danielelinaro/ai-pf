# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:05:58 2023

@author: Daniele Linaro
"""

import os
import sys
import json
from time import time as TIME
import numpy as np

import powerfactory as pf
from pfcommon import get_simulation_time, get_simulation_variables


__all__ = ['compute_fourier_coeffs', 'SinusoidalLoad']

def compute_fourier_coeffs(F, time, speed, mu=10):
    n_F = len(F)
    n_generators = speed[0].shape[1]
    gammac = np.zeros((n_F, n_generators))
    gammas = np.zeros((n_F, n_generators))
    for i,(f,t,spd) in enumerate(zip(F, time, speed)):
        dt = np.diff(t)
        dt = dt[dt >= 1e-6][0]
        T = 1/f
        n = 1
        while n*T < t[0]:
            n += 1
        μ = 1 + np.where((n+1+np.arange(mu)) * T <= t[-1])[0][-1]
        t0,t1 = n*T, (n+μ)*T
        idx = (t>=t0) & (t<=t1)
        for j in range(n_generators):
            gammac[i,j] = f/μ*dt*np.cos(2*np.pi*f*(t[np.newaxis,idx]-t0)) @ (spd[idx,j]-1)
            gammas[i,j] = f/μ*dt*np.sin(2*np.pi*f*(t[np.newaxis,idx]-t0)) @ (spd[idx,j]-1)
    return gammac,gammas


def find_in_contents(container, name):
    for obj in container.GetContents():
        if obj.loc_name == name:
            return obj
    return None

class SinusoidalLoad (object):
    def __init__(self, load, grid, app, library_name, user_models_name, frame_name, outdir='.'):
        self.load = load
        self.grid = grid
        self.app = app
        self.meas_filepath = os.path.join(os.path.abspath(outdir),
                                          load.loc_name.replace(' ', '_') + '_PQ.dat')

        library = find_in_contents(self.app.GetActiveProject(), library_name)
        if library is None:
            raise Exception('Cannot locate library')
        user_models = find_in_contents(library, user_models_name)
        if user_models is None:
            raise Exception('Cannot locate user models')
        self.frame = find_in_contents(user_models, frame_name)
        if self.frame is None:
            raise Exception('Cannot locate time-varying load frame')

        ld_name = load.loc_name.replace(' ', '_')
        self.meas_file = self.grid.CreateObject('ElmFile', 'meas_' + ld_name)
        self.meas_file.f_name = self.meas_filepath
        self.comp_model = self.grid.CreateObject('ElmComp', 'sinusoidal_' + ld_name)
        self.comp_model.typ_id = self.frame
        self.comp_model.SetAttribute("pelm", [self.meas_file, self.load])

    def write_load_file(self, n_samples, dt, F, P, Q, verbosity_level=0):
        tPQ = np.zeros((n_samples, 3))
        tPQ[:,0] = dt * np.arange(n_samples)
        tPQ[:,1] = P[0] + P[1] * np.sin(2*np.pi*F*tPQ[:,0])
        tPQ[:,2] = Q[0] + Q[1] * np.sin(2*np.pi*F*tPQ[:,0])
        if verbosity_level > 1:
            sys.stdout.write(f'Writing sinusoidal input to file (tend = {n_samples*dt:.2f} s)... ')
            sys.stdout.flush()
        with open(self.meas_filepath, 'w') as fid:
            fid.write('2\n')
            for row in tPQ:
                fid.write(f'{row[0]:.6f}\t{row[1]:.2f}\t{row[2]:.2f}\n')
        if verbosity_level > 1: sys.stdout.write('done.\n')

    def clean(self):
        print(f'Deleting measurement file `{self.meas_file.loc_name}`...')
        self.meas_file.Delete()
        print(f'Deleting composite model `{self.comp_model.loc_name}`...')
        self.comp_model.Delete()
        

def run_sim(sin_load, F, P, Q, dt, n_periods, ttran, verbosity_level=1):
    if verbosity_level > 1: print(f'Running simulation with F = {F:g} Hz.')
    T = 1/F
    tstop = ttran + n_periods * T
    dt = min(dt, T/100)
    n_samples = int(tstop / dt) + 1
    sin_load.write_load_file(n_samples, dt, F, P, Q, verbosity_level)
    
    ### compute the initial condition of the simulation
    inc = app.GetFromStudyCase('ComInc')
    inc.iopt_sim = 'rms'
    inc.iopt_coiref = 2
    inc.tstart = 0
    inc.dtgrd = dt     # [s]
    err = inc.Execute()
    if err:
        raise Exception('Cannot compute initial condition')
    elif verbosity_level > 1:
        print(f'Successfully computed initial condition (dt = {dt*1e3:.1f} ms).')

    ### run the transient simulation
    sim = app.GetFromStudyCase('ComSim')
    sim.tstop = ttran
    if verbosity_level > 1:
        sys.stdout.write(f'Running simulation until t = {ttran:.1f} sec... ')
        sys.stdout.flush()
    t0 = TIME()
    err = sim.Execute()
    t1 = TIME()
    if err:
        raise Exception('Cannot run transient simulation')
    if verbosity_level > 1:
        sys.stdout.write(f'done in {t1-t0:.0f} sec.\n')
    
    ### tell PowerFactory which variables should be saved to its internal file
    # speed, mechanical torque, electrical torque, terminal voltage, electrical power
    var_names = 's:xspeed', #'s:xme', 's:xmt', 's:ut', 's:pgt'
    res = app.GetFromStudyCase('*.ElmRes')
    gens = [gen for gen in app.GetCalcRelevantObjects('*.ElmSym') if not gen.outserv]
    if verbosity_level > 2: print('Adding the following quantities to the list of variables to be saved:')
    for gen in gens:
        for var_name in var_names:
            res.AddVariable(gen, var_name)
            if verbosity_level > 2: sys.stdout.write(f'{gen.loc_name}:{var_name} ')
        if verbosity_level > 2: sys.stdout.write('\n')

    ### run the transient simulation
    sim.tstop = tstop
    if verbosity_level > 1:
        sys.stdout.write(f'Running simulation until t = {tstop:.1f} sec... ')
        sys.stdout.flush()
    t0 = TIME()
    err = sim.Execute()
    t1 = TIME()
    if err:
        raise Exception('Cannot run transient simulation')
    if verbosity_level > 1:
        sys.stdout.write(f'done in {t1-t0:.0f} sec.\n')

    ### get the data
    interval = (0, None) if config['save_transient'] else (ttran, None)
    if verbosity_level > 1:
        sys.stdout.write('Loading data from PF internal file... ')
        sys.stdout.flush()
    t0 = TIME()
    res.Load()
    time = get_simulation_time(res, interval=interval, dt=dt)
    data = {}
    for var_name in var_names:
        data[var_name] = get_simulation_variables(res, var_name, interval=interval,
                                                  dt=dt, elements=gens)
        if verbosity_level > 2:
            sys.stdout.write(f'{var_name} ')
            sys.stdout.flush()
    t1 = TIME()
    if verbosity_level > 1:
        sys.stdout.write(f'done in {t1-t0:.0f} sec.\n')
        
    return time,data


progname = os.path.basename(sys.argv[0])

def usage(exit_code=None):
    print(f'usage: {progname} [-f | --force] [-o | --outfile <filename>] [-v | --verbose <level>] config_file')
    if exit_code is not None:
        sys.exit(exit_code)


if __name__ == '__main__':
    
    force = False
    outfile = None
    verbosity_level = 1

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
        elif arg in ('-v', '--verbose'):
            i += 1
            verbosity_level = int(sys.argv[i])
        elif arg[0] == '-':
            print(f'{progname}: unknown option `{arg}`')
            sys.exit(1)
        else:
            break
        i += 1
    
    if i == n_args:
        print('You must specify a configuration file')
        sys.exit(1)
    elif i == n_args-1:
        config_file = sys.argv[i]
    else:
        print('Arguments after project name are not allowed')
        sys.exit(1)

    if not os.path.isfile(config_file):
        print('{}: {}: no such file.'.format(progname, config_file))
        sys.exit(1)
    config = json.load(open(config_file, 'r'))

    F_start, F_stop = config['F_log10']
    steps_per_decade = config['steps_per_decade']
    project_name = config['project_name']
    
    if outfile is None:
        outfile = '{}_{}_{}_{}.npz'.format(project_name, F_start, F_stop, steps_per_decade)

    if os.path.isfile(outfile) and not force:
        print(f'{progname}: output file `{outfile}` exists: use -f to overwrite.')
        sys.exit(1)

    project_name = '\\Terna_Inerzia\\' + project_name

    ### Get the PowerFactory application
    app = pf.GetApplication()
    if app is None:
        print('Cannot get PowerFactory application.')
        sys.exit(2)
    if verbosity_level > 0: print('Got PowerFactory application.')
    
    ### Activate the project
    err = app.ActivateProject(project_name)
    if err:
        print(f'Cannot activate project {project_name}.')
        sys.exit(3)
    if verbosity_level > 0: print(f'Activated project "{project_name}".')
    
    ### Get the active project
    project = app.GetActiveProject()
    if project is None:
        print('Cannot get active project.')
        sys.exit(3)
    if verbosity_level > 0: print('Got active project.')
    
    ### Get some info over the network
    get_in_service_objects = lambda suffix: [obj for obj in app.GetCalcRelevantObjects(suffix) if not obj.outserv]
    generators = get_in_service_objects('*.ElmSym')
    lines = get_in_service_objects('*.ElmLne')
    buses = get_in_service_objects('*.ElmTerm')
    loads = get_in_service_objects('*.ElmLod')
    transformers = get_in_service_objects('*.ElmTr2')
    n_generators, n_lines, n_buses = len(generators), len(lines), len(buses)
    n_loads, n_transformers = len(loads), len(transformers)
    if verbosity_level > 0:
        print(f'There are {n_generators} generators.')
        print(f'There are {n_lines} lines.')
        print(f'There are {n_buses} buses.')
        print(f'There are {n_loads} loads.')
        print(f'There are {n_transformers} transformers.')
        
    DSLs = get_in_service_objects('*.ElmDsl')
    PSSs = []
    for dsl in DSLs:
        if 'PSS' in dsl.typ_id.loc_name:
            PSSs.append(dsl)
    n_PSSs = len(PSSs)
    if verbosity_level > 0: print(f'There are {n_PSSs} power system stabilizers.')
    
    turn_off_PSSs = True
    
    if turn_off_PSSs:
        for pss in PSSs:
            pss.outserv = True
        if verbosity_level > 0: print('Turned OFF the power system stabilizers.')

    if verbosity_level > 99:
        print('{:>5s} {:30s} {:>8s} {:>8s}'.format('NUM', 'LOAD NAME', 'P [MW]', 'Q [MVAr]'))

    load = None
    for i in range(n_loads):
        if verbosity_level > 99:
            print('[{:3d}] {:30s} {:8.2f} {:8.2f}'.format(i, loads[i].loc_name,
                                                  loads[i].plini, loads[i].qlini))
        if loads[i].loc_name == config['load_name']:
            load = loads[i]
            print('Found load named `{}`.'.format(config['load_name']))
            break
    if load is None:
        print('Cannot find a load named `{}`.'.format(config['load_name']))
        sys.exit(1)
        
    grids = app.GetCalcRelevantObjects('*.ElmNet')
    grid = None
    for i in range(len(grids)):
        if grids[i].loc_name == config['grid_name']:
            grid = grids[i]
            print('Found grid named `{}`.'.format(config['grid_name']))
            break
    if grid is None:
        print('Cannot find a grid named `{}`.'.format(config['grid_name']))
        sys.exit(1)

    n = int(F_stop - F_start) * steps_per_decade + 1
    F = np.logspace(F_start, F_stop, n)    
    if verbosity_level == 1:
        from tqdm import tqdm
        iter_fun = lambda x: tqdm(x, ascii=True, ncols=100)
    else:
        iter_fun = lambda x: x
        
    sin_load = SinusoidalLoad(load, grid, app, config['library_name'],
                              config['user_models_name'], config['frame_name'])
    dt = config['dt']
    ttran = config['ttran']
    n_periods = config['n_periods']
    P = (load.plini, config['dP'] * load.plini)
    Q = (load.qlini, 0.0)
    time,speed = [], []
    for f in iter_fun(F):
        t,data = run_sim(sin_load, f, P, Q, dt, n_periods, ttran, verbosity_level=verbosity_level)
        time.append(t)
        speed.append(data['s:xspeed'])

    sin_load.clean()

    if turn_off_PSSs:
        for pss in PSSs:
            pss.outserv = False
        if verbosity_level > 0: print('Turned ON the power system stabilizers.')
            
    gamma_c, gamma_s = compute_fourier_coeffs(F, time, speed, n_periods)
            
    data = {'F': F, 'n_periods': n_periods, 'ttran': ttran,
            'time': np.array(time, dtype=object),
            'speed': np.array(speed, dtype=object),
            'mu': n_periods, 'gamma_c': gamma_c, 'gamma_s': gamma_s}

    np.savez_compressed(outfile, **data)

    ### send an email to let the recipients know that the job has finished
    import smtplib
    from email.message import EmailMessage
    username = 'danielelinaro@gmail.com'
    password = 'inyoicyukfhlqebz'
    recipients = ['daniele.linaro@polimi.it']
    msg = EmailMessage()
    with open(config_file, 'r') as fid:
        msg.set_content(fid.read())
    msg['Subject'] = 'PowerFactory job finished - data saved to {}'.format(outfile)
    msg['To'] = ', '.join(recipients)
    msg['From'] = username
    
    smtp_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    smtp_server.login(username, password)
    smtp_server.sendmail(username, recipients, msg.as_string())
    smtp_server.quit()
