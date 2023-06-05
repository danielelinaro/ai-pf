# -*- coding: utf-8 -*-
"""
Created on Tue May 30 11:54:00 2023

@author: Daniele Linaro
"""

import re
import os
import sys
import json
from time import time as TIME
import numpy as np
from numpy.random import RandomState, SeedSequence, MT19937

import powerfactory as pf
from pfcommon import OU, get_simulation_time, get_simulation_variables


__all__ = ['compute_fourier_coeffs']

progname = os.path.basename(sys.argv[0])


def compute_fourier_coeffs(F, time, speed, mu=10):
    n_F = len(F)
    n_generators = speed[0].shape[1]
    gammac = np.zeros((n_F, n_generators))
    gammas = np.zeros((n_F, n_generators))
    for i,(f,t,spd) in enumerate(zip(F, time, speed)):
        dt = np.diff(t)
        dt = dt[dt >= 1e-6][0]
        idx = t > t[-1] - mu / f
        for j in range(n_generators):
            gammac[i,j] = f/mu*dt*np.cos(2*np.pi*f*t[np.newaxis,idx]) @ (spd[idx,j]-1)
            gammas[i,j] = f/mu*dt*np.sin(2*np.pi*f*t[np.newaxis,idx]) @ (spd[idx,j]-1)
    return gammac,gammas


############################################################
###                   UTILITY FUNCTIONS                  ###
############################################################


def _IC(dt, verbosity_level=0):
    ### compute the initial condition of the simulation
    inc = PF_APP.GetFromStudyCase('ComInc')
    inc.iopt_sim = 'rms'
    inc.iopt_coiref = 2
    inc.tstart = 0
    inc.dtgrd = dt    # [s]
    err = inc.Execute()
    if err:
        raise Exception('Cannot compute initial condition')
    elif verbosity_level > 1:
        print(f'Successfully computed initial condition (dt = {dt*1e3:.1f} ms).')
    return inc

def _tran(tstop, verbosity_level=0):    
    ### run the transient simulation
    sim = PF_APP.GetFromStudyCase('ComSim')
    sim.tstop = tstop
    if verbosity_level > 1:
        sys.stdout.write('Running simulation until t = {:.1f} sec... '.format(tstop))
        sys.stdout.flush()
    t0 = TIME()
    err = sim.Execute()
    t1 = TIME()
    if verbosity_level > 1:
        sys.stdout.write(f'done in {t1-t0:.0f} sec.\n')
    return sim, t1-t0, err

def _get_objects(suffix, keep_out_of_service=False):
    return [obj for obj in PF_APP.GetCalcRelevantObjects(suffix) \
            if not obj.outserv or keep_out_of_service]


def _find_object(suffix, obj_name, in_service_only=True):
    for obj in _get_objects(suffix, not in_service_only):
        if obj.loc_name == obj_name:
            return obj
    return None


def _find_in_contents(container, obj_name):
    for obj in container.GetContents():
        if obj.loc_name == obj_name:
            return obj
    return None


def _activate_project(project_name, verbosity_level=0):   
    ### Activate the project
    err = PF_APP.ActivateProject(project_name)
    if err:
        raise Exception(f'Cannot activate project {project_name}.')
    if verbosity_level > 0: print(f'Activated project "{project_name}".')
    ### Get the active project
    project = PF_APP.GetActiveProject()
    if project is None:
        raise Exception('Cannot get active project.')
    return project


def _turn_off_objects(to_turn_off, verbosity_level=0):
    out_of_service_objs = []
    for dev_type,loc_names in to_turn_off.items():
        objs = _get_objects('*.' + dev_type)
        for obj in objs:
            if obj.loc_name in loc_names:
                obj.outserv = True
                out_of_service_objs.append(obj)
                if verbosity_level > 1:
                    print(f"Turned off device `{obj.loc_name}`.")
    return out_of_service_objs


def _turn_on_objects(out_of_service_objs):
    for obj in out_of_service_objs:
        obj.outserv = False


def _set_vars_to_save(record_map, verbosity_level=0):
    ### tell PowerFactory which variables should be saved to its internal file
    # speed, electrical power, mechanical torque, electrical torque, terminal voltage
    res = PF_APP.GetFromStudyCase('*.ElmRes')
    device_names = {}
    if verbosity_level > 2: print('Adding the following quantities to the list of variables to be saved:')
    for dev_type in record_map:
        devices = _get_objects('*.' + dev_type)
        try:
            key = record_map[dev_type]['devs_name']
        except:
            key = dev_type
        device_names[key] = []
        for dev in devices:
            if record_map[dev_type]['names'] == '*' or dev.loc_name in record_map[dev_type]['names']:
                if verbosity_level > 2: sys.stdout.write(f'{dev.loc_name}:')
                for var_name in record_map[dev_type]['vars']:
                    res.AddVariable(dev, var_name)
                    if verbosity_level > 2: sys.stdout.write(f' {var_name}')
                device_names[key].append(dev.loc_name)
                if verbosity_level > 2: sys.stdout.write('\n')
    return res, device_names


def _get_attributes(record_map, verbosity_level=0):
    device_names = {}
    attributes = {}
    if verbosity_level > 2: print('Getting the following attributes:')
    for dev_type in record_map:
        devices = _get_objects('*.' + dev_type)
        try:
            key = record_map[dev_type]['devs_name']
        except:
            key = dev_type
        device_names[key] = []
        attributes[key] = {}
        for dev in devices:
            if record_map[dev_type]['names'] == '*' or dev.loc_name in record_map[dev_type]['names']:
                if verbosity_level > 2: sys.stdout.write(f'{dev.loc_name}:')
                if 'attrs' in record_map[dev_type]:
                    for attr_name in record_map[dev_type]['attrs']:
                        if attr_name not in attributes[key]:
                            attributes[key][attr_name] = []
                        if '.' in attr_name:
                            obj = dev
                            for subattr in attr_name.split('.'):
                                obj = obj.GetAttribute(subattr)
                            attributes[key][attr_name].append(obj)
                        else:
                            attributes[key][attr_name].append(dev.GetAttribute(attr_name))
                        if verbosity_level > 2: sys.stdout.write(f' {attr_name}')
                device_names[key].append(dev.loc_name)
                if verbosity_level > 2: sys.stdout.write('\n')
    return attributes, device_names


def _get_data(res, record_map, data_obj, interval=(0,None), dt=None, verbosity_level=0):
    # data_obj is a PowerFactor DataObject used to create an IntVec object
    # where the column data will be stored. If it is None, the (much slower)
    # GetValue function will be used, which gets one value at a time from the
    # ElmRes object
    if verbosity_level > 1:
        sys.stdout.write('Loading data from PF internal file... ')
        sys.stdout.flush()
    vec = data_obj.CreateObject('IntVec') if data_obj is not None else None
    t0 = TIME()
    res.Flush()
    res.Load()
    t1 = TIME()
    if verbosity_level > 1:
        sys.stdout.write(f'in memory in {t1-t0:.0f} sec... ')
        sys.stdout.flush()
    time = get_simulation_time(res, vec, interval, dt)
    t2 = TIME()
    if verbosity_level > 1:
        sys.stdout.write(f'read time in {t2-t1:.0f} sec... ')
        sys.stdout.flush()
    data = {}
    for dev_type in record_map:
        devices = _get_objects('*.' + dev_type)
        if isinstance(record_map[dev_type]['names'], list):
            devices = [dev for dev in devices if dev.loc_name in record_map[dev_type]['names']]
        try:
            key = record_map[dev_type]['devs_name']
        except:
            key = dev_type
        data[key] = {}
        for var_name in record_map[dev_type]['vars']:
            data[key][var_name] = get_simulation_variables(res, var_name, vec,
                                                           interval, dt, app=PF_APP,
                                                           elements=devices)
    res.Release()
    t3 = TIME()
    if vec is not None:
        vec.Delete()
    if verbosity_level > 1:
        sys.stdout.write(f'read vars in {t3-t2:.0f} sec (total: {t3-t0:.0f} sec).\n')
    return np.array(time), data


def _get_seed(config):
    if 'seed' in config:
        return config['seed']
    import time
    return time.time_ns() % 5061983


def _get_random_state(config):
    seed = _get_seed(config)
    rs = RandomState(MT19937(SeedSequence(seed)))
    return rs,seed


def _print_network_info():
    ### Get some info over the network
    generators = _get_objects('*.ElmSym')
    lines = _get_objects('*.ElmLne')
    buses = _get_objects('*.ElmTerm')
    loads = _get_objects('*.ElmLod')
    transformers = _get_objects('*.ElmTr2')
    n_generators, n_lines, n_buses = len(generators), len(lines), len(buses)
    n_loads, n_transformers = len(loads), len(transformers)
    print(f'There are {n_generators} generators.')
    print(f'There are {n_lines} lines.')
    print(f'There are {n_buses} buses.')
    print(f'There are {n_loads} loads.')
    print(f'There are {n_transformers} transformers.')


def _send_email(subject, content, recipients=['daniele.linaro@polimi.it']):
    import smtplib
    from email.message import EmailMessage
    if isinstance(recipients, str):
        recipients = [recipients]
    username = 'danielelinaro@gmail.com'
    password = 'inyoicyukfhlqebz'
    msg = EmailMessage()
    msg.set_content(content)
    msg['Subject'] = subject
    msg['To'] = ', '.join(recipients)
    msg['From'] = username
    smtp_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    smtp_server.login(username, password)
    smtp_server.sendmail(username, recipients, msg.as_string())
    smtp_server.quit()


############################################################
###                     CLASSES                          ###
############################################################


class TimeVaryingLoad(object):
    def __init__(self, load, app, grid, library_name, user_models_name, frame_name, outdir='.'):
        self.load = load
        self.app = app
        self.grid = grid
        self.meas_filepath = os.path.join(os.path.abspath(outdir),
                                          load.loc_name.replace(' ', '_') + '_PQ.dat')
        library = _find_in_contents(self.app.GetActiveProject(), library_name)
        if library is None:
            raise Exception('Cannot locate library')
        user_models = _find_in_contents(library, user_models_name)
        if user_models is None:
            raise Exception('Cannot locate user models')
        self.frame = _find_in_contents(user_models, frame_name)
        if self.frame is None:
            raise Exception('Cannot locate time-varying load frame')
        ld_name = load.loc_name.replace(' ', '_')
        self.meas_file = self.grid.CreateObject('ElmFile', 'meas_' + ld_name)
        self.meas_file.f_name = self.meas_filepath
        self.comp_model = self.grid.CreateObject('ElmComp', 'time_dep_' + ld_name)
        self.comp_model.typ_id = self.frame
        self.comp_model.SetAttribute("pelm", [self.meas_file, self.load])

    @classmethod
    def _write(cls, filename, tPQ, verbose=False):
        if verbose:
            sys.stdout.write(f'Writing load values to `{filename}`... ')
            sys.stdout.flush()
        with open(filename, 'w') as fid:
            fid.write('2\n')
            for row in tPQ:
                fid.write(f'{row[0]:.6f}\t{row[1]:.2f}\t{row[2]:.2f}\n')
        if verbose:
            sys.stdout.write('done.\n')

    def write_to_file(self, dt, P, Q, verbose=False):
        n_samples = P.size
        tPQ = np.zeros((n_samples, 3))
        tPQ[:,0] = dt * np.arange(n_samples)
        tPQ[:,1] = P
        tPQ[:,2] = Q
        TimeVaryingLoad._write(self.meas_filepath, tPQ, verbose)

    def clean(self, verbose=False):
        if verbose: print(f'Deleting measurement file `{self.meas_file.loc_name}`...')
        self.meas_file.Delete()
        if verbose: print(f'Deleting composite model `{self.comp_model.loc_name}`...')
        self.comp_model.Delete()


class SinusoidalLoad(TimeVaryingLoad):
    def __init__(self, load, app, grid, library_name, user_models_name, frame_name, outdir='.'):
        super().__init__(load, app, grid, library_name, user_models_name, frame_name, outdir)

    def write_to_file(self, dt, P, Q, F, n_samples, verbose=False):
        t = dt * np.arange(n_samples)
        super().write_to_file(dt, P[0] + P[1] * np.sin(2*np.pi*F*t),
                              Q[0] + Q[1] * np.sin(2*np.pi*F*t), verbose)

class NormalStochasticLoad(TimeVaryingLoad):
    def __init__(self, load, app, grid, library_name, user_models_name, frame_name, outdir='.', seed=None):
        super().__init__(load, app, grid, library_name, user_models_name, frame_name, outdir)
        self.seed = seed
        if seed is not None:
            self.rs = RandomState(MT19937(SeedSequence(seed)))
        else:
            self.rs = np.random

    def write_to_file(self, dt, P, Q, n_samples, verbose=False):
        super().write_to_file(dt,
                              P[0] + P[1] * self.rs.normal(size=n_samples),
                              Q[0] + Q[1] * self.rs.normal(size=n_samples),
                              verbose)

class OULoad(TimeVaryingLoad):
    def __init__(self, load, app, grid, library_name, user_models_name, frame_name, outdir='.', seed=None):
        super().__init__(load, app, grid, library_name, user_models_name, frame_name, outdir)
        self.seed = seed
        if seed is not None:
            self.rs = RandomState(MT19937(SeedSequence(seed)))
        else:
            self.rs = None

    def write_to_file(self, dt, P, Q, n_samples, tau, verbose=False):
        if np.isscalar(tau):
            tau = [tau, tau]
        super().write_to_file(dt,
                              OU(dt, P[0], P[1], tau[0], n_samples, random_state=self.rs),
                              OU(dt, Q[0], Q[1], tau[1], n_samples, random_state=self.rs),
                              verbose)


############################################################
###                    TRANSIENT                         ###
############################################################

def run_tran():
    
    def usage(exit_code=None):
        print(f'usage: {progname} tran [-f | --force] [-o | --outfile <filename>] [-v | --verbose <level>] config_file')
        if exit_code is not None:
            sys.exit(exit_code)
            
    force = False
    outfile = None
    verbosity_level = 1

    i = 2
    n_args = len(sys.argv)
    while i < n_args:
        arg = sys.argv[i]
        if arg in ('-h','--help'):
            usage(0)
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
        print('Arguments after configuration file name are not allowed')
        sys.exit(1)

    if not os.path.isfile(config_file):
        print('{}: {}: no such file.'.format(progname, config_file))
        sys.exit(1)
    config = json.load(open(config_file, 'r'))

    project_name = config['project_name']
    
    if outfile is None:
        outfile = '{}_tran.npz'.format(project_name)

    if os.path.isfile(outfile) and not force:
        print(f'{progname}: output file `{outfile}` exists: use -f to overwrite.')
        sys.exit(1)

    rs,seed = _get_random_state(config)
    if verbosity_level > 0: print(f'Seed: {seed}.')

    project_name = '\\Terna_Inerzia\\' + project_name
    project = _activate_project(project_name, verbosity_level)
    _print_network_info()
    out_of_service_objects = _turn_off_objects(config['out_of_service'],
                                               verbosity_level)

    def check_matches(load, patterns, limits):
        if all([re.match(pattern, load.loc_name) is None for pattern in patterns]):
            return False
        p,q = np.abs(load.plini), np.abs(load.qlini)
        if (p >= limits['P'][0] and p <= limits['P'][1]) or \
            (q >= limits['Q'][0] and q <= limits['Q'][1]):
               return True
        return False

    loads = list(filter(lambda load: check_matches(load,
                                                   config['stoch_loads'],
                                                   config['limits']),
                        _get_objects('*.ElmLod')))
    n_loads = len(loads)

    if verbosity_level > 0:
        print(f'{len(loads)} loads match the name pattern and have either P or Q within the limits.')
        if verbosity_level > 1:
            print('{:^5s} {:<30s} {:^10s} {:^10s}'.format('#', 'Name', 'P [MW]', 'Q [MVAr]'))
            print('=' * 58)
            for i,load in enumerate(loads):
                print('[{:3d}] {:30s} {:10.3f} {:10.3f}'.format(i+1, load.loc_name, load.plini, load.qlini))

    found = False
    for grid in PF_APP.GetCalcRelevantObjects('*.ElmNet'):
        if grid.loc_name == config['grid_name']:
            if verbosity_level > 0:
                print('Found grid named `{}`.'.format(config['grid_name']))
            found = True
            break
    if not found:
        print('Cannot find a grid named `{}`.'.format(config['grid_name']))
        sys.exit(1)

    seeds = rs.randint(0, 1000000, size=n_loads)
    stoch_loads = [OULoad(ld, PF_APP, grid, config['library_name'],
                          config['user_models_name'], config['frame_name'],
                          outdir='stoch_loads', seed=sd)
                   for ld,sd in zip(loads,seeds)]

    dt = config['dt']
    tstop = config['tstop']
    n_samples = int(np.ceil(tstop / dt)) + 1
    tau = [config['tau']['P'], config['tau']['Q']]
    for load,stoch_load in zip(loads, stoch_loads):
        P = load.plini, load.plini*config['sigma']['P']
        Q = load.qlini, load.qlini*config['sigma']['Q']
        stoch_load.write_to_file(dt, P, Q, n_samples, tau, verbosity_level>2)

    inc = _IC(dt, verbosity_level)
    res, _ = _set_vars_to_save(config['record'], verbosity_level)
    sim,dur,err = _tran(tstop, verbosity_level)

    for load in stoch_loads:
        load.clean(verbosity_level>2)
    _turn_on_objects(out_of_service_objects)

    interval = (0, None)
    time,data = _get_data(res, config['record'], project, interval, dt, verbosity_level)
    attributes, device_names = _get_attributes(config['record'], verbosity_level)

    blob = {'config': config,
            'seed': seed,
            'OU_seeds': seeds,
            'time': np.array(time, dtype=object),
            'data': data,
            'attributes': attributes,
            'device_names': device_names}

    np.savez_compressed(outfile, **blob)

    ### send an email to let the recipients know that the job has finished
    # with open(config_file, 'r') as fid:
    #     _send_email(subject=f'PowerFactory job finished - data saved to {outfile}',
    #                 content=fid.read())

############################################################
###                   AC ANALYSIS                        ###
############################################################


def run_AC_analysis():
    
    def usage(exit_code=None):
        print(f'usage: {progname} AC [-f | --force] [-o | --outfile <filename>] [-v | --verbose <level>] config_file')
        if exit_code is not None:
            sys.exit(exit_code)
            
    force = False
    outfile = None
    verbosity_level = 1

    i = 2
    n_args = len(sys.argv)
    while i < n_args:
        arg = sys.argv[i]
        if arg in ('-h','--help'):
            usage(0)
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
    project = _activate_project(project_name, verbosity_level)
    _print_network_info()
    out_of_service_objects = _turn_off_objects(config['out_of_service'],
                                               verbosity_level)

    load = _find_object('*.ElmLod', config['load_name'])
    if load is None:
        print('Cannot find a load named `{}`.'.format(config['load_name']))
        sys.exit(1)

    # we can't use _find_object as above because grids do not have an .outserv flag
    found = False
    for grid in PF_APP.GetCalcRelevantObjects('*.ElmNet'):
        if grid.loc_name == config['grid_name']:
            if verbosity_level > 0:
                print('Found grid named `{}`.'.format(config['grid_name']))
            found = True
            break
    if not found:
        print('Cannot find a grid named `{}`.'.format(config['grid_name']))
        sys.exit(1)

    n = int(F_stop - F_start) * steps_per_decade + 1
    F = np.logspace(F_start, F_stop, n)    
    if verbosity_level == 1:
        from tqdm import tqdm
        iter_fun = lambda x: tqdm(x, ascii=True, ncols=100)
    else:
        iter_fun = lambda x: x
        
    sin_load = SinusoidalLoad(load, PF_APP, grid,
                              config['library_name'],
                              config['user_models_name'],
                              config['frame_name'])
    dt = config['dt']
    ttran = config['ttran']
    n_periods = config['n_periods']
    P = (load.plini, config['dP'] * load.plini)
    Q = (load.qlini, 0.0)
    time,data = [], []
    for f in iter_fun(F):
        if verbosity_level > 1: print(f'Running simulation with F = {f:g} Hz.')
        T = 1/f
        tstop = ttran + n_periods * T
        dt = min(config['dt'], T/100)
        n_samples = int(tstop / dt) + 1
        sin_load.write_to_file(dt, P, Q, f, n_samples, verbosity_level>0)
        inc = _IC(dt, verbosity_level)
        sim,dur,err = _tran(ttran, verbosity_level)
        res, _ = _set_vars_to_save(config['record'], verbosity_level)
        sim,dur,err = _tran(tstop, verbosity_level)
        interval = (0, None) if config['save_transient'] else (ttran, None)
        t,d = _get_data(res, config['record'], project, interval, dt, verbosity_level)
        time.append(t)
        data.append(d)

    attributes, device_names = _get_attributes(config['record'], verbosity_level)
    sin_load.clean(verbosity_level>1)
    _turn_on_objects(out_of_service_objects)

    blob = {'config': config,
            'F': F,
            'time': np.array(time, dtype=object),
            'data': data,
            'attributes': attributes,
            'device_names': device_names}

    np.savez_compressed(outfile, **blob)

    ### send an email to let the recipients know that the job has finished
    with open(config_file, 'r') as fid:
        _send_email(subject=f'PowerFactory job finished - data saved to {outfile}',
                    content=fid.read())


############################################################
###                    LOAD STEP                         ###
############################################################

def run_load_step_sim():

    def usage(exit_code=None):
        print(f'usage: {progname} load-step [-f | --force] [-o | --outfile <filename>] [-v | --verbose <level>] config_file')
        if exit_code is not None:
            sys.exit(exit_code)

    force = False
    outfile = None
    verbosity_level = 1

    i = 2
    n_args = len(sys.argv)
    while i < n_args:
        arg = sys.argv[i]
        if arg in ('-h','--help'):
            usage(0)
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
        print('You must specify a configuration file.')
        sys.exit(1)
    elif i == n_args-1:
        config_file = sys.argv[i]
    else:
        print('Arguments after project name are not allowed.')
        sys.exit(1)

    if not os.path.isfile(config_file):
        print('{}: {}: no such file.'.format(progname, config_file))
        sys.exit(1)
    config = json.load(open(config_file, 'r'))

    if outfile is None:
        outfile = '{}_{}_{}_{}.npz'.format(config['project_name'],
                                           config['load_name'],
                                           config['dP'],
                                           config['dQ'])

    if os.path.isfile(outfile) and not force:
        print(f'{progname}: output file `{outfile}` exists: use -f to overwrite.')
        sys.exit(1)

    project_name = '\\Terna_Inerzia\\' + config['project_name']
    project = _activate_project(project_name, verbosity_level)
    _print_network_info()

    out_of_service_objects = _turn_off_objects(config['out_of_service'],
                                               verbosity_level)
    load = _find_object('*.ElmLod', config['load_name'])
    if load is None:
        print('Cannot find a load named `{}`.'.format(config['load_name']))
        sys.exit(1)

    ### create a load event
    sim_events = PF_APP.GetFromStudyCase('IntEvt')
    load_event = sim_events.CreateObject('EvtLod', 'my_load_event')
    if load_event is None:
        raise Exception('Cannot create load event')
    load_event.p_target = load
    load_event.time = config['tstep']
    load_event.dP = config['dP'] * 100
    load_event.dQ = config['dQ'] * 100
    
    inc = _IC(config['dt'], verbosity_level)
    res,_ = _set_vars_to_save(config['record'], verbosity_level)
    sim,dur,err = _tran(config['tstop'], verbosity_level)
    load_event.Delete()
    _turn_on_objects(out_of_service_objects)
    if err:
        print('Cannot run transient simulation.')
        sys.exit(1)

    attributes,device_names = _get_attributes(config['record'], verbosity_level)
    time,data = _get_data(res, config['record'], project, verbosity_level=verbosity_level)

    blob = {'config': config,
            'time': np.array(time, dtype=object),
            'data': data,
            'attributes': attributes,
            'device_names': device_names}

    np.savez_compressed(outfile, **blob)


############################################################
###                         HELP                         ###
############################################################


def help():
    if len(sys.argv) > 2 and sys.argv[2] in commands:
        cmd = sys.argv[2]
        sys.argv = [sys.argv[0], cmd, '-h']
        commands[cmd]()
    else:
        print('Usage: {} <command> [<args>]'.format(progname))
        print('')
        print('Available commands are:')
        print('   tran           Run a transient simulation (with stochastic loads)')
        print('   AC             Run a frequency scan analysis.')
        print('   load-step      Run a load step simulation.')
        print('')
        print('Type `{} help <command>` for help about a specific command.'.format(progname))


############################################################
###                         MAIN                         ###
############################################################


# all the commands currently implemented
commands = {'help': help, 'AC': run_AC_analysis, 'load-step': run_load_step_sim,
            'tran': run_tran}

if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] in ('-h','--help'):
        commands['help']()
        sys.exit(0)
    if not sys.argv[1] in commands:
        print('{}: {} is not a recognized command. See `{} --help`.'.
              format(progname, sys.argv[1], progname))
        sys.exit(1)
    ### Get the PowerFactory PF_APPlication
    global PF_APP
    PF_APP = pf.GetApplication()
    if PF_APP is None:
        print('Cannot get PowerFactory application.')
        sys.exit(1)
    commands[sys.argv[1]]()
