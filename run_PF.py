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

from pfcommon import OU, get_simulation_time, get_simulation_variables, \
    run_power_flow, parse_sparse_matrix_file, parse_Amat_vars_file, \
        parse_Jacobian_vars_file


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
###                   GLOBAL VARIABLES                   ###
############################################################

# the lists TO_TURN_ON and TO_TURN_OFF contain the objects that will have
# to be turned on and off at the end of the simulation
TO_TURN_ON = []
TO_TURN_OFF = []
# HVDCs contains the loads that model the HVDC connections in the Sardinia network 
HVDCs = []
# HVDC_P contains the default values of absorbed active powert of the HVDCs
HVDC_P = {}


############################################################
###                   UTILITY FUNCTIONS                  ###
############################################################


def _IC(dt, verbose=False):
    ### compute the initial condition of the simulation
    inc = PF_APP.GetFromStudyCase('ComInc')
    inc.iopt_sim = 'rms'
    inc.iopt_coiref = 2
    inc.tstart = 0
    inc.dtgrd = dt    # [s]
    err = inc.Execute()
    if err:
        raise Exception('Cannot compute initial condition')
    elif verbose:
        print(f'Successfully computed initial condition (dt = {dt*1e3:.1f} ms).')
    return inc


def _tran(tstop, verbose=False):    
    ### run the transient simulation
    sim = PF_APP.GetFromStudyCase('ComSim')
    sim.tstop = tstop
    if verbose:
        sys.stdout.write('Running simulation until t = {:.1f} sec... '.format(tstop))
        sys.stdout.flush()
    t0 = TIME()
    err = sim.Execute()
    t1 = TIME()
    if verbose:
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


def _activate_project(project_name, verbose=False):   
    ### Activate the project
    err = PF_APP.ActivateProject(project_name)
    if err:
        raise Exception(f'Cannot activate project {project_name}.')
    if verbose: print(f'Activated project "{project_name}".')
    ### Get the active project
    project = PF_APP.GetActiveProject()
    if project is None:
        raise Exception('Cannot get active project.')
    return project


def _find_objects(to_find, verbose=False):
    all_objs = []
    for dev_type,loc_names in to_find.items():
        objs = _get_objects('*.' + dev_type)
        for obj in objs:
            if obj.loc_name in loc_names:
                all_objs.append(obj)
                if verbose:
                    print(f'Found device `{obj.loc_name}`.')
    return all_objs


def _turn_on_off_objects(objs, outserv, verbose=False):
    for obj in objs:
        if obj.HasAttribute('outserv'):
            obj.outserv = outserv
            if verbose:
                print('Turned {} device `{}`.'.format('OFF' if outserv else 'ON', obj.loc_name))
        elif verbose:
            print('Device `{}` does not have an outserv attribute.'.format(obj.loc_name))
_turn_on_objects  = lambda objs, verbose=False: _turn_on_off_objects(objs, False, verbose)
_turn_off_objects = lambda objs, verbose=False: _turn_on_off_objects(objs, True, verbose)


def _find_SMs_to_toggle(synch_mach):
    in_service, out_of_service = [], []
    # synchronous machines that change their status
    SMs = [sm for sm in PF_APP.GetCalcRelevantObjects('*.ElmSym') \
           if sm.loc_name in synch_mach.keys() and sm.outserv == synch_mach[sm.loc_name]]
    if len(SMs) > 0:
        # substations
        substations = {obj.loc_name: substation for substation in \
                       PF_APP.GetCalcRelevantObjects('*.ElmSubstat') \
                       for obj in substation.GetContents() if obj in SMs}
        for name,substation in substations.items():
            for obj in substation.GetContents():
                if obj.HasAttribute('outserv'):
                    if obj.outserv:
                        out_of_service.append(obj)
                    else:
                        in_service.append(obj)
    return in_service, out_of_service


def _apply_configuration(config, verbosity_level):
    global TO_TURN_ON, TO_TURN_OFF, HVDCs, HVDC_P
    TO_TURN_ON = [obj for obj in _find_objects(config['out_of_service'], verbosity_level>1) \
                  if not obj.outserv]
    _turn_off_objects(TO_TURN_ON, verbosity_level>2)

    in_service = []
    if 'synch_mach' in config:
        SM_dict = {}
        for k,v in config['synch_mach'].items():
            if isinstance(v, int):
                SM_dict[k] = v
            elif isinstance(v, dict):
                SM = _find_object('*.ElmSym', k)
                if SM is not None:
                    for attr,value in v.items():
                        subattrs = attr.split('.')
                        obj = SM
                        for subattr in subattrs[:-1]:
                            obj = obj.GetAttribute(subattr)
                        obj.SetAttribute(subattrs[-1], value)
            else:
                raise Exception(f'Do not know how to deal with key `{k}` in config["synch_mach"]')
        in_service,TO_TURN_OFF = _find_SMs_to_toggle(SM_dict)
    TO_TURN_ON += in_service

    # switch off the objects that are currently in service
    _turn_off_objects(in_service)
    # switch on the objects that are currently out of service, i.e., they will
    # have to be turned off at the end
    _turn_on_objects(TO_TURN_OFF)

    # Run a power flow analysis
    PF1 = run_power_flow(PF_APP)
    P_to_distribute = 0
    slacks = []
    for SG in PF1['SGs']:
        if 'slack' in SG.lower():
            P_to_distribute += PF1['SGs'][SG]['P']
            slacks.append(_find_object('*.ElmGenStat', SG))
    if verbosity_level > 0: print(f'Total power to distribute from {len(slacks)} slack generators: {P_to_distribute:.2f} MW.')
    
    # Find the loads that model the HVDC connections
    HVDCs = [ld for ld in  _get_objects('ElmLod') if ld.typ_id.loc_name == 'HVDCload']
    idx = np.argsort([hvdc.plini for hvdc in HVDCs])[::-1]
    HVDCs = [HVDCs[i] for i in idx]
    HVDC_P = {hvdc.loc_name: hvdc.plini for hvdc in HVDCs}
    for hvdc in HVDCs:
        hvdc.plini = max(0., HVDC_P[hvdc.loc_name] - P_to_distribute)
        P_to_distribute -= HVDC_P[hvdc.loc_name] - hvdc.plini
        print('{}: {:.2f} -> {:.2f} MW'.format(hvdc.loc_name,
                                               HVDC_P[hvdc.loc_name],
                                               hvdc.plini))
        if P_to_distribute <= 1:
            break

    for slack in slacks:
        slack.outserv = True
        TO_TURN_ON.append(slack)
        
    PF2 = run_power_flow(PF_APP)
    for SM in PF1['SMs']:
        try:
            if verbosity_level > 0 and \
                (np.abs(PF1['SMs'][SM]['P'] - PF2['SMs'][SM]['P']) > 0.1 or \
                 np.abs(PF1['SMs'][SM]['Q'] - PF2['SMs'][SM]['Q']) > 0.1):
                print('{}: P = {:7.2f} -> {:7.2f} MW, Q = {:7.2f} -> {:7.2f} MVAr'.\
                      format(SM,
                             PF1['SMs'][SM]['P'],
                             PF2['SMs'][SM]['P'],
                             PF1['SMs'][SM]['Q'],
                             PF2['SMs'][SM]['Q']))
        except:
            pass

    return PF1, PF2


def _restore_network_state(verbose):
    global TO_TURN_ON, TO_TURN_OFF, HVDCs, HVDC_P
    for hvdc in HVDCs:
        hvdc.plini = HVDC_P[hvdc.loc_name]
    _turn_on_objects (TO_TURN_ON,  verbose)
    _turn_off_objects(TO_TURN_OFF, verbose)


def _compute_measures(fn, verbose=False):
    # synchronous machines
    cnt = 0
    Psm, Qsm = 0, 0
    H, S, J = {}, {}, {}
    for sm in PF_APP.GetCalcRelevantObjects('*.ElmSym'):
        if not sm.outserv:
            name = sm.loc_name
            Psm += sm.pgini
            Qsm += sm.qgini
            H[name],S[name] = sm.typ_id.h, sm.typ_id.sgn
            J[name],polepairs = sm.typ_id.J, sm.typ_id.polepairs
            cnt += 1
            if verbose:
                print('[{:2d}] {}: S = {:7.1f} MVA, H = {:5.2f} s, J = {:7.0f} kgm^2, polepairs = {}{}'.
                      format(cnt, sm.loc_name, S[name], H[name], J[name], polepairs, ' [SLACK]' if sm.ip_ctrl else ''))
            #num += H*S
            #den += S
        elif sm.ip_ctrl and verbose:
            print('{}: SM SLACK OUT OF SERVICE'.format(sm.loc_name))
    # static generators
    Psg, Qsg = 0, 0
    for sg in PF_APP.GetCalcRelevantObjects('*.ElmGenStat'):
        if not sg.outserv:
            Psg += sg.pgini
            Qsg += sg.qgini
        if sg.ip_ctrl and verbose:
            print('{}: SG SLACK{}'.format(sg.loc_name, ' OUT OF SERVICE' if sg.outserv else ''))
    # loads
    Pload, Qload = 0, 0
    for load in PF_APP.GetCalcRelevantObjects('*.ElmLod'):
        if not load.outserv:
            Pload += load.plini
            Qload += load.qlini
    Etot = np.sum([H[k]*S[k] for k in H])
    Stot = np.sum(list(S.values()))
    Htot = Etot / Stot
    Mtot = 2 * Etot / fn
    if verbose:
        print('  P load: {:8.1f} MW'.format(Pload))
        print('  Q load: {:8.1f} MVAr'.format(Qload))
        print('    P SM: {:8.1f} MW'.format(Psm))
        print('    Q SM: {:8.1f} MVAr'.format(Qsm))
        print('    P SG: {:8.1f} MW'.format(Psg))
        print('    Q SG: {:8.1f} MVAr'.format(Qsg))
        print('    Stot: {:8.1f} MVA'.format(Stot))
        print(' INERTIA: {:8.1f} s.'.format(Htot))
        print('  ENERGY: {:8.1f} MJ.'.format(Etot))
        print('MOMENTUM: {:8.1f} MJ s.'.format(Mtot))
    return Htot,Etot,Mtot,Stot,H,S,J,Pload,Qload,Psm,Qsm,Psg,Qsg


def _set_vars_to_save(record_map, verbose=False):
    ### tell PowerFactory which variables should be saved to its internal file
    # speed, electrical power, mechanical torque, electrical torque, terminal voltage
    res = PF_APP.GetFromStudyCase('*.ElmRes')
    device_names = {}
    if verbose: print('Adding the following quantities to the list of variables to be saved:')
    for dev_type in record_map:
        devices = _get_objects('*.' + dev_type)
        try:
            key = record_map[dev_type]['devs_name']
        except:
            key = dev_type
        device_names[key] = []
        for dev in devices:
            if record_map[dev_type]['names'] == '*' or dev.loc_name in record_map[dev_type]['names']:
                if verbose: sys.stdout.write(f'{dev.loc_name}:')
                for var_name in record_map[dev_type]['vars']:
                    res.AddVariable(dev, var_name)
                    if verbose: sys.stdout.write(f' {var_name}')
                device_names[key].append(dev.loc_name)
                if verbose: sys.stdout.write('\n')
    return res, device_names


def _get_attributes(record_map, verbose=False):
    device_names = {}
    attributes = {}
    if verbose: print('Getting the following attributes:')
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
                if verbose: sys.stdout.write(f'{dev.loc_name}:')
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
                        if verbose: sys.stdout.write(f' {attr_name}')
                device_names[key].append(dev.loc_name)
                if verbose: sys.stdout.write('\n')
    return attributes, device_names


def _get_data(res, record_map, data_obj, interval=(0,None), dt=None, verbose=False):
    # data_obj is a PowerFactor DataObject used to create an IntVec object
    # where the column data will be stored. If it is None, the (much slower)
    # GetValue function will be used, which gets one value at a time from the
    # ElmRes object
    if verbose:
        sys.stdout.write('Loading data from PF internal file... ')
        sys.stdout.flush()
    vec = data_obj.CreateObject('IntVec') if data_obj is not None else None
    t0 = TIME()
    res.Flush()
    res.Load()
    t1 = TIME()
    if verbose:
        sys.stdout.write(f'in memory in {t1-t0:.0f} sec... ')
        sys.stdout.flush()
    time = get_simulation_time(res, vec, interval, dt)
    t2 = TIME()
    if verbose:
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
    if verbose:
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
        self.comp_model.SetAttribute('pelm', [self.meas_file, self.load])

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
        print(f'usage: {progname} tran [-f | --force] [-o | --outfile <filename>]')
        print( '       ' + ' ' * len(progname) + '      [-v | --verbose <level>] [-m | --email] config_file')
        if exit_code is not None:
            sys.exit(exit_code)
            
    force = False
    outfile = None
    verbosity_level = 1
    send_email = False

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
        elif arg in ('-m', '--email'):
            send_email = True
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
    project = _activate_project(project_name, verbosity_level>0)
    _print_network_info()
    
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

    def check_matches(load, patterns, limits, out_of_service):
        if all([re.match(pattern, load.loc_name) is None for pattern in patterns]) or \
            load.loc_name in out_of_service:
            return False
        p,q = load.plini, load.qlini
        if (p >= limits['P'][0] and p <= limits['P'][1]) or \
            (q >= limits['Q'][0] and q <= limits['Q'][1]):
               return True
        return False

    loads = list(filter(lambda load: check_matches(load,
                                                   config['stoch_loads'],
                                                   config['limits'],
                                                   config['out_of_service']['ElmLod'] \
                                                       if 'ElmLod' in config['out_of_service'] else []),
                        _get_objects('*.ElmLod')))
    n_loads = len(loads)

    if verbosity_level > 0:
        if len(loads) == 1:
            print('One load matches the name pattern and has either P or Q within the limits.')
        else:
            print(f'{len(loads)} loads match the name pattern and have either P or Q within the limits.')
        if verbosity_level > 2:
            print('{:^5s} {:<30s} {:^10s} {:^10s}'.format('#', 'Name', 'P [MW]', 'Q [MVAr]'))
            print('=' * 58)
            for i,load in enumerate(loads):
                print('[{:3d}] {:30s} {:10.3f} {:10.3f}'.format(i+1, load.loc_name, load.plini, load.qlini))

    seeds = rs.randint(0, 1000000, size=n_loads)
    stoch_loads = [OULoad(ld, PF_APP, grid, config['library_name'],
                          config['user_models_name'], config['frame_name'],
                          outdir='stoch_loads', seed=sd)
                   for ld,sd in zip(loads,seeds)]

    dt = config['dt']
    tstop = config['tstop']
    n_samples = int(np.ceil(tstop / dt)) + 1
    tau = [config['tau']['P'], config['tau']['Q']]
    for i,(load,stoch_load) in enumerate(zip(loads, stoch_loads)):
        P = load.plini, np.abs(load.plini)*config['sigma']['P']
        Q = load.qlini, np.abs(load.qlini)*config['sigma']['Q']
        msg = 'Writing load file {:d}/{:d}...'.format(i+1, n_loads)
        sys.stdout.write(msg)
        sys.stdout.flush()
        stoch_load.write_to_file(dt, P, Q, n_samples, tau, verbosity_level>2)
        if i < n_loads-1:
            sys.stdout.write('\b' * len(msg))
    sys.stdout.write('\n')
    
    PF1, PF2 = _apply_configuration(config, verbosity_level)

    Htot,Etot,Mtot,Stot,H,S,J,Pload,Qload,Psm,Qsm,Psg,Qsg = \
        _compute_measures(grid.frnom, verbosity_level>0)

    try:
        inc = _IC(dt, verbosity_level>1)
        res, _ = _set_vars_to_save(config['record'], verbosity_level>2)
        sim,dur,err = _tran(tstop, verbosity_level>1)

        interval = (0, None)
        time,data = _get_data(res, config['record'], project, interval, dt, verbosity_level>1)
        attributes, device_names = _get_attributes(config['record'], verbosity_level>2)
    
        blob = {'config': config,
                'seed': seed,
                'OU_seeds': seeds,
                'inertia': Htot,
                'energy': Etot,
                'momentum': Mtot,
                'Stot': Stot,
                'H': H, 'S': S, 'J': J,
                'Psm': Psm, 'Qsm': Qsm,
                'Psg': Psg, 'Qsg': Qsg,
                'Pload': Pload, 'Qload': Qload,
                'PF_with_slack': PF1,
                'PF_without_slack': PF2,
                'time': np.array(time, dtype=object),
                'data': data,
                'attributes': attributes,
                'device_names': device_names}
    
        np.savez_compressed(outfile, **blob)

    except:
        print('Cannot run simulation...')

    for load in stoch_loads:
        load.clean(verbosity_level>2)

    _restore_network_state(verbosity_level>2)

    if send_email:
        ### send an email to let the recipients know that the job has finished
        with open(config_file, 'r') as fid:
            _send_email(subject=f'PowerFactory job finished - data saved to {outfile}',
                        content=fid.read())


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

    try:
        outdir = config['outdir']
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
    except:
        outdir = '.'
        
    if outfile is None:
        outfile = os.path.join(outdir, config['project_name'] + '_AC.npz')
    if os.path.isfile(outfile) and not force:
        print(f'{progname}: output file `{outfile}` exists: use -f to overwrite.')
        sys.exit(1)

    project_name = '\\Terna_Inerzia\\' + config['project_name']
    project = _activate_project(project_name, verbosity_level>0)
    _print_network_info()

    # we can't use _find_object because grids do not have an .outserv flag
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

    PF1, PF2 = _apply_configuration(config, verbosity_level)
    
    Htot,Etot,Mtot,Stot,H,S,J,Pload,Qload,Psm,Qsm,Psg,Qsg = \
        _compute_measures(grid.frnom, verbosity_level>0)
    
    modal_analysis = PF_APP.GetFromStudyCase('ComMod')
    # modal_analysis.cinitMode          = 1
    modal_analysis.iSysMatsMatl       = 1
    modal_analysis.iEvalMatl          = 1
    modal_analysis.output_type        = 1
    modal_analysis.repBufferAndExtDll = 1
    modal_analysis.repConstantStates  = 1
    modal_analysis.dirMatl            = outdir
    sys.stdout.write('Running modal analysis... ')
    sys.stdout.flush()
    err = modal_analysis.Execute()

    _restore_network_state(verbosity_level>2)

    if err:
        print('ERROR!')
    else:
        sys.stdout.write('done.\nSaving data... ')
        sys.stdout.flush()
        loads = _get_objects('*.ElmLod')
        load_buses, bus_equiv_terms = {}, {}
        for i,load in enumerate(loads):
            # the bus to which the load is directly connected
            bus = load.bus1.cterm
            # list of terminals that are equivalent to bus, i.e., those terminals
            # that are only connected via closed switchs or zero-length lines
            equiv_terms = bus.GetEquivalentTerminals()
            # get connected busbars
            busbars = [bb for bb in bus.GetConnectedMainBuses() if bb in equiv_terms]
            n_busbars = len(busbars)
            if n_busbars == 0:
                load_buses[load.loc_name] = bus.loc_name
            elif n_busbars == 1:
                load_buses[load.loc_name] = busbars[0].loc_name
                # this is probably not really necessary
                equiv_terms = busbars[0].GetEquivalentTerminals()
            else:
                raise Exception(f'Cannot figure out the bus ``{load.loc_name}`` is connected to.')
            # print('[{:03d}] {} -> {}'.format(i+1,load.loc_name,load_buses[load.loc_name]))
            equiv_terms_names = sorted([term.loc_name for term in equiv_terms])
            bus_equiv_terms[load_buses[load.loc_name]] = equiv_terms_names

        A = parse_sparse_matrix_file(os.path.join(outdir, 'Amat.mtl'))
        J = parse_sparse_matrix_file(os.path.join(outdir, 'Jacobian.mtl'))
        cols,var_names,model_names = \
            parse_Amat_vars_file(os.path.join(outdir,'VariableToIdx_Amat.txt'))
        vars_idx,state_vars,voltages,currents,signals = \
            parse_Jacobian_vars_file(os.path.join(outdir,'VariableToIdx_Jacobian.txt'))
        omega_col_idx, = np.where([name == 'speed' for name in var_names])
        gen_names = [os.path.splitext(os.path.basename(model_names[i]))[0] \
                     for i in omega_col_idx]
        data = {'config': config,
                'inertia': Htot,
                'energy': Etot,
                'momentum': Mtot,
                'Stot': Stot,
                'H': H, 'S': S, 'J': J,
                'Psm': Psm, 'Qsm': Qsm,
                'Psg': Psg, 'Qsg': Qsg,
                'Pload': Pload, 'Qload': Qload,
                'PF_with_slack': PF1,
                'PF_without_slack': PF2,
                'J': J, 'vars_idx': vars_idx,
                'state_vars': state_vars, 'voltages': voltages,
                'currents': currents, 'signals': signals,
                'A': A, 'var_names': var_names,
                'model_names': model_names,
                'omega_col_idx': omega_col_idx,
                'gen_names': gen_names,
                'load_buses': load_buses,
                'bus_equiv_terms': bus_equiv_terms}
        np.savez_compressed(outfile, **data)
        print('done.')


############################################################
###                 AC TRAN ANALYSIS                     ###
############################################################


def run_AC_tran_analysis():
    
    def usage(exit_code=None):
        print(f'usage: {progname} AC-tran [-f | --force] [-o | --outfile <filename>] [-v | --verbose <level>] config_file')
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
    if i == n_args-1:
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
    project = _activate_project(project_name, verbosity_level>0)
    _print_network_info()

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

    PF1, PF2 = _apply_configuration(config, verbosity_level)
    
    Htot,Etot,Mtot,Stot,H,S,J,Pload,Qload,Psm,Qsm,Psg,Qsg = \
        _compute_measures(grid.frnom, verbosity_level>0)

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
        sin_load.write_to_file(dt, P, Q, f, n_samples, verbosity_level>2)
        try:
            inc = _IC(dt, verbosity_level>1)
            sim,dur,err = _tran(ttran, verbosity_level>1)
            res, _ = _set_vars_to_save(config['record'], verbosity_level>2)
            sim,dur,err = _tran(tstop, verbosity_level>1)
            interval = (0, None) if config['save_transient'] else (ttran, None)
            t,d = _get_data(res, config['record'], project, interval, dt, verbosity_level>1)
            time.append(t)
            data.append(d)
        except:
            print('Cannot run simulation')

    if len(time) > 0:
        attributes, device_names = _get_attributes(config['record'], verbosity_level>2)
        blob = {'config': config,
                'inertia': Htot,
                'energy': Etot,
                'momentum': Mtot,
                'Stot': Stot,
                'H': H, 'S': S, 'J': J,
                'Psm': Psm, 'Qsm': Qsm,
                'Psg': Psg, 'Qsg': Qsg,
                'Pload': Pload, 'Qload': Qload,
                'PF_with_slack': PF1,
                'PF_without_slack': PF2,
                'F': F,
                'time': np.array(time, dtype=object),
                'data': data,
                'attributes': attributes,
                'device_names': device_names}    
        np.savez_compressed(outfile, **blob)

    sin_load.clean(verbosity_level>1)

    _restore_network_state(verbosity_level>2)

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
    project = _activate_project(project_name, verbosity_level>0)
    _print_network_info()

    out_of_service_objects = _turn_off_objects(config['out_of_service'],
                                               verbosity_level>2)
    load = _find_object('*.ElmLod', config['load_name'])
    if load is None:
        print('Cannot find a load named `{}`.'.format(config['load_name']))
        sys.exit(1)

    ### create a load event
    sim_events = PF_APP.GetFromStudyCase('Simulation Events/Fault.IntEvt')
    load_event = sim_events.CreateObject('EvtLod', '{}_{}'.
                                         format(load.loc_name, np.random.randint(0, 10000)))
    if load_event is None:
        raise Exception('Cannot create load event')
    load_event.p_target = load
    load_event.time = config['tstep']
    load_event.dP = config['dP'] * 100
    load_event.dQ = config['dQ'] * 100
    
    inc = _IC(config['dt'], verbosity_level>1)
    res,_ = _set_vars_to_save(config['record'], verbosity_level>2)
    sim,dur,err = _tran(config['tstop'], verbosity_level>1)
    # setting the load as out of service because for some reason the following
    # call to .Delete doesn't remove the object from the simulation events
    load_event.outserv = True
    load_event.Delete()
    _turn_on_objects(out_of_service_objects, verbosity_level>2)
    if err:
        print('Cannot run transient simulation.')
        sys.exit(1)

    attributes,device_names = _get_attributes(config['record'], verbosity_level>2)
    time,data = _get_data(res, config['record'], project, verbosity_level>1)

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
        print('   AC-tran        Run a frequency scan analysis using numerical integration.')
        print('   load-step      Run a load step simulation.')
        print('')
        print('Type `{} help <command>` for help about a specific command.'.format(progname))


############################################################
###                         MAIN                         ###
############################################################


# all the commands currently implemented
commands = {'help': help,
            'AC': run_AC_analysis,
            'AC-tran': run_AC_tran_analysis,
            'load-step': run_load_step_sim,
            'tran': run_tran}

if __name__ == '__main__':
    if len(sys.argv) == 1 or sys.argv[1] in ('-h','--help', 'help'):
        commands['help']()
        sys.exit(0)
    if not sys.argv[1] in commands:
        print('{}: {} is not a recognized command. See `{} --help`.'.
              format(progname, sys.argv[1], progname))
        sys.exit(1)
    ### Get the PowerFactory PF_APPlication
    import powerfactory as pf
    global PF_APP
    PF_APP = pf.GetApplication()
    if PF_APP is None:
        print('\nCannot get PowerFactory application.')
        sys.exit(1)
    PF_APP.ResetCalculation()
    commands[sys.argv[1]]()
