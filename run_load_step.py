# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:49:58 2023

@author: Daniele Linaro
"""

import os
import sys
import json
from time import time as TIME
import numpy as np

import powerfactory as pf
from pfcommon import get_simulation_time, get_simulation_variables


get_in_service_objects = lambda suffix: [obj for obj in app.GetCalcRelevantObjects(suffix)
                                         if not obj.outserv]


def run_sim(load, dP, dQ, tstep, tstop, dt, record, verbosity_level=1):
    
    sim_events = app.GetFromStudyCase('IntEvt')
    load_event = sim_events.CreateObject('EvtLod', 'my_load_event')
    if load_event is None:
        raise Exception('Cannot create load event')
    load_event.p_target = load
    load_event.time = tstep
    load_event.dP = dP * 100
    load_event.dQ = dQ * 100
    
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

    ### tell PowerFactory which variables should be saved to its internal file
    # speed, electrical power, mechanical torque, electrical torque, terminal voltage
    res = app.GetFromStudyCase('*.ElmRes')
    device_names = {}
    if verbosity_level > 2: print('Adding the following quantities to the list of variables to be saved:')
    for dev_type in record:
        devices = get_in_service_objects('*.' + dev_type)
        try:
            key = record[dev_type]["devs_name"]
        except:
            key = dev_type
        device_names[key] = []
        for dev in devices:
            if record[dev_type]['names'] == "*" or dev.loc_name in record[dev_type]['names']:
                if verbosity_level > 2: sys.stdout.write(f'{dev.loc_name}:')
                for var_name in record[dev_type]['vars']:
                    res.AddVariable(dev, var_name)
                    if verbosity_level > 2: sys.stdout.write(f' {var_name}')
                device_names[key].append(dev.loc_name)
                if verbosity_level > 2: sys.stdout.write('\n')

    ### run the transient simulation
    sim = app.GetFromStudyCase('ComSim')
    sim.tstop = tstop
    if verbosity_level > 1:
        sys.stdout.write(f'Running simulation until t = {tstop:.1f} sec... ')
        sys.stdout.flush()
    t0 = TIME()
    err = sim.Execute()
    t1 = TIME()
    load_event.Delete()
    
    if err:
        raise Exception('Cannot run transient simulation')
    if verbosity_level > 1:
        sys.stdout.write(f'done in {t1-t0:.0f} sec.\n')

    ### get the data
    if verbosity_level > 1:
        sys.stdout.write('Loading data from PF internal file... ')
        sys.stdout.flush()
    t0 = TIME()
    res.Load()
    time = get_simulation_time(res)
    data = {}
    for dev_type in record:
        devices = get_in_service_objects('*.' + dev_type)
        try:
            key = record[dev_type]["devs_name"]
        except:
            key = dev_type
        data[key] = {}
        for var_name in record[dev_type]['vars']:
            data[key][var_name] = get_simulation_variables(res,
                                                           var_name,
                                                           elements=devices)
    t1 = TIME()
    if verbosity_level > 1:
        sys.stdout.write(f'done in {t1-t0:.0f} sec.\n')
        
    return time, data, device_names


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

    project_name = config['project_name']
    load_name = config['load_name']
    dP,dQ = config['dP'], config['dQ']

    if outfile is None:
        outfile = '{}_{}_{}_{}.npz'.format(project_name, load_name, dP, dQ)

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
        
    if verbosity_level > 99:
        print('{:>5s} {:30s} {:>8s} {:>8s}'.format('NUM', 'LOAD NAME', 'P [MW]', 'Q [MVAr]'))

    out_of_service_devices = []
    for dev_type,loc_names in config['out_of_service'].items():
        devices = get_in_service_objects('*.' + dev_type)
        for dev in devices:
            if dev.loc_name in loc_names:
                dev.outserv = True
                out_of_service_devices.append(dev)
                if verbosity_level > 1:
                    print(f"Turned off device `{dev.loc_name}`.")

    load = None
    for i in range(n_loads):
        if verbosity_level > 99:
            print('[{:3d}] {:30s} {:8.2f} {:8.2f}'.format(i, loads[i].loc_name,
                                                  loads[i].plini, loads[i].qlini))
        if loads[i].loc_name == load_name:
            load = loads[i]
            print('Found load named `{}`.'.format(config['load_name']))
            break
    if load is None:
        print('Cannot find a load named `{}`.'.format(config['load_name']))
        sys.exit(1)

    dt = config['dt']
    tstep = config['tstep']
    tstop = config['tstop']
    time,data,device_names = run_sim(load,  dP, dQ, tstep, tstop, dt,
                                     config['record'],
                                     verbosity_level=verbosity_level)

    for dev in out_of_service_devices:
        dev.outserv = False

    data = {'config': config,
            'time': np.array(time),
            'data': data,
            'device_names': device_names}
            # 'speed': np.array(data['s:xspeed']),
            # 'power': np.array(data['s:pgt'])}

    np.savez_compressed(outfile, **data)

    ### send an email to let the recipients know that the job has finished
    # import smtplib
    # from email.message import EmailMessage
    # username = 'danielelinaro@gmail.com'
    # password = 'inyoicyukfhlqebz'
    # recipients = ['daniele.linaro@polimi.it']
    # msg = EmailMessage()
    # with open(config_file, 'r') as fid:
    #     msg.set_content(fid.read())
    # msg['Subject'] = 'PowerFactory job finished - data saved to {}'.format(outfile)
    # msg['To'] = ', '.join(recipients)
    # msg['From'] = username
    
    # smtp_server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    # smtp_server.login(username, password)
    # smtp_server.sendmail(username, recipients, msg.as_string())
    # smtp_server.quit()
