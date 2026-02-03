#!/usr/bin/env python
# coding: utf-8

import re
import sys
import numpy as np
import powerfactory as pf
from pfcommon import run_power_flow, print_power_flow
from pfobjects import SynchronousMachine, Load, Bus, Transformer, Line, PowerPlant

if __name__ == '__main__':
    app = pf.GetApplication()
    if app is None:
        raise Exception('Cannot get PowerFactory application')
    else:
        print('Successfully obtained PowerFactory application.')

    project_name = '\\Terna_Inerzia\\39 Bus New England System'
    err = app.ActivateProject(project_name)
    if err:
        raise Exception(f'Cannot activate project {project_name}')
    print(f'Successfully activated project {project_name}.')

    SMs = app.GetCalcRelevantObjects('*.ElmSym')
    DSLs = app.GetCalcRelevantObjects('*.ElmDsl')
    AVRs = [dsl for dsl in DSLs if re.match('^AVR', dsl.loc_name)]
    GOVs = [dsl for dsl in DSLs if re.match('^GOV', dsl.loc_name)]
    PSSs = [dsl for dsl in DSLs if re.match('^PSS', dsl.loc_name)]
    plants = [mod for mod in app.GetCalcRelevantObjects('*.ElmComp') \
              if 'Power Plant' in mod.loc_name]
    lines = app.GetCalcRelevantObjects('*.ElmLne')
    buses = app.GetCalcRelevantObjects('*.ElmTerm')
    loads = app.GetCalcRelevantObjects('*.ElmLod')
    transformers = app.GetCalcRelevantObjects('*.ElmTr2')
    n_sms, n_avrs, n_govs, n_psss, n_plants = len(SMs), len(AVRs), len(GOVs), len(PSSs), len(plants)
    n_lines, n_buses, n_loads, n_transformers = len(lines), len(buses), len(loads), len(transformers)
    print(f'There are {n_plants} power plants.')
    print(f'There are {n_sms} synchronous machines.')
    print(f'There are {n_avrs} automatic voltage regulators.')
    print(f'There are {n_govs} turbine governors.')
    print(f'There are {n_psss} power system stabilizers.')
    print(f'There are {n_lines} lines.')
    print(f'There are {n_buses} buses.')
    print(f'There are {n_loads} loads.')
    print(f'There are {n_transformers} transformers.')

    PF = run_power_flow(app)
    print_power_flow(PF)
    from scipy.io import savemat
    outfile = 'IEEE39_PF'
    np.savez_compressed(outfile + '.npz', PF)
    savemat(outfile + '.mat', PF)

    rotor_type = 1 # 0: salient pole, 1: round rotor
    # for gen in generators:
    #     if 'G 10' not in gen.loc_name:
    #         gen.typ_id.iturbo = rotor_type
    
    get_name = lambda elem: elem.name

    plants_ = sorted([PowerPlant(plant) for plant in plants], key=lambda elem: elem.sm.name)
    SMs_in_plants_names = [plant.sm.loc_name for plant in plants_]
    SMs_ = sorted([
        SynchronousMachine(sm) for sm in SMs \
            if sm.loc_name not in SMs_in_plants_names
    ], key=get_name)
    import pdb
    pdb.set_trace()
    loads_ = sorted([Load(load) for load in loads], key=get_name)
    buses_ = sorted([Bus(bus) for bus in buses], key=get_name)
    transformers_ = [Transformer(transformer) for transformer in transformers]
    lines_ = sorted([Line(line) for line in lines], key=get_name)
    
    with_power_buses = True
    if False:
        if rotor_type == 0:
            outfile += '_salient_pole'
        elif rotor_type == 1:
            outfile += '_round_rotor'
    with open(outfile + '.inc', 'w') as fid:
        for plant in plants_:
            fid.write(str(plant) + '\n\n')
        for sm in SMs_:
            fid.write(str(sm) + '\n\n')
        for load in loads_:
            fid.write(str(load) + '\n')
        fid.write('\n')
        for line in lines_:
            fid.write(str(line) + '\n')
        fid.write('\n')
        for trans in transformers_:
            fid.write(str(trans) + '\n')
        fid.write('\n')
        if with_power_buses:
            for bus in buses_:
                fid.write(str(bus) + '\n')
    
