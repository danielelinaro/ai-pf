#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import sys
import numpy as np

powerfactory_path = r'C:\Program Files\DIgSILENT\PowerFactory 2020 SP4\Python\3.8'
if powerfactory_path not in sys.path:
    sys.path.append(powerfactory_path)
import powerfactory as pf

try:
    from pfcommon import *
except:
    sys.path.append('..')
    from pfcommon import *


# In[ ]:


app = pf.GetApplication()
if app is None:
    raise Exception('Cannot get PowerFactory application')
else:
    print('Successfully obtained PowerFactory application.')


# In[ ]:


project_name = '\\Terna_Inerzia\\IEEE39_rnd_load'
# project_name = '\\Terna_Inerzia\\IEEE39_part_rnd_load'
# project_name = '\\Terna_Inerzia\\IEEE39_minimal'
err = app.ActivateProject(project_name)
if err:
    raise Exception(f'Cannot activate project {project_name}')
print(f'Successfully activated project {project_name}.')


# In[ ]:


generators = get_objects(app, '*.ElmSym')
DSLs = get_objects(app, '*.ElmDsl')
AVRs = [dsl for dsl in DSLs if 'AVR' in dsl.loc_name]
GOVs = [dsl for dsl in DSLs if 'GOV' in dsl.loc_name]
power_plants = [comp_mod for comp_mod in get_objects(app, '*.ElmComp')
                if 'Power Plant' in comp_mod.loc_name]
lines = get_objects(app, '*.ElmLne')
buses = get_objects(app, '*.ElmTerm')
loads = get_objects(app, '*.ElmLod')
transformers = get_objects(app, '*.ElmTr2')
n_generators, n_avrs, n_govs, n_plants = len(generators), len(AVRs), len(GOVs), len(power_plants)
n_lines, n_buses, n_loads, n_transformers = len(lines), len(buses), len(loads), len(transformers)
print(f'There are {n_plants} power plants.')
print(f'There are {n_generators} generators.')
print(f'There are {n_avrs} automatic voltage regulators.')
print(f'There are {n_govs} turbine governors.')
print(f'There are {n_lines} lines.')
print(f'There are {n_buses} buses.')
print(f'There are {n_loads} loads.')
print(f'There are {n_transformers} transformers.')


# In[ ]:


rotor_type = 1 # 0: salient pole, 1: round rotor
for gen in generators:
    if 'G 10' not in gen.loc_name:
        gen.typ_id.iturbo = rotor_type


# In[ ]:


load_flow = app.GetFromStudyCase('ComLdf')
err = load_flow.Execute()

get_name = lambda elem: elem.name

powerplants = sorted([PowerPlant(plant) for plant in power_plants], key=lambda elem: elem.gen.name)
generators_in_plants_names = [plant.gen.name for plant in powerplants]
powergenerators = sorted([PowerGenerator(gen) for gen in generators 
                          if gen.loc_name not in generators_in_plants_names], key=get_name)
powerloads = sorted([PowerLoad(load) for load in loads], key=get_name)
powerbuses = sorted([PowerBus(bus) for bus in buses], key=get_name)
powertransformers = [PowerTransformer(transformer) for transformer in transformers]
powerlines = sorted([PowerLine(line) for line in lines], key=get_name)


# In[ ]:


with_power_buses = False
output_file = project_name.split('\\')[-1].lower() + '_PF'
if False:
    if rotor_type == 0:
        output_file += '_salient_pole'
    elif rotor_type == 1:
        output_file += '_round_rotor'
with open(output_file + '.inc', 'w') as fid:
    for plant in powerplants:
        fid.write(str(plant) + '\n\n')
    for gen in powergenerators:
        fid.write(str(gen) + '\n\n')
    for load in powerloads:
        fid.write(str(load) + '\n')
    fid.write('\n')
    for line in powerlines:
        fid.write(str(line) + '\n')
    fid.write('\n')
    for trans in powertransformers:
        fid.write(str(trans) + '\n')
    fid.write('\n')
    if with_power_buses:
        for bus in powerbuses:
            fid.write(str(bus) + '\n')

