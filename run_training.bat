
@echo off

set /A start=1
set /A stop=100
set /A stoptraining=500
set /A step=1

set config=config\var_H_area_1\IEEE39_PF_config_test_set_01.json
for /l %%j in (%start%, %step%, %stop%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s test_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_validation_set_01.json
for /l %%j in (%start%, %step%, %stop%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s validation_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_training_set_01.json
for /l %%j in (%start%, %step%, %stoptraining%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s training_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_test_set_02.json
for /l %%j in (%start%, %step%, %stop%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s test_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_validation_set_02.json
for /l %%j in (%start%, %step%, %stop%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s validation_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_training_set_02.json
for /l %%j in (%start%, %step%, %stoptraining%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s training_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_test_set_03.json
for /l %%j in (%start%, %step%, %stop%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s test_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_validation_set_03.json
for /l %%j in (%start%, %step%, %stop%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s validation_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_training_set_03.json
for /l %%j in (%start%, %step%, %stoptraining%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s training_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_test_set_04.json
for /l %%j in (%start%, %step%, %stop%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s test_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_validation_set_04.json
for /l %%j in (%start%, %step%, %stop%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s validation_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_training_set_04.json
for /l %%j in (%start%, %step%, %stoptraining%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s training_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_test_set_05.json
for /l %%j in (%start%, %step%, %stop%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s test_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_validation_set_05.json
for /l %%j in (%start%, %step%, %stop%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s validation_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_training_set_05.json
for /l %%j in (%start%, %step%, %stoptraining%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s training_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_test_set_06.json
for /l %%j in (%start%, %step%, %stop%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s test_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_validation_set_06.json
for /l %%j in (%start%, %step%, %stop%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s validation_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_training_set_06.json
for /l %%j in (%start%, %step%, %stoptraining%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s training_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_test_set_07.json
for /l %%j in (%start%, %step%, %stop%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s test_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_validation_set_07.json
for /l %%j in (%start%, %step%, %stop%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s validation_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_training_set_07.json
for /l %%j in (%start%, %step%, %stoptraining%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s training_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_test_set_08.json
for /l %%j in (%start%, %step%, %stop%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s test_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_validation_set_08.json
for /l %%j in (%start%, %step%, %stop%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s validation_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_training_set_08.json
for /l %%j in (%start%, %step%, %stoptraining%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s training_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_test_set_09.json
for /l %%j in (%start%, %step%, %stop%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s test_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_validation_set_09.json
for /l %%j in (%start%, %step%, %stop%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s validation_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_training_set_09.json
for /l %%j in (%start%, %step%, %stoptraining%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s training_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_test_set_10.json
for /l %%j in (%start%, %step%, %stop%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s test_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_validation_set_10.json
for /l %%j in (%start%, %step%, %stop%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s validation_set %config%
)

set config=config\var_H_area_1\IEEE39_PF_config_training_set_10.json
for /l %%j in (%start%, %step%, %stoptraining%) do (
	python run_pf_simulation.py -a -o data\IEEE39\var_H_area_1 -p inertia -s training_set %config%
)
