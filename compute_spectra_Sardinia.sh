#!/bin/bash

datadir="/Users/daniele/Research/ai-pf/data/Sardinia/SM_configs_from_data/stoch_loads_no_GFM"
datafile="Sardegna_stoch_loads_no_GFM_AC.npz"
inputs="input_names_Sardinia_no_GFM_loads.json"
varnames="vars_to_save_Sardinia_small.txt"
N=50
fmin=-6.0
fmax=2.0
force=0
cores=0
maxcores=8
for d in $datadir/* ; do
    if [ ! -f ${d}/UNIQUE_SM_CONFIG ] ; then
	continue
    fi
    outfile="$d/Sardegna_stoch_loads_no_GFM_TF_${fmin}_${fmax}_${N}.npz"
    if [ ! -f $outfile -o $force -eq 1 ] ; then
	echo "$d"
	python3 compute_spectra.py \
		--force \
		--no-matrixes \
		--ref-sm Sardinia  \
		-N $N \
		--fmin $fmin \
		--fmax $fmax \
		-I $inputs \
		-V $varnames \
		-o $outfile \
		$d/$datafile >$d/spectra.log 2>&1 &
	let cores=cores+1
    fi
    if [ $cores -ge $maxcores ] ; then
	cores=0
	wait
    fi
done
wait
