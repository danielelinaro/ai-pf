#!/bin/bash

export OPENBLAS_NUM_THREADS=4
N=100
#refSM="CODCTI0201GGR1____GEN_____"
refSM="Sardinia"
#dP=0.1
inputs="input_names_Sardinia_multiple_GFM.json"
vars="vars_to_save_Sardinia.txt"
datadir="data/Sardinia/SM_configs_from_data/multiple_GFM_multitone_opt/Genstat_01"
fname="Sardegna_2021_06_03cr_AC.npz"
maxcores=40
ncores=0
for dir in ${datadir}/* ; do
    for Ta in 4 40 ; do
	if [ ! -f "${dir}/Ta_${Ta}/${fname%.npz}_TF_-6.0_2.0_${N}.npz" ] ; then
	    datafile="${dir}/Ta_${Ta}/${fname}"
	    echo "Processing $datafile..."
	    python3 compute_spectra.py -v -N $N --ref-sm $refSM -I $inputs -V $vars $datafile >${dir}/Ta_${Ta}/TF.log 2>&1 &
	    let ncores=ncores+1
	    sleep 1
	    if [ $ncores -eq $maxcores ] ; then
		wait
		ncores=0
	    fi
	fi
    done
done
wait

