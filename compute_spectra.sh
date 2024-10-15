#!/bin/bash

export OPENBLAS_NUM_THREADS=4
N=100
refSM="CODCTI0201GGR1____GEN_____"
dP=0.1
loads="EqX_BNFC_I0601TRR_____LOAD____,EqX_MIMC_I2201TR1_____LOAD____,EqX_ALIC_I1501TR1_____LOAD____,EqX_CA4CDI1501TRV_____LOAD____"
datadir="data/Sardinia/SM_configs_from_data"
fname="V2020_Rete_Sardegna_2021_06_03cr_AC.npz"
maxcores=20
ncores=0
for dir in ${datadir}/* ; do
    if [ ! -f "$dir/${fname%.npz}_TF_-6.0_2.0_$N.npz" ] ; then
	datafile="$dir/$fname"
	echo "Processing $datafile..."
	python3 compute_spectra.py -N $N --ref-sm $refSM --P --dP $dP -L $loads $datafile > $dir/TF.log 2>&1 &
	let ncores=ncores+1
	if [ $ncores -eq $maxcores ] ; then
	    wait
	    ncores=0
	fi
    fi
done
