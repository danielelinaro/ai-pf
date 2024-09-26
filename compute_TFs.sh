#!/bin/bash

export OPENBLAS_NUM_THREADS=4
N=100
refSM="CODCTI0201GGR1____GEN_____"
dP=0.1
load="EqX_BNFC_I0601TRR_____LOAD____"
datadir="data/Sardinia/SM_configs_from_data"
fname="V2020_Rete_Sardegna_2021_06_03cr_AC.npz"
maxcores=5
ncores=0
for dir in ${datadir}/* ; do
    datafile="$dir/$fname"
    echo "Processing $datafile..."
    python3 compute_TFs.py -N $N --ref-sm $refSM --P --dP $dP -L $load $datafile > $dir/TF.log 2>&1 &
    let ncores=ncores+1
    if [ $ncores -eq $maxcores ] ; then
	wait
	ncores=0
    fi
done
