#!/bin/bash

export OPENBLAS_NUM_THREADS=1
N=100
datadir="data/Sardinia/SM_configs_from_data"
fname="V2020_Rete_Sardegna_2021_06_03cr_AC_TF_-6.0_2.0_${N}.npz"
maxcores=40
ncores=0

for dir in ${datadir}/* ; do
    datafile="$dir/$fname"
    echo "Using TF's from $datafile..."
    #python3 filter_OU_inputs.py --suffix training_set --tend 60000 --data-file $datafile > $dir/training_set.log 2>&1 &
    #let ncores=ncores+1
    python3 filter_OU_inputs.py --suffix test_set --tend 6000 --data-file $datafile > $dir/test_set.log 2>&1 &
    let ncores=ncores+1
    python3 filter_OU_inputs.py --suffix validation_set --tend 6000 --data-file $datafile > $dir/validation_set.log 2>&1 &
    let ncores=ncores+1
    if [ $ncores -ge $maxcores ] ; then
	wait
	ncores=0
    fi
done
