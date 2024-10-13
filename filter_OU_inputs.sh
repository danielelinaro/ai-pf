#!/bin/bash

export OPENBLAS_NUM_THREADS=1
N=100
datadir="data/Sardinia/SM_configs_from_data"
fname="V2020_Rete_Sardegna_2021_06_03cr_AC_TF_-6.0_2.0_${N}.npz"
config="config/filter_OU_config.json"
maxcores=20

datadirs=("004" "007" "017" "018" "041" "061" "062" "071" "092" "094"
	  "102" "103" "137" "140" "143" "156" "157" "164" "170" "174"
	  "184" "196" "198" "217" "224" "253" "256" "271" "273" "288"
	  "312" "314" "322" "327" "332" "333" "334")

### TRAINING SET
echo "====================== TRAINING SETS ======================"
ncores=0
for dir in ${datadirs[*]} ; do
    datafile="$datadir/$dir/$fname"
    echo "Using TF's from $datafile..."
    python3 filter_OU_inputs.py --suffix training_set --tend 60000 --data-file $datafile $config > $datadir/$dir/training_set.log 2>&1 &
    let ncores=ncores+1
    if [ $ncores -ge $maxcores ] ; then
	wait
	ncores=0
    fi
done
wait

### TEST AND VALIDATION SETS
echo "====================== TEST AND VALIDATION SETS ======================"
ncores=0
for dir in ${datadirs[*]} ; do
    datafile="$datadir/$dir/$fname"
    echo "Using TF's from $datafile..."
    python3 filter_OU_inputs.py --suffix test_set --tend 6000 --data-file $datafile $config > $datadir/$dir/test_set.log 2>&1 &
    let ncores=ncores+1
    python3 filter_OU_inputs.py --suffix validation_set --tend 6000 --data-file $datafile $config > $datadir/$dir/validation_set.log 2>&1 &
    let ncores=ncores+1
    if [ $ncores -ge $maxcores ] ; then
	wait
	ncores=0
    fi
done
wait
