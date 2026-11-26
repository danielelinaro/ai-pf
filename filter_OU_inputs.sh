#!/bin/bash

export OPENBLAS_NUM_THREADS=1
N=50
#datadir="data/Sardinia/SM_configs_from_data"
#fname="V2020_Rete_Sardegna_2021_06_03cr_AC_TF_-6.0_2.0_${N}.npz"
#config="config/filter_OU_config.json"
datadir="data/Sardinia/SM_configs_from_data/stoch_loads_no_GFM"
fname="Sardegna_stoch_loads_no_GFM_TF_-6.0_2.0_${N}.npz"
config="config/filter_OU_inputs_config.json"

datadirs=("004" "007" "017" "018" "041" "061" "062" "071" "092" "094"
	  "102" "103" "137" "140" "143" "156" "157" "164" "170" "174"
	  "184" "196" "198" "217" "224" "253" "256" "271" "273" "288"
	  "312" "314" "322" "327" "332" "333" "334")
datadirs=("001" "003" "004" "010" "012" "013" "014" "016" "022" "024"
	  "029" "030" "032" "034" "042" "045" "058" "059" "061" "063"
	  "095" "130" "137" "143" "146" "225" "227" "236" "237" "238"
	  "240" "242" "246" "247" "248" "249" "250" "269")

maxcores=40

### TRAINING SET
trainingset=1
tend=43200
if [ $trainingset -eq 1 ] ; then
    echo "====================== TRAINING SET ======================"
    ncores=0
    for dir in ${datadirs[@]} ; do
	datafile="${datadir}/${dir}/${fname}"
	echo "Using TF's from ${datafile}..."
	python3 filter_OU_inputs.py --suffix training_set --tend $tend \
		--data-file $datafile $config >${datadir}/${dir}/training_set.log 2>&1 &
	sleep 1
	let ncores=ncores+1
	if [ $ncores -ge $maxcores ] ; then
	    ncores=0
	    wait
	fi
    done
    wait
fi

### TEST AND VALIDATION SETS
echo "====================== TEST AND VALIDATION SETS ======================"
ncores=0
tend=7200
for dir in ${datadirs[@]} ; do
    datafile="${datadir}/${dir}/${fname}"
    echo "Using TF's from $datafile..."
    python3 filter_OU_inputs.py --suffix test_set --tend $tend \
	    --data-file $datafile $config >${datadir}/${dir}/test_set.log 2>&1 &
    sleep 1
    let ncores=ncores+1
    python3 filter_OU_inputs.py --suffix validation_set --tend $tend \
	    --data-file $datafile $config >${datadir}/${dir}/validation_set.log 2>&1 &
    sleep 1
    let ncores=ncores+1
    if [ $ncores -ge $maxcores ] ; then
	ncores=0
	wait
    fi
done
wait
