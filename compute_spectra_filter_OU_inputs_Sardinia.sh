#!/bin/bash

fmin=-6
fmax=2
N=100
dP=0.05
datadir="data/Sardinia/SM_configs_from_data/default"
datafile="V2020_Rete_Sardegna_2021_06_03cr_stoch_AC.npz"
infile="${datadir}/$datafile"
nloads=5
if [ $nloads -eq 1 ] ; then
    loadsfile="config/Sardinia/stoch_load.txt"
    suffix="1_load"
else
    loadsfile="config/Sardinia/stoch_loads.txt"
    suffix="${nloads}_loads"
fi
outfile="${datadir}/${datafile%.npz}_TF_${fmin}.0_${fmax}.0_${N}_${dP}_${suffix}.npz"

#python3 compute_spectra.py -N $N --fmin $fmin --fmax $fmax -o $outfile --dP $dP -L $loadsfile $infile

infile=$outfile
outfile="${datadir}/${datafile%_AC.npz}_ss_${dP}_${suffix}.h5"
config="config/Sardinia/Sardinia_filter_OU_config_${suffix}.json"
python3 filter_OU_inputs.py --data-file $infile -o $outfile --dB 10 $config
