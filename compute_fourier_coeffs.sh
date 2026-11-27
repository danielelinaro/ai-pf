#!/bin/bash

progname=$(basename $0)
usage() {
    echo "$progname [-h] [-N <# of tones>] [-F <base frequency>] [-T <# of periods>] [-A <multitone algo>] [-n] [-f]"
}

basedir="data/Sardinia/SM_configs_from_data/multiple_GFM_multitone_opt/Genstat_01"
network_name="Sardegna_2021_06_03cr"

algo="friese"
N_tones=50
F0=0.01
N_periods=1
dP=0.0004
dt=0.0005
force=0
dry_run=0

while getopts "hA:N:F:T:fn" arg ; do
    case "${arg}" in
	h)
	    usage
	    exit 0
	    ;;
	n)
	    dry_run=1
	    ;;
	A)
	    algo=$(echo "${OPTARG}" | tr '[:upper:]' '[:lower:]')
	    if [ "$algo" != "friese" -a "$algo" != "boyd" ] ; then
		echo "Unknown multitone algorithm '${OPTARG}'."
		exit 1
	    fi
	    ;;
	N)
	    N_tones=${OPTARG}
	    if [ ${N_tones} -le 0 ] ; then
		echo "${progname}: number of tones must be > 0."
		exit 1
	    fi
	    ;;
	F)
	    F0=${OPTARG}
	    if (( $(echo "$F0 <= 0" | bc -l) )) ; then
		echo "${progname}: base frequency must be > 0."
		exit 1
	    fi
	    ;;
	T)
	    N_periods=${OPTARG}
	    if [ ${N_periods} -le 0 ] ; then
		echo "${progname}: number of periods must be > 0."
		exit 1
	    fi
	    ;;
	f)
	    force=1
	    ;;
	*)
	    usage
	    exit 1
	    ;;
    esac
done

for config in ${basedir}/* ; do
    for d in ${config}/Ta_* ; do
	infile="${d}/${network_name}_tran_multitone_${algo}_N_${N_tones}_F0_${F0}_Hz_dP_${dP}_dt_${dt}.npz"
	if [ -f $infile ] ; then
	    outfile="${infile%.npz}_fourier_coeffs.npz"
	    if [ $force -eq 1 -o ! -f $outfile ] ; then
		if [ ${dry_run} -eq 1 ] ; then
		    echo "$infile"
		else
		    python3 compute_fourier_coeffs.py -f --f0 $F0 --N-tones ${N_tones} --N-periods ${N_periods} --save-mat $infile
		fi
	    fi
	fi
    done
done
