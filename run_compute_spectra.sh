#!/bin/bash

network="simple_case_globalH"
network="WSCC"
network="IEEE39"
#network="Sardinia"
fmin=-6
fmax=2
dP=0.01

if [ "$network" = "Sardinia" ] ; then
    slack="CODCTI0201GGR1____GEN_____"
    loads="load_names_Sardinia.txt"
    vars="vars_to_save_Sardinia.txt"
    npz="V2020_Rete_Sardegna_2021_06_03cr_ORIG_AC.npz"
    N=50
    conds=(
	"default_config"
    )
    H=(2.9 7.9)
    for cond in ${conds[@]} ; do
	for h in ${H[@]} ; do
	    dir="data/${network}/${cond}/FSACTI0201GGR3_h_${h}"
	    infile="${dir}/${npz}"
	    if [ -f "$infile" ] ; then
		outfile=`echo ${infile%.npz}_TF_${fmin}.0_${fmax}.0_${N}_${dP}.npz | tr ' ' '_'`
		python3 compute_spectra.py -N $N --fmin $fmin --fmax $fmax --ref-sm "$slack" \
			--dP $dP -L $loads -V $vars -o "$outfile" --save-mat "$infile"
	    else
		echo "$infile: no such file."
	    fi
	done
    done
elif [ "$network" = "IEEE39" ] ; then
    slack="G 02"
    loads="load_names_IEEE39.txt"
    vars="vars_to_save_IEEE39.txt"
    npz="39 Bus New England System_AC.npz"
    N=100
    conds=(
	# "general_load"
	"const_P"
	"const_Z"
    )
    H=(3.771 8.771)
    for cond in ${conds[@]} ; do
	if [ "$cond" = "const_Z" ] ; then
	    exp=2
	elif [ "$cond" = "const_I" ] ; then
	    exp=1
	else
	    exp=0
	fi
	echo "Load exponent: ${exp}."
	for h in ${H[@]} ; do
	    dir="data/${network}/${cond}/modified_network/G_07_h_${h}"
	    infile="${dir}/${npz}"
	    if [ -f "$infile" ] ; then
		outfile=`echo ${infile%.npz}_TF_${fmin}.0_${fmax}.0_${N}_${dP}.npz | tr ' ' '_'`
		python3 compute_spectra.py -N $N --fmin $fmin --fmax $fmax --ref-sm "$slack" \
			--dP $dP -L $loads -V $vars -o "$outfile" --save-mat --load-exp $exp -f "$infile"
	    else
		echo "$infile: no such file."
	    fi
	done
    done
elif [ "$network" = "WSCC" ] ; then
    slack="G1"
    loads="Load A,Load B,Load C"
    npz="9 Bus WSCC_AC.npz"
    N=100
    conds=(
	"loads_A_B_C_const_I"
	"loads_A_B_C_const_Z"
	"loads_A_B_C_const_P"
	"loads_A_B_C_const_P_stable"
	"load_A_const_P_loads_B_C_const_Z"
	"load_B_const_P_loads_A_C_const_Z"
	"load_C_const_P_loads_A_B_const_Z"
	"loads_A_B_const_P_load_C_const_Z"
	"loads_A_C_const_P_load_B_const_Z"
	"loads_B_C_const_P_load_A_const_Z"
	"loads_A_B_C_const_P_load_D_const_Z"
	"loads_B_C_D_const_P_load_A_const_Z"
    )
    for cond in ${conds[@]} ; do
	[[ $cond =~ "const_Z" ]]
	if [ ${#BASH_REMATCH[@]} -gt 0 ] ; then
	    exp=2
	else
	    [[ $cond =~ "const_I" ]]
	    if [ ${#BASH_REMATCH[@]} -gt 0 ] ; then
		exp=1
	    else
		exp=0
	    fi
	fi
	echo "Load exponent for condition '${cond}': ${exp}."
	for h in 3.33 6.66 ; do
	    dir="data/${network}/${cond}/G2_h_${h}"
	    infile="${dir}/${npz}"
	    if [ -f "$infile" ] ; then
		suffix=`echo $loads | tr ', ' '_'`
		outfile=`echo ${infile%.npz}_TF_${fmin}.0_${fmax}.0_${N}_${dP}_${suffix}.npz | tr ' ' '_'`
		python3 compute_spectra.py -N $N --fmin $fmin --fmax $fmax --ref-sm "$slack" \
			--dP $dP -L "$loads" -o "$outfile" --save-mat --load-exp $exp -f "$infile"
	    else
		echo "$infile: no such file."
	    fi
	done
    done
elif [ "$network" = "simple_case_globalH" ] ; then
    slack="G1"
    loads="General Load"
    npz="simple_case_globalH_AC.npz"
    N=1000
    for cond in "P" "Z" "I" ; do
	for h in 2.5 3.5 ; do
	    dir="data/${network}/const_${cond}/G2_h_${h}"
	    infile="${dir}/${npz}"
	    if [ -f "$infile" ] ; then
		outfile=`echo ${infile%.npz}_TF_${fmin}.0_${fmax}.0_${N}_${dP}_${loads}.npz | tr ' ' '_'`
		python3 compute_spectra.py -N $N --fmin $fmin --fmax $fmax --ref-sm "$slack" \
			--dP $dP -L "$loads" -o "$outfile" --save-mat "$infile"
	    else
		echo "$infile: no such file."
	    fi
	done
    done
fi

