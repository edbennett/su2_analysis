#!/bin/bash

for folder in `find raw_data/nf2_FUN raw_data/nf0_FUN raw_data/nf0_ASY -type d | sed -E 's:raw_data/?::' | sort`
do
    echo ==========================================================================
    # Setup variables
    parameters=`echo ${folder} | sed -E 's:[_/a-z]: :g'`
    NF=`echo $parameters | awk '{print $1}'`
    rep=`echo $parameters | awk '{print $2}'`
    T=`echo $parameters | awk '{print $3}'`
    L=`echo $parameters | awk '{print $4}'`
    beta=`echo $parameters | awk '{print $7}'`
    m=`echo $parameters | awk '{print $8}'`

    if [ -z "${m}" ]
    then
	continue
    fi

    echo "β = ${beta}; am = ${m}; L = ${L}; T = ${T}"

    # Prepare folders if necessary
    if [ ! -d processed_data/${folder} ]
    then
	mkdir -p processed_data/${folder}
    fi
    if [ ! -d processing_params/${folder} ]
    then
	mkdir -p processing_params/${folder}
    fi

    # Create metadata file for easy table generation later
    echo -e "nf\trepresentation\tbeta\tbare_mass\tL\tT\n${NF}\t${rep}\t${beta}\t-${m}\t${L}\t${T}" > processed_data/${folder}/meta.dat

    if [ -f raw_data/${folder}/out_corr ]
    then
	echo Mesons:
	# Extract bare correlators from raw output file
	cat raw_data/${folder}/out_corr | ./mes_filter.sh > processed_data/${folder}/correlators
	
	# Process each channel to get masses in these channels
	for channel in g5 gk g5gk
	do
	    if [ -f processed_data/${folder}/mass_${channel}.dat ]
	    then
		echo ${channel} already analysed
		continue
	    fi

	    echo ${channel}
	    if [ "$channel" == "g5" ]
	    then
		intensity=intense
	    else
		intensity=default
	    fi
	    params=processing_params/${folder}/${channel}_params
	    if [ ! -f $params ]
	    then
		params=processing_params/default_corr_params
	    fi
	    python fit_correlation_function.py \
		   --correlator_filename=processed_data/${folder}/correlators \
		   --channel=$channel \
		   --NT=$T \
		   --NS=$L \
		   --optimizer_intensity=default \
		   --output_filename_prefix=processed_data/${folder}/ \
		   `cat $params`
	done
    else
	echo No correlators to analyse.
    fi

    if [ -f raw_data/${folder}/out_hmc ]
    then
	if [ -f processed_data/${folder}/plaquette.dat ]
	then
	    echo Plaquette already analysed
	else
	    echo 'Plaquette:'
	    params=processing_params/${folder}/plaquette_params
	    if [ ! -f $params ]
	    then
		params=processing_params/default_plaquette_params
	    fi
	    
	    python avr_plaquette.py \
		   --filename raw_data/${folder}/out_hmc \
		   --output_filename_prefix processed_data/${folder}/ \
		   --beta ${beta} \
		   `cat $params`
	fi
    else
	echo No HMC logs to analyse for plaquette.
    fi
	
    if [ -f raw_data/${folder}/out_wflow ]
    then
	if [ -f processed_data/${folder}/w0.dat ]
	then
	   echo w0 already analysed
	else
	    echo 'w0:'
	    params=processing_params/${folder}/wflow_params
	    if [ ! -f $params ]
	    then
		params=processing_params/default_wflow_parms
	    fi
	    
	    awk 'BEGIN {cfg_index = 1;} /WILSONFLOW/ {if ($5 < last_t) {cfg_index = cfg_index + 1;} print cfg_index, $5, $6, $8, $10; last_t = $5}' raw_data/${folder}/out_wflow > processed_data/${folder}/wflow
	    
	    python w0.py \
		   --filename=processed_data/${folder}/wflow \
		   --output_filename_prefix=processed_data/${folder}/
	fi
    else
	echo No gradient flow data to analyse for w0
    fi

    all_observables=processed_data/${folder}/all_observables.dat
    if [ -f ${all_observables} ]
    then
       rm ${all_observables}
    fi

    if stat -t processed_data/${folder}/*.dat >/dev/null 2>&1
    then
	paste processed_data/${folder}/*.dat > ${all_observables}
    fi

    echo
    echo
done

echo
echo ==========================================================================
echo Processing Nf=3 antisymmetric with both hot and cold starts
echo
for folder in `find raw_data/nf3_ASY -type d | sed -E 's:raw_data/?::' | sort`
do
    # Setup variables
    parameters=`echo ${folder} | sed -E 's:[_/a-z]: :g'`
    NF=`echo $parameters | awk '{print $1}'`
    rep=`echo $parameters | awk '{print $2}'`
    T=`echo $parameters | awk '{print $3}'`
    L=`echo $parameters | awk '{print $4}'`
    beta=`echo $parameters | awk '{print $7}'`
    m=`echo $parameters | awk '{print $8}'`

    if [ -z "${m}" ]
    then
	continue
    fi

    echo "β = ${beta}; am = ${m}; L = ${L}; T = ${T}"

    # Prepare folders if necessary
    if [ ! -d processed_data/${folder} ]
    then
	mkdir -p processed_data/${folder}
    fi
    if [ ! -d processing_params/${folder} ]
    then
	mkdir -p processing_params/${folder}
    fi

    # Create metadata file for easy table generation later
    echo -e "nf\trepresentation\tbeta\tbare_mass\tL\tT\n${NF}\t${rep}\t${beta}\t-${m}\t${L}\t${T}" > processed_data/${folder}/meta.dat

    for start in cold hot
    do
	echo ${start}:
	if [ -f raw_data/${folder}/${start} ]
	then
	    if [ -f processed_data/${folder}/${start}_plaquette.dat ]
	    then
		echo Plaquette already analysed
	    else
		echo 'Plaquette:'
		params=processing_params/${folder}/plaquette_params
		if [ ! -f $params ]
		then
		    params=processing_params/default_plaquette_params
		fi
		
		python avr_plaquette.py \
		       --filename raw_data/${folder}/${start} \
		       --output_filename_prefix processed_data/${folder}/${start}_ \
		       --beta ${beta} \
		       --column_header_prefix ${start} \
		       `cat $params`
	    fi
	else
	    echo No ${start} HMC logs to analyse for plaquette.
	    touch processed_data/${folder}/${start}_plaquette.dat
	fi
    done
    all_observables=processed_data/${folder}/all_observables.dat
    if [ -f ${all_observables} ]
    then
       rm ${all_observables}
    fi

    if stat -t processed_data/${folder}/*.dat >/dev/null 2>&1
    then
	paste processed_data/${folder}/*.dat > ${all_observables}
    fi
done

echo
echo ==========================================================================
echo Collating results:

for beta in `ls processed_data/nf2_FUN/*/all_observables.dat | sed 's:[/_xbm]: :g' | awk '{print $9}' | sort | uniq`
do
    echo ${beta}
    python concatenate.py processed_data/nf2_FUN/*b${beta}*/all_observables.dat --output processed_data/nf2_FUN/b${beta}.dat
done

python concatenate.py processed_data/nf3_ASY/*/*/*/all_observables.dat --output processed_data/nf3_ASY/all.dat

echo ==========================================================================
echo generating plots:
for fignum in {1..6}
do
    echo Figure ${fignum}
    python fig${fignum}.py
done
