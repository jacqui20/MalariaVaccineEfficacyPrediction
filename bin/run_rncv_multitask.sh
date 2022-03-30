#!/bin/bash

source "/home/breuter/anaconda3/etc/profile.d/conda.sh"
conda activate malaria_env

maindir='/home/breuter/MalariaVaccineEfficacyPrediction/results/multitaskSVM'
data_maindir='/home/breuter/MalariaVaccineEfficacyPrediction/data/precomputed_multitask_kernels'
combinations=('RPP' 'RPR' 'RRP' 'RRR' 'SPP' 'SPR' 'SRP' 'SRR')

for dataset in 'whole' 'selective'; do
    if [ "$dataset" = 'whole' ]; then
        identifier='kernel_matrix'
    elif [ "$dataset" = 'selective' ]; then
        identifier='kernel_matrix_SelectiveSet'
    fi
    mkdir "${maindir}/${dataset}"

    for combination in "${combinations[@]}"; do
        mkdir "${maindir}/${dataset}/${combination}"

        for scaling in 'unscaled'; do
	    timestamp=$(date +%d-%m-%Y_%H-%M-%S)
            err="runRNCV_${dataset}_${combination}_${scaling}_${timestamp}.err"
            out="runRNCV_${dataset}_${combination}_${scaling}_${timestamp}.out"
            ana_dir="${maindir}/${dataset}/${combination}/${scaling}"
            data_dir="${data_maindir}/${scaling}"
            mkdir "${ana_dir}"
            cd "${ana_dir}" || { echo "Couldn't cd into ${ana_dir}"; exit 1; }
            cp /home/breuter/MalariaVaccineEfficacyPrediction/bin/rncv_multitask.py . || { echo "cp /home/breuter/MalariaVaccineEfficacyPrediction/bin/rncv_multitask.py . failed"; exit 1; }
            python -u rncv_multitask.py --analysis-dir "${ana_dir}" --data-dir "${data_dir}" --combination "${combination}" --identifier "${identifier}" 1> "${out}" 2> "${err}"
        done
    done
done
