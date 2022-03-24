#!/bin/bash

source "/home/breuter/anaconda3/etc/profile.d/conda.sh"
conda activate malaria_env

maindir='/home/breuter/MalariaVaccineEfficacyPrediction/results/RLR'
data_dir='/home/breuter/MalariaVaccineEfficacyPrediction/data/timepoint-wise'

for dataset in 'whole' 'selective'; do

    timestamp=$(date +%d-%m-%Y_%H-%M-%S)
    err="runRGSCV_${dataset}_${timestamp}.err"
    out="runRGSCV_${dataset}_${timestamp}.out"
    ana_dir="${maindir}/${dataset}"
    mkdir "${ana_dir}"
    cd "${ana_dir}" || { echo "Couldn't cd into ${ana_dir}"; exit 1; }
    cp /home/breuter/MalariaVaccineEfficacyPrediction/bin/rgscv_RLR.py . || { echo "cp /home/breuter/MalariaVaccineEfficacyPrediction/bin/rgscv_RLR.py . failed"; exit 1; }
    python -u rgscv_RLR.py --analysis-dir "${ana_dir}" --data-dir "${data_dir}" --identifier "${dataset}" 1> "${out}" 2> "${err}"

done
