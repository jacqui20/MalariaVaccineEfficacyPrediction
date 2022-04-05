#!/bin/bash

# Replace the following line by the path of your own conda.sh file:
source "/Users/schmidtj/anaconda3/etc/profile.d/conda.sh"
# This is intended to run in the malaria_env conda environment:
conda activate malaria_env

# This is intended to run in the bin folder of the MalariaVaccineEfficacyPrediction package.
# The MalariaVaccineEfficacyPrediction package should be situated in the users home directory.
topdir="${HOME}/MalariaVaccineEfficacyPrediction"
if [ ! -d "$topdir" ]; then
    { echo "${topdir} doesn't exists."; exit 1; }
fi
maindir="${topdir}/results/RLR"
if [ ! -d "$maindir" ]; then
    mkdir "$maindir"
fi
data_dir="${topdir}/data/timepoint-wise"

for dataset in 'whole' 'selective'; do

    if [ "$dataset" = 'whole' ]; then
        rgscv_path="${maindir}/${dataset}/RGSCV/RepeatedGridSearchCV_results_24.03.2022_09-23-48.tsv"
    else
        rgscv_path="${maindir}/${dataset}/RGSCV/RepeatedGridSearchCV_results_24.03.2022_12-47-24.tsv"
    fi

    for timepoint in 'III14' 'C-1' 'C28'; do

        timestamp=$(date +%d-%m-%Y_%H-%M-%S)
        err="runFeatureEvalRLR_${dataset}_${timestamp}.err"
        out="runFeatureEvalRLR_${dataset}_${timestamp}.out"
        ana_dir="${maindir}/${dataset}/featureEvaluation"
        if [ ! -d "$ana_dir" ]; then
            mkdir "$ana_dir"
        fi
        cd "${ana_dir}" || { echo "Couldn't cd into ${ana_dir}"; exit 1; }
        cp "${topdir}/bin/featureEvalRLR.py" . || { echo "cp ${topdir}/bin/featureEvalRLR.py . failed"; exit 1; }
        python -u featureEvalRLR.py --data-path "${data_dir}/${dataset}_data_${timepoint}.csv" --identifier "$dataset" --rgscv-path "$rgscv_path" --out-dir "$ana_dir" --timepoint "$timepoint" 1> "${out}" 2> "${err}"

    done

done
