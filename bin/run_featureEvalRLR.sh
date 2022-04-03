#!/bin/bash

# Replace the following line by the path of your own conda.sh file:
source "/Users/schmidtj/anaconda3/etc/profile.d/conda.sh"
# This is intended to run in the malaria_env conda environment:
conda activate malaria_env

# This is intended to run in the bin folder of the MalariaVaccineEfficacyPrediction package:
cd .. || { echo "Couldn't cd one level up"; exit 1; }
topdir=$PWD
maindir="${topdir}/results/RLR"
data_dir="${topdir}/data/proteome_data"  # is it REALLY intended to run on the whole data and NOT on the timepoint-wise data???

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
        mkdir "${ana_dir}"
        cd "${ana_dir}" || { echo "Couldn't cd into ${ana_dir}"; exit 1; }
        cp "${topdir}/bin/featureEvalRLR.py" . || { echo "cp ${topdir}/bin/featureEvalRLR.py . failed"; exit 1; }
        python -u featureEvalRLR.py --data-path "${data_dir}/preprocessed_${dataset}_data.csv" --identifier "$dataset" --rgscv-path "$rgscv_path" --out-dir "$ana_dir" --timepoint "$timepoint" 1> "${out}" 2> "${err}"

    done

done
