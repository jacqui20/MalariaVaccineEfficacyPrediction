#!/bin/bash

# Replace the following line by the path of your own conda.sh file:
source "/home/breuter/anaconda3/etc/profile.d/conda.sh"
# This is intended to run in the malaria_env conda environment:
conda activate malaria_env

# This is intended to run in the bin folder of the MalariaVaccineEfficacyPrediction package.
# The MalariaVaccineEfficacyPrediction package should be situated in the users home directory.
topdir="${HOME}/MalariaVaccineEfficacyPrediction"
if [ ! -d "$topdir" ]; then
    { echo "${topdir} doesn't exists."; exit 1; }
fi
maindir="${topdir}/results/SVM"
if [ ! -d "$maindir" ]; then
    mkdir "$maindir"
fi
data_dir="${topdir}/data/timepoint-wise"

for dataset in 'whole' 'selective'; do

    timestamp=$(date +%d-%m-%Y_%H-%M-%S)
    err="runRNCV_${dataset}_${timestamp}.err"
    out="runRNCV_${dataset}_${timestamp}.out"
    ana_dir="${maindir}/${dataset}"
    if [ ! -d "$ana_dir" ]; then
	mkdir "$ana_dir"
    fi
    cd "$ana_dir" || { echo "Couldn't cd into ${ana_dir}"; exit 1; }
    cp "${topdir}/bin/rncv_SVM.py" . || { echo "cp ${topdir}/bin/rncv_SVM.py . failed"; exit 1; }
    python -u rncv_SVM.py --analysis-dir "${ana_dir}" --data-dir "${data_dir}" --identifier "${dataset}" 1> "${out}" 2> "${err}"

done
