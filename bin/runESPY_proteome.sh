#!/bin/bash

# Copyright (c) 2022 Jacqueline Wistuba-Hamprecht and Bernhard Reuter.
# ------------------------------------------------------------------------------------------------
# If you use this code or parts of it, cite the following reference:
# ------------------------------------------------------------------------------------------------
# Jacqueline Wistuba-Hamprecht and Bernhard Reuter (2022)
# https://github.com/jacqui20/MalariaVaccineEfficacyPrediction
# ------------------------------------------------------------------------------------------------
# This is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along with this program.
# If not, see <http://www.gnu.org/licenses/>.
# ------------------------------------------------------------------------------------------------

# @Author: Bernhard Reuter

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
maindir="${topdir}/results/multitaskSVM"
if [ ! -d "$maindir" ]; then
    mkdir "$maindir"
fi
data_dir="${topdir}/data/proteome_data"
kernel_dir="${topdir}/data/precomputed_multitask_kernels/unscaled"

for dataset in 'whole' 'selective'; do

    if [ "$dataset" = 'whole' ]; then
        rgscv_path="${maindir}/${dataset}/RRR/unscaled/RGSCV/RepeatedGridSearchCV_results_24.03.2022_16-16-36.tsv"
    else
        rgscv_path="${maindir}/${dataset}/RRR/unscaled/RGSCV/RepeatedGridSearchCV_results_24.03.2022_19-19-18.tsv"
    fi

    for timepoint in 'III14' 'C-1' 'C28'; do

        timestamp=$(date +%d-%m-%Y_%H-%M-%S)
        err="runESPY_${dataset}_${timestamp}.err"
        out="runESPY_${dataset}_${timestamp}.out"
        ana_dir="${maindir}/${dataset}/RRR/unscaled/featureEvaluation"
        if [ ! -d "$ana_dir" ]; then
            mkdir "$ana_dir"
        fi
        cd "${ana_dir}" || { echo "Couldn't cd into ${ana_dir}"; exit 1; }
        cp "${topdir}/bin/ESPY.py" . || { echo "cp ${topdir}/bin/ESPY.py . failed"; exit 1; }
        python -u ESPY.py --data-dir "$data_dir" --out-dir "$ana_dir" --identifier "$dataset" --lower-percentile 25 --upper-percentile 75 --kernel-dir "$kernel_dir" --rgscv-path "$rgscv_path" --timepoint "$timepoint" 1> "${out}" 2> "${err}"

    done

done
