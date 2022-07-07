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

# # Replace the following line by the path of your own conda.sh file:
# source "/home/breuter/anaconda3/etc/profile.d/conda.sh"
# # This is intended to run in the malaria_env conda environment:
# conda activate malaria_env

# This is intended to run in the bin folder of the MalariaVaccineEfficacyPrediction package.
# The MalariaVaccineEfficacyPrediction package should be situated in the users home directory.
topdir="${HOME}/MalariaVaccineEfficacyPrediction"
if [ ! -d "$topdir" ]; then
    { echo "${topdir} doesn't exists."; exit 1; }
fi
maindir="${topdir}/results/filtered/multitaskSVM"
if [ ! -d "$maindir" ]; then
    mkdir "$maindir"
fi
data_dir="${topdir}/data/proteome_data"
kernel_dir="${topdir}/data/precomputed_multitask_kernels/filtered"
combinations=('SPP' 'SPR' 'SRP' 'SRR' 'RPP' 'RPR' 'RRP' 'RRR')

for dataset in 'whole' 'selective'; do
    if [ "$dataset" = 'whole' ]; then
        identifier='kernel_matrix'
    elif [ "$dataset" = 'selective' ]; then
        identifier='kernel_matrix_SelectiveSet'
    fi
    mkdir "${maindir}/${dataset}"

    for combination in "${combinations[@]}"; do
        mkdir "${maindir}/${dataset}/${combination}"
	    timestamp=$(date +%d-%m-%Y_%H-%M-%S)
        err="runRGSCV_${dataset}_${combination}_${timestamp}.err"
        out="runRGSCV_${dataset}_${combination}_${timestamp}.out"
        ana_dir="${maindir}/${dataset}/${combination}"
        mkdir "${ana_dir}"
        cd "${ana_dir}" || { echo "Couldn't cd into ${ana_dir}"; exit 1; }
        cp /home/breuter/MalariaVaccineEfficacyPrediction/bin/rgscv_multitask.py . || { echo "cp /home/breuter/MalariaVaccineEfficacyPrediction/bin/rgscv_multitask.py . failed"; exit 1; }
        python -u rgscv_multitask.py --analysis-dir "${ana_dir}" --data-file "${data_dir}/preprocessed_${dataset}_data.csv" --kernel-dir "$kernel_dir" --combination "${combination}" --identifier "${identifier}" 1> "${out}" 2> "${err}"
    done
done
