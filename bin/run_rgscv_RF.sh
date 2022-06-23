#!/bin/bash

# Copyright (c) 2022 Bernhard Reuter and Jacqueline Wistuba-Hamprecht.
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
maindir="${topdir}/results/RF"
if [ ! -d "$maindir" ]; then
    mkdir "$maindir"
fi
data_dir="${topdir}/data/timepoint-wise"

for dataset in 'whole' 'selective'; do

    timestamp=$(date +%d-%m-%Y_%H-%M-%S)
    err="runRGSCV_${dataset}_${timestamp}.err"
    out="runRGSCV_${dataset}_${timestamp}.out"
    ana_dir="${maindir}/${dataset}"
    mkdir "${ana_dir}"
    cd "${ana_dir}" || { echo "Couldn't cd into ${ana_dir}"; exit 1; }
    cp /home/breuter/MalariaVaccineEfficacyPrediction/bin/rgscv_RF.py . || { echo "cp /home/breuter/MalariaVaccineEfficacyPrediction/bin/rgscv_RF.py . failed"; exit 1; }
    python -u rgscv_RF.py --analysis-dir "${ana_dir}" --data-dir "${data_dir}" --identifier "${dataset}" 1> "${out}" 2> "${err}"

done
