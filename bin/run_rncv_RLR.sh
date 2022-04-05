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

source "/home/breuter/anaconda3/etc/profile.d/conda.sh"
conda activate malaria_env

maindir='/home/breuter/MalariaVaccineEfficacyPrediction/results/RLR'
data_dir='/home/breuter/MalariaVaccineEfficacyPrediction/data/timepoint-wise'

for dataset in 'whole' 'selective'; do

    timestamp=$(date +%d-%m-%Y_%H-%M-%S)
    err="runRNCV_${dataset}_${timestamp}.err"
    out="runRNCV_${dataset}_${timestamp}.out"
    ana_dir="${maindir}/${dataset}"
    mkdir "${ana_dir}"
    cd "${ana_dir}" || { echo "Couldn't cd into ${ana_dir}"; exit 1; }
    cp /home/breuter/MalariaVaccineEfficacyPrediction/bin/rncv_RLR.py . || { echo "cp /home/breuter/MalariaVaccineEfficacyPrediction/bin/rncv_RLR.py . failed"; exit 1; }
    python -u rncv_RLR.py --analysis-dir "${ana_dir}" --data-dir "${data_dir}" --identifier "${dataset}" 1> "${out}" 2> "${err}"

done
