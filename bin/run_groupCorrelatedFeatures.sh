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
# source "/Users/schmidtj/opt/anaconda3/etc/profile.d/conda.sh"
# # This is intended to run in the malaria_env conda environment:
# conda activate malaria_env

# This is intended to run in the bin folder of the MalariaVaccineEfficacyPrediction package.
# The MalariaVaccineEfficacyPrediction package should be situated in the users home directory.
topdir="${HOME}/MalariaVaccineEfficacyPrediction"
if [ ! -d "$topdir" ]; then
    { echo "${topdir} doesn't exists."; exit 1; }
fi
ana_dir="${topdir}/results/correlationAnalysis"
if [ ! -d "$ana_dir" ]; then
    mkdir "$ana_dir"
fi
data_dir="${topdir}/data/timepoint-wise"
method='spearman'

cd "${ana_dir}" || { echo "Couldn't cd into ${ana_dir}"; exit 1; }
cp "${topdir}/bin/groupCorrelatedFeatures.py" . || { echo "cp ${topdir}/bin/groupCorrelatedFeatures.py . failed"; exit 1; }

for dataset in 'whole' 'selective'; do

    for timepoint in 'III14' 'C-1' 'C28'; do

        for threshold in '0.95' '0.98' '0.99'; do

            timestamp=$(date +%d-%m-%Y_%H-%M-%S)
            err="run_groupCorrelatedFeatures_${dataset}_${timepoint}_${method}_threshold${threshold}_${timestamp}.err"
            out="run_groupCorrelatedFeatures_${dataset}_${timepoint}_${method}_threshold${threshold}_${timestamp}.out"
            python -u groupCorrelatedFeatures.py --data-dir "${data_dir}" --identifier "$dataset" --out-dir "$ana_dir" --timepoint "$timepoint" --correlation_threshold "$threshold" --correlation_method "$method" 1> "${out}" 2> "${err}"

        done

    done

done
