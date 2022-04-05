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

"""
Evaluation of informative features from RLR models

Will save the results to various .tsv files

@Author: Jacqueline Wistuba-Hamprecht and Bernhard Reuter

"""

import numpy as np
import pandas as pd
import scipy
import sklearn
import sys
import os
import argparse
# maindir = '/'.join(os.getcwd().split('/')[:-1])
# sys.path.append(maindir)
from source.featureEvaluationRLR import featureEvaluationRLR


def main(
    data_path: str,
    identifier: str,
    rgscv_path: str,
    out_dir: str,
    timepoint: str,
):
    """
    Evaluation of informative features from RLR.
    """
    proteome_data = pd.read_csv(data_path, sep=',', index_col=0)
    rgscv_results = pd.read_csv(rgscv_path, sep="\t", index_col=0)

    coefs = featureEvaluationRLR(
        data=proteome_data,
        rgscv_results=rgscv_results,
        timepoint=timepoint)

    fn = os.path.join(out_dir, f"RLR_informative_features_{identifier}_data_{timepoint}.tsv")
    pd.DataFrame(data=coefs).to_csv(fn, sep='\t', na_rep='nan')

    print(f"Results are saved in: {fn}")


if __name__ == "__main__":
    print('sys.path:', sys.path)
    print('scikit-learn version:', sklearn.__version__)
    print('pandas version:', pd.__version__)
    print('numpy version:', np.__version__)
    print('scipy version:', scipy.__version__)

    parser = argparse.ArgumentParser(
        description=('Function to run an analysis of informative features from RLR.')
    )
    parser.add_argument(
        '--data-path', dest='data_path', metavar='FILE', required=True,
        help='Path to the proteome data file.'
    )
    parser.add_argument(
        '--identifier', dest='identifier', required=True,
        help=('Prefix to identify the proteome dataset.')
    )
    parser.add_argument(
        '--rgscv-path', dest='rgscv_path', metavar='FILE', required=True,
        help='Path to the File were the RGSCV results are stored.'
    )
    parser.add_argument(
        '--out-dir', dest='out_dir', metavar='DIR', required=True,
        help='Path to the directory to wihich the output shall be written.'
    )
    parser.add_argument(
        '--timepoint', dest='timepoint', required=True, type=str,
        help='Time point for which the analysis shall be performed.'
    )
    args = parser.parse_args()

    main(
        args.data_path,
        args.identifier,
        args.rgscv_path,
        args.out_dir,
        args.timepoint,
    )
