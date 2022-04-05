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
Parser for the feature selection approach

This module computes the ESPY value of each single feature.
The ESPY value is the distances of each single features to the classification boundary in the
multitask-SVM model and compares the change of the distance with a consensus sample.

This module requires the output file of the Parser_multitask_SVM.py module
and the Feature_Evaluation_multitask_SVM.py script.

@Author: Jacqueline Wistuba-Hamprecht and Bernhard Reuter
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Optional
from source.FeatureEvaluationESPY import ESPY_measurement, svm_model, multitask_model, make_plot
from source.utils import DataSelector, get_parameters
from source.utils import select_timepoint


def main(
    data_dir: str,
    out_dir: str,
    identifier: str,
    uq: int,
    lq: int,
    rgscv_path: Optional[str] = None,
    kernel_dir: Optional[str] = None,
    timepoint: Optional[str] = None,
) -> None:
    """
    Call ESPY measurement.
    """
    print(f"ESPY value measurement started on {identifier} data.")
    print("with the following parameters:")
    print("value of upper percentile      = ", str(uq))
    print("value of lower percentile      = ", str(lq))
    print("at time point                = ", str(timepoint))
    print("\n")

    if identifier == 'simulated':

        data = pd.read_csv(os.path.join(data_dir, 'simulated_data.csv'))
        output_filename = f"ESPY_values_on_{identifier}_data"

        X_train, X_test, y_train, y_test = train_test_split(
            data.iloc[:, :1000].to_numpy(),
            data.iloc[:, 1000].to_numpy(),
            test_size=0.3,
            random_state=123,
        )

        rbf_svm_model = svm_model(
            X_train_data=X_train,
            y_train_data=y_train,
            X_test_data=X_test,
            y_test_data=y_test,
        )

        distance_result = ESPY_measurement(
            identifier=identifier,
            data=pd.DataFrame(X_test),
            model=rbf_svm_model,
            lq=lq,
            up=uq,
        )
        print(distance_result)

        make_plot(
            data=distance_result.iloc[:, :25],
            name=output_filename,
            outputdir=out_dir,
        )

        distance_result.to_csv(
            os.path.join(out_dir, output_filename + ".tsv"),
            sep='\t',
            na_rep='nan',
        )
        print("results are saved in: " + os.path.join(out_dir, output_filename))

    elif identifier in ['whole', 'selective']:

        if timepoint == 'III14':
            t = 2
        elif timepoint == 'C-1':
            t = 3
        elif timepoint == 'C28':
            t = 4
        else:
            raise ValueError(
                "The string given via the '--timepoint' argument must "
                "be one of 'III14', 'C-1', or 'C28'."
            )

        data = pd.read_csv(os.path.join(data_dir, f'preprocessed_{identifier}_data_sorted.csv'))
        rgscv_results = pd.read_csv(rgscv_path, delimiter="\t", header=0, index_col=0)

        output_filename = f"ESPY_values_on_{identifier}_data_{timepoint}"

        timepoint_results = select_timepoint(rgscv_results, timepoint)
        params = get_parameters(timepoint_results, "multitask")

        y = data.loc[:, 'Protection'].to_numpy()

        # initialize running index array for DataSelector
        assert y.size * y.size < np.iinfo(np.uint32).max, \
            f"y is to large: y.size * y.size >= {np.iinfo(np.uint32).max}"
        X = np.array(
            [x for x in range(y.size * y.size)],
            dtype=np.uint32
        ).reshape((y.size, y.size))

        if identifier == 'whole':
            matrix_identifier = 'kernel_matrix_RRR'
        elif identifier == 'selective':
            matrix_identifier = 'kernel_matrix_SelectiveSet_RRR'

        kernel_matrix = DataSelector(
            kernel_directory=kernel_dir,
            identifier=matrix_identifier,
            SA=params['SA'],
            SO=params['SO'],
            R0=params['R0'],
            R1=params['R1'],
            R2=params['R2'],
            P1=params['P1'],
            P2=params['P2'],
        ).fit(X, y).transform(X)

        multitask_classifier = multitask_model(
            kernel_matrix=kernel_matrix,
            kernel_parameters=params,
            y_label=y,
        )

        print(
            "Are values in proteome data floats: "
            f"{np.all(np.isin(data.dtypes.to_list()[5:], ['float64']))}"
        )

        data_at_timePoint = data.loc[data["TimePointOrder"] == t]

        distance_result = ESPY_measurement(
            identifier=identifier,
            data=data_at_timePoint.iloc[:, 3:],
            model=multitask_classifier,
            lq=lq,
            up=uq,
            proteome_data=data,
            kernel_parameters=params,
        )

        print("Distances for features:")
        print(distance_result)

        make_plot(
            data=distance_result.iloc[:, :50],
            name=output_filename,
            outputdir=out_dir,
        )

        distance_result.to_csv(
            os.path.join(out_dir, output_filename + ".tsv"),
            sep='\t',
            na_rep='nan'
        )
        print(f'Results were saved in: {os.path.join(out_dir, output_filename)}')

    else:

        raise ValueError(
            "The string given via the '--identifier' argument must be "
            "one of 'whole', 'selective', or 'simulated'."
        )


if __name__ == "__main__":
    print('pandas version:', pd.__version__)
    print('numpy version:', np.__version__)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-dir',
        dest='data_dir',
        metavar='DIR',
        required=True,
        help=('Path to the directory were the simulated data or '
              'preprocessed proteome data is located.'),
    )
    parser.add_argument(
        '--out-dir',
        dest='out_dir',
        metavar='DIR',
        required=True,
        help='Path to the directory were the results shall be saved.',
    )
    parser.add_argument(
        '--identifier',
        dest='identifier',
        required=True,
        choices=['whole', 'selective', 'simulated'],
        help=("String to identify the proteome dataset. "
              "Must be one of 'whole', 'selective', or 'simulated'."),
    )
    parser.add_argument(
        '--lower-percentile',
        dest='lq',
        type=int,
        default=25,
        help='Lower percentile given as int, by default 25%.',
    )
    parser.add_argument(
        '--upper-percentile',
        dest='uq',
        type=int,
        default=75,
        help='Upper percentile given as int, by default 75%.',
    )
    parser.add_argument(
        '--kernel-dir',
        dest='kernel_dir',
        metavar='DIR',
        help='Path to the directory were the precomputed kernel matrices are stored.',
    )
    parser.add_argument(
        '--rgscv-path',
        dest='rgscv_path',
        metavar='FILEPATH',
        help='Path to the File were the RGSCV results are stored.',
    )
    parser.add_argument(
        '--timepoint',
        dest='timepoint',
        choices=['III14', 'C-1', 'C28'],
        help='Time point for which the analysis shall be performed.',
    )

    args = parser.parse_args()

    if args.identifier == 'simulated':
        main(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            identifier=args.identifier,
            uq=args.uq,
            lq=args.lq,
        )

    elif args.identifier in ['whole', 'selective']:

        main(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            identifier=args.identifier,
            uq=args.uq,
            lq=args.lq,
            rgscv_path=args.rgscv_path,
            kernel_dir=args.kernel_dir,
            timepoint=args.timepoint,
        )

    else:

        raise ValueError(
            "The string given via the '--identifier' argument must be "
            "one of 'whole', 'selective', or 'simulated'."
        )
