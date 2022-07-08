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

"""
Find the best parameters using 10-times repeated NestedGridsearchCV.

Will save the results to a various .tsv/.xslx files.

Activate associated environment first: conda activate malaria_env

@Author: Bernhard Reuter

"""

import argparse
import numpy as np
import os.path
import pandas as pd
import pathlib
from pprint import pprint
import scipy
import sklearn
import sys
import traceback
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import nestedcv as ncv
from source.utils import CustomPredefinedSplit, assign_folds


def main(
    ana_dir: str,
    data_file: str,
    identifier: str,
) -> None:

    # number of repetitions:
    Nexp = 10

    param_grid = {
        'logisticregression__l1_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'logisticregression__C': [1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.e0, 1.e1, 1.e2, 1.e3, 1.e4],
    }

    # Generate a timestamp
    timestamp = ncv.generate_timestamp()

    print('')
    print("Identifier:", identifier)
    print('parameter grid:')
    print(param_grid)
    print('')
    print('Start:', timestamp)
    print('')

    # Create directories for the output files
    maindir = os.path.join(ana_dir, 'RGSCV/')
    pathlib.Path(maindir).mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_file, header=0, index_col=0)
    # Move dose column to the rightmost metadata columns:
    dose = data['Dose']
    data.drop(columns=['Dose'], inplace=True)
    data.insert(loc=3, column='Dose', value=dose)

    groups_all = data.loc[:, 'group'].to_numpy()
    y_all = data.loc[:, 'Protection'].to_numpy()

    # initialize result dict
    results = dict()
    results['time'] = []
    results['scoring'] = []
    results['best_params'] = []
    results['best_score'] = []

    times = ['III14', 'C-1', 'C28']
    scorings = ['mcc', 'precision_recall_auc', 'roc_auc']
    for step, time in enumerate(times):

        print('=================================================================================')
        print(f'{time} start: {ncv.generate_timestamp()}')
        print('')

        # define prefix for filenames:
        prefix = f'{time}'

        if time == 'III14':
            t = 2
        elif time == 'C-1':
            t = 3
        elif time == 'C28':
            t = 4
        else:
            raise ValueError(f"Unknown timepoint {time}.")

        data_at_timePoint = data.loc[data["TimePointOrder"] == t, :]
        X = data_at_timePoint.iloc[:, 3:].to_numpy()  # including dose
        groups = data_at_timePoint.loc[:, 'group'].to_numpy()
        y = data_at_timePoint.loc[:, 'Protection'].to_numpy()

        print('shape of binary response array:', y.size)
        print('number of positives:', np.sum(y))
        print('number of positives divided by total number of samples:', np.sum(y)/y.size)
        print('')

        # initialize test folds and CV splitters for outer CV
        cv = []
        print('Predefined CV folds:')
        print('---------------------------------------------------------------------------------')
        for rep in range(Nexp):
            print(f'CV folds for repetition {rep}:')
            test_fold, train_fold = assign_folds(
                y_all, groups_all, 40, step, random_state=rep, print_info=False
            )
            test_fold = test_fold[40 * step: 40 * (step + 1)]
            train_fold = train_fold[40 * step: 40 * (step + 1)]
            assert np.all(test_fold == train_fold), \
                f"test_fold != train_fold: {test_fold} != {train_fold}"
            cv.append(CustomPredefinedSplit(test_fold, train_fold))
            print(
                f"train_fold: {train_fold} "
                f"test_fold: {test_fold}"
            )
            cps = CustomPredefinedSplit(test_fold, train_fold)
            for i, (train_index, test_index) in enumerate(cps.split()):
                print(
                    f"TRAIN (len={len(train_index)}): {train_index} "
                    f"TEST (len={len(test_index)}): {test_index}"
                )
            print('')
        print('---------------------------------------------------------------------------------')
        print('')

        estimator = make_pipeline(
            StandardScaler(
                with_mean=True,
                with_std=True,
            ),
            LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                max_iter=10000,
            ),
        )

        gs = ncv.RepeatedGridSearchCV(
            estimator,
            param_grid,
            scoring=scorings,
            cv=cv,
            n_jobs=8,
            Nexp=Nexp,
            save_to=None,
            reproducible=False,
        )
        gs.fit(X, y, groups)

        filename = f"{prefix}_RepeatedGridSearchCV_cv_results"
        fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=False)
        pd.DataFrame(data=gs.cv_results_).to_csv(fn, sep='\t', na_rep='nan')
        fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=False)
        pd.DataFrame(data=gs.cv_results_).to_excel(fn, na_rep='nan')

        opt_params = gs.opt_params_
        print(f'{time} opt_params:')
        pprint(opt_params)
        print('')

        opt_scores = gs.opt_scores_
        print(f'{time} opt_scores:')
        pprint(opt_scores)
        print('')

        for scoring in scorings:
            results['time'].append(time)
            results['scoring'].append(scoring)
            results['best_params'].append(opt_params[scoring])
            results['best_score'].append(opt_scores[scoring])

    print('results:')
    pprint(results)
    print('')

    filename = "RepeatedGridSearchCV_results"
    fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=False)
    pd.DataFrame(data=results).to_csv(fn, sep='\t', na_rep='nan')
    fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=False)
    pd.DataFrame(data=results).to_excel(fn, na_rep='nan')

    print('End:', ncv.generate_timestamp())


if __name__ == "__main__":
    print('sys.path:', sys.path)
    print('scikit-learn version:', sklearn.__version__)
    print('pandas version:', pd.__version__)
    print('numpy version:', np.__version__)
    print('scipy version:', scipy.__version__)
 
    warning_file = open("warnings.log", "w")

    def warn_with_traceback(
        message, category, filename, lineno, file=None, line=None
    ):
        try:
            warning_file.write(warnings.formatwarning(message, category, filename, lineno, line))
            traceback.print_stack(file=warning_file)
        except Exception:
            sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
            traceback.print_stack(file=sys.stderr)

    warnings.showwarning = warn_with_traceback
    warnings.simplefilter("default")

    parser = argparse.ArgumentParser(
        description=('Function to run repeated cross-validated grid-search for RLR models')
    )
    parser.add_argument(
        '--analysis-dir', dest='analysis_dir', metavar='DIR', required=True,
        help='Path to the directory were the analysis shall be performed and stored.'
    )
    parser.add_argument(
        '--data-file', dest='data_file', metavar='FILE', required=True,
        help='Path to the proteome data file.'
    )
    parser.add_argument(
        '--identifier', dest='identifier', required=True,
        help=(
            "Prefix to identify the proteome data files and name the output files, "
            "use either 'whole' or 'selective'."
        )
    )
    args = parser.parse_args()

    try:
        main(
            args.analysis_dir,
            args.data_file,
            args.identifier,
        )
    finally:

        warning_file.close()
