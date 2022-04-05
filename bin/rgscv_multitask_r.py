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
# from shutil import rmtree
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from typing import Dict, List, Any, Union
# from tempfile import mkdtemp
sys.path.append('/home/breuter/NestedGridSearchCV')
import nestedcv as ncv
sys.path.append('/home/breuter/MalariaVaccineEfficacyPrediction')
from source.utils import DataSelector, CustomPredefinedSplit, assign_folds


def main(
    ANA_PATH: str,
    DATA_PATH: str,
    identifier: str,
    combination: str,
    param_grid: Union[Dict[str, List[Any]], List[Dict[str, List[Any]]]],
) -> None:

    # number of repetitions:
    Nexp = 10

    # Generate a timestamp used for file naming
    timestamp = ncv.generate_timestamp()

    print('')
    print("Identifier:", identifier)
    print('Kernel combination:', combination)
    print('parameter grid:')
    print(param_grid)
    print('')
    print('Start:', timestamp)
    print('')

    # Create directories for the output files
    maindir = os.path.join(ANA_PATH, 'RGSCV/')
    pathlib.Path(maindir).mkdir(parents=True, exist_ok=True)

    labels = pd.read_table(os.path.join(DATA_PATH, 'target_label_vec.csv'), sep=',', index_col=0)
    groups = labels.loc[:, 'group'].to_numpy()
    y = labels.loc[:, 'Protection'].to_numpy()

    print('shape of binary response array:', y.size)
    print('number of positives:', np.sum(y))
    print('number of positives divided by total number of samples:', np.sum(y)/y.size)
    print('')

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
        prefix = f'{time}_{combination}'

        # initialize test folds and CV splitters for outer CV
        cv = []
        print('Predefined CV folds:')
        print('---------------------------------------------------------------------------------')
        for rep in range(Nexp):
            print(f'CV folds for repetition {rep}:')
            test_fold, train_fold = assign_folds(y, groups, 40, step, random_state=rep)
            cv.append(CustomPredefinedSplit(test_fold, train_fold))
            print('')
        print('---------------------------------------------------------------------------------')
        print('')

        estimator = make_pipeline(
            DataSelector(
                kernel_directory=DATA_PATH,
                identifier=f'{identifier}_{combination}',
            ),
            SVC(
                kernel='precomputed',
                probability=True,
                random_state=1337,
                cache_size=500,
            ),
            # memory=cachedir,
        )

        # initialize running index array for DataSelector
        assert y.size * y.size < np.iinfo(np.uint32).max, \
            f"y is to large: y.size * y.size >= {np.iinfo(np.uint32).max}"
        X = np.array(
            [x for x in range(y.size * y.size)],
            dtype=np.uint32
        ).reshape((y.size, y.size))
        print('shape of running index array:', X.shape)
        print('')

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
        fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=timestamp)
        pd.DataFrame(data=gs.cv_results_).to_csv(fn, sep='\t', na_rep='nan')
        fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=timestamp)
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
    fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=timestamp)
    pd.DataFrame(data=results).to_csv(fn, sep='\t', na_rep='nan')
    fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=timestamp)
    pd.DataFrame(data=results).to_excel(fn, na_rep='nan')

    print('End:', ncv.generate_timestamp())


if __name__ == "__main__":
    print('sys.path:', sys.path)
    print('scikit-learn version:', sklearn.__version__)
    print('pandas version:', pd.__version__)
    print('numpy version:', np.__version__)
    print('scipy version:', scipy.__version__)
    print('Start:', ncv.generate_timestamp())

    timestamp = ncv.generate_timestamp()
    warning_file = open(f"warnings_{timestamp}.log", "w")

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
        description=('Function to run nested cross-validated grid-search for OligoSVM models')
    )
    parser.add_argument(
        '--analysis-dir', dest='analysis_dir', metavar='DIR', required=True,
        help='Path to the directory were the analysis shall be performed and stored.'
    )
    parser.add_argument(
        '--data-dir', dest='data_dir', metavar='DIR', required=True,
        help='Path to the directory were the data is located.'
    )
    parser.add_argument(
        '--combination', dest='combination', required=True,
        help='Kernel combination. Supply SPP, SPR, SRP, SRR, RPP, RPR, RRP, or RRR.'
    )
    parser.add_argument(
        '--identifier', dest='identifier', required=True,
        help=('Prefix to identify the precomputed kernel matrices (stored as .npy files).'
              'E.g. kernel_matrix_rescale.')
    )
    args = parser.parse_args()

    # cachedir = mkdtemp()
    combination = args.combination
    if combination == 'SPP':
        param_grid = {
            "dataselector__SA": [0.25, 0.5, 0.75],
            "dataselector__SO": [-2.0, -1.0, 0.0, 1.0, 2.0],
            "dataselector__R0": ["X"],
            "dataselector__R1": ["X"],
            "dataselector__R2": ["X"],
            "dataselector__P1": [2.0, 3.0, 4.0, 5.0],
            "dataselector__P2": [2.0, 3.0, 4.0, 5.0],
            "svc__C": [1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.e0, 1.e1, 1.e2, 1.e3, 1.e4],
        }
    elif combination == 'SPR':
        param_grid = {
            "dataselector__SA": [0.25, 0.5, 0.75],
            "dataselector__SO": [-2.0, -1.0, 0.0, 1.0, 2.0],
            "dataselector__R0": ["X"],
            "dataselector__R1": ["X"],
            "dataselector__R2": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 1e1, 1e2, 1e3, 1e4, 1e5],
            "dataselector__P1": [2.0, 3.0, 4.0, 5.0],
            "dataselector__P2": ["X"],
            "svc__C": [1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.e0, 1.e1, 1.e2, 1.e3, 1.e4],
        }
    elif combination == 'SRP':
        param_grid = {
            "dataselector__SA": [0.25, 0.5, 0.75],
            "dataselector__SO": [-2.0, -1.0, 0.0, 1.0, 2.0],
            "dataselector__R0": ["X"],
            "dataselector__R1": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 1e1, 1e2, 1e3, 1e4, 1e5],
            "dataselector__R2": ["X"],
            "dataselector__P1": ["X"],
            "dataselector__P2": [2.0, 3.0, 4.0, 5.0],
            "svc__C": [1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.e0, 1.e1, 1.e2, 1.e3, 1.e4],
        }
    elif combination == 'SRR':
        param_grid = {
            "dataselector__SA": [0.25, 0.5, 0.75],
            "dataselector__SO": [-2.0, -1.0, 0.0, 1.0, 2.0],
            "dataselector__R0": ["X"],
            "dataselector__R1": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 1e1, 1e2, 1e3, 1e4, 1e5],
            "dataselector__R2": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 1e1, 1e2, 1e3, 1e4, 1e5],
            "dataselector__P1": ["X"],
            "dataselector__P2": ["X"],
            "svc__C": [1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.e0, 1.e1, 1.e2, 1.e3, 1.e4],
        }
    elif combination == 'RPP':
        param_grid = {
            "dataselector__SA": ["X"],
            "dataselector__SO": ["X"],
            "dataselector__R0": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 1e1, 1e2, 1e3, 1e4, 1e5],
            "dataselector__R1": ["X"],
            "dataselector__R2": ["X"],
            "dataselector__P1": [2.0, 3.0, 4.0, 5.0],
            "dataselector__P2": [2.0, 3.0, 4.0, 5.0],
            "svc__C": [1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.e0, 1.e1, 1.e2, 1.e3, 1.e4],
        }
    elif combination == 'RPR':
        param_grid = {
            "dataselector__SA": ["X"],
            "dataselector__SO": ["X"],
            "dataselector__R0": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 1e1, 1e2, 1e3, 1e4, 1e5],
            "dataselector__R1": ["X"],
            "dataselector__R2": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 1e1, 1e2, 1e3, 1e4, 1e5],
            "dataselector__P1": [2.0, 3.0, 4.0, 5.0],
            "dataselector__P2": ["X"],
            "svc__C": [1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.e0, 1.e1, 1.e2, 1.e3, 1.e4],
        }
    elif combination == 'RRP':
        param_grid = {
            "dataselector__SA": ["X"],
            "dataselector__SO": ["X"],
            "dataselector__R0": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 1e1, 1e2, 1e3, 1e4, 1e5],
            "dataselector__R1": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 1e1, 1e2, 1e3, 1e4, 1e5],
            "dataselector__R2": ["X"],
            "dataselector__P1": ["X"],
            "dataselector__P2": [2.0, 3.0, 4.0, 5.0],
            "svc__C": [1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.e0, 1.e1, 1.e2, 1.e3, 1.e4],
        }
    elif combination == 'RRR':
        param_grid = {
            "dataselector__SA": ["X"],
            "dataselector__SO": ["X"],
            "dataselector__R0": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 1e1, 1e2, 1e3, 1e4, 1e5],
            "dataselector__R1": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 1e1, 1e2, 1e3, 1e4, 1e5],
            "dataselector__R2": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 1e1, 1e2, 1e3, 1e4, 1e5],
            "dataselector__P1": ["X"],
            "dataselector__P2": ["X"],
            "svc__C": [1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.e0, 1.e1, 1.e2, 1.e3, 1.e4],
        }

    try:
        main(
            args.analysis_dir,
            args.data_dir,
            args.identifier,
            combination,
            param_grid,
        )
    finally:

        warning_file.close()
        # rmtree(cachedir)
