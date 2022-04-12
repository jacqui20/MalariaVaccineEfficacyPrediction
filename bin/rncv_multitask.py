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
import re
import scipy
import sklearn
import sys
import traceback
import warnings
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from typing import Dict, List, Any, Union
import nestedcv as ncv
from source.utils import DataSelector, CustomPredefinedSplit, assign_folds


def main(
    ANA_PATH: str,
    DATA_PATH: str,
    identifier: str,
    combination: str,
    param_grid: Union[Dict[str, List[Any]], List[Dict[str, List[Any]]]],
) -> None:

    # number of repetitions:
    Nexp2 = 10

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
    maindir = os.path.join(ANA_PATH, 'RNCV/')
    pathlib.Path(maindir).mkdir(parents=True, exist_ok=True)
    rfiledir = os.path.join(ANA_PATH, 'RNCV/results/')
    pathlib.Path(rfiledir).mkdir(parents=True, exist_ok=True)
    # efiledir = os.path.join(ANA_PATH, 'RNCV/estimators/')
    # pathlib.Path(efiledir).mkdir(parents=True, exist_ok=True)
    # pfiledir = os.path.join(ANA_PATH, 'GridSearchCV/plots/')
    # pathlib.Path(pfiledir).mkdir(parents=True, exist_ok=True)
    # ifiledir = os.path.join(ANA_PATH, 'GridSearchCV/results/RGSCV/')
    # pathlib.Path(ifiledir).mkdir(parents=True, exist_ok=True)
    # ofiledir = os.path.join(ANA_PATH, 'GridSearchCV/results/RNCV/')
    # pathlib.Path(ofiledir).mkdir(parents=True, exist_ok=True)

    labels = pd.read_table(os.path.join(DATA_PATH, 'target_label_vec.csv'), sep=',', index_col=0)
    groups = labels.loc[:, 'group'].to_numpy()
    y = labels.loc[:, 'Protection'].to_numpy()

    print('shape of binary response array:', y.size)
    print('number of positives:', np.sum(y))
    print('number of positives divided by total number of samples:', np.sum(y)/y.size)
    print('')

    # initialize the key result collector:
    key_results: Dict[str, List[Any]] = {}
    key_results['kernel_combination'] = []
    key_results['time'] = []
    key_results['scoring'] = []
    key_results['top_params'] = []
    key_result_keys = [
        'ncv_mcc',
        'test_mcc',
        'ncv_precision_recall_auc',
        'test_precision_recall_auc',
        'ncv_roc_auc',
        'test_roc_auc',
        'mean_ncv_mcc',
        'mean_test_mcc',
        'mean_ncv_precision_recall_auc',
        'mean_test_precision_recall_auc',
        'mean_ncv_roc_auc',
        'mean_test_roc_auc',
    ]
    for key in key_result_keys:
        key_results[key] = []

    for step, time in enumerate(['III14', 'C-1', 'C28']):

        print('=================================================================================')
        print(f'{time} start: {ncv.generate_timestamp()}')
        print('')

        # define prefix for filenames:
        prefix = f'{time}_{combination}'

        rng = np.random.RandomState(0)

        # initialize test folds and CV splitters for outer CV
        outer_cv = []
        print('Predefined CV folds:')
        print('---------------------------------------------------------------------------------')
        for rep in range(Nexp2):
            print(f'CV folds for repetition {rep}:')
            test_fold, train_fold = assign_folds(y, groups, 40, step, random_state=rep)
            outer_cv.append(CustomPredefinedSplit(test_fold, train_fold))
            print('')
        print('---------------------------------------------------------------------------------')
        print('')

        # define identifiers used for naming of output files
        rfile = f'{prefix}_RGSCV'
        efile = f'{prefix}_RNCV'
        # pfile = f'{prefix}_RNCV'
        # ifile = f'{prefix}_RGSCV'
        # ofile = f'{prefix}_RNCV'

        # set options for NestedGridSearchCV
        cv_options = {
            'scoring': ['precision_recall_auc',
                        'roc_auc',
                        'mcc'],
            'refit': False,
            'tune_threshold': False,
            # 'threshold_tuning_scoring': ['f1', 'f1', 'f1', 'J', None],
            'save_to': {'directory': rfiledir, 'ID': rfile},
            # 'save_best_estimator': {'directory': efiledir, 'ID': efile},
            # 'save_pr_plots': {'directory': pfiledir, 'ID': pfile},
            # 'save_tt_plots': {'directory': pfiledir, 'ID': pfile},
            'save_pred': None,
            'save_inner_to': None,
            'reproducible': False,
            'inner_cv': StratifiedGroupKFold(n_splits=5, random_state=rng, shuffle=True),
            'outer_cv': outer_cv,
            'Nexp1': 1,
            'Nexp2': Nexp2,
            'n_jobs': 8,
        }

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

        clf_grid = ncv.RepeatedStratifiedNestedCV(
            estimator=estimator,
            param_grid=param_grid,
            cv_options=cv_options
        )
        clf_grid.fit(X, y, None, groups)

        # save the fitted RNCV instance
        if cv_options['refit']:
            if isinstance(cv_options['refit'], str):
                filename = f"{efile}_{cv_options['refit']}_RepeatedStratifiedNestedCV"
            else:
                filename = f"{efile}_{cv_options['scoring']}_RepeatedStratifiedNestedCV"
            ncv.save_model(
                clf_grid, efiledir, filename, timestamp=timestamp, compress=False, method='joblib'
            )

        result = clf_grid.repeated_nested_cv_results_

        try:
            ncv.save_json(
                {key: dict(sorted(value.items())) for key, value in result.items()},
                rfiledir,
                f'{prefix}_repeated_nested_cv_results',
                timestamp=timestamp
            )
        except TypeError:
            warnings.warn('Saving the results as JSON file failed due to a TypeError.')

        print(f'{time} results:')
        pprint(result, width=200)
        print('')

        scorings = sorted(result.keys())
        # collect the key results:
        for scoring in scorings:
            key_results['kernel_combination'].append(combination)
            key_results['time'].append(time)
            key_results['scoring'].append(scoring)
            key_results['top_params'].append(result[scoring]['ranked_best_inner_params'][0])
            for key in key_result_keys:
                key_results[key].append(result[scoring][key])

        performances = []
        ncv_performances = []
        for scoring in scorings:

            performance = {}
            performance['best_inner_params'] = result[scoring]['best_inner_params']
            scores = []
            for score in result[scoring].keys():
                if (bool(re.match(r'best', score)) and not
                        bool(re.search(r'ncv|best_inner|best_params', score))):
                    performance[score] = result[scoring][score]
                elif (bool(re.match(r'test|train', score)) and not
                        bool(re.search(r'ncv|best_inner|best_params', score))):
                    scores.append(re.sub(r'test_|train_', '', score, count=1))
            for score in sorted(scores):
                performance[f'test_{score}'] = result[scoring][f'test_{score}']
                performance[f'train_{score}'] = result[scoring][f'train_{score}']

            assert len(performance['best_inner_params']) % cv_options['Nexp2'] == 0, \
                'len(best_inner_params[%s] modulo Nexp2 != 0 for %s.)' % (prefix, scoring)
            index = []
            n_cv_splits = len(performance['best_inner_params']) // cv_options['Nexp2']
            for x2 in range(cv_options['Nexp2']):
                for x1 in range(n_cv_splits):
                    index.append('outer-repetition' + str(x2) + '_outer-split' + str(x1))
            performances.append(pd.DataFrame(performance, index=index))

            ncv_performance = {}
            scores = []
            for score in result[scoring].keys():
                if bool(re.match(r'ncv', score)):
                    scores.append(re.sub(r'ncv_', '', score, count=1))
            for score in sorted(scores):
                ncv_performance[f'ncv_{score}'] = result[scoring][f'ncv_{score}']
            index = []
            for x2 in range(cv_options['Nexp2']):
                index.append('outer-repetition' + str(x2))
            ncv_performances.append(pd.DataFrame(ncv_performance, index=index))

        # save train/test performance scores to an excel file with one sheet per scoring
        fn = ncv.filename_generator(
            f'{prefix}_train_test_performance_scores',
            extension=".xlsx",
            directory=rfiledir,
            timestamp=timestamp
        )
        if not os.path.exists(fn):
            with pd.ExcelWriter(fn) as writer:
                for dataframe, scoring in zip(performances, scorings):
                    dataframe.to_excel(writer, sheet_name=scoring, na_rep='nan')
        else:
            warnings.warn(f"Overwriting already existing file {fn}.")
            with pd.ExcelWriter(fn) as writer:
                for dataframe, scoring in zip(performances, scorings):
                    dataframe.to_excel(writer, sheet_name=scoring, na_rep='nan')

        # save ncv performance scores to an excel file with one sheet per scoring
        fn = ncv.filename_generator(
            f'{prefix}_ncv_performance_scores',
            extension=".xlsx",
            directory=rfiledir,
            timestamp=timestamp
        )
        if not os.path.exists(fn):
            with pd.ExcelWriter(fn) as writer:
                for dataframe, scoring in zip(ncv_performances, scorings):
                    dataframe.to_excel(writer, sheet_name=scoring, na_rep='nan')
        else:
            warnings.warn(f"Overwriting already existing file {fn}.")
            with pd.ExcelWriter(fn) as writer:
                for dataframe, scoring in zip(ncv_performances, scorings):
                    dataframe.to_excel(writer, sheet_name=scoring, na_rep='nan')
        del performances, ncv_performances

        # collect and save outer cross-validation train- and test-split performance scores
        for i, scoring in enumerate(scorings):

            performance_mean = {}
            performance_min_max_nice = {}
            scores = []
            counter = 0
            threshold_tuning_score = None

            for score in result[scoring].keys():
                if (bool(re.match(r'best', score)) and not
                        bool(re.search(r'ncv|best_inner|best_params', score))):
                    if bool(re.fullmatch(r'best_threshold', score)):
                        dummy = score
                    else:
                        threshold_tuning_score = re.sub(r'best_', '', score)
                        dummy = 'best_threshold_tuning_score'
                        counter += 1
                    if counter > 1:
                        warnings.warn(f'Unexpected "best" score {score}')
                    mean = f'mean_{score}'
                    std = f'std_{score}'
                    min_ = f'min_{score}'
                    max_ = f'max_{score}'
                    dummy_mean = f'mean_{dummy}'
                    dummy_min_max = f'min_max_{dummy}'
                    performance_mean[dummy_mean] = [
                        "%.3f" % result[scoring][mean] + " " + u"\u00B1"
                        + " " + "%.6f" % result[scoring][std]
                    ]
                    performance_min_max_nice[dummy_min_max] = [
                        "[%.3f, %.3f]" % (result[scoring][min_], result[scoring][max_])
                    ]
                elif (bool(re.match(r'test|train', score)) and not
                        bool(re.search(r'ncv|best_inner|best_params', score))):
                    scores.append(re.sub(r'test_|train_', '', score, count=1))

            if not threshold_tuning_score:
                performance_mean['mean_best_threshold_tuning_score'] = [np.nan]
                performance_min_max_nice['min_max_best_threshold_tuning_score'] = ['[nan, nan]']
                performance_mean['mean_best_threshold'] = [np.nan]
                performance_min_max_nice['min_max_best_threshold'] = ['[nan, nan]']

            for score in sorted(scores):
                for x in ['test', 'train']:
                    mean = f'mean_{x}_{score}'
                    std = f'std_{x}_{score}'
                    min_ = f'min_{x}_{score}'
                    max_ = f'max_{x}_{score}'
                    min_max = f'min_max_{x}_{score}'
                    if bool(re.search(r'brier_loss', score)):
                        performance_mean[mean] = [
                            "%.5f" % result[scoring][mean] + " " + u"\u00B1"
                            + " " + "%.6f" % result[scoring][std]
                        ]
                        performance_min_max_nice[min_max] = [
                            "[%.5f, %.5f]" % (result[scoring][min_], result[scoring][max_])
                        ]
                    elif bool(re.search(r'log_loss', score)):
                        performance_mean[mean] = [
                            "%.4f" % result[scoring][mean] + " " + u"\u00B1"
                            + " " + "%.6f" % result[scoring][std]
                        ]
                        performance_min_max_nice[min_max] = [
                            "[%.4f, %.4f]" % (result[scoring][min_], result[scoring][max_])
                        ]
                    else:
                        performance_mean[mean] = [
                            "%.3f" % result[scoring][mean] + " " + u"\u00B1"
                            + " " + "%.6f" % result[scoring][std]
                        ]
                        performance_min_max_nice[min_max] = [
                            "[%.3f, %.3f]" % (result[scoring][min_], result[scoring][max_])
                        ]

            print(f'{time} mean outer cross-validation train- and test-split performance '
                  f'scores ({scoring} optimized hyperparameters):')
            pprint(performance_mean)
            print('')

            if threshold_tuning_score:
                index = [f'{scoring} | {threshold_tuning_score}']
            else:
                index = [scoring]

            if i == 0:
                performance_mean_df = pd.DataFrame(
                    performance_mean, index=index
                )
                performance_min_max_nice_df = pd.DataFrame(
                    performance_min_max_nice, index=index
                )
            else:
                performance_mean_df = performance_mean_df.append(
                    pd.DataFrame(performance_mean, index=index),
                    verify_integrity=True
                )
                performance_min_max_nice_df = performance_min_max_nice_df.append(
                    pd.DataFrame(performance_min_max_nice, index=index),
                    verify_integrity=True
                )

        filename = f"{prefix}_collected_mean_train_test_performance_scores"
        fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=timestamp)
        performance_mean_df.to_csv(fn, sep='\t', na_rep='nan')
        fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=timestamp)
        performance_mean_df.to_excel(fn, na_rep='nan')
        filename = f"{prefix}_collected_formatted_min_max_train_test_performance_scores"
        fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=timestamp)
        performance_min_max_nice_df.to_csv(fn, sep='\t', na_rep='nan')
        fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=timestamp)
        performance_min_max_nice_df.to_excel(fn, na_rep='nan')
        del performance_mean, performance_mean_df
        del performance_min_max_nice, performance_min_max_nice_df

        # collect and save nested cross-validation performance scores
        for i, scoring in enumerate(scorings):

            performance_mean = {}
            performance_min_max_nice = {}
            scores = []
            for score in result[scoring].keys():
                if bool(re.match(r'ncv', score)):
                    scores.append(re.sub(r'ncv_', '', score, count=1))
            for score in sorted(scores):
                for x in ['ncv']:
                    mean = f'mean_{x}_' + score
                    std = f'std_{x}_' + score
                    min_ = f'min_{x}_' + score
                    max_ = f'max_{x}_' + score
                    min_max = f'min_max_{x}_' + score
                    if bool(re.search(r'brier_loss', score)):
                        performance_mean[mean] = [
                            "%.5f" % result[scoring][mean] + " " + u"\u00B1"
                            + " " + "%.6f" % result[scoring][std]
                        ]
                        performance_min_max_nice[min_max] = [
                            "[%.5f, %.5f]" % (result[scoring][min_], result[scoring][max_])
                        ]
                    elif bool(re.search(r'log_loss', score)):
                        performance_mean[mean] = [
                            "%.4f" % result[scoring][mean] + " " + u"\u00B1"
                            + " " + "%.6f" % result[scoring][std]
                        ]
                        performance_min_max_nice[min_max] = [
                            "[%.4f, %.4f]" % (result[scoring][min_], result[scoring][max_])
                        ]
                    else:
                        performance_mean[mean] = [
                            "%.3f" % result[scoring][mean] + " " + u"\u00B1"
                            + " " + "%.6f" % result[scoring][std]
                        ]
                        performance_min_max_nice[min_max] = [
                            "[%.3f, %.3f]" % (result[scoring][min_], result[scoring][max_])
                        ]

            print(f'{time} mean nested cross-validation performance scores '
                  f'({scoring} optimized hyperparameters):')
            pprint(performance_mean)
            print('')

            if i == 0:
                performance_mean_df = pd.DataFrame(
                    performance_mean, index=[scoring]
                )
                performance_min_max_nice_df = pd.DataFrame(
                    performance_min_max_nice, index=[scoring]
                )
            else:
                performance_mean_df = performance_mean_df.append(
                    pd.DataFrame(performance_mean, index=[scoring]),
                    verify_integrity=True
                )
                performance_min_max_nice_df = performance_min_max_nice_df.append(
                    pd.DataFrame(performance_min_max_nice, index=[scoring]),
                    verify_integrity=True
                )

        filename = f"{prefix}_collected_mean_ncv_performance_scores"
        fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=timestamp)
        performance_mean_df.to_csv(fn, sep='\t', na_rep='nan')
        fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=timestamp)
        performance_mean_df.to_excel(fn, na_rep='nan')
        filename = f"{prefix}_collected_formatted_min_max_ncv_performance_scores"
        fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=timestamp)
        performance_min_max_nice_df.to_csv(fn, sep='\t', na_rep='nan')
        fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=timestamp)
        performance_min_max_nice_df.to_excel(fn, na_rep='nan')
        del performance_mean, performance_mean_df
        del performance_min_max_nice, performance_min_max_nice_df
        print(f'{time} end: {ncv.generate_timestamp()}')
        print('=================================================================================')
        print('')

    filename = f"{combination}_collected_key_results"
    fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=timestamp)
    pd.DataFrame(key_results).to_csv(fn, sep='\t', na_rep='nan')
    fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=timestamp)
    pd.DataFrame(key_results).to_excel(fn, na_rep='nan')

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
    warnings.filterwarnings(
        "ignore",
        message=(
            "`np.object` is a deprecated alias for the builtin `object`. "
            "To silence this warning, use `object` by itself. "
            "Doing this will not modify any behavior and is safe."
        ),
        category=DeprecationWarning,
        module=r".*nestedcv"
    )
    warnings.filterwarnings(
        "ignore",
        message=(
            "Divide through zero encountered while trying to calculate the MCC. MCC is set to zero."
        ),
        category=UserWarning,
        module=r".*nestedcv"
    )

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
        help='Kernel combination. Supply SPP, SPR, SRP, or SRR.'
    )
    parser.add_argument(
        '--identifier', dest='identifier', required=True,
        help=('Prefix to identify the precomputed kernel matrices (stored as .npy files).'
              'E.g. kernel_matrix_rescale.')
    )
    args = parser.parse_args()

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
