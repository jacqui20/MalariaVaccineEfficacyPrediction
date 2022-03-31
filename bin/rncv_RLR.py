'''
Find the best parameters using 10-times repeated NestedGridsearchCV.

Will save the results to a various .tsv/.xslx files.

Activate associated environment first: conda activate malaria_env
'''
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
# from shutil import rmtree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from typing import Dict, List, Any
# from tempfile import mkdtemp
sys.path.append('/home/breuter/NestedGridSearchCV')
import nestedcv as ncv
sys.path.append('/home/breuter/MalariaVaccineEfficacyPrediction')
from source.utils import CustomPredefinedSplit, assign_folds


def main(
    ANA_PATH: str,
    DATA_PATH: str,
    identifier: str,
) -> None:

    # number of repetitions:
    Nexp2 = 10

    param_grid = {
        'logisticregression__l1_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'logisticregression__C': [1.e-4, 1.e-3, 1.e-2, 1.e-1, 1.e0, 1.e1, 1.e2, 1.e3, 1.e4],
    }

    # Generate a timestamp used for file naming
    timestamp = ncv.generate_timestamp()

    print('')
    print("Identifier:", identifier)
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

    labels_all = pd.read_table(
        ('/home/breuter/MalariaVaccineEfficacyPrediction/data/'
         'precomputed_multitask_kernels/unscaled/target_label_vec.csv'),
        sep=',',
        index_col=0
    )
    groups_all = labels_all.loc[:, 'group'].to_numpy()
    y_all = labels_all.loc[:, 'Protection'].to_numpy()

    # initialize the key result collector:
    key_results: Dict[str, List[Any]] = {}
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
        prefix = f'{time}'

        rng = np.random.RandomState(0)

        data = pd.read_table(
            os.path.join(DATA_PATH, f'{identifier}_data_{time}.csv'),
            sep=',',
            index_col=0
        )
        X = data.iloc[:, 2:].to_numpy()
        groups = data.loc[:, 'group'].to_numpy()
        y = data.loc[:, 'Protection'].to_numpy()

        print('shape of binary response array:', y.size)
        print('number of positives:', np.sum(y))
        print('number of positives divided by total number of samples:', np.sum(y)/y.size)
        print('')

        # initialize test folds and CV splitters for outer CV
        outer_cv = []
        print('Predefined CV folds:')
        print('---------------------------------------------------------------------------------')
        for rep in range(Nexp2):
            print(f'CV folds for repetition {rep}:')
            test_fold, train_fold = assign_folds(
                y_all, groups_all, 40, step, random_state=rep, print_info=False
            )
            test_fold = test_fold[40 * step: 40 * (step + 1)]
            train_fold = train_fold[40 * step: 40 * (step + 1)]
            assert np.all(test_fold == train_fold), \
                f"test_fold != train_fold: {test_fold} != {train_fold}"
            outer_cv.append(CustomPredefinedSplit(test_fold, train_fold))
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
            'Nexp1': 5,
            'Nexp2': Nexp2,
            'n_jobs': 8,
        }

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
            # memory=cachedir,
        )

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

    filename = "collected_key_results"
    fn = ncv.filename_generator(filename, '.tsv', directory=maindir, timestamp=timestamp)
    pd.DataFrame(key_results).to_csv(fn, sep='\t', na_rep='nan')
    fn = ncv.filename_generator(filename, '.xlsx', directory=maindir, timestamp=timestamp)
    pd.DataFrame(key_results).to_excel(fn, na_rep='nan')

    print('End:', ncv.generate_timestamp())
    # return key_results


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
        '--identifier', dest='identifier', required=True,
        help=('Prefix to identify the precomputed kernel matrices (stored as .npy files).'
              'E.g. kernel_matrix_rescale.')
    )
    args = parser.parse_args()

    # cachedir = mkdtemp()

    try:
        main(
            args.analysis_dir,
            args.data_dir,
            args.identifier,
        )
    finally:

        warning_file.close()
        # rmtree(cachedir)
