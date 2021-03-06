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
This module contains the main functionality of the ESPY approach.

@Author: Jacqueline Wistuba-Hamprecht and Bernhard Reuter

"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from typing import Dict, List, Optional, Tuple, Union
import time
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os
from source.utils import make_kernel_matrix


def svm_model(
    *,
    X_train_data: np.ndarray,
    y_train_data: np.ndarray,
    X_test_data: np.ndarray,
    y_test_data: np.ndarray,
) -> SVC:
    """ Initialize SVM model on simulated data
    Initialize SVM model with a rbf kernel on simulated data and
    perform a grid search for kernel parameter evaluation
    Returns the SVM model with the best parameters based on the highest mean AUC score
    Parameters
    ----------
    X_train_data : np.ndarray
        matrix of trainings data
    y_train_data : np.ndarray
        y label for training
    X_test_data : np.ndarray
        matrix of test data
    y_test_data : np.ndarray
        y label for testing
    Returns
    -------
    model : sklearn.svm.SVC object
        trained SVM model on evaluated kernel parameter
    """

    # Initialize SVM model, rbf kernel
    C_range = np.logspace(-3, 3, 7)
    gamma_range = np.logspace(-6, 6, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    scoring = {"AUC": "roc_auc"}

    svm = SVC(kernel="rbf")

    # grid search on simulated data
    # grid search on simulated data
    clf = GridSearchCV(
        SVC(kernel="rbf"),
        param_grid,
        scoring=scoring,
        refit="AUC"
    )
    clf.fit(X_train_data, y_train_data)

    print(
        "The best parameters are %s with a mean AUC score of %0.2f"
        % (clf.best_params_, clf.best_score_)
    )

    # run rbf SVM with parameters fromm grid search,
    # probability has to be TRUE to evaluate features via SHAP
    svm = SVC(
        kernel="rbf",
        gamma=clf.best_params_.get("gamma"),
        C=clf.best_params_.get("C"),
        probability=True
    )

    model = svm.fit(X_train_data, y_train_data)

    y_pred = model.predict(X_test_data)

    AUC = roc_auc_score(y_test_data, y_pred)

    print("AUC score on unseen data:" + " " + str(AUC))

    return model


def multitask_model(
    *,
    kernel_matrix: np.ndarray,
    kernel_parameters: Dict[str, Union[str, float]],
    y_label: np.ndarray
) -> SVC:
    """Initialize multitask-SVM model based on the output of file of the rgscv_multitask.py.

    initialize multitask-SVM model based on evaluated kernel combinations

    Parameter
    ---------
    kernel_matrix : np.ndarray,
        gram matrix
    kernel_parameters : dict,
        parameter combination to initialize multitask-SVM model
    y_label : np.ndarray
        y labels

    Returns
    --------
    multitaskModel: sklearn.svm.SVC object
        trained multitask-SVM model on evaluated kernel parameter
    """

    # set up multitask model based on evaluated parameter
    multitaskModel = SVC(
        kernel="precomputed",
        C=kernel_parameters['C'],
        probability=True,
        random_state=1337,
        cache_size=500,
    )
    multitaskModel.fit(kernel_matrix, y_label)

    return multitaskModel


def make_feature_combination(
    X: pd.DataFrame,
    upperValue: int,
    lowerValue: int,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, List[np.ndarray]]]:
    """Generate vector of feature combination.

    Generate for each single feature a vector based on upper- and lower quantile value.

    Parameter
    ---------
    X : pd.DataFrame
        Data (n_samples x n_features).
    upperValue : int
        Upper percentile given as int.
    lowerValue : int
        Lower percentile given as int.

    Returns
    --------
    statistics : pd.Dataframe
        Statistics (median, lower and upper percentile) of features.
    median : np.ndarray
        The median (over the samples) vector `m`.
    combinations : dict
        Dict of feature combinations.
        `combinations["lower_combinations"]` is a list of length n_features,
        with `combinations["lower_combinations"][i]` equal the median vector
        but having the i-th element `m[i]` replaced by the lower percentile,
        `combinations["upper_combinations"]` is a list of length n_features,
        with `combinations["upper_combinations"][i]` equal the median vector
        but having the i-th element `m[i]` replaced by the upper percentile.
    """
    assert isinstance(upperValue, int), "`upperValue` must be int"
    assert isinstance(lowerValue, int), "`lowerValue` must be int"
    assert 0 <= upperValue <= 100, "`upperValue` must be in [0, 100]"
    assert 0 <= lowerValue <= upperValue, "`lowerValue` must be in [0, upperValue]"

    statistics = X.median(axis=0).to_frame(name="Median")
    statistics["UpperQuantile"] = X.quantile(float(upperValue) / 100.)
    statistics["LowerQuantile"] = X.quantile(float(lowerValue) / 100.)
    statistics = statistics.T

    median = statistics.loc["Median", :].to_numpy().copy()
    lower_quantile = statistics.loc["LowerQuantile", :].to_numpy().copy()
    upper_quantile = statistics.loc["UpperQuantile", :].to_numpy().copy()

    temp_lq = []
    temp_uq = []
    for i in range(len(median)):

        temp1 = median.copy()
        temp1[i] = upper_quantile[i]
        temp_uq.append(temp1)

        temp2 = median.copy()
        temp2[i] = lower_quantile[i]
        temp_lq.append(temp2)

    combinations = {
        "lower_combinations": temp_lq,
        "upper_combinations": temp_uq,
    }

    return statistics, median, combinations


def compute_distance_hyper(
    median: np.ndarray,
    combinations: Dict[str, List[np.ndarray]],
    model: SVC,
    labels: List[str],
    data: Optional[pd.DataFrame] = None,
    kernel_parameters: Optional[Dict[str, Union[float, str]]] = None,
    simulated: bool = False,
) -> pd.DataFrame:
    """Evaluate distance of each single feature to the classification boundary.

    Compute distance of support vectors to classification boundary for each feature
    and the change of each feature by upper and lower quantile on proteome data.

    Parameter
    ---------
    median : np.ndarray
        Median (over the samples) vector.
    combinations : list
        List of combinations of feature values and their upper and lower percentile.
    model : sklearn.svm.SVC
        SVC model.
    labels : list
        List of feature labels.
    data : pd.DataFrame
        Preprocessed proteome data (n_samples x n_features).
    kernel_parameters : dict
        Combination of kernel parameters.
    simulated: bool, default=False
        If True, the ESPY measurement is performed on simulated data.

    Returns
    --------
    get_distance_df : pd.DataFrame
        Dataframe of distance values for each feature per time point.
    """

    # empty array for lower and upper quantile
    get_distance_lower = []
    get_distance_upper = []

    # calc distances for all combinations
    for m in range(len(median)):

        if simulated:

            get_distance_lower.append(
                model.decision_function(combinations["lower_combinations"][m].reshape(1, -1))[0]
            )
            get_distance_upper.append(
                model.decision_function(combinations["upper_combinations"][m].reshape(1, -1))[0]
            )

            d_cons = model.decision_function(median.reshape(1, -1))

        else:

            if isinstance(data, pd.DataFrame) and isinstance(kernel_parameters, dict):

                params = (
                    kernel_parameters['SA'],
                    kernel_parameters['SO'],
                    kernel_parameters['R0'],
                    kernel_parameters['R1'],
                    kernel_parameters['R2'],
                    kernel_parameters['P1'],
                    kernel_parameters['P2'],
                )

                if (params[0] == 'X' or params[1] == 'X') and params[2] != 'X':
                    if params[0] == params[1]:
                        kernel_time_series = 'rbf_kernel'
                    else:
                        raise ValueError("Time series kernel is neither rbf nor sigmoid.")
                elif params[0] != 'X' and params[1] != 'X' and params[2] == 'X':
                    kernel_time_series = 'sigmoid_kernel'
                else:
                    raise ValueError("Time series kernel is neither rbf nor sigmoid.")

                if params[3] != 'X' and params[5] == 'X':
                    kernel_dosage = 'rbf_kernel'
                elif params[3] == 'X' and params[5] != 'X':
                    kernel_dosage = 'poly_kernel'
                else:
                    raise ValueError("Dosage kernel is neither rbf nor polynomial.")

                if params[4] != 'X' and params[6] == 'X':
                    kernel_abSignals = 'rbf_kernel'
                elif params[4] == 'X' and params[6] != 'X':
                    kernel_abSignals = 'poly_kernel'
                else:
                    raise ValueError("Antibody signal kernel is neither rbf nor polynomial.")

                # for lower quantile combination:
                # add test combination as new sample to data
                data.loc["eval_feature", :] = combinations["lower_combinations"][m]
                # calculate gram matrix
                gram_matrix = make_kernel_matrix(
                    data=data,
                    model=params,
                    kernel_time_series=kernel_time_series,
                    kernel_dosage=kernel_dosage,
                    kernel_abSignals=kernel_abSignals,
                )
                single_feature_sample = gram_matrix[0][-1, :len(gram_matrix[0])-1]
                distance = model.decision_function(single_feature_sample.reshape(1, -1))
                get_distance_lower.append(distance[0])

                # for upper quantile combination:
                # add test combination as new sample to data
                data.loc["eval_feature", :] = combinations["upper_combinations"][m]
                # calculate gram matrix
                gram_matrix = make_kernel_matrix(
                    data=data,
                    model=params,
                    kernel_time_series=kernel_time_series,
                    kernel_dosage=kernel_dosage,
                    kernel_abSignals=kernel_abSignals,
                )
                single_feature_sample = gram_matrix[0][-1, :len(gram_matrix[0])-1]
                distance = model.decision_function(single_feature_sample.reshape(1, -1))
                get_distance_upper.append(distance[0])

                # generate consensus feature
                data.loc["eval_feature", :] = median
                gram_matrix = make_kernel_matrix(
                    data=data,
                    model=params,
                    kernel_time_series=kernel_time_series,
                    kernel_dosage=kernel_dosage,
                    kernel_abSignals=kernel_abSignals,
                )
                feature_consensus_sample = gram_matrix[0][-1, :len(gram_matrix[0])-1]

                # compute distance for consensus sample
                d_cons = model.decision_function(feature_consensus_sample.reshape(1, -1))

            else:

                raise ValueError("You must supply `data` and `kernel_parameters`.")

    # get data frame of distances values for median, lower and upper quantile
    get_distance_df = pd.DataFrame(
        [get_distance_upper, get_distance_lower],
        columns=labels,
        index=["UpperQuantile [d]", "LowerQuantile [d]"],
    )

    # add distance of consensus feature
    get_distance_df.loc["consensus [d]"] = np.repeat(d_cons, len(get_distance_df.columns))
    print("Number of evaluated features:")
    print(len(get_distance_df.columns))

    # calculate absolute distance value |d| based on lower and upper quantile
    for col in get_distance_df.columns:
        # get evaluated distance based on upper quantile minus consensus
        d_up = (get_distance_df[col].loc["UpperQuantile [d]"] -
                get_distance_df[col].loc["consensus [d]"])
        get_distance_df.loc["UQ - consensus [d]", col] = d_up

        # get evaluated distance based on lower quantile minus consensus
        d_down = (get_distance_df[col].loc["LowerQuantile [d]"] -
                  get_distance_df[col].loc["consensus [d]"])
        get_distance_df.loc["LQ - consensus [d]", col] = d_down

        # calculate maximal distance value from distance_based on lower quantile
        # and distance_based on upper quantile
        if d_up > 0 and d_down < 0:
            direction = 1.
            d_value = direction * (abs(d_up) + abs(d_down))
        elif d_up < 0 and d_down > 0:
            direction = -1.
            d_value = direction * (abs(d_up) + abs(d_down))
        elif d_up != 0 and d_down == 0:
            d_value = d_up
        elif d_up == 0 and d_down != 0:
            d_value = d_down
        elif d_up == d_down == 0:
            d_value = 0.0
        else:
            d_value = np.nan
        get_distance_df.loc['d', col] = d_value

    # set up final data frame for distance evaluation
    get_distance_df.loc['|d|'] = abs(get_distance_df.loc['d'].values)
    # sort values by abs-value of |d|
    get_distance_df = get_distance_df.T.sort_values(
        by='|d|', axis=0, kind='stable', ascending=False, na_position='last'
    ).T

    # Normalize the distance values
    bool_idx = get_distance_df.loc['d'].notna()
    sum_of_distance = get_distance_df.loc['|d|', :].sum(skipna=True)
    for col in get_distance_df.columns[bool_idx]:
        get_distance_df.loc['d_norm', col] = get_distance_df.loc['d', col] / sum_of_distance
    get_distance_df.loc['|d_norm|'] = abs(get_distance_df.loc['d_norm'].values)
    assert np.allclose(get_distance_df.loc['|d_norm|', :].sum(skipna=True), 1.0), \
        "np.allclose(get_distance_df.loc['|d_norm|', :].sum(skipna=True), 1.0) is False."

    print("Dimension of distance matrix:")
    print(get_distance_df.shape)
    print('')

    return get_distance_df


def make_plot(
    data: pd.DataFrame,
    name: str,
    outputdir: str,
) -> None:
    """

    Parameter
    ---------
    data : pd.DataFrame
        Dataframe of distances.
    name : str
        Output filename.
    outputdir : str
        Directory where the plots are stored as .png and .pdf.
    """
    plt.figure(figsize=(20, 10))
    labels = data.columns

    ax = plt.subplot(111)
    w = 0.3
    opacity = 0.6

    values = np.array(data.T.d_norm)
    clrs = ['red' if (x > 0) else 'blue' for x in values ]

    index = np.arange(len(labels))
    ax.bar(
        index,
        abs(data.loc["|d_norm|"].values),
        width=w,
        color=clrs,
        align="center",
        alpha=opacity
    )
    ax.xaxis_date()

    plt.xlabel('number of features', fontsize=20)
    plt.ylabel('ESPY value', fontsize=20)
    plt.xticks(index, labels, fontsize=10, rotation=90)

    plt.savefig(os.path.join(outputdir, name + ".png"), dpi=600)
    plt.savefig(os.path.join(outputdir, name + ".pdf"), format="pdf", bbox_inches="tight")
    plt.close()


def ESPY_measurement(
    *,
    identifier: str,
    data: pd.DataFrame,
    model: SVC,
    lq: int,
    up: int,
    proteome_data: Optional[pd.DataFrame] = None,
    kernel_parameters: Optional[Dict[str, Union[str, float]]] = None,
 ) -> pd.DataFrame:
    """ESPY measurement.

    Calculate ESPY value for each feature on proteome or simulated data.

    Parameter
    -----------
    identifier : str
        A str that defines, if the input data is real ('whole', 'selective')
        proteome data or simulated ('simulated') data.
    data : pd.Dataframe
        Dataframe of input data.
    model : sklearn.svm.SVC
        SVC model.
    lq : int
        Lower percentile value.
    up : int
        Upper percentile value.
    proteome_data : pd.DataFrame, default=None
        Full proteome dataset.
    kernel_parameters : pd.DataFrame, default=None
        Kernel parameters for real data.
    Returns
    --------
    distance_matrix_for_all_feature_comb : pd.Dataframe
        Dataframe of ESPY values |d| for each feature in simulated data.

    """

    start = time.time()

    statistics, median, combinations = make_feature_combination(
        X=data,
        lowerValue=lq,
        upperValue=up
    )

    print("Statistics of features:")
    print(statistics)
    print('')

    if identifier == 'simulated':

        distance_matrix_for_all_feature_comb = compute_distance_hyper(
            median=median,
            combinations=combinations,
            model=model,
            labels=statistics.columns.to_list(),
            simulated=True,
        )

    elif identifier in ['whole', 'selective']:

        if isinstance(proteome_data, pd.DataFrame) and isinstance(kernel_parameters, dict):

            distance_matrix_for_all_feature_comb = compute_distance_hyper(
                median=median,
                combinations=combinations,
                model=model,
                labels=statistics.columns.to_list(),
                data=proteome_data.iloc[:, 3:],
                kernel_parameters=kernel_parameters,
            )

        else:

            raise ValueError(
                "`proteome_data` and `kernel_parameters` must be supplied."
            )

    else:

        raise ValueError(
            "`identifier` must be one of 'whole', 'selective', or 'simulated'."
        )

    end = time.time()
    print(f"end of computation after: {end - start} seconds.\n")

    return distance_matrix_for_all_feature_comb
