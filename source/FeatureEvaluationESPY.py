"""
This module contains the main functionality of the ESPY approach.

@Author: Jacqueline Wistuba-Hamprecht
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
# import sys
# maindir = '/'.join(os.getcwd().split('/')[:-1])
# sys.path.append(maindir)
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
) -> Tuple[pd.DataFrame, List[float]]:
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
    feature_comb : pd.Dataframe
        Combination of features.
    get_features_comb : list
        List of feature combinations.
    """
    assert isinstance(upperValue, int), "`upperValue` must be int"
    assert isinstance(lowerValue, int), "`lowerValue` must be int"
    assert 0 <= upperValue <= 100, "`upperValue` must be in [0, 100]"
    assert 0 <= lowerValue <= upperValue, "`lowerValue` must be in [0, upperValue]"

    feature_comb = X.median().to_frame(name="Median")
    feature_comb["UpperQuantile"] = X.quantile(float(upperValue) / 100.)
    feature_comb["LowerQuantile"] = X.quantile(float(lowerValue) / 100.)
    feature_comb = feature_comb.T

    feature_comb_arr = feature_comb.to_numpy().copy()

    get_features_comb = []
    get_features_comb.append(feature_comb_arr[0])
    for i in range(len(feature_comb_arr[0])):
        temp1 = feature_comb_arr[0].copy()
        temp1[i] = feature_comb_arr[1][i]
        get_features_comb.append(temp1)

        temp2 = feature_comb_arr[0].copy()
        temp2[i] = feature_comb_arr[2][i]
        get_features_comb.append(temp2)

    return feature_comb, get_features_comb


def compute_distance_hyper(
    combinations: List[float],
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
    for m in range(1, len(combinations)):

        if simulated:
            distance = model.decision_function(combinations[m].reshape(1, -1))
            if m % 2:
                get_distance_upper.append(distance[0])
            else:
                get_distance_lower.append(distance[0])

            d_cons = model.decision_function(combinations[0].reshape(1, -1))

        else:
            params = (
                kernel_parameters['SA'],
                kernel_parameters['SO'],
                kernel_parameters['R0'],
                kernel_parameters['R1'],
                kernel_parameters['R2'],
                kernel_parameters['P1'],
                kernel_parameters['P2'],
            )

            # add test combination as new sample to data
            data.loc["eval_feature", :] = combinations[m]

            gram_matrix = make_kernel_matrix(
                data=data,
                model=params,
                kernel_time_series='rbf_kernel',
                kernel_dosage='rbf_kernel',
                kernel_abSignals='rbf_kernel',
            )
            single_feature_sample = gram_matrix[0][-1, :len(gram_matrix[0])-1]
            # print(single_feature_sample.reshape(1,-1))

            distance = model.decision_function(single_feature_sample.reshape(1, -1))
            # print(distance)
            if m % 2:
                get_distance_upper.append(distance[0])
            else:
                get_distance_lower.append(distance[0])
            # print(m)
            # print(distance)

            # generate consensus feature
            data.loc["eval_feature", :] = combinations[0]
            gram_matrix = make_kernel_matrix(
                data=data,
                model=params,
                kernel_time_series='rbf_kernel',
                kernel_dosage='rbf_kernel',
                kernel_abSignals='rbf_kernel',
            )
            feature_consensus_sample = gram_matrix[0][-1, :len(gram_matrix[0])-1]
            # print(feature_consensus_sample.shape)

            # compute distance for consensus sample
            d_cons = model.decision_function(feature_consensus_sample.reshape(1, -1))
            # print(d_cons)

    # get data frame of distances values for median, lower and upper quantile
    # print("Matrix of distances for Upper-/Lower- quantile per feature")
    get_distance_df = pd.DataFrame([get_distance_upper, get_distance_lower], columns=labels)
    # print(get_distance_df.shape)

    # add distance of consensus feature
    get_distance_df.loc["consensus [d]"] = np.repeat(d_cons, len(get_distance_df.columns))
    # print(get_distance_df["med"].iloc[0])
    # print(get_distance_df.shape)
    print("Number of evaluated features:")
    print(len(get_distance_df.columns))

    # calculate absolute distance value |d| based on lower and upper quantile
    d_value = 0
    for col in get_distance_df:
        # get evaluated distance based on upper quantile minus consensus
        val_1 = get_distance_df[col].iloc[0] - get_distance_df[col].iloc[2]
        # print(val_1)
        get_distance_df.loc["UQ - consensus [d]", col] = val_1

        # get evaluated distance based on lower quantile minus consensus
        val_2 = get_distance_df[col].iloc[1] - get_distance_df[col].iloc[2]
        # print(val_2)
        get_distance_df.loc["LQ - consensus [d]", col] = val_2

        # calculate maximal distance value from distance_based on lower quantile
        # and distance_based on upper quantile
        if val_1 >= 0 or val_1 < 0 and val_2 > 0 or val_2 <= 0:
            a = max(abs(val_1), abs(val_2))
            if a == abs(val_1):
                d_value = val_1
            else:
                d_value = val_2

        get_distance_df.loc["|d|", col] = d_value

    # set up final data frame for distance evaluation
    get_distance_df = get_distance_df.rename(
        {0: "UpperQuantile [d]", 1: "LowerQuantile [d]"},
        axis='index'
    )
    get_distance_df.loc['|d|'] = abs(get_distance_df.loc["|d|"].values)
    get_distance_df = get_distance_df.T.sort_values(by="|d|", ascending=False).T
    # Norm distance value by distance_nome = distance_value/sum(distance_values)
    sum_of_distance = get_distance_df.loc['|d|', :].sum()

    for col in get_distance_df.columns:
        get_distance_df.loc['|d|', col] = get_distance_df.loc['|d|', col]/sum_of_distance

    assert get_distance_df.loc['|d|', :].sum() == 1.0, "get_distance_df.loc['|d|', :].sum() != 1.0"
    # sort values by abs-value of |d|
    get_distance_df.loc["sort"] = abs(get_distance_df.loc["|d|"].values)
    print("Dimension of distance matrix:")
    print(get_distance_df.shape)
    print("end computation")

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

    index = np.arange(len(labels))
    ax.bar(
        index,
        abs(data.loc["|d|"].values),
        width=w,
        color="darkblue",
        align="center",
        alpha=opacity
    )
    ax.xaxis_date()

    plt.xlabel('number of features', fontsize=20)
    plt.ylabel('ESPY value', fontsize=20)
    plt.xticks(index, labels, fontsize=10, rotation=90)

    plt.savefig(os.path.join(outputdir, name + ".png"), dpi=600)
    plt.savefig(os.path.join(outputdir, name + ".pdf"), format="pdf", bbox_inches="tight")
    plt.show()


def ESPY_measurement(
    *,
    identifier: str,
    data: pd.DataFrame,
    model: SVC,
    lq: int,
    up: int,
    proteome_data: Optional[pd.DataFrame] = None,
    kernel_parameters: Optional[pd.DataFrame] = None,
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

    combinations, all_feature_combinations = make_feature_combination(
        X=data,
        lowerValue=lq,
        upperValue=up
    )

    print("Combination of features:")
    print(combinations)

    if identifier == 'simulated':

        distance_matrix_for_all_feature_comb = compute_distance_hyper(
            combinations=all_feature_combinations,
            model=model,
            labels=combinations.columns.to_list(),
            simulated=True,
        )

    elif identifier in ['whole', 'selective']:

        distance_matrix_for_all_feature_comb = compute_distance_hyper(
            combinations=all_feature_combinations,
            model=model,
            labels=combinations.columns.to_list(),
            data=proteome_data.iloc[:, 3:],
            kernel_parameters=kernel_parameters,
        )
    else:

        raise ValueError(
            "`identifier` must be one of 'whole', 'selective', or 'simulated'."
        )

    end = time.time()
    print("end of computation after: ", str(end - start), "seconds")

    return distance_matrix_for_all_feature_comb
