"""
This module contains the main functionality of the ESPY approach.

@Author: Jacqueline Wistuba-Hamprecht
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from typing import Dict, List, Optional, Tuple, Union
import time
import os
import sys
maindir = '/'.join(os.getcwd().split('/')[:-1])
sys.path.append(maindir)
from source.utils import make_kernel_matrix


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
        Upper quantile in percent given as int.
    lowerValue : int
        Lower quantile in precent given as int.

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

    feature_comb_arr = feature_comb.values.copy()

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


def compute_distance_hyper_proteome(
    combinations: List[float],
    model: SVC,
    input_labels: pd.DataFrame,
    data: pd.DataFrame,
    kernel_parameters: Dict[str, Union[float, str]],
):
    """Evaluate distance of each single feature to classification boundary

    Compute distance of support vectors to classification boundary for each feature
    and the change of each feature by upper and lower quantile on proteome data

    Parameter
    ---------
    combinations : list
        combination of feature value and its upper and lower quantile
    model : sklearn.svm.SVC
        SVC model
    input_labels : pd.DataFrame
        list of feature labels
    data : pd.DataFrame
        pre-processed proteome data, n x m matrix (n = samples as rows, m = features as columns)
    kernel_paramters : dict
        combination of kernel parameter

    Returns
    --------
    get_distance_df : pd.DataFrame
     dataframe of distance values for each feature per time point
    """
    params = (
        kernel_parameters['SA'],
        kernel_parameters['SO'],
        kernel_parameters['R0'],
        kernel_parameters['R1'],
        kernel_parameters['R2'],
        kernel_parameters['P1'],
        kernel_parameters['P2'],
    )

    # get labels, start with first PF-Antigen name
    labels = list(input_labels.columns.values)

    # empty array for lower and upper quantile
    get_distance_lower = []
    get_distance_upper = []

    # calc distances for all combinations
    for m in range(1, len(combinations)):

        # add test combination as new sample to
        data.loc["eval_feature", :] = combinations[m]

        gram_matrix = make_kernel_matrix(
            data=data,
            model=params,
            kernel_time_series='rbf_kernel',
            kernel_dosage='rbf_kernel',
            kernel_abSignals='rbf_kernel',
        )
        single_feature_sample = gram_matrix[-1, :len(gram_matrix[0])-1]
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
    feature_consensus_sample = gram_matrix[-1, :len(gram_matrix[0])-1]
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
    # sort values by abs-value of |d|
    get_distance_df.loc["sort"] = abs(get_distance_df.loc["|d|"].values)
    print("Dimension of distance matrix:")
    print(get_distance_df.shape)
    print("end computation")

    return get_distance_df


def compute_disctance_hyper_simulateddata(
        combinations: List[float],
        model: SVC,
        input_labels: pd.DataFrame,
):
    """Evaluate distance of each single feature to classification boundary

    Compute distance of support vectors to classification boundary for each feature
    and the change of each feature by upper and lower quantile on simulated data

    Paramter
    --------
    combinations : list
        dataframe of combination of feature value, itself, upper and lower quantile
    model : sklearn.svm.SVC
        SVC model
    input-labels : pd.DatFrame
        list of feature labels

    Returns
    -------
    get_distance_df : pd.Dataframe
     dataframe of ESPY values for each feature per time point
    """
    # reshape test data
    combinations = np.asarray(combinations)
    # print(combinations)
    # get labels
    labels = list(input_labels)
    # get distance
    get_distance_lower = []
    get_distance_upper = []
    # calc distances for all combinations
    for m in range(1, len(combinations)):
        distance = model.decision_function(combinations[m].reshape(1, -1))
        if m % 2:
            get_distance_upper.append(distance[0])
        else:
            get_distance_lower.append(distance[0])
        # print(distance)

    # calc distance for consensus sample
    d_cons = model.decision_function(combinations[0].reshape(1, -1))
    # print(d_cons)
    # get data frame of distances values for median, lower and upper quantile
    get_distance_df = pd.DataFrame([get_distance_upper, get_distance_lower], columns=labels)

    # add median
    get_distance_df.loc["Median"] = np.repeat(d_cons, len(get_distance_df.columns))
    # print(get_distance_df)

    # calculate absolute distance value |d| from lower and upper quantile
    for col in get_distance_df:
        # print(col)
        # distance_%75
        val_1 = get_distance_df[col].iloc[0] - get_distance_df[col].iloc[2]
        # print(val_1)
        # distance_%75
        val_2 = get_distance_df[col].iloc[1] - get_distance_df[col].iloc[2]
        # print(val_2)

        # calculate maximal distance value from distance_25% and distance_75%
        # TODO what if both values are the same size?
        if val_1 > 0 or val_1 < 0 and val_2 > 0 or val_2 < 0:
            a = max(abs(val_1), abs(val_2))

        if a == abs(val_1):
            d_value = abs(val_1)
        else:
            d_value = abs(val_2)
        # print(d_value)
        get_distance_df.loc["|d|", col] = d_value
    # rename dataframe rows

    get_distance_df = get_distance_df.rename(
        {0: "UpperQuantile", 1: "LowerQuantile"},
        axis='index'
    )
    get_distance_df = get_distance_df.T.sort_values(by="|d|", ascending=False).T
    # sort values by abs-value of |d|
    get_distance_df.loc["sort"] = abs(get_distance_df.loc["|d|"].values)

    return get_distance_df


def ESPY_measurement(
        *,
        identifier: str,
        data: pd.DataFrame,
        model: SVC,
        lq: int,
        up: int,
        proteome_data: Optional[pd.DataFrame],
        kernel_parameters: Optional[pd.DataFrame],
 ) -> pd.DataFrame:
    """ESPY measurement
    Calculate ESPY value for each feature on simulated data

    Parameter
    -----------
    identifier : str
        A str that defines, if the input data is real ('whole', 'selective')
        proteome data or simulated ('simulated') data.
    data : pd.Dataframe
        dataframe of input data
    model : sklearn.svm.SVC
        SVC model
    lq:  int
        value of lower quantile
    up : int
        value of upper quantile
    proteome_data : Optional[pd.DataFrame]
        Full proteome dataset.
    kernel_parameters : Optional[pd.DataFrame]
        Kernel parameters for real data.
    Returns
    --------
    distance_matrix_for_all_feature_comb : pd.Dataframe
        dataframe of ESPY value |d| for each feature in simulated data

    """

    start = time.time()

    combinations, all_feature_combinations = make_feature_combination(
        X=data,
        lowerValue=lq,
        upperValue=up
    )

    print("Combination of feature")
    print(combinations)

    if identifier == 'simulated':

        distance_matrix_for_all_feature_comb = compute_disctance_hyper_simulateddata(
            combinations=all_feature_combinations,
            model=model,
            input_labels=combinations)

    elif identifier in ['whole', 'selective']:

        distance_matrix_for_all_feature_comb = compute_distance_hyper_proteome(
            combinations=all_feature_combinations,
            model=model,
            input_labels=combinations,
            data=proteome_data.iloc[:, 3:],
            kernel_paramter=kernel_parameters
        )

    else:

        raise ValueError(
            "`identifier` must be one of 'whole', 'selective', or 'simulated'."
        )

    end = time.time()
    print("end of computation after: ", str(end - start), "seconds")

    return distance_matrix_for_all_feature_comb
