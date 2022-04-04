"""
This module contains the main functionality of the ESPY measurment approach.
First the multitask-SVM classification model is initialized with the kernel matrix based on the evaluated kernel
parameter from 10time repeated 5-fold grid search CV. Second informative features are evaluated based on the initialized
multitask-SVM model. 
Created on: 25.05.2019

@Author: Jacqueline Wistuba-Hamprecht


"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel, sigmoid_kernel, polynomial_kernel
from sklearn.model_selection import train_test_split
import time
import os
import sys
maindir = '/'.join(os.getcwd().split('/')[:-1])
sys.path.append(maindir)
from source.utils import make_symmetric_matrix_psd, normalize, multitask, select_timepoint, get_parameters, sort_proteome_data,\
    make_plot, multitask_model, initialize_svm_model


def get_kernel_paramter(kernel_parameter):
    """ Returns the combination of kernel parameters from the results of the
        multitask-SVM approach based on the highest mean AUC.

        see results of the Parser_multitask_SVM.py module

        Args: kernel_parameter: results of the multitask-SVM approach as .csv file

        Returns:
            pam (list): Combination of kernel parameter for the combination of kernel functions for the
            multitask-SVM classifier
            based on the highest mean AUC value
            best_AUC (float): highest mean AUC value
        """
    pam_roc_auc = kernel_parameter[kernel_parameter['scoring'].isin(['roc_auc'])]
    pam = pam_roc_auc['best_params']
    #print(pam)
    pam = pam.str.split(" ", expand=True)
    print(pam)
    return pam


def make_feature_combination(
        X: pd.DataFrame,
        upperValue: int,
        lowerValue: int
):
    """Generate vector of feature combination.

    Generate for each single feature a vector based on Upper- and LowerQuantile value

    Parameter
    ---------
    X: pd.DataFrame
        dataframe of simulated features
    upperValue: int
        value of upper quantile
    lowerValue: int
        value of lower quantile

    Returns
    --------
    feature_comb: pd.Dataframe
        combination of features
    get_features_comb: dict
        dictionary of feature combinations
    """

    feature_comb = X.median().to_frame(name="Median")
    feature_comb["UpperQuantile"] = X.quantile(upperValue / 100)
    feature_comb["LowerQuantile"] = X.quantile(lowerValue / 100)
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


def feature_gram_matrix(
        data: pd.DataFrame,
        kernel_parameter: dict
):
    """Initialize feature as gram matrix of dimension 1x120

    Define feature for distance measure on multitask-SVM classification model

    Parameter
    ---------
    data: pd.DataFrame,
        dataframe of feature (n= samples, m= feature)
    kernel_parameter: dict
        combination of kernel parameter

    Returns
    --------
    test_sample: np.ndarray
        matrix of defined feature with dimension 1x120

            """
    # get start point of antibody reactivity signals in data (n_p)
    AB_signal_start = data.columns.get_loc("TimePointOrder") + 1
    AB_signals = data[data.columns[AB_signal_start:]]

    # extract time points to vector (y_t)
    time_series = data["TimePointOrder"]

    # extract dosage to vector (y_d)
    dose = data["Dose"]

    # kernel parameter for rbf kernel function for time points
    kernel_pamR0 = pd.to_numeric(kernel_parameter[5].str.split(",", expand=True)[0])
    # print("gamma value for rbf kernel for time points" + str(kernel_pamR0))

    # kernel parameter for rbf kernel function for dosis
    kernel_pamR1 = pd.to_numeric(kernel_parameter[7].str.split(",", expand=True)[0])
    # print("gamma value for rbf kernel for dosis" + str(kernel_pamR1))

    # kernel parameter for rbf kernel function for ab signals
    kernel_pamR2 = pd.to_numeric(kernel_parameter[9].str.split(",", expand=True)[0])
    # print("gamma value for rbf kernel for ab signals" + str(kernel_pamR2))

    # set up kernel matrix for time series
    time_series_kernel_matrix = rbf_kernel(time_series.values.reshape(len(time_series), 1), gamma=kernel_pamR0)

    # set up kernel matrix for time series
    dose_kernel_matrix = rbf_kernel(dose.values.reshape(len(dose), 1), gamma=kernel_pamR1)

    # set up kernel matrix for ab signals
    AB_signals_kernel_matrix = rbf_kernel(AB_signals, gamma= kernel_pamR2)

    # pre-compute multitask kernel matrix K((np, nt),(np', nt'))
    multi_AB_signals_time_series_kernel_matrix = multitask(
        AB_signals_kernel_matrix,
        time_series_kernel_matrix,
    )
    # pre-compute multitask kernel matrix K((np, nt, nd),(np', nt', nd'))
    multi_AB_signals_time_dose_kernel_matrix, c_list, info_list = make_symmetric_matrix_psd(
        multitask(
            multi_AB_signals_time_series_kernel_matrix,
            dose_kernel_matrix,
        )
    )

    if c_list:
        print(
            "multi_AB_signals_time_dose_kernel_matrix kernel had to be corrected.\n"
            f"model: {kernel_parameter}"
        )
    # print("Dimension of kernel matrix with feature")
    # print(AB_signals_kernel_matrix.shape)

    # set up feature test sample for distance evaluation
    test_sample = AB_signals_kernel_matrix[-1, :len(AB_signals_kernel_matrix[0])-1]

    return test_sample


def compute_distance_hyper_proteome(
        combinations: np.ndarray,
        model: SVC,
        input_labels: list,
        data: pd.DataFrame,
        kernel_paramter: dict
):
    """Evaluate distance of each single feature to classification boundary

    Compute distance of support vectors to classification boundary for each feature and the change of
    each feature by upper and lower quantile on proteome data

    Parameter
    ---------
    combinations: np.ndarray
        combination of feature value and its upper and lower quantile
    model: sklearn.svm.SVC
        SVC model
    input_labels: list
        list of feature labels
    data: pd.DataFrame
        pre-processed proteome data, n x m matrix (n = samples as rows, m = features as columns)
    kernel_paramter: dict
        combination of kernel parameter

    Returns
    --------
    get_distance_df: pd.DataFrame
     dataframe of distance values for each feature per time point
"""


    # get labels, start with first PF-Antigen name
    labels = list(input_labels.columns.values)

    # empty array for lower and upper quantile
    get_distance_lower = []
    get_distance_upper = []

    # calc distances for all combinations
    for m in range(1, len(combinations)):
        # add test combination as new sample to
        data.loc["eval_feature", :] = combinations[m]
        single_feature_sample = feature_gram_matrix(data, kernel_paramter)
        #print(single_feature_sample.reshape(1,-1))
        distance = model.decision_function(single_feature_sample.reshape(1, -1))
        #print(distance)
        if m % 2:
            get_distance_upper.append(distance[0])
        else:
            get_distance_lower.append(distance[0])
        #print(m)
        #print(distance)

    # generate consensus feature
    data.loc["eval_feature", :] = combinations[0]
    feature_consensus_sample = feature_gram_matrix(data, kernel_paramter)
    #print(feature_consensus_sample.shape)
    # compute distance for consensus sample
    d_cons = model.decision_function(feature_consensus_sample.reshape(1, -1))
    #print(d_cons)
    # get data frame of distances values for median, lower and upper quantile
    # print("Matrix of distances for Upper-/Lower- quantile per feature")
    get_distance_df = pd.DataFrame([get_distance_upper, get_distance_lower], columns=labels)
    # print(get_distance_df.shape)

    # add distance of consensus feature
    get_distance_df.loc["consensus [d]"] = np.repeat(d_cons, len(get_distance_df.columns))
    #print(get_distance_df["med"].iloc[0])
    #print(get_distance_df.shape)
    print("Number of evaluated features:")
    print(len(get_distance_df.columns))

    # calculate absolute distance value |d| based on lower and upper quantile
    d_value = 0
    for col in get_distance_df:
        # get evaluated distance based on upper quantile minus consensus
        val_1 = get_distance_df[col].iloc[0] - get_distance_df[col].iloc[2]
        #print(val_1)
        get_distance_df.loc["UQ - consensus [d]", col] = val_1

        # get evaluated distance based on lower quantile minus consensus
        val_2 = get_distance_df[col].iloc[1] - get_distance_df[col].iloc[2]
        #print(val_2)
        get_distance_df.loc["LQ - consensus [d]", col] = val_2

        # calculate maximal distance value from distance_based on lower quantile and distance_based on upper quantile
        if val_1 >= 0 or val_1 < 0 and val_2 > 0 or val_2 <= 0:
            a = max(abs(val_1), abs(val_2))
            if a == abs(val_1):
                d_value = val_1
            else:
                d_value = val_2

        get_distance_df.loc["|d|", col] = d_value

    # set up final data frame for distance evaluation
    get_distance_df = get_distance_df.rename({0: "UpperQuantile [d]", 1: "LowerQuantile [d]"}, axis='index')
    get_distance_df.loc['|d|'] = abs(get_distance_df.loc["|d|"].values)
    get_distance_df = get_distance_df.T.sort_values(by="|d|", ascending=False).T
    #sort values by abs-value of |d|
    get_distance_df.loc["sort"] = abs(get_distance_df.loc["|d|"].values)
    print("Dimension of distance matrix:")
    print(get_distance_df.shape)
    print("end computation")

    return get_distance_df


def compute_disctance_hyper_simulateddata(
        combinations: pd.DataFrame,
        model: SVC,
        input_labels: list
):
    """Evaluate distance of each single feature to classification boundary

    Compute distance of support vectors to classification boundary for each feature and the change of
    ach feature by upper and lower quantile on simulated data

    Paramter
    --------
    combinations: pd.Dataframe
        dataframe of combination of feature value, itself, upper and lower quantile
    model: sklearn.svm.SVC
        SVC model
    input-labels: list
        list of feature labels

    Returns
    -------
    get_distance_df: pd.Dataframe
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
    temp = []
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

    get_distance_df = get_distance_df.rename({0: "UpperQuantile", 1: "LowerQuantile"}, axis='index')
    get_distance_df = get_distance_df.T.sort_values(by="|d|", ascending=False).T
    # sort values by abs-value of |d|
    get_distance_df.loc["sort"] = abs(get_distance_df.loc["|d|"].values)

    return get_distance_df


def ESPY_measurment(
        data: pd.DataFrame,
        model: SVC,
        lq: int,
        up: int,
        filename: str,
        proteome_data=None,
        kernel_paramter=None):
    """ESPY measurement
    Calculate ESPY value for each feature on simulated data

    Parameter
    -----------
    data: pd.Dataframe
        dataframe of input data
    model: sklearn.svm.SVC
        SVC model
    lq: int
        value of lower quantile
    up: int
        value of upper quantile
    filename: str
        name of input file

    Returns
    --------
    distance_matrix_for_all_feature_comb: pd.Dataframe
        dataframe of ESPY value |d| for each feature in simulated data

    """

    start = time.time()

    combinations, all_feature_combinations = make_feature_combination(
            X=data,
            lowerValue=lq,
            upperValue=up)

    print("Combination of feature")
    print(combinations)

    if filename == "simulated_data.csv":
        distance_matrix_for_all_feature_comb = compute_disctance_hyper_simulateddata(
            combinations=all_feature_combinations,
            model=model,
            input_labels=combinations)

    elif filename == "preprocessed_whole_data.csv" or "preprocessed_selective_data.csv":
        distance_matrix_for_all_feature_comb = compute_distance_hyper_proteome(
            combinations=all_feature_combinations,
            model=model,
            input_labels=combinations,
            data=proteome_data.iloc[:, 3:],
            kernel_paramter=kernel_paramter)

    end = time.time()
    print("end of computation after: ", str(end - start), "seconds")

    return distance_matrix_for_all_feature_comb


