"""
This module contains the main functionality of the feature selection approach.
First the multitask-SVM classification model is initilaized with the parameter combinations evaluated based
on the results from Parser_multitask_SVM.py module. Second informative features are evaluated.
Created on: 25.05.2019

@Author: Jacqueline Wistuba-Hamprecht


"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics.pairwise import rbf_kernel, sigmoid_kernel, polynomial_kernel
import time
import os
import sys
maindir = '/'.join(os.getcwd().split('/')[:-1])
sys.path.append(maindir)
from utils import make_symmetric_matrix_psd, normalize




def multitask(
        a: np.ndarray,
        b: np.ndarray,
) -> np.ndarray:
    """ Multitask approach

    Combination of two kernel matrices are combined by element-wise multiplication to one kernel matrix.

    Parameters
    ----------
    a : np.ndarray
        Kernel matrix a.
    b : np.ndarray
        Kernel matrix b.

    Returns
    -------
    a * b : np.ndarray
        Element-wise multiplicated kernel matrix a * b.
    """
    return a * b


def multitask_model(kernel_matrix, kernel_parameter, y_label):
    """Set up multitask-SVM model based on the output of file of the rgscv_multitask.py.

    set up multitask-SVM model based on evaluated kernel combinations from Parser_multitask_SVM.py module.

    Args: data (matrix): n x m matrix (n = samples as rows, m = features as columns)
          kernel_parameter (dictionary): combination of kernel parameter

    Returns: multitaskModel (SVM model): classification model based on the multitask-SVM approach

    """

    # extract cost value C
    #print(kernel_parameter[15])
    C_reg = pd.to_numeric(kernel_parameter[15].str.split("}", expand=True)[0])
    #print(C_reg)

    # set up multitask model based on evaluated parameter
    multitaskModel = svm.SVC(kernel="precomputed",
                             C=C_reg,
                             probability=True,
                             random_state=1337,
                             cache_size=500)
    multitaskModel.fit(kernel_matrix, y_label)

    return multitaskModel


def select_time_point(kernel_parameter, time_point):
    """
    Selection of the time point to run ESPY measurment

    Parameter:
    ---------
    kernel_parameter: dataframe
        performance results per time point
    time_point: str
        preferable time point

    Returns:
    --------
    X: np.darray
        matrix of performance scores per time point

    """
    X = kernel_parameter[kernel_parameter['time'].isin([time_point])]
    #print(X)
    return X

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


def sort(
        data: pd.DataFrame,
) -> pd.DataFrame:
    """ Sorting

    Input data is sorted by time point to keep same patient over all four time points in order.

    Parameters
    ----------
    data : pd.DataFrame
        Raw proteome data, n x m pd.DataFrame (n = samples as rows, m = features as columns)

    Returns
    -------
    data : pd.DataFrame
        Returns sorted DataFrame
    """
    data.sort_values(by=["TimePointOrder","Patient"], inplace=True)
    data.reset_index(inplace = True)
    data.drop(columns = ['index'], inplace=True)
    return data

def make_feature_combination(data, upperValue, lowerValue):
    """Generate vector of feature combination.

        Generate for each single feature a vector based on Upper- and LowerQuantile value

        Args: data (matrix): matrix of proteome data, n x m matrix (n = samples as rows, m = features as columns)
              upperValue (int): value of upper quantile
              lowerValue (int): value of lower quantile

        Returns: feature_comb (matrix): combination of features
                 get_features_comb (series):  series of feature combinations
        """
    # get start point of antibody reactivity signals in data
    feature_start = data.columns.get_loc("Protection") + 1
    # get matrix of feature values
    features = data[data.columns[feature_start:]]
    #print(feature_start)

    # generate feature combination based on median signal intensity, intensity for upper quantile and lower quantile
    # x^p = {x1, x2, ... , x_j-1, x^p_j, X_j+1, ...., x_m}^T
    feature_comb = features.median().to_frame(name = "Median")
    feature_comb["UpperQuantile"] = features.quantile(upperValue/100)
    feature_comb["LowerQuantile"] = features.quantile(lowerValue/100)
    feature_comb = feature_comb.T
    feature_comb_arr = feature_comb.values.copy()

    # concatenate feature combinations in series
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


def feature_gram_matrix(data, kernel_parameter):
    """Define feature

            Define feature for prediction in multitask-SVM classification model

            Args: data (matrix): matrix of feature (n= samples, m= feature)
                  kernel_parameter (dictionary): combination of kernel parameter

            Returns: test_sample (matrix): matrix of defined feature

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


def compute_distance_hyper(combinations, model, input_labels, data, kernel_paramter):
    """Evaluate distance of each single feature to classification boundary

                Compute distance of support vectors to classification boundery for each feature and the change of
                each feature by upper and lower quantile

                Args: combinations (vector): vector of combination of feature value, itself, upper and lower quantile
                      model (SVM model): multitask-SVM model
                      input-labels (list) = list of feature names
                      data (matrix): pre-processed proteome data, n x m matrix (n = samples as rows, m = features as columns)
                      kernel_parameter (dictionary): combination of kernel parameter

                Returns: get_distance_df (matrix): matrix of distance values for each feature per time point

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
    get_distance_df = get_distance_df.T.sort_values(by="|d|", ascending=False).T
    #sort values by abs-value of |d|
    get_distance_df.loc["sort"] = abs(get_distance_df.loc["|d|"].values)
    print("Dimension of distance matrix:")
    print(get_distance_df.shape)
    print("end computation")

    return get_distance_df


def ESPY_measurment(target_label,
                    kernel_parameter,
                    kernel_matrix,
                    timePoint,
                    t_nm,
                    proteome_data,
                    uq,
                    lq,
                    outputdir,
                    output_filename):
    """ MAIN function of the feature evaluation approach.

                the distances of each feature to the classification boundery are evaluated.


                Paramters:
                ----------
                target_label (dataframe): dataframe of target values
                kernel_parameter (dictionary): combination of kernel parameter
                kernel_matrix (matrix): 120x120 gram matrix
                timePoint (str): timepoint
                t_nm (int): int of time point
                proteome_data (dataframe): dataframe of proteome data (n = rows of samples, m = columns of features)
                uq (int): value of upper quantile
                lq (int): value of lower quantile
                outputdir (path): path where results are stored
                output_filename(str): name of output file


                Returns: distances_for_all_feature_comb (matrix): matrix of distance values for each feature per time point
                         combinations (matrix): combination of features
                         timePoint (int): time point of feature evaluation
                """

    start = time.time()

    time_point = select_time_point(kernel_parameter, timePoint)
    param_combi_RRR = get_kernel_paramter(time_point)

    kernel_pamR0 = pd.to_numeric(param_combi_RRR[5].str.split(",", expand=True)[0])
    print("gamma value for rbf kernel for time point series " + str(kernel_pamR0))
    kernel_pamR1 = pd.to_numeric(param_combi_RRR[7].str.split(",", expand=True)[0])
    print("gamma value for rbf kernel for dose " + str(kernel_pamR1))
    kernel_pamR2 = pd.to_numeric(param_combi_RRR[9].str.split(",", expand=True)[0])
    print("gamma value for rbf kernel for ab signals " + str(kernel_pamR2))
    print('')

    y_label = target_label.loc[:, 'Protection'].to_numpy()
    multitask_classifier = multitask_model(kernel_matrix, param_combi_RRR, y_label)
    proteome_data = sort(proteome_data)
    print("Are values in proteome data floats: " + str(np.all(np.isin(proteome_data.dtypes.to_list()[5:], ['float64']))))
    data_at_timePoint = proteome_data.loc[proteome_data["TimePointOrder"] == t_nm]
    combinations, all_feature_combinations = make_feature_combination(data_at_timePoint, uq, lq)
    print("ESPY measurment on proteome data at time point " + str(timePoint) + " started")
    distances_for_all_feature_comb = compute_distance_hyper(all_feature_combinations, multitask_classifier, combinations, proteome_data.iloc[:,3:], param_combi_RRR)
    # print(distances_for_all_feature_comb)
    distances_for_all_feature_comb.to_csv(os.path.join(outputdir, output_filename), sep='\t', na_rep='nan')
    print('results are saved in: ' + os.path.join(outputdir, output_filename))

    end = time.time()
    print("end of computation after: ", str(end - start), "seconds")

    return distances_for_all_feature_comb, combinations

