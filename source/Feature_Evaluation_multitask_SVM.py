"""
This module contains the main functionality of the feature selection approach.
First the multitask-SVM classification model is initilaized with the parameter combinations evaluated based
on the results from Parser_multitask_SVM.py module. Second informative features are evaluated.
Created on: 25.05.2019

@Author: Jacqueline Wistuba-Hamprecht


"""

# required packages
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics.pairwise import rbf_kernel,sigmoid_kernel,polynomial_kernel
import time

"""
  Set up multitask-SVM model based on the output of file of the Parser_multitask_SVM.py.

"""

# Extracting time point based on highest mean AUC value from prediction performance analysis
def get_time_point(timepoint):
    """ Returns time point

        Returns time point from the results of the
        multitask-SVM approach based on the highest mean AUC.
        see results of the Parser_multitask_SVM.py module

        Args: time_point:

        Returns: time (int): time point

            """

    best_AUC = max(timepoint["mean AUC"])
    time = timepoint.loc[timepoint['mean AUC'] == best_AUC, 'time point']
    return time.iat[0]


def get_kernel_paramter(kernel_parameter):
    """ Returns the combination of kernel parameters from the results of the
            multitask-SVM approach based on the highest mean AUC.

        Returns the combination of kernel parameters from the results of the
        multitask-SVM approach based on the highest mean AUC.
        see results of the Parser_multitask_SVM.py module

        Args: kernel_parameter (matrix): results of the multitask-SVM approach as .csv file

        Returns:
                pam (list): Combination of kernel parameter for the combination of kernel functions for the multitask-SVM classifier
                based on the highest mean AUC value
        """
    best_AUC = max(kernel_parameter["mean AUC"])
    pam = kernel_parameter.loc[kernel_parameter['mean AUC'] == best_AUC, "label"]
    pam = pam.str.split(" ", expand=True)
    return pam



def multitask(a, b):
    """ Multitask approach

            Combination of two kernel matrices are combined by element-wise multiplication to one kernel matrix.

            Args: a (matrix): kernel matrix a
                  b (matrix): kernel matrix b

            Returns: element-wise multiplicated kernel matrix a * b
    """
    return a*b



def make_svm_positive(mat):
    """ Spectral Translation approach

            proofs if the kernel matrix is non positive (semi-) definite, if yes,
            the spectral translation approach (Schölkopf et al, 2002) is applied to make the kernel matrix
            positive (semi-) definite.

            Args: mat (matrix): kernel matrix (matrix)

            Returns: mat (matrix): positive semi-definite kernel matrix
            """
    if not np.all(np.linalg.eigvals(mat) > 0):
        eigen_values = np.linalg.eigvals(mat)
        minimum_eigenvalue = np.min(eigen_values.real)
        #print(minimum_eigenvalue)
        np.fill_diagonal(mat, np.diag(mat) - minimum_eigenvalue)
    if np.all(np.linalg.eigvals(mat) < 0):
        print("is not psd")
    return mat



def make_kernel_matrix(train_set, kernel_parameter):
    """ Compute combined kernel matrix

        for each task, time, PfSPZ dose and antibody signal intensity a kernel matrix is pre-computed given the pre-defined
        kernel function and combined to one matrix by the multitask approach

        Args: trainings data (matrix): matrix of trainings data, n x m matrix (n = samples as rows, m = features as columns)
              kernel_parameter (dictionary): combination of kernel parameter


        Returns: multi_AB_signals_time_dose_kernel_matrix (matrix): returns a combined matrix
        """
    # get start point of antibody reactivity signals in data (n_p)
    AB_signal_start = train_set.columns.get_loc("TimePointOrder") + 1
    AB_signals = train_set[train_set.columns[AB_signal_start:]]

    # extract time points to vector (y_t)
    time_series = train_set["TimePointOrder"]

    # extract dosage to vector (y_d)
    dose = train_set["Dose"]

    #kernel paramters
    # for time_points
    scale = pd.to_numeric(kernel_parameter.iat[0, 5])
    #print(scale)
    offset = pd.to_numeric(kernel_parameter.iat[0, 7])

    # for AB signals
    if kernel_parameter.iat[0, 2] == "gamma:":
        ABgamma = pd.to_numeric(kernel_parameter.iat[0, 3])
        print("selected kernel for AB signals is rbf kernel with a value of: ", ABgamma)
        #set up kernel matrix_rank
        AB_signals_kernel_matrix = rbf_kernel(AB_signals, gamma=ABgamma)
        AB_signals_kernel_matrix = make_svm_positive(AB_signals_kernel_matrix)
    elif kernel_parameter.iat[0, 2] == "P:":
        ABdegree = pd.to_numeric(kernel_parameter.iat[0, 3])
        print("selected kernel for AB signals is polynomial kernel with a value of: ", ABdegree)
        #set up kernel matrix_rank
        AB_signals_kernel_matrix = polynomial_kernel(AB_signals, degree=ABdegree)
        AB_signals_kernel_matrix = make_svm_positive(AB_signals_kernel_matrix)

    #for Dosis
    if kernel_parameter.iat[0, 8] == "O:":
        Dgamma = pd.to_numeric(kernel_parameter.iat[0, 9])
        print("selected kernel for dose is rbf kernel with a value of: " ,Dgamma)
        #set up kernel matrix
        dose_kernel_matrix = rbf_kernel(dose.values.reshape(len(dose), 1), gamma=Dgamma)
        dose_kernel_matrix = make_svm_positive(dose_kernel_matrix)
    elif kernel_parameter.iat[0, 8] == "P:":
        Ddegree = pd.to_numeric(kernel_parameter.iat[0, 9])
        print("selected kernel for dose is polynomial kernel with a value of: ", Ddegree)
        #set up kernel matrix
        dose_kernel_matrix = polynomial_kernel(dose.values.reshape(len(dose), 1), degree=Ddegree)
        dose_kernel_matrix = make_svm_positive(dose_kernel_matrix)


    # pre-computed kernel matrix of time points K(n_t,n_t')
    time_series_kernel_matrix = sigmoid_kernel(time_series.values.reshape(len(time_series), 1), gamma=scale, coef0=offset)
    # test for psd
    time_series_kernel_matrix = make_svm_positive(time_series_kernel_matrix)
    # show_heatmap(time_series_kernel_matrix)

    # pre-computed multitask kernel matrix K((np, nt),(np', nt'))
    multi_AB_signals_time_series_kernel_matrix = multitask(AB_signals_kernel_matrix, time_series_kernel_matrix)
    # show_heatmap(multi_AB_signals_time_series_kernel_matrix)

    # pre-computed multitask kernel matrix K((np, nt, nd),(np', nt', nd'))
    multi_AB_signals_time_dose_kernel_matrix = multitask(multi_AB_signals_time_series_kernel_matrix, dose_kernel_matrix)
    # show_heatmap(multi_AB_signals_time_dose_kernel_matrix)

    # proof dimension and rank of kernel matrix
    print("Dimension of kernel matrix")
    print(multi_AB_signals_time_dose_kernel_matrix.shape)
    print("Rank of kernel matrix")
    print(np.linalg.matrix_rank(multi_AB_signals_time_dose_kernel_matrix))

    return multi_AB_signals_time_dose_kernel_matrix



def multitask_model(data, kernel_parameter):
    """Set up multitask-SVM model based on the output of file of the Parser_multitask_SVM.py.

    set up multitask-SVM model based on evaluated kernel combinations from Parser_multitask_SVM.py module.

    Args: data (matrix): n x m matrix (n = samples as rows, m = features as columns)
          kernel_parameter (dictionary): combination of kernel parameter

    Returns: multitaskModel (SVM model): classification model based on the multitask-SVM approach

    """

    # extract kernel parameter combination based on highest mean AUC value from prediction performance analysis
    print("hello")
    hyperparamter= get_kernel_paramter(kernel_parameter)
    print("hello2")

    # pre-compute multitask kernel matrix on all input samples
    multi_kernel_matrix = make_kernel_matrix(data, hyperparamter)
    y_label = data["Protection"].values

    # extract cost value C based on highest mean AUC value from prediction performance analysis
    C_reg = pd.to_numeric(hyperparamter.iat[0, 1])

    # set up multitask model based on evaluated parameter
    multitaskModel = svm.SVC(kernel="precomputed", C=C_reg)
    multitaskModel.fit(multi_kernel_matrix, y_label)

    return multitaskModel

#######################################################################################################################
"""
  Evaluate feature importance.
"""

# Generate vector of feature combination based on Upper- and LowerQuantile value of each feature
# INPUT: data = feature values per time point, upperValue = defined upper quantile, lowerValue = defined lower quantile
# OUTPUT: combination of features as matrix and series of feature combinations
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
    feature_start = data.columns.get_loc("TimePointOrder") + 1
    # get matrix of feature values
    features = data[data.columns[feature_start:]]

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



def feature_gram_matrix(data, kernel_paramter):
    """Define feature

            Define feature for prediction in multitask-SVm classification model

            Args: data (matrix): matrix of feature (n= samples, m= feature)
                  kernel_parameter (dictionary): combination of kernel parameter

            Returns: test_sample (matrix): matrix of defined feature

            """

    #  extract kernel parameter combination based on highest mean AUC value from prediction performance analysis
    kernel_pam = get_kernel_paramter(kernel_paramter)

    #kernel paramters
    if kernel_pam.iat[0,2] == "gamma:":
        ABgamma = pd.to_numeric(kernel_pam.iat[0,3])
        #set up kernel matrix_rank
        AB_signals_kernel_matrix = rbf_kernel(data, gamma= ABgamma)
        AB_signals_kernel_matrix = make_svm_positive(AB_signals_kernel_matrix)
        #print("Dimension of kernel matrix with feature")
        #print(AB_signals_kernel_matrix.shape)
    elif kernel_pam.iat[0,2] == "P:":
        ABdegree = pd.to_numeric(kernel_pam.iat[0,3])
        #set up kernel matrix_rank
        AB_signals_kernel_matrix = polynomial_kernel(data, degree=ABdegree)
        AB_signals_kernel_matrix = make_svm_positive(AB_signals_kernel_matrix)
        #print("Dimension of kernel matrix with feature")
        #print(AB_signals_kernel_matrix.shape)

    # set up feature test sample for distance evaluation
    test_sample = AB_signals_kernel_matrix[-1,:len(AB_signals_kernel_matrix[0])-1]

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
    print("Matrix of distances for Upper-/Lower- quantile per feature")
    get_distance_df = pd.DataFrame([get_distance_upper, get_distance_lower], columns=labels)
    #print(get_distance_df.shape)

    # add distance of consensus feature
    get_distance_df.loc["consensus [d]"] = np.repeat(d_cons, len(get_distance_df.columns))
    #print(get_distance_df["med"].iloc[0])
    #print(get_distance_df.shape)
    print("Number of columns of evaluated distance matrix")
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
    #print("Dimension of distance matrix")
    #print(get_distance_df.shape)
    #print("end computation")

    return get_distance_df


# MAIN FUNCTION
# Evaluate the influence of each single feature by calculating the distance change of support vectors to the
# hyperplane
# INPUT: proteome data, kernel parameter combination, value for upper quantile, value for lower quantile
def feature_evaluation(data, kernel_parameter, UpperValue, LowerValue):
    """ MAIN MODUL of the feature evaluation approach.

                This main modul evaluates the distances of each feature to the classification boundery.


                Args: data (matrix): preprocessed proteome matrix as n x m matrix (where n = samples in rows, m = features
                              as columns) concatenated with the column "time point", the column "protection state" and the column
                              "dosage"

                      kernel_parameter (dictionary): combination of kernel parameter
                      UpperValue (int): value of upper quantile
                      LowerValue (int): value of lower quantile


                Returns: distances_for_all_feature_comb (matrix): matrix of distance values for each feature per time point
                         combinations (matrix): combination of features
                         timePoint (int): time point of feature evaluation
                """
    start = time.time()
    #print("dimension of input proteome data ", data.shape)


    # select time point for evaluation
    timePoint= pd.to_numeric(get_time_point(kernel_parameter))
    print("ESPY value computation at time point: ", timePoint)

    # set up model for best parameter
    multitaskModel = multitask_model(data, kernel_parameter)

    # generate feature combinations given by Median, UpperQuantile and LowerQuantile value
    # over all samples per time point
    if timePoint == 2:
        data_t2 = data.loc[data["TimePointOrder"] == 2]
        print("Dimension of proteome data at time point" + ' ' + str(timePoint), "is ", data_t2.shape)
        combinations, all_feature_combinations = make_feature_combination(data_t2, UpperValue, LowerValue)
        print("Length of matrix of feature combination at time point" + ' ' + str(timePoint) + ' ' +
              "for importance eval: ", len(all_feature_combinations))
    elif timePoint == 3:
        data_t3 = data.loc[data["TimePointOrder"] == 3]
        print("Dimension of proteome data at time point" + ' ' + str(timePoint), "is ", data_t3.shape)
        combinations, all_feature_combinations = make_feature_combination(data_t3, UpperValue, LowerValue)
        print("Length of matrix of feature combination at time point" + ' ' + str(timePoint) + ' '
              + "for importance eval: ", len(all_feature_combinations))
    elif timePoint == 4:
        data_t4 = data.loc[data["TimePointOrder"] == 4]
        print("Dimension of proteome data at time point" + ' ' + str(timePoint), "is ", data_t4.shape)
        combinations, all_feature_combinations = make_feature_combination(data_t4, UpperValue, LowerValue)
        print("length of matrix of feature combination at time point" + ' ' + str(timePoint) + ' '
              + "for importance eval:", len(all_feature_combinations))

    # pre-process data - delete, sample ID, groups, protection columns
    eval_features_start = data.columns.get_loc("TimePointOrder") + 1
    X = data[data.columns[eval_features_start:]]
    print("Feature evaluation is running....")
    distances_for_all_feature_comb = compute_distance_hyper(all_feature_combinations, multitaskModel, combinations, X,
                                                            kernel_parameter)

    end = time.time()
    print("end of computation after: ", str(end - start), "seconds")
    return distances_for_all_feature_comb, combinations, timePoint
