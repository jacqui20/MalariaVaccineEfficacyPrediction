"""
This module evaluates the informative features from the simulated data
using the ESPY value measurement.
Created on: 25.01.2022

@Author: Jacqueline Wistuba-Hamprecht


"""

#required packages
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import shap
import time
import random


def train_testSplit(data, label):
    """ Split data into train and test data

            Returns a set of trainings data and a set of test data
            based on a split of 70% for training and 30% for testing

            Args: data (matrix) : matrix of simulated data
                  label (matrix): y label

            Returns: X_train (matrix): matrix of trainings data
                     X_test (matrix): matrix of test data
                     Y_train (vector): y label for training
                     Ytest (vector): y label for testing

    """
    X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.3, random_state=123)

    #print("Dimension of X_train:")
    #print(X_train.shape)
    #print("Dimension of X_test:")
    #print(X_test.shape)

    return X_train, X_test, Y_train, Y_test


def initialize_svm_model(X_Train_data, y_train_data, X_test_data, y_test_data):
    """ Initialize SVM model on simulated data

                Initialize SVM model with an rbf kernel on simulated data and
                perform a grid search for kernel parameter evaluation
                Returns the SVM model with the best parameters based on the highest mean accurary score

                Args: X_train_data (matrix) : matrix of trainings data
                      y_train_data (vector): y label for training
                      X_test_data (matrix): matrix of test data
                      y_test_data (vector): y label for testing


                Returns: model : trained SVM model on evaluated kernel parameter
    """

    # Initialize SVM model, rbf kernel
    C_range = np.logspace(-3, 3, 7)
    gamma_range = np.logspace(-6, 6, 13)
    param_grid = dict(gamma=gamma_range, C=C_range)
    scoring = {"AUC": "roc_auc"}

    svm = SVC(kernel="rbf")

    # grid search on simulated data
    clf = GridSearchCV(svm, param_grid, scoring=scoring, refit="AUC")
    svm_model_cv = clf.fit(X_Train_data.values, y_train_data)

    print(
            "The best parameters are %s with a mean AUC score of %0.2f"
            % (svm_model_cv.best_params_, svm_model_cv.best_score_)
    )

    # run rbf SVM with paramters fromm grid search, probaility has to be TRUE to evaluate features via SHAP
    svm3 = SVC(kernel="rbf", gamma=svm_model_cv.best_params_.get("gamma"), C=svm_model_cv.best_params_.get("C"),
               probability=True)
    model = svm3.fit(X_Train_data.values, y_train_data)

    y_pred = model.predict(X_test_data.values)

    AUC = roc_auc_score(y_test_data, y_pred)

    print("AUC score on unseen data:" + " " + str(AUC))

    return model


def make_feature_combination(X, upperValue, lowerValue):
    """Generate vector of feature combination.

            Generate for each single feature a vector based on Upper- and LowerQuantile value

            Args: X (matrix): matrix of simulated features
                  upperValue (int): value of upper quantile
                  lowerValue (int): value of lower quantile

            Returns: feature_comb (matrix): combination of features
                     get_features_comb (series):  series of feature combinations
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


def compute_disctance_hyper(combinations, model, input_labels):
    """Evaluate distance of each single feature to classification boundary

                    Compute distance of support vectors to classification boundery for each feature and the change of
                    each feature by upper and lower quantile

                    Args: combinations (vector): vector of combination of feature value, itself, upper and lower quantile
                          model (SVM model): SVM model
                          input-labels (list) = list of feature names

                    Returns: get_distance_df (matrix): matrix of ESPY values for each feature per time point

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



def ESPY_measurment(data, lq, up):
    """ MAIN MODUL of the feature evaluation approach.

                    This main modul evaluates the distances of each feature to the classification boundery.


                    Args: data (matrix): simulated data matrix n x m (where n = samples in rows, m = features in columns)
                          UpperValue (int): value of upper quantile
                          LowerValue (int): value of lower quantile


                    Returns: distances_for_all_feature_comb (matrix): matrix of ESPY value for each feature
                    """

    #print(data.head())
    X_train, X_test, Y_train, Y_test = train_testSplit(data.iloc[:, :1000], data.iloc[:, 1000])
    #print(X_test.head())
    #print(Y_test)
    rbf_svm_model = initialize_svm_model(X_Train_data=X_train, y_train_data= Y_train, X_test_data=X_test, y_test_data=Y_test)
    combinations, all_feature_combinations = make_feature_combination(X_test, up, lq)
    distance_matrix_for_all_feature_comb = compute_disctance_hyper(all_feature_combinations, rbf_svm_model,
                                                                   combinations)
    return distance_matrix_for_all_feature_comb




