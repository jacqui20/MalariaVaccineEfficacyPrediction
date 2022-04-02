"""
This module contains functions to evaluate the informative features 
from simulated data using the ESPY value measurement.

@Author: Jacqueline Wistuba-Hamprecht
"""

#required packages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def initialize_svm_model(
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


def compute_disctance_hyper(
        combinations: pd.DataFrame,
        model: SVC,
        input_labels: list
        ):
    """Evaluate distance of each single feature to classification boundary

    Compute distance of support vectors to classification boundary for each feature and the change of
    ach feature by upper and lower quantile

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


def make_plot(
        data: pd.DataFrame,
        name: str,
        outputdir: str):
    """

    Paramter
    ---------
    data: pd.DataFrame
        dataframe of distances
    name: str
        name of outputfile
    outputdir: str
        Path where the plots are stored as .png and .pdf
    """
    plt.figure(figsize=(20, 10))
    labels = data.columns

    ax = plt.subplot(111)
    w = 0.3
    opacity = 0.6

    index = np.arange(len(labels))
    ax.bar(index, abs(data.loc["|d|"].values), width=w, color="darkblue", align="center", alpha=opacity)
    ax.xaxis_date()

    plt.xlabel('number of features', fontsize=20)
    plt.ylabel('ESPY value', fontsize=20)
    plt.xticks(index, labels, fontsize=10, rotation=90)

    plt.savefig(os.path.join(outputdir, name + ".png"), dpi=600)
    plt.savefig(os.path.join(outputdir, name + ".pdf"), format="pdf", bbox_inches="tight")
    plt.show()


def ESPY_measurment(
        simulated_data: pd.DataFrame,
        lq: int,
        up: int,
        outputname: str,
        outputdir: str):
    """ESPY measurement
    Calculate ESPY value for each feature on simulated data

    Parameter
    -----------
    simulated_data: pd.Dataframe
        dataframe of simulated data
    lq: int
        value of lower quantile
    up: int
        value of upper quantile
    outputname: str
        name of output file
    outputdir: str
        path to store output files

    Returns
    --------
    distance_matrix_for_all_feature_comb: pd.Dataframe
        dataframe of ESPY value |d| for each feature in simulated data

    """


    X_train, X_test, Y_train, Y_test = train_test_split(
        simulated_data.iloc[:, :1000].to_numpy(),
        simulated_data.iloc[:, 1000].to_numpy(),
        test_size=0.3,
        random_state=123)

    rbf_svm_model = initialize_svm_model(X_train_data=X_train, y_train_data=Y_train,
                                         X_test_data=X_test, y_test_data=Y_test)

    combinations, all_feature_combinations = make_feature_combination(
        X=pd.DataFrame(X_test), lowerValue=lq, upperValue=up)

    distance_matrix_for_all_feature_comb = compute_disctance_hyper(
        combinations=all_feature_combinations,
        model=rbf_svm_model,
        input_labels=combinations)

    make_plot(data=distance_matrix_for_all_feature_comb.iloc[:, :25],
              name=outputname,
              outputdir=outputdir)

    return distance_matrix_for_all_feature_comb




