"""
This Module contains the evaluation of informative features based on the SHAP (SHapley Additive exPlanations) framework
on simulated data.
Created on: 18.01.2022

@Author: Jacqueline Wistuba-Hamprecht

"""


# required packages
import numpy as np
import pandas as pd
import os
import sys
import os.path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import shap
import random
cwd = os.getcwd()
datadir = '/'.join(cwd.split('/')[:-1]) + '/data/simulated_data'
outputdir = '/'.join(cwd.split('/')[:-1]) + '/results/simulated_data'



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

    return X_train, X_test, Y_train, Y_test


def initialize_svm_model(X_Train_data, y_train_data, X_test_data, y_test_data):
    """ Initialize SVM model on simulated data

                Initialize SVM model with a rbf kernel on simulated data and
                perform a grid search for kernel parameter evaluation
                Returns the SVM model with the best parameters based on the highest mean AUC score

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

    # run rbf SVM with parameters fromm grid search, probability has to be TRUE to evaluate features via SHAP
    svm3 = SVC(kernel="rbf", gamma=svm_model_cv.best_params_.get("gamma"), C=svm_model_cv.best_params_.get("C"),
               probability=True)
    model = svm3.fit(X_Train_data.values, y_train_data)

    y_pred = model.predict(X_test_data.values)

    AUC = roc_auc_score(y_test_data, y_pred)

    print("AUC score on unseen data:" + " " + str(AUC))

    return model


def SHAP_value(model, x_train, x_test, outputdir):
    explainer = shap.KernelExplainer(model.predict_proba, x_train)
    shap_values = explainer.shap_values(x_test)

    shap.initjs()
    shap.summary_plot(shap_values, x_test, show=False)
    plt.savefig(os.path.join(outputdir, "SHAP_value_simulated_data.png"), dpi=600)


if __name__ == "__main__":
    data_path = os.path.join(datadir, 'simulated_data.csv')
    simulated_data = pd.read_csv(data_path)

    X_train, X_test, Y_train, Y_test = train_testSplit(simulated_data.iloc[:, :1000], simulated_data.iloc[:, 1000])
    rbf_SVM_model = initialize_svm_model(X_Train_data=X_train, y_train_data=Y_train, X_test_data=X_test,
                                         y_test_data=Y_test)
    print("Evaluation of informative features based on SHAP values started")
    SHAP_value(model=rbf_SVM_model, x_train=X_train, x_test=X_test, outputdir=outputdir)
    print("Evaluation terminated and results are saved in ./results as SHAP_value_simulated_data.png ")
