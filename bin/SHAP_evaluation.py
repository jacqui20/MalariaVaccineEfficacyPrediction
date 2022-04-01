"""
This Module contains the evaluation of informative features based on
the SHAP (SHapley Additive exPlanations) framework on simulated data.

@Author: Jacqueline Wistuba-Hamprecht
"""

import numpy as np
import pandas as pd
import os
import os.path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import shap
cwd = os.getcwd()
datadir = '/'.join(cwd.split('/')[:-1]) + '/data/simulated_data'
outputdir = '/'.join(cwd.split('/')[:-1]) + '/results/simulated_data'


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


def SHAP_value(
    model: SVC,
    X_train: np.ndarray,
    X_test: np.ndarray,
    outputdir: str,
) -> None:
    explainer = shap.KernelExplainer(model.predict_proba, X_train)
    shap_values = explainer.shap_values(X_test)

    shap.initjs()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(outputdir, "SHAP_value_simulated_data.png"), dpi=600)


if __name__ == "__main__":
    data_path = os.path.join(datadir, 'simulated_data.csv')
    simulated_data = pd.read_csv(data_path)

    X_train, X_test, Y_train, Y_test = train_test_split(
        simulated_data.iloc[:, :1000].to_numpy(),
        simulated_data.iloc[:, 1000].to_numpy(),
        test_size=0.3,
        random_state=123
    )
    rbf_SVM_model = initialize_svm_model(
        X_train_data=X_train,
        y_train_data=Y_train,
        X_test_data=X_test,
        y_test_data=Y_test
    )

    print(
        "Evaluation of informative features based on SHAP values started"
    )

    SHAP_value(model=rbf_SVM_model, X_train=X_train, X_test=X_test, outputdir=outputdir)

    print(
        "Evaluation terminated and results are saved in "
        "./results as SHAP_value_simulated_data.png."
    )
