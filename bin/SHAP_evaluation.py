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
This Module contains the evaluation of informative features based on
the SHAP (SHapley Additive exPlanations) framework on simulated data.

@Author: Jacqueline Wistuba-Hamprecht and Bernhard Reuter

"""

from datetime import datetime
import numpy as np
import pandas as pd
import os
import pathlib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import shap
import warnings


def svm_model(
    *,
    X_train_data: np.ndarray,
    y_train_data: np.ndarray,
    X_test_data: np.ndarray,
    y_test_data: np.ndarray,
) -> SVC:
    """ Initialize SVM model on simulated data

    Initialize SVM model with a rbf kernel on simulated data and
    perform a grid search for kernel parameter evaluation.
    Returns the SVM model with the best parameters based on the highest mean AUC score.

    Parameters
    ----------
    X_train_data : np.ndarray
        Training data.
    y_train_data : np.ndarray
        y labels for training.
    X_test_data : np.ndarray
        Test data.
    y_test_data : np.ndarray
        y labels for testing.

    Returns
    -------
    model : sklearn.svm.SVC object
        Trained SVM model with best kernel parameters found via GridSearchCV.
    """

    # Initialize SVM model, rbf kernel
    param_grid = {
        'gamma': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 1e1, 1e2, 1e3, 1e4, 1e5, 1e6],
        'C': [1.e-3, 1.e-2, 1.e-1, 1.e0, 1.e1, 1.e2, 1.e3],
    }

    # grid search on simulated data
    clf = GridSearchCV(
        SVC(kernel='rbf', probability=True),
        param_grid,
        scoring='roc_auc',
        refit=True,
    )
    clf.fit(X_train_data, y_train_data)

    print(
            "The best parameters are %s with a mean AUC score of %0.2f"
            % (clf.best_params_, clf.best_score_)
    )

    # run rbf SVM with parameters fromm grid search,
    # probability has to be TRUE to evaluate features via SHAP
    svm = SVC(
        kernel='rbf',
        gamma=clf.best_params_.get('gamma'),
        C=clf.best_params_.get('C'),
        probability=True
    )

    model = svm.fit(X_train_data, y_train_data)

    y_pred = model.predict(X_test_data)

    AUC = roc_auc_score(y_test_data, y_pred)

    print(f"AUC score on unseen data: {AUC}")

    return model


def SHAP_values(
    model: SVC,
    X_train: np.ndarray,
    X_test: np.ndarray,
    outputdir: str,
) -> None:

    timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")

    explainer = shap.KernelExplainer(model.predict_proba, X_train)
    shap_values = explainer.shap_values(X_test, l1_reg='num_features(25)')

    shap.initjs()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(outputdir, f"SHAP_values_simulated_data_{timestamp}.png"), dpi=600)
    plt.savefig(os.path.join(outputdir, f"SHAP_values_simulated_data_{timestamp}.pdf"),
                format="pdf", bbox_inches="tight")

    if isinstance(shap_values, np.ndarray):
        np.save(os.path.join(outputdir, f"SHAP_values_simulated_data_{timestamp}.npy"),
                shap_values)
    elif isinstance(shap_values, list):
        np.savez(os.path.join(outputdir, f"SHAP_values_simulated_data_{timestamp}.npz"),
                 *shap_values)
    else:
        warnings.warn("Couldn't save shap values, since they were of "
                      f"unexpected type {type(shap_values)}.")


if __name__ == "__main__":
    cwd = os.getcwd()
    datadir = '/'.join(cwd.split('/')[:-1]) + '/data/simulated_data'
    outputdir = '/'.join(cwd.split('/')[:-1]) + '/results/SVM/simulated/SHAP'
    pathlib.Path(outputdir).mkdir(parents=True, exist_ok=True)
    simulated_data = pd.read_csv(os.path.join(datadir, 'simulated_data.csv'))

    X_train, X_test, y_train, y_test = train_test_split(
        simulated_data.iloc[:, :1000].to_numpy(),
        simulated_data.iloc[:, 1000].to_numpy(),
        test_size=0.3,
        random_state=123,
    )
    rbf_SVM_model = svm_model(
        X_train_data=X_train,
        y_train_data=y_train,
        X_test_data=X_test,
        y_test_data=y_test,
    )

    timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
    print(
        f"Evaluation of informative features based on SHAP values has started at {timestamp}."
    )

    SHAP_values(model=rbf_SVM_model, X_train=X_train, X_test=X_test, outputdir=outputdir)

    timestamp = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
    print(
        f"Evaluation has terminated at {timestamp} and results are saved in "
        "../results/SVM/simulated/SHAP as SHAP_value_simulated_data.png."
    )
