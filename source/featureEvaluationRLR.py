"""
Evaluation of informative features from RLR models

Will save the results to various .tsv files


"""
import numpy as np
import pandas as pd
import scipy
import sklearn
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def get_kernel_parameter(
        kernel_parameter: np.ndarray
        ):
    """ Return combination of kernel parameter to initialize Elastic Net model

        Parameters
        ----------
        kernel_parameter : np.ndarray
            matrix of kernel parameter per time point and the evaluated mean AUC value

        Returns
        --------
        pam : dict
            Dictionary of kernel values
        """
    pam_roc_auc = kernel_parameter[kernel_parameter['scoring'].isin(['roc_auc'])]
    pam = pam_roc_auc['best_params']
    #print(pam)
    pam = pam.str.split(" ", expand=True).values
    #print(pam)
    return pam


def select_time_point(
        kernel_parameter: np.ndarray,
        time_point: str
        ):
    """ Select time point to evaluate informative features from RLR

    Parameter
    ---------
    kernel_parameter : np.ndarrray
        matrix of kernel parameter per time point and the evaluated mean AUC value
    time_point : str
        preferable time point

    Returns
    --------
    x: np.ndarray
        matrix of performance and kernel values per time point
    """
    x = kernel_parameter[kernel_parameter['time'].isin([time_point])]
    #print(X)
    return x


def rearagne_columns(
        data: pd.DataFrame
):
    """ Re-arrange column order of dataframe

    Move column Dose to the start position of features

    Parameters
    ----------
    data: pd.Dataframe
        dataframe of proteome data

    Returns
    --------
    df : pd.Dataframe
        re-arranged column order of dataframe

    """

    df = data.copy()
    dose = df['Dose']
    df = df.drop(columns=['Dose'])
    df.insert(loc=4, column='Dose', value=dose)
    #print(df)

    return df


def RLR_model(
        *,
        X_data: np.ndarray,
        y_labels: np.ndarray,
        kernel_parameter: dict,
        feature_labels: list
) -> LogisticRegression:
    """Initialize Elastic Net model on proteome data

    Initialize Elastic Net model with kernel parameter from grid search on proteome data and
    evaluate coefficients.
    Returns the evaluated coefficients.

    Parameters
    ---------
    X_data: np.ndarray,
        matrix of input data
    y_labels: np.ndarray,
        y label
    kernel_parameter: dict,
        dictionary of kernel parameter to initialize Elastic Net model
    feature_labels: list
        list of feature labels

    Returns
    -------
    model: sklearn.linear_model.LogisticRegression object
        initialized Elastic net model on pre-defined kernel parameter
    cdf_nonzeros: pd.Dataframe
        dataframe of features with evaluated non-zero coefficient weights

    """
    # get parameter for Elastic Net model
    c = pd.to_numeric(kernel_parameter[0][1].split(",")[0])
    print("C-value:" + str(c))
    print(type(c))
    l1_value = pd.to_numeric(kernel_parameter[0][3].split("}")[0])
    print("l1_value:" + str(l1_value))

    # Initialize Elastic Net model
    estimator = make_pipeline(
        StandardScaler(
            with_mean=True,
            with_std=True,
        ),
        LogisticRegression(
            penalty='elasticnet',
            C=c,
            solver='saga',
            l1_ratio=l1_value,
            max_iter=10000,
        ),
        # memory=cachedir,
    )
    estimator.fit(X_data, y_labels)
    print(estimator)
    model = estimator[1]

    # Extract non-zero coefficients
    print("Non Zero weights:", np.count_nonzero(model.coef_))
    cdf = pd.concat([pd.DataFrame(feature_labels), pd.DataFrame(np.transpose(model.coef_))], axis=1)
    cdf.columns = ['Pf_antigen_ID', 'weight']
    cdf = cdf.sort_values(by=['weight'], ascending=True)
    cdf_nonzeros = cdf[cdf['weight'] != 0]
    return model, cdf_nonzeros


def featureEvaluationRLR(
        data: pd.DataFrame,
        results_rgscv: np.ndarray,
        timepoint: str
):
    """Evaluation of informative features from Elastic Net model

    Parameter
    ---------
    data: pd.DataFrame
        Dataframe of proteome data
    results_rgscv: np.ndarray
        kernel combination for Elastic Net model per time point based on mean AUC score
    timepoint: str
        time point to evaluate informative features

    Returns
    -------
    coefficients: pd.Dataframe
        Dataframe of non-zero coefficients

    """

    data = rearagne_columns(data)

    print("Paramter combination for best mean AUC at time point" + str(timepoint) + ":")

    time_point = select_time_point(
        kernel_parameter=results_rgscv,
        time_point=timepoint)
    kernel_param = get_kernel_parameter(
        kernel_parameter=time_point)
    print(kernel_param)
    print('')

    print("Start feature evaluation with dose as auxellary feature:")
    X_data = data.iloc[:, 4:].to_numpy()
    y_labels = data.loc[:, 'Protection'].to_numpy()
    feature_labels = data.iloc[:, 4:].columns

    model, coeffiecients = RLR_model(
        X_data=X_data,
        y_labels=y_labels,
        kernel_parameter=kernel_param,
        feature_labels=feature_labels)
    print(coeffiecients)

    return coeffiecients
