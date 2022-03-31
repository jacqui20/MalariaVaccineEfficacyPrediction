"""
Evaluation of informative features from RLR models

Will save the results to various .tsv/.csv files


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

cwd = os.getcwd()
resultsdir = '/'.join(cwd.split('/')[:-1]) + '/results/RLR'
datadir = '/'.join(cwd.split('/')[:-1]) + '/data/timepoint-wise'
outputdir_whole = '/'.join(cwd.split('/')[:-1]) + '/results/RLR/whole/Informative_features'
outputdir_selective = '/'.join(cwd.split('/')[:-1]) + '/results/RLR/selective/Informative_features'


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
    pam = pam.str.split(" ", expand=True).values
    #print(pam)
    return pam


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
    x: np.darray
        matrix of performance scores per time point

    """
    x = kernel_parameter[kernel_parameter['time'].isin([time_point])]
    #print(X)
    return x


def rearagne_columns(data):
    df = data.copy()
    dose = df['Dose']
    df = df.drop(columns=['Dose'])
    df.insert(loc=4, column='Dose', value=dose)
    #print(df)

    return df


def RLR_model(X_data, y_labels, kernel_parameter, feature_labels):

    c = pd.to_numeric(kernel_parameter[0][1].split(",")[0])
    print("C-value:" + str(c))
    print(type(c))
    l1_value = pd.to_numeric(kernel_parameter[0][3].split("}")[0])
    print("l1_value:" + str(l1_value))

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
    print("Non Zero weights:", np.count_nonzero(model.coef_))
    cdf = pd.concat([pd.DataFrame(feature_labels), pd.DataFrame(np.transpose(model.coef_))], axis=1)
    cdf.columns = ['Pf_antigen_ID', 'weight']
    cdf = cdf.sort_values(by=['weight'], ascending=True)
    cdf_nonzeros = cdf[cdf['weight'] != 0]
    return model, cdf_nonzeros


def featureEvaluationRLR(data, results_rgscv, timepoint):

    data = rearagne_columns(data)
    print("Paramter combination for best mean AUC at timepoint" + str(timepoint) + ":")
    time_point = select_time_point(results_rgscv, timepoint)
    kernel_param = get_kernel_paramter(time_point)
    print(kernel_param)
    print('')
    print("Start feature evaluation with dose as auxellary feature:")
    X_data = data.iloc[:, 4:].to_numpy()
    #print(X_data)
    y_labels = data.loc[:, 'Protection'].to_numpy()
    #print(y_labels)
    feature_labels = data.iloc[:, 4:].columns

    model, coeffiecients = RLR_model(X_data, y_labels, kernel_param, feature_labels)
    print(coeffiecients)
    return coeffiecients





