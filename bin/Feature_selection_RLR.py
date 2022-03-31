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


def main(data, kernel_parameter):
    print("Feature evaluation with dose as auxellary feature:")
    X_data = data.iloc[:, 4:].to_numpy()
    #print(X_data)
    y_labels = data.loc[:, 'Protection'].to_numpy()
    #print(y_labels)
    feature_labels = data.iloc[:, 4:].columns

    model, coeffiecients = RLR_model(X_data, y_labels, kernel_parameter, feature_labels)
    print(coeffiecients)
    return coeffiecients


if __name__ == "__main__":
    print('sys.path:', sys.path)
    print('scikit-learn version:', sklearn.__version__)
    print('pandas version:', pd.__version__)
    print('numpy version:', np.__version__)
    print('scipy version:', scipy.__version__)

    result_path_whole = os.path.join(resultsdir, 'whole/RGSCV/RepeatedGridSearchCV_results_24.03.2022_09-23-48.tsv')
    result_path_selective = os.path.join(resultsdir, 'selective/RGSCV/'
                                                     'RepeatedGridSearchCV_results_24.03.2022_12-47-24.tsv')

    data_path_whole_III14 = os.path.join(datadir, "whole_data_III14.csv")
    data_path_whole_C1 = os.path.join(datadir, "whole_data_C-1.csv")
    data_path_whole_C28 = os.path.join(datadir, "whole_data_C28.csv")

    data_path_selective_III14 = os.path.join(datadir, "selective_data_III14.csv")
    data_path_selective_C1 = os.path.join(datadir, "selective_data_C-1.csv")
    data_path_selective_C28 = os.path.join(datadir, "selective_data_C28.csv")

    result_whole = pd.read_csv(result_path_whole, sep='\t', index_col=0)
    result_selective = pd.read_csv(result_path_selective, sep='\t', index_col=0)

    data_whole_III14 = pd.read_csv(data_path_whole_III14)
    data_whole_III14 = rearagne_columns(data_whole_III14)
    # print(data_whole_III14.iloc[:5, :5])
    data_whole_C1 = pd.read_csv(data_path_whole_C1)
    data_whole_C1 = rearagne_columns(data_whole_C1)
    data_whole_C28 = pd.read_csv(data_path_whole_C28)
    data_whole_C28 = rearagne_columns(data_whole_C28)

    data_selective_III14 = pd.read_csv(data_path_selective_III14)
    data_selective_III14 = rearagne_columns(data_selective_III14)
    # print(data_selective_III14.iloc[:5, :5])
    data_selective_C1 = pd.read_csv(data_path_selective_C1)
    data_selective_C1 = rearagne_columns(data_selective_C1)
    data_selective_C28 = pd.read_csv(data_path_selective_C28)
    data_selective_C28 = rearagne_columns(data_selective_C28)
    
    print("Paramter combination for best mean AUC:")
    print("for whole chip")
    print('at time point III14:')
    time_point_whole_III14 = select_time_point(result_whole, "III14")
    kernel_param_whole_III14 = get_kernel_paramter(time_point_whole_III14)
    print(kernel_param_whole_III14)
    print('')
    print('at time point C-1:')
    time_point_whole_C1 = select_time_point(result_whole, "C-1")
    kernel_param_whole_C1 = get_kernel_paramter(time_point_whole_C1)
    print(kernel_param_whole_C1)
    print('')
    print('at time point C28:')
    time_point_whole_C28 = select_time_point(result_whole, "C28")
    kernel_param_whole_C28 = get_kernel_paramter(time_point_whole_C28)
    print(kernel_param_whole_C28)
    print('')
    print("for selective:")
    print('at time point III14:')
    time_point_sel_III14 = select_time_point(result_selective, "III14")
    kernel_param_sel_III14 = get_kernel_paramter(time_point_sel_III14)
    print(kernel_param_sel_III14)
    print('')
    print('at time point C-1:')
    time_point_sel_C1 = select_time_point(result_selective, "C-1")
    kernel_param_sel_C1 = get_kernel_paramter(time_point_sel_C1)
    print(kernel_param_sel_C1)
    print('')
    print('at time point C28:')
    time_point_sel_C28 = select_time_point(result_selective, "C28")
    kernel_param_sel_C28 = get_kernel_paramter(time_point_sel_C28)
    print(kernel_param_sel_C28)
    print('')
    print('=================================================================================')
    print('')
    print("Feature evaluation at III14 for whole data")
    coef_whole_III14 = main(data_whole_III14, kernel_param_whole_III14)
    print('=================================================================================')
    print("Feature evaluation at C-1 for whole data")
    coef_whole_C1 = main(data_whole_C1, kernel_param_whole_C1)
    print('=================================================================================')
    print("Feature evaluation at C28 for whole data")
    coef_whole_C28 = main(data_whole_C28, kernel_param_whole_C28)
    print('=================================================================================')
    print('')
    print("Feature evaluation at III14 for selective data")
    coef_selective_III14 = main(data_selective_III14, kernel_param_sel_III14)
    print('=================================================================================')
    print("Feature evaluation at C-1 for selective data")
    coef_selective_C1 = main(data_selective_C1, kernel_param_sel_C1)
    print('=================================================================================')
    print("Feature evaluation at C28 for selective data")
    coef_selective_C28 = main(data_selective_C28, kernel_param_sel_C28)
    print('')
    print('=================================================================================')


    pd.DataFrame(data=coef_whole_III14).to_csv(os.path.join(outputdir_whole, r'coefficients_whole_III14.tsv'),
                                                  sep='\t', na_rep='nan')
    pd.DataFrame(data=coef_whole_C1).to_csv(os.path.join(outputdir_whole, r'coefficients_whole_C1.tsv'),
                                               sep='\t', na_rep='nan')
    pd.DataFrame(data=coef_whole_C28).to_csv(os.path.join(outputdir_whole, r'coefficients_whole_C28.tsv'),
                                                sep='\t', na_rep='nan')

    pd.DataFrame(data=coef_selective_III14).to_csv(os.path.join(outputdir_selective,
                                                      r'coefficients_selective_III14.tsv'),
                                                      sep='\t', na_rep='nan')
    pd.DataFrame(data=coef_selective_C1).to_csv(os.path.join(outputdir_selective,
                                                                r'coefficients_selective_C1.tsv'),
                                                   sep='\t', na_rep='nan')
    pd.DataFrame(data=coef_selective_C28).to_csv(os.path.join(outputdir_selective,
                                                                r'coefficients_selective_C28.tsv'),
                                                   sep='\t', na_rep='nan')
