"""
This module contains the preprocessing of the raw proteome array data
Created on: 25.05.2019

@Author: Jacqueline Wistuba-Hamprecht

"""

# required packages
from typing import Callable, Any, Union

import numpy as np
import pandas as pd
import os
import sys
import os.path

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


def substract_preimmunization_baseline(data):
    """ Substract baseline immunity

            The baseline immunity (TimePointOrder = 1) is substracted from the data generated at post-immunization
            (TimePointOrder = 2, 3 and 4) for each sample x_i in data X to focus on PfSPZ-CVac induced antibody
            responses.

            Args: data (matrix): matrix raw proteome data, n x m matrix (n = samples as rows, m = features as columns)


            Returns: data_minusBS (matrix): returns data X minus baseline immunity
            """
    # baseline: preimmunization data I-1
    data_baseline = data.loc[data["TimePointOrder"] == 1].copy()
    data_baseline.reset_index(inplace=True)

    # after immunization III 14
    data_III14 = data.loc[data["TimePointOrder"] == 2].copy()
    data_III14.reset_index(inplace=True)

    # before re-infection C-1
    data_C1 = data.loc[data["TimePointOrder"] == 3].copy()
    data_C1.reset_index(inplace=True)

    # after re-infection C+ 28
    data_C28 = data.loc[data["TimePointOrder"] == 4].copy()
    data_C28.reset_index(inplace=True)

    # substract baseline
    start_indx = data.columns.get_loc("TimePointOrder") + 2
    data_III14.iloc[:, start_indx:] = data_III14.iloc[:, start_indx:].sub(data_baseline.iloc[:, start_indx:],
                                                                          fill_value=0)
    data_C1.iloc[:, start_indx:] = data_C1.iloc[:, start_indx:].sub(data_baseline.iloc[:, start_indx:],
                                                                    fill_value=0)
    data_C28.iloc[:, start_indx:] = data_C28.iloc[:, start_indx:].sub(data_baseline.iloc[:, start_indx:],
                                                                      fill_value=0)
    data_minusBS = [data_III14, data_C1, data_C28]
    data_minusBS = pd.concat(data_minusBS, ignore_index=True)
    data_minusBS = data_minusBS.drop(columns=['index'])
    return data_minusBS


def normalization(data):
    """ normalization

            Antibody signal intensities were normalized as described by Mordmüller et al.
            Negative Pf-specific antibody signal intensities were set to 1 and then transformed by
            the base 2 logarithm.

            Args: data (matrix): matrix raw proteome data, n x m matrix (n = samples as rows, m = features as columns)


            Returns: data (matrix): returns normalized data matrix X
            """
    start_indx = data.columns.get_loc("TimePointOrder") + 1
    # replace all negative values by 1
    data[data.columns[start_indx:]] = data[data.columns[start_indx:]].clip_lower(1)
    # log2 tranformation
    data[data.columns[start_indx:]] = np.log2(data[data.columns[start_indx:]])
    return data


def sort(data):
    """ Sorting

            Input data is sorted by patient IDs to keep same patient over all four time points in order.

            Args: data (matrix): matrix raw proteome data, n x m matrix (n = samples as rows, m = features as columns)


            Returns: data (matrix): returns sorted matrix X
            """
    data.sort_values(by=["Patient"], inplace=True)
    data.reset_index(inplace=True)
    data.drop(columns=['index'], inplace=True)
    return data


def preprocessing(data_path):
    """ Preprocessing of raw proteome data

            Raw proteome array data is preprocessed in the following way:
            - Substraction of baseline immunity
            - Normalization (log2 normalization)
            - Sorted by patients ID

            Args: data (matrix): matrix raw proteome data, n x m matrix (n = samples as rows, m = features as columns)


            Returns: data (matrix): returns preprocessed data matrix X
            """
    data_minusBS = substract_preimmunization_baseline(data_path)
    sort(data_minusBS)
    normalization(data_minusBS)
    return data_minusBS


if __name__ == "__main__":
    data_path = lambda d: os.path.join(os.path.dirname(os.path.realpath(__file__)), d)
    proteome_whole_array = pd.read_csv(data_path('whole_proteomearray_rawdata.csv'))
    proteome_selective_array = pd.read_csv(data_path('surface_proteomearray_rawdata.csv'))

    preprocessed_whole_data = preprocessing(data_path=proteome_whole_array)
    preprocessed_whole_data.to_csv("preprocessed_whole_data.csv", index=False)

    preprocessed_selective_data = preprocessing(data_path=proteome_selective_array)
    preprocessed_selective_data.to_csv("preprocessed_selective_data.csv", index=False)

    print("\n")
    print("the preprocessed data is now saved in /Data as:" + "\n")
    print("preprocessed_whole_data.csv" + ' and ' + "preprocessed_selective_data.csv")
