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
This module contains the preprocessing of the raw proteome array data

@Author: Jacqueline Wistuba-Hamprecht and Bernhard Reuter

"""

# required packages
import numpy as np
import pandas as pd
import os

cwd = os.getcwd()
datadir = '/'.join(cwd.split('/')[:-1]) + '/data'
outputdir = '/'.join(cwd.split('/')[:-1]) + '/data/proteome_data'


def substract_preimmunization_baseline(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """ Substract baseline immunity

    The baseline immunity (TimePointOrder = 1) is substracted from the data generated at
    post-immunization (TimePointOrder = 2, 3 and 4) for each sample x_i in data X
    to focus on PfSPZ-CVac induced antibody responses.

    Parameters
    ----------
    data : pd.DataFrame
        Raw proteome data, n x m matrix (n = samples as rows, m = features as columns)

    Returns
    -------
    data_minusBS : pd.DataFrame
        Data minus baseline immunity.
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
    data_III14.iloc[:, start_indx:] = data_III14.iloc[:, start_indx:].sub(
        data_baseline.iloc[:, start_indx:], fill_value=0
    )
    data_C1.iloc[:, start_indx:] = data_C1.iloc[:, start_indx:].sub(
        data_baseline.iloc[:, start_indx:], fill_value=0
    )
    data_C28.iloc[:, start_indx:] = data_C28.iloc[:, start_indx:].sub(
        data_baseline.iloc[:, start_indx:], fill_value=0
    )
    data_minusBS = pd.concat([data_III14, data_C1, data_C28], ignore_index=True)
    data_minusBS = data_minusBS.drop(columns=['index'])
    return data_minusBS


def normalization(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """ normalization

    Antibody signal intensities were normalized as described by Mordmüller et al.
    Negative Pf-specific antibody signal intensities were set to 1 and then transformed by
    the base 2 logarithm.

    Parameters
    ----------
    data : pd.DataFrame
        Raw proteome data, n x m matrix (n = samples as rows, m = features as columns).

    Returns
    -------
    data : pd.DataFrame
        Normalized proteome data.
    """
    start_indx = data.columns.get_loc("TimePointOrder") + 1
    # replace all negative values by 1
    data[data.columns[start_indx:]] = data[data.columns[start_indx:]].clip(lower=1)
    # log2 transformation
    data[data.columns[start_indx:]] = np.log2(data[data.columns[start_indx:]])
    return data


def sort(
    data: pd.Dataframe,
) -> pd.DataFrame:
    """ Sorting

    Input data is sorted by patient IDs to keep same patient over all four time points in order.

    Parameters
    ----------
    data : pd.DataFrame
        Raw proteome data, n x m matrix (n = samples as rows, m = features as columns).

    Returns
    -------
    data : np.ndarray
        Sorted proteome data.
    """
    data.sort_values(by=["Patient"], inplace=True)
    data.reset_index(inplace=True)
    data.drop(columns=['index'], inplace=True)
    return data


def preprocessing(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """ Preprocessing of raw proteome data

    Raw proteome array data is preprocessed in the following way:
    - Substraction of baseline immunity
    - Normalization (log2 normalization)
    - Sorted by patients ID

    Parameters
    ----------
    data : pd.DataFrame
        Raw proteome data, n x m matrix (n = samples as rows, m = features as columns).

    Returns
    -------
    data : pd.DataFrame
        Preprocessed proteome data.
    """
    data_minusBS = substract_preimmunization_baseline(data)
    sort(data_minusBS)
    normalization(data_minusBS)
    return data_minusBS


if __name__ == "__main__":
    data_path_whole = os.path.join(datadir, 'proteome_data/whole_proteomearray_rawdata.csv')
    data_path_selective = os.path.join(datadir, 'proteome_data/surface_proteomearray_rawdata.csv')

    proteome_whole_array = pd.read_csv(data_path_whole)
    proteome_selective_array = pd.read_csv(data_path_selective)

    preprocessed_whole_data = preprocessing(data=proteome_whole_array)
    preprocessed_whole_data.to_csv(
        os.path.join(outputdir, r'preprocessed_whole_data.csv'),
        index=False,
    )

    preprocessed_selective_data = preprocessing(data=proteome_selective_array)
    preprocessed_selective_data.to_csv(
        os.path.join(outputdir, r'preprocessed_selective_data.csv'),
        index=False,
    )

    print("\n")
    print("the preprocessed data is now saved in ./data/proteome_data as:" + "\n")
    print("preprocessed_whole_data.csv" + ' and ' + "preprocessed_selective_data.csv")
