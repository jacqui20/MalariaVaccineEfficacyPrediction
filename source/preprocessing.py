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
This module contains functions used to preprocess the raw proteome data.

@Author: Jacqueline Wistuba-Hamprecht and Bernhard Reuter

"""

import numpy as np
import pandas as pd


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
    data_baseline = data.loc[data["TimePointOrder"] == 1, :].copy()
    data_baseline.reset_index(inplace=True, drop=True)

    # after immunization III 14
    data_III14 = data.loc[data["TimePointOrder"] == 2, :].copy()
    data_III14.reset_index(inplace=True, drop=True)

    # before re-infection C-1
    data_C1 = data.loc[data["TimePointOrder"] == 3, :].copy()
    data_C1.reset_index(inplace=True, drop=True)

    # after re-infection C+ 28
    data_C28 = data.loc[data["TimePointOrder"] == 4, :].copy()
    data_C28.reset_index(inplace=True, drop=True)

    # substract baseline
    start_indx = data.columns.get_loc("TimePointOrder") + 1
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


def sort_proteome_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """ Sorting

    Input data is sorted by time point and patient ID
    to keep same patient over all four time points in order.

    Parameters
    ----------
    data : pd.DataFrame
        Raw proteome data, n x m pd.DataFrame (n = samples as rows, m = features as columns)

    Returns
    -------
    data : pd.DataFrame
        Returns sorted DataFrame
    """
    data.sort_values(by=["TimePointOrder", "Patient"], inplace=True)
    data.reset_index(inplace=True, drop=True)
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
    data = substract_preimmunization_baseline(data)
    sort_proteome_data(data)
    normalization(data)
    return data
