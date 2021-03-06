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
@Author: Jacqueline Wistuba-Hamprecht and Bernhard Reuter

"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from source.utils import select_timepoint, get_parameters


def rearrange_columns(
    data: pd.DataFrame
) -> pd.DataFrame:
    """ Re-arrange column order of proteome dataframe.

    Move column 'Dose' to the beginning of the feature columns.

    Parameters
    ----------
    data : pd.Dataframe
        Proteome dataframe.

    Returns
    --------
    df : pd.Dataframe
        Proteome dataframe with re-arranged columns.

    """

    df = data.copy()
    dose = df['Dose']
    df = df.drop(columns=['Dose'])
    df.insert(loc=4, column='Dose', value=dose)
    return df


def RLR_model(
        *,
        X: np.ndarray,
        y: np.ndarray,
        params: Dict[str, float],
        feature_labels: List[str],
) -> Tuple[LogisticRegression, pd.DataFrame]:
    """Fit RLR model on proteome data.

    Fit RLR model with parameters given by `params` on
    proteome data and find non-zero coefficients.

    Parameters
    ---------
    X : np.ndarray
        Data array.
    y : np.ndarray
        Label array
    params : dict
        Parameter dictionary used to fit RLR.
    feature_labels : list
        List of feature labels.

    Returns
    -------
    model : sklearn.linear_model.LogisticRegression object
        Fitted RLR model with parameters given by `params`.
    coefs_nonzero : pd.Dataframe
        Non-zero RLR coefficients.

    """

    # Initialize and fit RLR
    estimator = make_pipeline(
        StandardScaler(
            with_mean=True,
            with_std=True,
        ),
        LogisticRegression(
            penalty='elasticnet',
            C=params['logisticregression__C'],
            solver='saga',
            l1_ratio=params['logisticregression__l1_ratio'],
            max_iter=10000,
        ),
    )
    estimator.fit(X, y)
    print(estimator)
    model = estimator[1]

    # Extract non-zero coefficients
    print("Number of non-zero weights:", np.count_nonzero(model.coef_))
    coefs = pd.concat(
        [pd.DataFrame(feature_labels), pd.DataFrame(np.transpose(model.coef_))],
        axis=1
    )
    coefs = coefs.set_axis(['Pf_antigen_ID', 'weight'], axis='columns')
    coefs.sort_values(by=['weight'], ascending=True, inplace=True)
    coefs_nonzero = coefs[coefs['weight'] != 0]
    return model, coefs_nonzero


def featureEvaluationRLR(
        data: pd.DataFrame,
        rgscv_results: pd.DataFrame,
        timepoint: str,
):
    """Evaluation of informative features from RLR.

    Parameter
    ---------
    data : pd.DataFrame
        Dataframe containing proteome data.
    rgscv_results : pd.DataFrame
        DataFrame containing optimal parameters and mean AUROC values
        per time point as found via Repeated Grid-Search CV (RGSCV).
    timepoint : str
        Time point to evaluate informative features for.

    Returns
    -------
    coefs : pd.Dataframe
        Dataframe of non-zero coefficients.

    """

    print(f"Parameter combination for best mean AUC at time point {timepoint} :")
    timepoint_results = select_timepoint(
        rgscv_results=pd.DataFrame(rgscv_results),
        timepoint=timepoint)

    params = get_parameters(
        timepoint_results=timepoint_results,
        model='RLR',
    )
    print(f"Parameters: {params}")
    print('')

    if all(isinstance(x, float) for x in params.values()):

        print("Start feature evaluation with dose as auxillary feature:")
        data = rearrange_columns(data)
        X = data.iloc[:, 4:].to_numpy()
        y = data.loc[:, 'Protection'].to_numpy()
        feature_labels = data.iloc[:, 4:].columns.to_list()

        _, coefs = RLR_model(
            X=X,
            y=y,
            params=params,
            feature_labels=feature_labels,
        )
        print(coefs)

    else:

        raise ValueError("All parameter values must be of type float.")

    return coefs
