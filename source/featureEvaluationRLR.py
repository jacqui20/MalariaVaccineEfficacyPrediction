import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def get_RLR_parameters(
    timepoint_results: pd.DataFrame,
) -> Dict[str, float]:
    """Return combination of parameters to initialize RLR.

    Parameters
    ----------
    timepoint_results : pd.DataFrame
        DataFrame containing optimal parameters and mean AUROC values
        for a particular time point as found via Repeated Grid-Search CV (RGSCV).

    Returns
    --------
    params : dict
        Parameter dictionary.
    """
    roc_results = timepoint_results[timepoint_results['scoring'].isin(['roc_auc'])]
    assert roc_results.shape == (1, 4), \
        f"roc_results.shape != (1, 4): {roc_results.shape} != (1, 4)"
    params_string = roc_results['best_params'].iloc[0]
    assert type(params_string) == str, \
        f"type(params_string) != str: {type(params_string)} != str"
    params = eval(params_string)
    assert set(params.keys()) == {'logisticregression__l1_ratio', 'logisticregression__C'}, \
        ("set(params.keys()) != {'logisticregression__l1_ratio', 'logisticregression__C'}:"
         f"{set(params.keys())} != {'logisticregression__l1_ratio', 'logisticregression__C'}")
    return params


def select_timepoint(
    rgscv_results: pd.DataFrame,
    timepoint: str
) -> pd.DataFrame:
    """ Select time point to evaluate informative features from RLR.

    Parameter
    ---------
    rgscv_results : pd.DataFrame
        DataFrame containing optimal parameters and mean AUROC values
        per time point as found via Repeated Grid-Search CV (RGSCV).

    timepoint : str
        Time point to extract parameters and AUROC values for.

    Returns
    --------
    timepoint_results: pd.DataFrame
        DataFrame containing optimal parameters and mean AUROC values
        for the selected time point as found via Repeated Grid-Search CV (RGSCV).
    """
    timepoint_results = rgscv_results[rgscv_results['time'].isin([timepoint])]
    return timepoint_results


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
        params: dict,
        feature_labels: list
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
    coefs.set_axis(['Pf_antigen_ID', 'weight'], axis='columns')
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
        rgscv_results=rgscv_results,
        timepoint=timepoint)
    params = get_RLR_parameters(
        timepoint_results=timepoint_results
    )
    print(params)
    print('')

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

    return coefs
