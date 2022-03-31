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
from source import featureEvaluationRLR


def main():
    """
    Evaluation of informative features from RLR

    """
    coefs = featureEvaluationRLR(proteome_data, results_rgscv, timepoint)
    pd.DataFrame(data=coefs).to_csv(os.path.join(outputdir, output_file_name + ".tsv"), sep='\t', na_rep='nan')

    print(
        "Results are saved in: "
        + os.path.join(outputdir, output_file_name + ".tsv")
    )


if __name__ == "__main__":
    print('sys.path:', sys.path)
    print('scikit-learn version:', sklearn.__version__)
    print('pandas version:', pd.__version__)
    print('numpy version:', np.__version__)
    print('scipy version:', scipy.__version__)

    main()
