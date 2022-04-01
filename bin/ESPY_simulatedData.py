"""
Parser for the feature selection approach

This module computes the ESPY value of each single feature. The ESPY value is the distances of each single features to
the classification boundary in the multitask-SVM model and compares the change of the distance with a consensus sample.

This module requires the output file of the Parser_multitask_SVM.py module and the Feature_Evaluation_multitask_SVM.py
script.

Created on: 25.05.2019

@Author: Jacqueline Wistuba-Hamprecht


"""

# required packages
import argparse
import os
import pandas as pd
from pathlib import Path
import sys
sys.path.append("/Users/schmidtj/Documents/Doktorarbeit/Publications/"
                "Predicting_malaria_vaccine_efficacy_from_anti-plasmodial_ab_profiles/"
                "Proteome_chip_analysis_publication/plos-latex-template/Resubmission_02_22/"
                "Local_Code_Github/MalariaVaccineEfficacyPrediction")
import FeatureEvaluation_simulatedData


cwd = os.getcwd()
outputdir = '/'.join(cwd.split('/')[:-1]) + '/results/simulated_data'


def main(
        data: pd.DataFrame,
        uq: int,
        lq: int,
        filename: str,
        outputdir: Path
) -> object:

    # Print value setting for the feature selection approach
    print("\n")
    print("The feature selection approach is initialized with the following parameters:")
    print("value of upper quantile      = ", str(uq))
    print("value of lower quantile      = ", str(lq))
    print("\n")

    """
    Call ESPY measurement.
    """
    print("ESPY value measurement started on " + str(filename))
    print("\n")
    output_filename = "ESPY_value_of_features_on_simulated_data"
    distance_result = Feature_Evaluation_simulated_data.ESPY_measurment(
        simulated_data=data,
        lq=lq,
        up=uq,
        outputdir=outputdir,
        outputname=output_filename)
    distance_result.to_csv(os.path.join(outputdir, output_filename + ".tsv"), sep='\t', na_rep='nan')
    print("results are saved in: " + os.path.join(outputdir, output_filename))


if __name__ == "__main__":
    print('sys.path:', sys.path)
    print('pandas version:', pd.__version__)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-infile", '--infile',  type=Path,  required=True,
        help="Path to the directory were the simulated data is located.")

    parser.add_argument(
        "-uq", "--Upper_quantile", type=int, default=75,
        help="define percentage for upper quantile as int, by default 75%")

    parser.add_argument(
        "-lq", "--Lower_quantile", type=int, default=25,
        help="define percentage for lower quantile as int, by default 25%")

    args = parser.parse_args()

    # path to preprocessed malaria proteome data
    file = args.infile
    name_input_file = args.infile.name
    if not os.path.exists(file):
        print("input file does not exist:", file)

    data = pd.read_csv(file)

    main(data=data,
         uq=args.Upper_quantile,
         lq=args.Lower_quantile,
         filename=args.infile.name,
         outputdir=outputdir)
