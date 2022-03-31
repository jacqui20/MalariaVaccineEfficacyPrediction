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
from source import Feature_Evaluation_multitask_SVM
from source import Feature_Evaluation_simulated_data
from source import Normal_Distribution_Fitting

cwd = os.getcwd()
outputdir = '/'.join(cwd.split('/')[:-1]) + '/results/simulated_data'

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
    best_AUC = max(kernel_parameter["mean AUC"])
    pam = kernel_parameter.loc[kernel_parameter['mean AUC'] == best_AUC, 'label']
    pam = pam.str.split(" ", expand=True)
    return pam, best_AUC


def main():
    """
    MAIN FUNCTION of the feature selection approach.

    This module contains the main function of the feature selection approach. Here, the main idea is to calculate the
    distances of each single feature to the classification boundary of the previously evaluated best multitask-SVM
    classifier.

    Args: infile (path): path to the preprocessed proteome matrix as n x m matrix
                        (where n = samples in rows, m = features as columns) concatenated with the column "time point",
                        the column "protection state" and the column "dosage" as .csv file

          results_of_multitask-SVM_approach (path): the path to the resulted file of Parser_multitask_SVM.py

          -uq (float): the value of the upper quantile value, by default 0.75
          -lq (float): the value of the upper quantile value, by default 0.25

    Returns: .csv file: conainting the evaluated distance measurements per feature for the consensus sample (median
                        quantile), the upper- and lower-Quantile and the resulting absolute distance |d| of each feature
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-infile", '--infile', help="path to the input file, which is structured as preprocessed "
                                                    "proteome matrix as n x m matrix" "(where n = samples in rows, "
                                                    "m = features as columns) concatenated with the column "
                                                    "'time point', the column 'protection state' and "
                                                    "the column 'dosage' as .csv file",
                        type=Path,
                        required=True)
    parser.add_argument("-results_of_multitask_SVM_approach", "--results_of_multitask_SVM_approach",
                        help="path to results of the multitask-SVM approach",
                        type=Path)
    parser.add_argument("-uq", "--Upper_quantile", help="define percentage for upper quantile as int, by default 75%",
                        type=int, default=75)  # Upper quantile
    parser.add_argument("-lq", "--Lower_quantile", help="define percentage for lower quantile as int, by default 25%",
                        type=int, default=25)  # Lower quantile
    parser.add_argument("-NormalDistributionFitting", "--NormalDistributionFitting", action="store_true")

    args = parser.parse_args()

    # set default value of 75 for upper quantile, if not provided
    uq = args.Upper_quantile
    if uq is None:
        uq = 75

    # set default value of 25 for upper quantile, if not provided
    lq = args.Lower_quantile
    if lq is None:
        lq = 25

    # path to preprocessed malaria proteome data
    file = args.infile
    name_input_file = args.infile.name
    if not os.path.exists(file):
        print("input file does not exist:", file)
        return
    data = pd.read_csv(file)

    # load results of the multitask-SVM approach
    # kernel_evaluation_file = args.results_of_multitask_SVM_approach
    # name_result_evaluation_file = args.results_of_multitask_SVM_approach.name
    # if not os.path.exists(kernel_evaluation_file):
    #    print("input file does not exist:", kernel_evaluation_file)
    #    return
    # kernel_param = pd.read_csv(kernel_evaluation_file)

    # Print value setting for the feature selection approach
    print("\n")
    print("The feature selection approach is initialized with the following parameters:")
    print("value of upper quantile      = ", str(uq))
    print("value of lower quantile      = ", str(lq))
    # print("name of loaded result file from multitask-SVM evaluation: " + args.results_of_multitask_SVM_approach.name)
    print("\n")

    """
    Call ESPY measurement.

    """
    if args.results_of_multitask_SVM_approach:
        kernel_evaluation_file = args.results_of_multitask_SVM_approach
        # name_result_evaluation_file = args.results_of_multitask_SVM_approach.name
        # if not os.path.exists(kernel_evaluation_file):
        #    print("input file does not exist:", kernel_evaluation_file)
        # return
        kernel_param = pd.read_csv(kernel_evaluation_file)
        # get combination of kernel values for highest mean AUC value based on the results from
        # the multitask-SVM approach
        kernel_comb, highest_AUC = get_kernel_paramter(kernel_param)
        print("input file: ", args.infile.name)
        print("name of loaded result file from multitask-SVM evaluation: " +
              args.results_of_multitask_SVM_approach.name)
        print("setting of kernel parameter combination", "\n", kernel_comb, "for highest mean AUC value of ",
              highest_AUC)
        print('\n')
        print("ESPY value measurement started on proteome array data:")
        print("\n")

        distance_result, combination_table, time = Feature_Evaluation_multitask_SVM.feature_evaluation(data,
                                                                                                       kernel_param,
                                                                                                       uq,
                                                                                                       lq)

        # generate output file name per time point
        output_filenameD = "Result_of_feature_distance_at_timePoint" + "_" + str(time) + ".csv"
        # output_filename_CombTable = "Combination_of_features_at_timePoint" + "_" + str(time) + ".csv"

        # save and show location of output files
        distance_result.to_csv(os.path.join(outputdir, output_filenameD), index=True)
        # combination_table.to_csv(output_filename_CombTable, index=True)
        print("results are saved in: " + os.path.join(outputdir, output_filenameD))

        if args.NormalDistributionFitting:
            print('\n')
            print("Normal distribution fitting is running to evaluate informative features with p-value < 0.05 "
                  "on proteome data at time point: " + str(time))
            print("\n")
            nfitting_result = Normal_Distribution_Fitting.main_function(distance_result, outputdir)
            output_filename_nf = "Evaluated_significant_features_on_proteome_data_at_timePoint_" + str(time) + ".csv"
            nfitting_result.to_csv(output_filename_nf, index=True)
            print('results are saved in: ' + output_filename_nf)
    else:
        print("input file: ", args.infile.name)
        print("ESPY value measurement started on simulated data:")
        print("\n")
        distance_result = Feature_Evaluation_simulated_data.ESPY_measurment(data, lq=lq, up=uq)
        output_filename = "ESPY_value_of_features_on_simulated_data.csv"
        distance_result.to_csv(os.path.join(outputdir, output_filename), index=True)
        print("results are saved in: " + os.path.join(outputdir, output_filename))

        if args.NormalDistributionFitting:
            print('\n')
            print("Normal distribution fitting is running to evaluate informative features with p-value < 0.05 "
                  "on simulated data:")
            print("\n")
            nfitting_result = Normal_Distribution_Fitting.main_function(distance_result, outputdir)
            output_filename_nf = "Evaluated_significant_features_on_simulated_data.csv"
            nfitting_result.to_csv(os.path.join(outputdir, output_filename_nf), index=True)
            print('results are saved in: ' + os.path.join(outputdir, output_filename_nf) + " and "
                  "Evaluated_significant_features.png")


if __name__ == "__main__":
    main()
