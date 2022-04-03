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
from sklearn.model_selection import train_test_split
import sys
maindir = '/'.join(os.getcwd().split('/')[:-1])
sys.path.append(maindir)
from source.FeatureEvaluation import ESPY_measurment, get_kernel_paramter
from source.utils import initialize_svm_model, make_plot, select_timepoint, get_parameters, sort_proteome_data, \
    multitask_model


cwd = os.getcwd()
outputdir = '/'.join(cwd.split('/')[:-1]) + '/results/simulated_data'


def main(
        data: pd.DataFrame,
        uq: int,
        lq: int,
        filename: str,
        target_label=None,
        kernel_parameter=None,
        kernel_matrix=None,
        timePoint=None,
        t_nm=None
) -> object:


        data = pd.read_csv(data)

        """
        Call ESPY measurement.
        """
        print("ESPY value measurement started on " + str(filename))
        print("with the following parameters:")
        print("value of upper quantile      = ", str(uq))
        print("value of lower quantile      = ", str(lq))
        print("at time point                = ", str(timePoint))
        print("\n")
        output_filename = "ESPY_value_of_features_on_simulated_data"

        if filename == "simulated_data.csv":
            X_train, X_test, Y_train, Y_test = train_test_split(
                data.iloc[:, :1000].to_numpy(),
                data.iloc[:, 1000].to_numpy(),
                test_size=0.3,
                random_state=123)

            rbf_svm_model = initialize_svm_model(X_train_data=X_train,
                                                 y_train_data=Y_train,
                                                 X_test_data=X_test,
                                                 y_test_data=Y_test)

            distance_result = ESPY_measurment(
                    data=pd.DataFrame(X_test),
                    model = rbf_svm_model,
                    lq=lq,
                    up=uq,
                    outputdir=outputdir,
                    outputname=output_filename,
                    filename=filename)

            make_plot(data=distance_result.iloc[:, :25],
                      name=output_filename,
                      outputdir=outputdir)

            distance_result.to_csv(os.path.join(outputdir, output_filename + ".tsv"), sep='\t', na_rep='nan')
            print("results are saved in: " + os.path.join(outputdir, output_filename))

        elif filename == "preprocessed_whole_data.csv" or "preprocessed_selective_data.csv":
            time_point = select_timepoint(kernel_parameter, timePoint)
            param_combi_RRR = get_kernel_paramter(time_point)

            kernel_pamR0 = pd.to_numeric(param_combi_RRR[5].str.split(",", expand=True)[0])
            print("gamma value for rbf kernel for time point series " + str(kernel_pamR0))
            kernel_pamR1 = pd.to_numeric(param_combi_RRR[7].str.split(",", expand=True)[0])
            print("gamma value for rbf kernel for dose " + str(kernel_pamR1))
            kernel_pamR2 = pd.to_numeric(param_combi_RRR[9].str.split(",", expand=True)[0])
            print("gamma value for rbf kernel for ab signals " + str(kernel_pamR2))
            print('')

            y_label = target_label.loc[:, 'Protection'].to_numpy()
            multitask_classifier = multitask_model(
                                    kernel_matrix=kernel_matrix,
                                    kernel_parameter=param_combi_RRR,
                                    y_label=y_label)

            proteome_data = sort_proteome_data(
                                data=data)
            print("Are values in proteome data floats: " + str(np.all(np.isin(proteome_data.dtypes.to_list()[5:], ['float64']))))
            data_at_timePoint = proteome_data.loc[proteome_data["TimePointOrder"] == t_nm]

            distance_result = ESPY_measurment(
                data=data_at_timePoint.iloc[:, :3],
                model = multitask_classifier,
                lq=lq,
                up=uq,
                outputdir=outputdir,
                outputname=output_filename,
                proteome_data=proteome_data,
                kernel_paramter=param_combi_RRR)

            #distance_result.to_csv(os.path.join(outputdir, output_filename), sep='\t', na_rep='nan')
            #print('results are saved in: ' + os.path.join(outputdir, output_filename))


if __name__ == "__main__":
    print('sys.path:', sys.path)
    print('pandas version:', pd.__version__)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-infile", '--infile',  type=Path,  required=True,
        help="Path to the directory were the simulated data or preprocessed proteome data is located.")

    parser.add_argument(
        "-uq", "--Upper_quantile", type=int, default=75,
        help="define percentage for upper quantile as int, by default 75%")

    parser.add_argument(
        "-lq", "--Lower_quantile", type=int, default=25,
        help="define percentage for lower quantile as int, by default 25%")

    parser.add_argument(
        '-target-label-path', '--target-label-path', action='store',
        help="Path to the File were the target labels of proteome data are stored."
    )

    parser.add_argument(
        '-kernel-matrix-path', '--kernel-matrix-path',  action='store',
        help="Path to the File were the precomputed kernel matrices are stored.'"
    )

    parser.add_argument(
        '-rgscv-path', '--rgscv-path', action='store',
        help='Path to the File were the RGSCV results are stored.'
    )

    parser.add_argument(
        '-timepoint', '--timepoint', action='store',
        help='Time point for which the analysis shall be performed.'
    )

    args = parser.parse_args()

    if args.infile.name == "simulated_data.csv":
        main(data=args.infile,
             uq = args.Upper_quantile,
             lq = args.Lower_quantile,
             filename=args.infile.name)

    elif args.infile.name == "preprocessed_whole_data.csv" or "preprocessed_selective_data.csv":
        if args.timepoint == 'III14':
            t = 2
        elif args.timepoint == 'C-1':
            t = 3
        elif args.timepoint == "C28":
            t = 4

        main(data=args.infile,
             uq = args.Upper_quantile,
             lq = args.Lower_quantile,
             filename=args.infile.name,
             target_label = args.target_label_path,
             kernel_parameter= args.rgscv_path,
             kernel_matrix = args.kernel_matrix_path,
             timePoint = args.timepoint,
             t_nm = t)
    else:
        print("not correct input file was loaded - please load either simulated_data.csv, preprocessed_whole_data.csv "
              "or preprocessed_selective_data.csv")





