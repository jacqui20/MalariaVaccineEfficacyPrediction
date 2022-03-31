import argparse
import os
import pandas as pd
import sys
maindir = '/'.join(os.getcwd().split('/')[:-1])
sys.path.append(maindir)
import Normal_Distribution_Fitting

cwd = os.getcwd()
datadir = '/'.join(cwd.split('/')[:-1]) + '/results/multitaskSVM/whole/RRR/unscaled/Informative features'
outputdir_whole = '/'.join(cwd.split('/')[:-1]) + '/results/multitaskSVM/whole/RRR/unscaled/Informative features'


if __name__ == "__main__":
    distance_path = os.path.join(datadir, 'ESPY_values_whole_chip_III14.tsv')
    distance_whole_III14 = pd.read_csv(distance_path, sep='\t', index_col=0)
    #print(distance_whole_III14.head())
    output_name_III14 = "Evaluated_significant_features_whole_chip_III14"

    distance_path = os.path.join(datadir, 'ESPY_values_whole_chip_C1.tsv')
    distance_whole_C1 = pd.read_csv(distance_path, sep='\t', index_col=0)
    print(distance_whole_C1.head())
    output_name_C1 = "Evaluated_significant_features_whole_chip_C1"

    distance_path = os.path.join(datadir, 'ESPY_values_whole_chip_C28.tsv')
    distance_whole_C28 = pd.read_csv(distance_path, sep='\t', index_col=0)
    print(distance_whole_C28.head())
    output_name_C28 = "Evaluated_significant_features_whole_chip_C28"


    nfitting_result = Normal_Distribution_Fitting.main_function(distance_whole_C1, outputdir_whole, output_name_C1)
    print(nfitting_result)
    output_filename_nf = output_name_III14 + ".tsv"
    nfitting_result.to_csv(os.path.join(outputdir_whole, output_filename_nf), sep='\t', na_rep='nan')
    print('results are saved in: ' + output_filename_nf)