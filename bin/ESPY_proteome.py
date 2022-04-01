"""
Identification of informative features based on EPSY measurement from proteome data

All results are saved as .tsv files


"""

import os
import sys
maindir = '/'.join(os.getcwd().split('/')[:-1])
sys.path.append(maindir)
from source.FeatureEvaluation_multitask import ESPY_measurment



if __name__ == "__main__":



    distances_for_all_feature_comb, combinations = ESPY_measurment(
        target_label = target_label,
        kernel_parameter= kernel_parameter,
        proteome_data = proteome_data,
        uq = upper_value,
        lq = lower_value,
        kernel_matrix = kernel_matrix
        TimePoint = time_point,
        t_nm = t)

    distances_for_all_feature_comb.to_csv(os.path.join(outputdir, output_filename), sep='\t', na_rep='nan')
    print('results are saved in: ' + os.path.join(outputdir, output_filename))

