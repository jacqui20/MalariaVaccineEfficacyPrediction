#!/bin/bash

source "/Users/schmidtj/anaconda3/etc/profile.d/conda.sh"
conda activate malaria_env

maindir='/Users/schmidtj/Documents/GitHub/MalariaVaccineEfficacyPrediction'
data_maindir='/Users/schmidtj/Documents/GitHub/MalariaVaccineEfficacyPrediction/data/proteome_data'
kernelmatrix_maindir='/Users/schmidtj/Documents/GitHub/MalariaVaccineEfficacyPrediction/data/
                      precomputed_multitask_kernels/unscaled'
kernel_pam_maindir= "/Users/schmidtj/Documents/GitHub/MalariaVaccineEfficacyPrediction/results/multitaskSVM"


outputdir_whole = '/Users/schmidtj/Documents/GitHub/MalariaVaccineEfficacyPrediction/results/multitaskSVM/whole/RRR'
outputdir_selective = '/Users/schmidtj/Documents/GitHub/MalariaVaccineEfficacyPrediction/results/multitaskSVM/selective/
                        RRR'



combinations_timePoints = ('III14', "C-1", "C28") # cobinations of timepoints as str to get kernel parameter combination
combinations_t = (2,3,4) # combinations of timepoints as int to select data per time point

// get target label from kernelmatrix_maindir
// get kernel parameter combination for whole or selective data from kernel_pam_maindir
// get kernelmatrix for selected kernel parameter per data set (whole or selective) from kernelmatrix_maindir

// extract kernel parameter from kernelmatrix description and from kernel parameter combination file --> compare those
// parameter and check if they are the same

// get proteome data for whole and seletive data from data_maindir

// set up output folder "Informative features" to outputdir_whole and outputdir_selective


// run ESPY measurment with ESPY_proteome.py
    for selective or whole proteome data
    for time in combinations_timePoints
    for t in combinations_t #(where "III14" has to run with '2',
                          #"C-1" has to run with '3' and "C28" has to run with '4')
//  ESPY_proteome.ESPY_measrument(
        target_label = target_label, # is the same for all time points for whole/ selective data
        kernel_parameter= kernel_parameter, # is the same for all time points for whole/ selective data
        proteome_data = proteome_data, # is the same for all time points for whole/ selective data
        uq = 75, # is the same for all runs
        lq = 25, # is the same for all runs
        kernel_matrix = kernel_matrix # varies for kernel_parameter combination at time point time and proteome data
        TimePoint = time, # varies for whole and selecctive proteome data in combination_TimePoints
        t_nm = t # varies for whole and selecctive proteome data in combination_t
)
//  set up output file name for ESPY measurment: "ESPY_value" + time + whole or selective data + '.tsv'
//  set up output file name for NormalDistribution fitting measurment: "Significant_features" + time +
    whole or selective data + '.tsv'
//  pass outputdir_selective or outputdir_whole






