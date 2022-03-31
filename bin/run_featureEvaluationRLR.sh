#!/bin/bash

source "/Users/schmidtj/anaconda3/etc/profile.d/conda.sh"
conda activate malaria_env

maindir='/Users/schmidtj/Documents/GitHub/MalariaVaccineEfficacyPrediction/results/RLR'
data_maindir='/Users/schmidtj/Documents/GitHub/MalariaVaccineEfficacyPrediction/data/proteome_data'

time_points["III14", "C-1", "C28"]

// generate output folders for whole and selective in ./results/RLR
//outputdir_whole = '/'.join(cwd.split('/')[:-1]) + '/results/RLR/whole/Informative_features'
//outputdir_selective = '/'.join(cwd.split('/')[:-1]) + '/results/RLR/selective/Informative_features'
// output_file_name = "Evaluated_features" + "whole" or "selective" + "data" + time_point{III14 or C1 or C28}


// load data for selective and whole proteome data
// load results of RLR gridsearch for selective and whole dataset
//Run:
// for the whole proteome data run featureEvalRLR.py with the results of RLR gridsearch for the whole proteome data and
// add the combinations of time_points
// e.g. featureEvalRLR(proteome_data, results_RLR, time_point, outputdir, output_file_name)

// for the selective proteome data run featureEvalRLR.py with the results of RLR gridsearch for the selective proteome data and
// add the combinations of time_points

