# Multitask-SVM
This repository contains the source code for the adapted Multitask-SVM approach,
a method for classifying and interpreting the immunisation status of PfSPZ-CVac
vaccinated volunteers based on antibody profiles recovered from protein arrays.

The details of the method are published here:

## Overview of the Multitask-SVM approach

The **Multitask-SVM** model is structured into **three methods**:<br>
1) **the prediction model** multitask_SVM_for_time_series.py, which set-up
the Multitask-SVM model based on the antibody profiles. The prediction model
can be executed using the Parser_multitask_SVM.py parser. <br>
2) **the feature selection method** Feature_Evaluation_multitask_SVM.py,
which apply the informative distance measurement to evaluate informative features based on their respective
ESPY score. The underlying multitask-SVM
model for evaluating the informative features uses the evaluated parameters
from part 1). The feature selection method can be executed using
the Parser_feature_selection.py parser.<br>
3) **the normal distribution fitting model** Normal_Distribution_Fitting.py,
which fits the evaluated absolute distances for each features from part 2)
to a normal distribution to determine significant features by their respective
p-value < 0.05. The normal distribution fitting model can be executed using
the Parser_normal_distribution_fitting.py parser.

## Requirements

This project requires python == 3.6.5

## Structure of input data
The following data table gives an overview of the input data structure: <br>
(file format: .csv file)

| Patient | group | Protection | Dose | TimePointOrder | feature 1 | ... | feature n|
--------| ------| ---------- | ---- | ---------------| -------- | ----| ---------|
| ID-01 III14  | 01 | 1   | 1 | 2 | 300.5 |... |... |
| ID-01 C1  | 01    | 1   | 1 | 3 | 320.0 |... |... |
| ID-01 C28  | 01   | 1   | 1 | 4 | 400.0 |... |... |
| ID-02 III14  | 02 | 0   | 0 | 2 | 1000.5 |... |... |
| ID-02 C1  | 02    | 0   | 0 | 3 | 1200.0 |... |... |
| ID-02 C28  | 02   | 0   | 0 | 4 | 1100.5 |... |... |
| ID-03 III14  | 03 | 1   | 4 | 2 | 600.3 |... |... |
| ID-03 C1  | 03    | 1   | 4 | 3 | 400.3 |... |... |
| ID-03 C28  | 03   | 1   | 4 | 4 | 200.0 |... |... |
| ......  | ....    | ..  | .. | ..| ... |... |... |


Column **group** represents the ID of the patient, necessary for the stratified
k-fold cross-validation. In column **Dose** the int value represents the respective
PfSPZ dose: {0 : placebo; 1 : 3.2 * 10^3 PfSPZ; 2 : 1.28 * 10^4 PfSPZ; 4 : 5.12 * 10^4 PfSPZ}
and the int value of column **TimePointOrder** represents the day of taken blood sera {2 : III14 post-immunization;
3 : C-1 pre-CHMI; 4 : C+28 post-CHMI}.


You can find the raw data of the underlying Pf-specific proteome
microarray-based antibody reactivity profile containing all 7,455 Pf-specific
antigens (features) in /Data and the raw data of the pre-selected
Pf-specific cell surface proteins (m = 1,199) in ./Data.
Both files contain the raw data and has to be baseline and log2-fold normalized
using the Datapreprocessing.py script in ./DataPreprocessing.

### Preprocessing of the Data
Datapreprocessing.py has to be executed in the ./DataPreprocessing folder.
```python
python Datapreprocessing.py
```
Output:
```
the preprocessed data is now saved in ./Data as:

preprocessed_whole_data.csv and preprocessed_selective_data.csv
```

## Train and evaluate the Multitask-SVM prediction model
### Arguments
<span style = "color: orange">Parser_multitask_SVM</span>( --infile = <span style = "color: lightblue">path_to_proteome_array</span>, --kernel_parameter = <span style = "color: lightblue">path_to_kernel_parameter</span>, --output_file_name = <span style = "color: lightblue">name_of_output_file</span>, --k_fold = <span style = "color: lightblue">number_of_folds</span>, --partitions = <span style = "color: lightblue">partition</span>, --iterations = <span style = "color: lightblue">number_of_iteration</span>, --AUC_threshold = <span style = "color: lightblue">threshold_AUC_value</span>,
  --kernel_for_dosage = <span style = "color: lightblue">kernel_parameter_for_dosage</span>, --kernel_for_ab_signal = <span style = "color: lightblue">kernel_parameter_for_ab_signal</span>)

| Name    | Type    | Description                             | Default   |
| -----   | -----   | ------------                           | ---  |
| --infile  | str     | path to pre-processed proteome matrix as n x m matrix (where n = samples in rows, m = features as columns) concatenated with auxallery information: the 'group' of each patient, the 'protection state', the 'dosage' and the 'time point',as .csv file (comma-delimited file) |
|--kernel_parameter | .csv file | .csv file with kernel parameter ranges  |
|--output_file_name | str | name of output file name, self-defined| Result_multitask_SVM_|
|--k_fold| int | number of k folds for stratified cross-validation| 5|
| --partition  | float | number of partitions for train test split| 0.3
| --iterations   | int   | number of iterations of prediction on unseen data | 10
| --kernel_for_dosage  | str   | defines the kernel function for the vector of PfSPZ dose                                        in the kernel combination evaluation. <span style = "color: lightblue">rbf_dosage</span> for rbf kernel for the PfSPZ dose and <span style = "color: lightblue">poly_dosage</span> the polynomial kernel. | poly_dosage
| --kernel_for_ab_signal | str | defines the kernel function for the antibody signal intensity in the kernel combination evaluation. <span style = "color: lightblue">rbf_ab_signal</span> for rbf kernel for the antibody signal intensity and <span style = "color: lightblue">poly_ab_signal</span> the polynomial kernel. | rbf_ab_signal


Example: to run the multitask-SVM method the preprocessed proteome data <span style = "color: lightblue">preprocessed_whole_data.csv</span> and <span style = "color: lightblue">preprocessed_selective_data.csv</span> in /Data after applying the Datapreprocessing.py script can be used.

Multitask-SVM on preprocessed proteome data:

```python
 python Parser_multitask_SVM.py --infile /Users/.../preprocessed_whole_data.csv
  --kernel_parameter /Users/../kernel_parameter.csv Results_multitask_SVM
  --output_file_name Result_multitask_SVM_ --k_fold 5 --partition 0.3
  --iterations 10 --kernel_for_dosage poly_dosage
  --kernel_abSignals poly_ab_signal
```

Output:
```
  The multitask-SVM approach is initialized with the following parameters:
  number of k-folds        =  5
  number of partition      = 0.3
  number of iteration      = 10
  kernel function for antibody signal intensities = poly_kernel
  kernel function for dose = poly_kernel
  name of output file = Result_multitask_SVM_

  loaded input file:
      Patient       group    Protection  Dose        ...           
0    T2-002 C -1      2           1       1          ...                  
1    T2-002 C 28      2           1       1          ...                   
2  T2-002 III 14      2           1       1          ...                   
3    T2-005 C -1      5           0       0          ...                               
4    T2-005 C 28      5           0       0          ...
[5 rows x 7460 columns]

  Evaluation of prediction performance starts
  Computation of multitask-SVM performance is running
  Prediction performance evaluation stopped after ~ 500 seconds


  results saved in: Result_multitask_SVM_at_t2_poly_kernel_ABsignals_poly_kernel_dosage.csv
  results saved in: Result_multitask_SVM_at_t3_poly_kernel_ABsignals_poly_kernel_dosage.csv
  results saved in: Result_multitask_SVM_at_t3_poly_kernel_ABsignals_poly_kernel_dosage.csv

```

## Apply the feature selection method
### Arguments
<span style = "color: orange">Parser_feature_selection</span>( infile = <span style = "color: lightblue">path_to_proteome_array</span>, results_of_multitask-SVM_approach = <span style = "color: lightblue">path_to_the_evaluated_results_of_the_multitask-SVM_approach</span>, up = <span style = "color: lightblue">value_of_upper_quantile</span>, lq = <span style = "color: lightblue">value_of_lower_quantile</span>)

| Name    | Type    | Description                             | Default   |
| -----   | -----   | ------------                           | ---  |
| --infile  | path     | path to pre-processed proteome matrix as n x m matrix (where n = samples in rows, m = features as columns) concatenated with auxiliary information: the 'group' of each patient, the 'protection state', the 'dosage' and the 'time point',as .csv file (comma-delimited file) |
| --results_of_multitask-SVM_approach | path | path to the evaluated results from the multitask-SVM approach |
| --up | int | percentage for the upper quantile | 75 |
| --lq | int | percentage for the lower quantile | 25 |

Example: example data to run the feature selection method can be found here

ESPY measurement on proteome array data:

```python
  python Parser_feature_selection.py --infile /Users/../pre_processed_proteome_data.csv
  --results_of_multitask-SVM_approach /Users/../
  --up 75 --lq 25
```
Output:
```
The feature selection approach is initialized with the following parameters:
value of upper quantile      =  75
value of lower quantile      =  25

input file:  preprocessed_malaria_minusBS_data.csv
name of loaded result file from multitask-SVM evaluation: Results_multitask_classification_wholeChip_minusBS_atC-1_ABsignals_rbf_DOSEpoly_10_iterations_paramter_validation


ESPY value measurement started on proteome array data:

selected time point to start feature analysis with respect to evaluated prediction performance is  3
selected kernel for AB signals is rbf kernel with a value of:  1e-06
selected kernel for dose is polynomial kernel with a value of:  5.0
Dimension of proteome data at time point 3 is  (40, 7460)
Length of matrix of feature combination at time point 3 for importance eval:  14911
Number of columns of evaluated distance matrix
7455
end computation
end of computation after:  xxx seconds
results are saved in: Result_of_feature_distance_at_timePoint_3.csv
```
ESPY measurement on simulated data:

```python
  python Parser_feature_selection.py --infile /Users/../simulated_data.csv
```
Output:
```
The feature selection approach is initialized with the following parameters:
value of upper quantile      =  75
value of lower quantile      =  25


input file:  simulated_data.csv
ESPY value measurement started on simulated data:

The best parameters are {'C': 100.0, 'gamma': 0.0001} with a mean AUC score of 0.81
Accuracy score on unseen data:0.82
AUC score on unseen data:0.8193456614509246


results are saved in: Evaluated_features_on_simulated_data.csv

```

## Apply the normal normal distribution fitting model
### Arguments
<span style = "color: orange">Parser_normal_distribution_fitting</span>( path_distances = <span style = "color: lightblue">path_to_the_results_of_feature_evaluation</span>, output_file_name = <span style = "color: lightblue">name_of_output_file</span>)

| Name    | Type    | Description                             | Default   |
| -----   | -----   | ------------                           | ---  |
| path_distances | str | path to the results of the feature evaluation | |
| output_file_name | str | name of output file name, self-defined | "Evaluated_informative_features" |

Example: example data to run the normal distribution fitting method can be found here

```
python Normal_Distribution_Fitting.py
```
Output:
```
```


# Environment activation

```
 conda env create -f environment.yml
 conda activate -n Vaccine_efficacy_prediction
```

# Help

You can run

$ python Parser_multitask_SVM.py --help

to get the list of the positional and conditional arguments to run the
prediction model.

$ python Parser_feature_selection.py --help

gives you the list of positional and conditional arguments to run the
feature selection approach.

$ $ python Parser_normal_distribution_fitting.py --help

gives you the ist of positional and conditional arguments to evaluate the
significant features fitting to a normal distribution.
