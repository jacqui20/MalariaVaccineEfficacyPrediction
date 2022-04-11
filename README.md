# multitask SVM
This repository contains the source code for the adapted multitask SVM approach,
a method for classifying and interpreting the immunisation status of PfSPZ-CVac
vaccinated volunteers based on antibody profiles recovered from protein arrays.

## Overview of the multitask SVM approach
The multitask SVM approach is set up into two main parts. The first part is the
assessment of performance measurement in comparison to state-of-the-art methods.
In the second part the ESPY method is used to quantify informative features from
the non-linear multitask SVM model in comparison to the state-of-the art methods.
All executable code can be found in the
[./bin](https://github.com/jacqui20/MalariaVaccineEfficacyPrediction/tree/main/bin) folder.

## Requirements

This project requires the malaria_environment.yml environment. Please install
and activate the malaria_environment.yml environment before executing any code. <br>
All code can be executed via the terminal from the ./bin folder.

```
conda env create -f malaria_environment.yml
conda activate malaria_environment
```

## Structure of proteome data
The following data table gives an overview of the data structure of
the proteome raw data: <br>
(file format: .csv file)

| Patient     | group | Protection | Dose | TimePointOrder | feature 1 | ... | feature n |
|-------------|-------|------------|------|----------------|-----------|-----|-----------|
| ID-01 III14 | 01    | 1          | 1    | 2              | 300.5     | ... | ...       |
| ID-01 C1    | 01    | 1          | 1    | 3              | 320.0     | ... | ...       |
| ID-01 C28   | 01    | 1          | 1    | 4              | 400.0     | ... | ...       |
| ID-02 III14 | 02    | 0          | 0    | 2              | 1000.5    | ... | ...       |
| ID-02 C1    | 02    | 0          | 0    | 3              | 1200.0    | ... | ...       |
| ID-02 C28   | 02    | 0          | 0    | 4              | 1100.5    | ... | ...       |
| ID-03 III14 | 03    | 1          | 4    | 2              | 600.3     | ... | ...       |
| ID-03 C1    | 03    | 1          | 4    | 3              | 400.3     | ... | ...       |
| ID-03 C28   | 03    | 1          | 4    | 4              | 200.0     | ... | ...       |
| ......      | ....  | ..         | ..   | ..             | ...       | ... | ...       |


Column **group** represents the ID of the patient, necessary for the 10-times nested stratified
5-fold cross-validation. In column **Dose** the int value represents the respective
PfSPZ dose: {0 : placebo; 1 : 3.2 * 10^3 PfSPZ; 2 : 1.28 * 10^4 PfSPZ; 4 : 5.12 * 10^4 PfSPZ}
and the int value of column **TimePointOrder** represents the day of taken blood sera {2 : III14 post-immunization;
3 : C-1 pre-CHMI; 4 : C+28 post-CHMI}.


You can find the raw data of the underlying Pf-specific proteome
microarray-based antibody reactivity profile containing all 7,455 Pf-specific
antigens (features) and the raw data of the pre-selected
Pf-specific cell surface proteins (m = 1,194) in
[./data/proteome_data](https://github.com/jacqui20/MalariaVaccineEfficacyPrediction/tree/main/data/proteome_data).
Both files contain the raw data and has to be baseline and log2-fold normalized
using the Datapreprocessing.py script.

### Preprocessing of the Data
Datapreprocessing.py has to be executed in the ./bin folder.
```python
python Datapreprocessing.py
```
Output:
```
the preprocessed data is now saved in ./data/proteome_data as:

preprocessed_whole_data.csv and preprocessed_selective_data.csv
```

### Prediction performance assesment of the multitask SVM in comparison to state-of-the-art methods
Here we give a short introduction how to run the 10-times repeated nested stratisfied 5-fold cross-validation
for the multitask SVM and the two state-of-the-art-methods, namely regularized logistic regression (RLR) and
single-task-SVM. The Random Forest (RF) approach from Veletta and Recker et. al can be found
[here](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005812). However, we <span style="color:orange;">strongly</span> advise
not to run the prediction performance evaluation on a simple machine because it is computationally intensive.
### Arguments

```
# execute multitask SVM
./run_rncv_multitask_r.sh

# execute RLR
./run_rncv_RLR.sh

#execute single-task SVM
./run_rncv_SVM.sh
```

## Apply ESPY
Here we show how to apply the ESPY method to the simulated data and the proteome data sets.

### On simulated data
To run ESPY on simulated data you have to specify the following parameters:

| Name         | Type | Description                                                                               |
|--------------|------|-------------------------------------------------------------------------------------------|
| --data-dir   | DIR  | Path to the directory where the simulated data is located.                                |     
| --out-dir    | DIR  | Path to the directory were the results shall be saved.                                    |     
| --identifier | str  | String to identify the dataset. <br/>Must be one of 'whole', 'selective', or 'simulated'. |    



```python
  python ESPY.py --data-dir /Users/.../MalariaVaccineEfficacyPrediction/results/SVM/simulated/ESPY
--out-dir /Users/.../MalariaVaccineEfficacyPrediction/results/SVM/simulated/ESPY  
--identifier "simulated"
```
Output:
```
The feature selection approach is initialized with the following parameters:
value of upper quantile      =  75
value of lower quantile      =  25


input file:  simulated_data.csv
ESPY value measurement started on simulated data:

The best parameters are {'C': 10.0, 'gamma': 0.001} with a mean AUC score of 0.90
AUC score on unseen data: 0.879800853485064


results are saved in: /.../MalariaVaccineEfficacyPrediction/results/simulated_data/Evaluated_features_on_simulated_data.csv

```

### On proteome data
To run ESPY on the proteome data sets you can easily run the shell script runESPY_proteome.sh via
```
./runESPY_proteome.sh
```
and the output files for the whole and the selective proteome array per
time point are generated automatically. The output files are stored
[here](https://github.com/jacqui20/MalariaVaccineEfficacyPrediction/tree/main/results/multitaskSVM) in the
pre-defined folders for the whole and selective data. <br>


## Apply feature evaluation of RLR
To run the feature evaluation of the RLR method you can easily run the shell script run_featureEvalRLR.sh via
```
./run_featureEvalRLR.sh
```
and the output files for the whole and selective proteome array data are automatically generated per time point.
The output files are stored [here](https://github.com/jacqui20/MalariaVaccineEfficacyPrediction/tree/main/results/RLR)
in the pre-defined folders for the whole and selective data.
## Apply SHAP framework on simulated data
To run the [SHAP](https://github.com/slundberg/shap) (SHapley Additive exPlanations) framework from Lundberg et al.
execute SHAP_evaluation.py

```python
python SHAP_evaluation.py
```

Output:
```
The best parameters are {'C': 10.0, 'gamma': 0.001} with a mean AUC score of 0.90
AUC score on unseen data: 0.879800853485064
Evaluation of informative features based on SHAP values has started at 10.04.2022_19-21-49.

Evaluation terminated and results are saved in ./results
```
