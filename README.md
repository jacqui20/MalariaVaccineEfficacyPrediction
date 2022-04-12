# multitask SVM
This repository contains code and data for an adapted multitask SVM approach, a method for classifying and interpreting the immunization status of PfSPZ-CVac vaccinated volunteers based on antibody profiles recovered from protein arrays.

## Overview of the multitask SVM approach
The multitask SVM approach is set up into two main parts.
The first part is a performance assessment in comparison to state-of-the-art methods.
In the second part, a new ESPY (fEature diStance exPlainabilitY) method is used to quantify informative features from the non-linear multitask SVM model in comparison to state-of-the art methods.
All executable code can be found in the [./bin](https://github.com/jacqui20/MalariaVaccineEfficacyPrediction/tree/main/bin) folder.

## Requirements

The code should be run in a customized conda environment.
Please install the conda environment via
```bash
conda env create -f malaria_environment.yml
```
and activate it with
```bash
conda activate malaria_env
```
before executing any code.

## Structure of proteome data
The following data table gives an overview of the data structure of the proteome raw data: <br>
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


Column **group** represents the ID of the patient.
In column **Dose** the integer values represent the respective PfSPZ dose:
{0 : placebo; 1 : 3.2 * 10^3 PfSPZ; 2 : 1.28 * 10^4 PfSPZ; 4 : 5.12 * 10^4 PfSPZ}
The integer values of the column **TimePointOrder** represent the day of blood serum extraction:
{2 : III14 post-immunization; 3 : C-1 pre-CHMI; 4 : C+28 post-CHMI}.

You can find the raw data of the underlying Pf-specific proteome microarray-based antibody reactivity profile containing all 7,455 Pf-specific antigens (features) and the raw data of the pre-selected Pf-specific cell surface proteins (m = 1,194) in [./data/proteome_data](https://github.com/jacqui20/MalariaVaccineEfficacyPrediction/tree/main/data/proteome_data).
Both files contain the raw data that has to be baseline and log2-fold normalized using the Datapreprocessing.py script.

### Preprocessing of the Data
Datapreprocessing.py has to be executed in the ./bin folder.
```python
python Datapreprocessing.py
```
Output:
```bash
the preprocessed data is now saved in ./data/proteome_data as:

preprocessed_whole_data.csv and preprocessed_selective_data.csv
```

### Prediction performance assesment of the multitask SVM in comparison to state-of-the-art methods
Here we give a short introduction how to run the 10-times repeated nested stratified 5-fold cross-validation for the multitask SVM and two state-of-the-art-methods, namely elastic net regularized logistic regression (RLR) and single-task SVM with a RBF kernel.
However, we <span style="color:orange;">strongly</span> advise to not run the performance assessment, since it is computationally demanding and time-consuming.
The Random Forest (RF) approach from Veletta and Recker et. al can be found [here](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005812).

### Arguments
If you are entirely sure that you want to run the performance assessment (Be sure to know what you are doing!), you can execute it (in ./bin) as follows:

```bash
# run multitask SVM performance assessment
./run_rncv_multitask.sh

# run RLR performance assessment
./run_rncv_RLR.sh

# run single-task SVM performance assessment
./run_rncv_SVM.sh
```

## Apply ESPY
Here we show how to apply the ESPY method to the simulated data and the proteome data sets.

### On simulated data
To run ESPY on simulated data, you have to specify the following parameters:

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
```bash
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
To run ESPY on the proteome data sets, you can easily run the shell script runESPY_proteome.sh via
```bash
./runESPY_proteome.sh
```
and the output files for the whole and the selective proteome array per time point will be generated automatically. 
The output files are stored [here](https://github.com/jacqui20/MalariaVaccineEfficacyPrediction/tree/main/results/multitaskSVM) in pre-defined subfolders for the whole and selective data. <br>


## Apply RLR feature evaluation
To run feature evaluation based on RLR you can easily run the shell script run_featureEvalRLR.sh via
```bash
./run_featureEvalRLR.sh
```
and the output files for the whole and selective proteome array data will be automatically generated per time point.
The output files are stored [here](https://github.com/jacqui20/MalariaVaccineEfficacyPrediction/tree/main/results/RLR) in pre-defined folders for the whole and selective data.

## Apply the SHAP framework on simulated data
To apply the [SHAP](https://github.com/slundberg/shap) (SHapley Additive exPlanations) framework from Lundberg et al. on simulated data, execute SHAP_evaluation.py
```python
python SHAP_evaluation.py
```

Output:
```bash
The best parameters are {'C': 10.0, 'gamma': 0.001} with a mean AUC score of 0.90
AUC score on unseen data: 0.879800853485064
Evaluation of informative features based on SHAP values has started at 07.04.2022_10-39-31.
<IPython.core.display.HTML object>
Evaluation has terminated at 07.04.2022_18-30-51 and results are saved in ../results/SVM/simulated/SHAP as SHAP_value_simulated_data.png.
```
