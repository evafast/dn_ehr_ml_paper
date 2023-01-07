# dn_ehr_ml_paper
Analysis code for the diabetic nephropathy paper. This is a collaboration with Goldfinch and the Providence Healthcare system. 

Inital analysis was done within Goldfinch within Sagemaker studio. Analysis was later on recreated with open source tools (not dependent on AWS computational environment) - shown here.

**00_preprocessing_and_aggregating_ds**
**00_preprocessing_harmonized**
Inital data cleaning of clinical datasets was done within Goldifnch computational environment. Data preprocessing scripts read from Goldfinch clinical database and aggregated clinical data into tabular format (patients = rows, features = columns)

**01_preprocessing_outputvariable_sampling**
Further prepocessing of features and sampling of test dataset.

**02_preprocess_for_ML**
Feature selection and transformation in preparation for machine learning. 

**03_modelling**
XGBoost modelling and hyperparameter optimization using hyperopt.

**04a_model_output_analysis_roc_validation**
**05a_model_output_analysis_roc_test**
Plotting of ROC and confusion matrix using validation/test datasets

**04b_model_output_analysis_validation**
**05b_model_output_analysis_test**
Shap value analysis of misclassified patients.

**06_EDAoffeatures**
EDA with features that were used for machine learning, generated report using pandas profiler

**helper_functions**
some helper functions used throughout the analysis

**docker**
docker file used for analysis, image is available on dockerhub