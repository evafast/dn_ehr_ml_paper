# various helper functions

# load packages
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import colors
import seaborn as sns
import datetime


import shap

import tarfile
import pickle as pkl

import xgboost as xgb

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn import metrics

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold, permutation_test_score


import os
from urllib.parse import urlparse

def permutscoring(model, merged, base_dir, output_suffix, file_name_base, save_fig = False):
    ## permutation scoring
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    score_model, perm_scores_model, pvalue_model = permutation_test_score(
        model, merged[:,1:], merged[:,0], scoring="roc_auc", cv=kfold, n_permutations=1000)

    # plot
    fig, ax = plt.subplots()
    ax.hist(perm_scores_model, bins=20, density=True)
    ax.axvline(score_model, ls="--", color="r")
    score_label = f"Score on original\ndata: {score_model:.2f}\n(p-value: {pvalue_model:.3f})"
    ax.text(0.7, 10, score_label, fontsize=12)
    ax.set_xlabel("ROC-AUC")
    _ = ax.set_ylabel("Probability")

    if save_fig:
        full_filename = f'{base_dir}{output_suffix}{file_name_base}_permutation.pdf'
        plt.savefig(full_filename, bbox_inches = "tight")
    
    # show the plot
    plt.show()
    plt.close()


def loglosscurve(model, validation, base_dir, output_suffix, file_name_base, save_fig = False):
    # evaluate performance
    yhat = model.predict(validation[:,1:])
    score = accuracy_score(validation[:,0], yhat)
    print('Accuracy: %.3f' % score)
    # retrieve performance metrics
    results = model.evals_result()
    # plot learning curves
    plt.plot(results['validation_0']['logloss'], label='train')
    plt.plot(results['validation_1']['logloss'], label='test')
    plt.ylabel('log loss')
    plt.xlabel('estimators (# of boosting rounds)')
    # show the legend
    plt.legend()
   
    if save_fig:
        full_filename = f'{base_dir}{output_suffix}{file_name_base}_logloss.pdf'
        plt.savefig(full_filename, bbox_inches = "tight")
        
    # show the plot
    plt.show()
    plt.close()

def conf_class_report(y_pred, Y_test, filepath, filename, save_fig = False):
    """A function to plot and save a confusion matrix.  
    # Inputs:
    #    y_pred          - y values predicted from model (in binary format)
    #    y_test          - actual real y values
    #    filepath        - where to save the output
    #    filename        - name of output
   
    """
    print(classification_report(Y_test, y_pred))
    report = classification_report(Y_test, y_pred, output_dict=True)
    report = pd.DataFrame(report).transpose()
    full_filename = filepath + filename + '_report.csv'
    report.to_csv(full_filename)
    
    cm = confusion_matrix(Y_test, y_pred)
    plt.rc('font', size=18)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='gist_heat')
    #plt.rcParams.update({'font.size': 22})
    
    if save_fig:
        full_filename = f'{filepath}{filename}_cm.pdf'
        plt.savefig(full_filename, bbox_inches = "tight")  
    
    plt.show()
    plt.close()

def plot_roc_curve(preds, Y_truth, filepath, filename, save_fig = False):
    """A function to plot and save an ROC curve  
    # Inputs:
    #    y_pred          - y values predicted from model probablities
    #    y_test          - actual real y values
    #    filepath        - where to save the output
    #    filename        - name of output
   
    """
    # fpr means false-positive-rate
    # tpr means true-positive-rate
    fpr, tpr, _ = metrics.roc_curve(Y_truth, preds)

    auc_score = metrics.auc(fpr, tpr)

    plt.title('ROC Curve')
    plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc_score))

    # it's helpful to add a diagonal to indicate where chance 
    # scores lie (i.e. just flipping a coin)
    plt.plot([0,1],[0,1],'r--')

    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.legend(loc='lower right')
    plt.rc('font', size=18)
    #plt.rcParams.update({'font.size': 18})
    plt.gca().set_aspect('equal')
    
    if save_fig:
        full_filename = f'{filepath}{filename}_roc.pdf'
        plt.savefig(full_filename, bbox_inches = "tight")  
    
    plt.show()
    plt.close()
    
def testhyperparameterranges(train, validation, parameter, ranges):
    parameterdic = {'n_estimators': 200, 
              'learning_rate': 0.05,
              'colsample_bytree': 0.5, 
              'max_depth': 3, 
              'min_child_weight': 10, 
              'reg_alpha': 5}
    
    for value in ranges:
        parameterdic[parameter] = value
        model = XGBClassifier(n_estimators = parameterdic['n_estimators'], 
                      max_depth = parameterdic['max_depth'], 
                      learning_rate = parameterdic['learning_rate'],
                      reg_alpha = parameterdic['reg_alpha'],
                      min_child_weight = parameterdic['min_child_weight'],
                      colsample_bytree = parameterdic['colsample_bytree'])

        # define the datasets to evaluate each iteration
        evalset = [(train[:,1:],train[:,0]), (validation[:,1:],validation[:,0])]
        # fit the model
        model.fit(train[:,1:],train[:,0], early_stopping_rounds=50, eval_metric='logloss', eval_set=evalset, verbose=False)
        print(value)

        loglosscurve(model, validation, "", "", "")
    
    
# calculates shap values and backconverts the unit
def shap_values_convert_units(dataset, stand, explainer):
    """A function to convert the scaled numeric units into the orginal ones (for plotting)  
    # Inputs:
    #    dataset         - pandas dataframe
    #    stand           - pandas dataframe (same number and order of columns as dataset df), 1st row = mean and 2nd row = std
    #    explainer       - is the shap model that creates the shap values
    """
    
    shap_values = explainer(dataset) 
    
    #extend and make a full df
    std_a = pd.concat([stand.loc['std'].to_frame().T]*dataset.shape[0]).to_numpy()
    mean_a = pd.concat([stand.loc['mean'].to_frame().T]*dataset.shape[0]).to_numpy()
    
    #fill the categorical values with either 1s or 0s
    fill_1 = np.full((dataset.shape[0], dataset.shape[1] - stand.shape[1]), 1)
    fill_0 = np.full((dataset.shape[0], dataset.shape[1] - stand.shape[1]), 0)
    
    extend_array_std = np.append(std_a, fill_1, axis=1)
    extend_array_mean = np.append(mean_a, fill_0, axis=1)
    step1 = np.multiply(shap_values.data, extend_array_std)
    step2 = step1 + extend_array_mean
    shap_values.data = step2
    return shap_values

def waterfall_enrich_plot(shap_values, name_output):
    """A function to plot and save waterfall plots for all individuals.  
    # Inputs:
    #    shap_values          - shap value objects that contain all individuals
    #    name_output          - name of output file
   
    """
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(name_output)

    for instance in range(len(shap_values)):

        fig=plt.gcf()
        shap.plots.waterfall(shap_values[instance])
        fig.savefig(pp, format='pdf', bbox_inches = "tight")


    # close the multipage pdf
    pp.close()
    

class OutlierRemover(BaseEstimator,TransformerMixin):
    """A class that removes/trimms outliers, was also used for ML in sagemaker, makes sure that the ouliers don't skew predictions too much"""
    
    def __init__(self,factor=1.5):
        self.factor = factor

    def outlier_detector(self,X,y=None):
        X = pd.Series(X).copy()
        q1 = X.quantile(0.1)
        q3 = X.quantile(0.9)
        iqr = q3 - q1
        self.lower_bound.append(q1 - (self.factor * iqr))
        self.upper_bound.append(q3 + (self.factor * iqr))

    def fit(self,X,y=None):
        self.lower_bound = []
        self.upper_bound = []
        X.apply(self.outlier_detector)
        return self

    def transform(self,X,y=None):
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            x = X.iloc[:, i].copy()
            x.loc[x < self.lower_bound[i]] = self.lower_bound[i]
            x.loc[x > self.upper_bound[i]] = self.upper_bound[i]
            X.iloc[:, i] = x
        return X

outlier_remover = OutlierRemover()

def standardize_test_ds(dataset, stand):
    """A function to standardize the test set numeric values based on training dataset mean and std 
    # Inputs:
    #    dataset         - pandas dataframe
    #    stand           - pandas dataframe (same number and order of columns as dataset df), 1st row = mean and 2nd row = std
    """
    
    # sort to make sure that columns have same order
    stand = stand.reindex(sorted(stand.columns), axis=1)
    dataset = dataset.reindex(sorted(dataset.columns), axis=1)
    
    std_a = pd.concat([stand.loc['std'].to_frame().T]*dataset.shape[0]).to_numpy()
    mean_a = pd.concat([stand.loc['mean'].to_frame().T]*dataset.shape[0]).to_numpy()
    
    #fill the categorical values with either 1s or 0s
    fill_1 = np.full((dataset.shape[0], dataset.shape[1] - stand.shape[1]), 1)
    fill_0 = np.full((dataset.shape[0], dataset.shape[1] - stand.shape[1]), 0)
    
    extend_array_std = np.append(std_a, fill_1, axis=1)
    extend_array_mean = np.append(mean_a, fill_0, axis=1)
    step1 = dataset - extend_array_mean
    step2 = step1/extend_array_std
    return step2

#select test subject ids

def preprocess_test_ds(executionid, base_dir, suffix):
    """Preprocess the test set to make it ready for inferring the predicted outcomes from ML model. This was a shockingly tedious proceedure. Mainly recreated the preprocessing for the training dataset. Left out outlier trimming and removed the pipeline part. Did the standard scaling based on the mean from the training dataset. Also no imputation of numeric values. 
    # Inputs:
    #    executionid (str)         - execution ID from reprocessed model file
    #    base_dir (str)    - base directory where files are saved
    #    local_base_directory(str)  - based directory where files are saved
    """
    # read the model files
    df_model = pd.read_csv(f'{base_dir}raw_data/reprocessed_model_log.csv')

    #read the parquet file original before filtering
    file_name_base = str(df_model[df_model['ExecutionID'] == executionid][['aggregation_time','gap']].values[0][0]) + '_years_' + str(df_model[df_model['ExecutionID'] == executionid][['aggregation_time','gap']].values[0][1])

    print(file_name_base)
    input_df_path = f'{base_dir}raw_data/preprocessed_data/ef_agg_before_biopsy_' + file_name_base + '_gap.parquet'
    df = pd.read_parquet(input_df_path, engine='pyarrow')

    #read the test variables
    test_df = pd.read_csv(f'{base_dir}{suffix}/ML_target_variables_test.csv')

    #read the colnames
    colnames = pd.read_csv(f'{base_dir}{suffix}{file_name_base}/column_header.csv', header = None)
    colnames = colnames[0].tolist()

    #reads dataset for standardization
    stand = pd.read_csv(f'{base_dir}{suffix}{file_name_base}/standardization.csv', index_col = 0)

    #select test subject ids
    df = pd.merge(df, test_df, left_on = 'gfb_subject_id', right_on = 'gfb_subject_id', how = 'inner')

    #select relevant columns
    clean_colnames = list(set(colnames) - set(df.columns.tolist()))
    clean_colnames = pd.Series(clean_colnames).str.replace('_TRUE|_>10 years|_>5 years|_Not Reported|_Yes|_White/Caucasian|_Male|_Not Hispanic or Latino',\
                                    '', regex=True)
    clean_colnames = list(set(clean_colnames)) #this gets rid of duplicates
    clean_colnames = list(clean_colnames) + list(set(colnames) & set(df.columns.tolist()))

    df_clean = df[clean_colnames]

    mask = df_clean.applymap(type) == float
    d = {True: 'TRUE'}
    df_clean = df_clean.where(mask, df_clean.fillna('Not Reported'))
    df_clean = df_clean.where(mask, df_clean.replace(d))
    df_clean = df_clean.where(mask, df_clean.astype(str))

    #drop duplicate column names - not sure how this could have happenend
    df_clean = df_clean.loc[:,~df_clean.columns.duplicated()]

    df_num = df_clean.select_dtypes(include='float')
    df_cat = df_clean.select_dtypes(include='object')
    #one hot encoder
    enc = OneHotEncoder(drop='first').fit(df_cat)
    #print(enc)
    df_cat = enc.transform(df_cat).toarray()
    #print(df_cat)
    all_cat_feat = enc.get_feature_names_out()
    #print(all_cat_feat)
    df_cat_tr = pd.DataFrame(df_cat, columns = all_cat_feat)
    df_cat_tr = df_cat_tr.reindex(sorted(df_cat_tr.columns), axis=1)

    #print(df_cat_tr)
    # standardize columns numeric dataframe
    df_num = standardize_test_ds(df_num, stand)  

    # merge categorical and numeric df
    df_transf = pd.concat([df_num, df_cat_tr.reindex(df_num.index)], axis=1)

    # replace outcome variable name
    colnames = ['is_DN_TRUE' if item == 'is_DN' else item for item in colnames]

    # select colname (should also induce order)
    df_transf = df_transf[colnames]

    #select outcome variable
    Y_test = df_transf[['is_DN_TRUE']]
    X_test = df_transf.drop(['is_DN_TRUE'], axis=1)
    
    model = xgb.XGBClassifier()
    model.load_model(f'{base_dir}{suffix}{file_name_base}/model.json')

    X_testD = X_test.values
    #print(X_test.values)
    y_pred_bin = model.predict(X_testD) > 0.5

    y_pred = model.predict_proba(X_testD)
    
    return y_pred_bin, y_pred, Y_test, model, X_test, stand, file_name_base


def replace_proceedure_code(path_to_csv, column_list):
    """Replace the proceedure codes with actual names and replace redundant names with unique names manually
    # Inputs:
    #    path_to_csv (str)         - path to the proceedure mapping file, stored locally
    #    column_list (list)        - list of columns that should be mapped
    """
    # read the proceedure dictionary
    proc_df = pd.read_csv(path_to_csv)
    proc_dic = proc_df.set_index('procedure_code')['short_description'].to_dict()
    proc_dic.update({"791.0": "Proteinuria", "250.00": "Type 2 diabetes mellitus without complications", "99211": "Office/outpatient visit-99211" , "99212": "Office/outpatient visit-99212", 
                    "99213": "Office/outpatient visit-99213", "99214": "Office/outpatient visit-99214", "99215": "Office/outpatient visit-99215"})
    
    # removed '_TRUE' from colnames and adding proceedure names instead of codes
    column_list = [i.rsplit('_TRUE', 1)[0] for i in column_list]
    column_list = [proc_dic.get(item,item)  for item in column_list]
    return column_list
