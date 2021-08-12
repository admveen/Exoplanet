# PACKAGE LOADING

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pickle

import re 

# this class is needed for loading the preprocessing pipeline 
import sys

import glob 

# append src directory location here.

sys.path.append("..\\..\\src\\")
from exo_preprocess import SqrtLogZeroExceptionTransformer

# necessary packages for loading our optimal models.
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier 

# evaulation metrics we'll be using
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef as mcc

#----------------------------------------------------------

# LOAD OPTIMIZED MODELS

model_path_list = glob.glob("..\\..\\models\\*.pkl")
model_list = [pickle.load(open(model_path, 'rb')) for model_path in model_path_list]

# LOAD DATA
processed_data_path = "..\\..\\data\\processed\\"

X_train = pd.read_csv(processed_data_path + 'X_train.csv', index_col=['KIC_ID', 'TCE_num'] )
y_train = pd.read_csv(processed_data_path + 'y_train.csv', index_col = ['KIC_ID', 'TCE_num'] )
X_test = pd.read_csv(processed_data_path + 'X_test.csv', index_col=['KIC_ID', 'TCE_num'] )
y_test = pd.read_csv(processed_data_path + 'y_test.csv', index_col=['KIC_ID', 'TCE_num'] )

#------------------------------------------------------------
# SCRIPT TO AUTO-GENERATE MODEL METRICS MARKDOWN
model_image_filepath = "..\\..\\reports\\presentation\\"
model_metrics_filepath = "..\\..\\reports\\model_metrics.md"
# flatten y vectors
y_train_flat = y_train.to_numpy().flatten()
y_test_flat = y_test.to_numpy().flatten()

with open(model_metrics_filepath, 'w') as f:
    f.write("## Model Metrics Report" + "\n") 
    
# run loop to fit/predict on each model and output test metrics to file

for model in model_list:
    model.fit(X_train,y_train_flat)
    y_pred = model.predict(X_test)
    class_metrics = pd.DataFrame(classification_report(y_test_flat,y_pred, output_dict = True)).T
    mkdowntable = class_metrics.loc[['1','2','3'], ['precision','recall', 'f1-score']].to_markdown()
    cfm_table = pd.DataFrame(confusion_matrix(y_test_flat, y_pred)).to_markdown()
    print(class_metrics.to_latex())
   
    with open(model_metrics_filepath, 'a+') as f:
        f.write("### " + str(model.steps[-1]) + "\n" + "#### Classification report" + "\n" + mkdowntable + "\n")
        f.write("#### Confusion Matrix" + "\n" + cfm_table + "\n" )
        f.write("#### Matthews Correlation Coefficient" +"\n" + "MCC:" + str(mcc(y_test_flat, y_pred)) + "\n")
        
    # let's generate confusion matrix images
    sns.set_context("poster")
    cfmatrix_fig = plot_confusion_matrix(model, X_test, y_test_flat)
    final_path = model_image_filepath + str(model.steps[-1][0]) + "_cfmat.png"
    plt.title(str(model.steps[-1][0]))
    plt.tight_layout()
    plt.savefig(final_path)
    plt.clf()
    
    # check if xgboost or randomforest. if so, output feature importances:
    if model.steps[-1][0] == 'exo_randforest':
        feat_cols = ['LPP_1', 'LPP_2', 'Period', 'Duration', 'EOS', 'WSS', 'max', 'min']
        feature_importance_series = pd.Series(model['exo_randforest'].feature_importances_, index = feat_cols).sort_values()
        plt.figure(figsize = (7.5,7))
        feature_importance_series.plot(kind = 'bar')
        plt.ylabel('Feature Importance')
        plt.title('Random Forest: Feature Importances')
        plt.tight_layout()
        plt.savefig(model_image_filepath + "randforest_feature_imp.png")
        plt.clf()
    elif model.steps[-1][0] == 'exo_xgb':
        feat_cols = ['LPP_1', 'LPP_2', 'Period', 'Duration', 'EOS', 'WSS', 'max', 'min']
        feature_importance_series = pd.Series(model['exo_xgb'].feature_importances_, index = feat_cols).sort_values()
        plt.figure(figsize = (7,7))
        feature_importance_series.plot(kind = 'bar')
        plt.ylabel('Feature Importance')
        plt.title('XGBoost: Feature Importances')
        plt.tight_layout()
        plt.savefig(model_image_filepath + "xgb_feature_imp.png")
        plt.clf()
        
