# PACKAGE LOADING

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
from copy import deepcopy

# this class is needed for loading the preprocessing pipeline 
import sys
import os
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

model_metrics_filepath = "..\\..\\reports\\model_metrics.md"
# run loop to fit/predict on each model and output test metrics to file

model = model_list[0]
y_train_flat = y_train.to_numpy().flatten()
y_test_flat = y_test.to_numpy().flatten()
model.fit(X_train,y_train_flat)
y_pred = model.predict(X_test)

class_metrics = pd.DataFrame(classification_report(y_test_flat,y_pred, output_dict = True)).T


    
mkdowntable = class_metrics.loc[['1','2','3'], ['precision','recall', 'f1-score']].to_markdown()
   
with open(model_metrics_filepath, 'a+') as f:
    f.write("##" + str(model.steps[-1]) + "\n" + mkdowntable) 
