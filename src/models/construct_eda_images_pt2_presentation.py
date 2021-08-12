# PACKAGE LOADING 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import pickle

os.chdir("..\\..\\src\\")
sys.path.append(os.getcwd())
from KOIclass import KOIObject
from exo_preprocess import SqrtLogZeroExceptionTransformer

from sklearn.manifold import TSNE

sns.set_context("poster")
sns.set_palette('tab10')

image_folder = "..\\reports\\presentation\\"

Xtrain_path = "..\\data\\processed\\X_train.csv"
Xtest_path = "..\\data\\processed\\X_test.csv"
ytrain_path = "..\\data\\processed\\y_train.csv"
ytest_path = "..\\data\\processed\\y_test.csv"

X_exo = pd.read_csv(Xtrain_path, index_col = ['KIC_ID', 'TCE_num']).append(pd.read_csv(Xtest_path, index_col = ['KIC_ID', 'TCE_num']))
y_exo = pd.read_csv(ytrain_path, index_col = ['KIC_ID', 'TCE_num']).append(pd.read_csv(ytest_path, index_col = ['KIC_ID', 'TCE_num']))

full_untrans_data = X_exo.join(y_exo)

TSNE_model = TSNE(learning_rate = 50)
transformed = TSNE_model.fit_transform(full_untrans_data.loc[:,'LCBIN_0':'LCBIN_140'])

tsne_df = pd.DataFrame(transformed, columns = ['TSNE_1', 'TSNE_2'])
tsne_df['target_label'] = full_untrans_data['target_label'].to_numpy()
tsne_df['target_label'] = tsne_df['target_label'].astype('category')
sns.jointplot(x = 'TSNE_1', y = 'TSNE_2', hue='target_label', data = tsne_df)
plt.savefig(image_folder+"lppTSNEjointplot.png")
plt.clf()

preprocess_path = "..\\models\\preprocessing\\preprocess_pipeline.pkl"
preprocessor = pickle.load(open(preprocess_path, 'rb'))


data_transformer = preprocessor.fit(X_exo)
X_trans = data_transformer.transform(X_exo)

colnames = ['LPP_1', 'LPP_2', 'Period', 'Duration', 'even_odd_stat', 'p_secondary', 'max', 'min']
X_trans_df = pd.DataFrame(X_trans, index = X_exo.index, columns = colnames)

full_transformed_data = X_trans_df.join(y_exo)
print(full_transformed_data.head())

# LPP Jointplot
sns.jointplot(x = 'LPP_1', y = 'LPP_2', hue = 'target_label', data = full_transformed_data, palette = "tab10")
plt.savefig(image_folder+"lppjointplot.png")
plt.clf()

def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=50, cmap=cmap, **kwargs)

with sns.axes_style("whitegrid"):
    g = sns.FacetGrid( full_transformed_data, hue="target_label", col="target_label", height=5)
g.map(hexbin, "LPP_1", "LPP_2", extent=[0, 0.5, 0.4, 0.8])
plt.savefig(image_folder + 'lpp_hexbinplot.png')
plt.clf()

# p_sec
plt.figure(figsize = (7,5))
sns.histplot(x = 'p_secondary', data = full_transformed_data, hue = 'target_label',  palette = "tab10", bins = 50)
plt.xlabel('Weak Secondary Statistic')

plt.tight_layout()

plt.savefig(image_folder+'psec_hist.png')
plt.clf()

full_transformed_data['WSS'] = (full_transformed_data.p_secondary >= 0.17).astype('int').replace({0: '< 0.17', 1: '> 0.17' })
sns.catplot(x = 'target_label', data = full_transformed_data, col = 'WSS', kind = 'count', palette = 'tab10')
plt.tight_layout()
plt.savefig(image_folder+'psec_class_diff.png')
plt.clf()

# odd_even_stat
plt.figure(figsize = (7,5))
sns.histplot(x = 'even_odd_stat', data = full_transformed_data, hue = 'target_label',  palette = "tab10", bins = 50)
plt.xlabel('Even Odd Statistic')
plt.tight_layout()
plt.savefig(image_folder+'evenodd_hist.png')
plt.clf()

full_transformed_data['EOS'] = (full_transformed_data.even_odd_stat >= 0.1).astype('int').replace({0: '< 0.1', 1: '> 0.1' })
sns.catplot(x = 'target_label', data = full_transformed_data, col = 'EOS', kind = 'count', palette = "tab10")
plt.savefig(image_folder+'evenodd_class_diff.png')
plt.clf()

#transformed feature correlations
sns.heatmap(full_transformed_data[colnames].corr())
plt.tight_layout()
plt.savefig(image_folder+'final_feature_corr.png')
plt.clf()

sns.heatmap(full_transformed_data[colnames].corr())
plt.tight_layout()
plt.savefig(image_folder+'final_feature_corr.png')
plt.clf()

# EDA on other features (min, max, period, duration)
cols_to_plot = ['min', 'max', 'Period', 'Duration']

def generate_ecdf_and_hist(colname):
    plt.figure(figsize = (7,5))
    sns.ecdfplot(x = colname, hue = 'target_label', data = full_transformed_data, palette = 'tab10')
    plt.title('Class-Resolved ECDF from ' + colname)
    plt.tight_layout()
    plt.savefig(image_folder+'ecdf_' + colname +'.png')
    plt.clf()
    
    plt.figure(figsize = (7,5))
    sns.histplot(x = colname, hue = 'target_label', bins = 50, data = full_transformed_data, palette = 'tab10')
    plt.title('Class-Resolved ECDF from ' + colname)
    plt.tight_layout()
    plt.savefig(image_folder+'hist_' + colname +'.png')
    plt.clf()
    
for col in cols_to_plot:
    generate_ecdf_and_hist(col)
    