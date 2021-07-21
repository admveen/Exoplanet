import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

from lpproj import LocalityPreservingProjection

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector

#transformer class to perform sqrt log transform on p_secondary and even_odd_stat

class SqrtLogZeroExceptionTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        super().__init__()
        self.trans_evenodd_max_ = None
        self.trans_psec_max_ = None

    def fit(self, X, y = None):
        cols_to_take = ['even_odd_stat', 'p_secondary'] #these are the columns for which we'll do the sqrt log transformation

        with np.errstate(divide='ignore'):
            X_trans = np.sqrt(np.abs(np.log10(X[cols_to_take])))
        # cuts out infinities and gets maxes --> will use this to impute infinities (which are zeros in the original feature set)
        maxsqrtlog = X_trans[~(X_trans == np.inf)].max()

        self.trans_evenodd_max_ = maxsqrtlog['even_odd_stat']
        self.trans_psec_max_ = maxsqrtlog['p_secondary']

        return self
    
    def transform(self, X, y = None):
        cols_to_take = ['even_odd_stat', 'p_secondary'] #these are the columns for which we'll do the sqrt log transformation

        # this will issue some divide by zero warnings. im going to suppress this warning.
        with np.errstate(divide='ignore'):
            X_trans = np.sqrt(np.abs(np.log10(X[cols_to_take])))


        X_toreturn = deepcopy(X)
        X_toreturn['even_odd_stat'] = X_trans['even_odd_stat'].replace(np.inf, self.trans_evenodd_max_)
        X_toreturn['p_secondary'] = X_trans['p_secondary'].replace(np.inf, self.trans_psec_max_)

        return X_toreturn

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)