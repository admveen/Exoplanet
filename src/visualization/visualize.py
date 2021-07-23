import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
#import customized Kepler Object of Interest class
import sys
sys.path.append("..\\src\\")
from KOIclass import KOIObject


# loads feature_df_path
feature_df_path = "..\\data\\interim\\feat_df_tot.csv"
df_exo = pd.read_csv(feature_df_path, index_col = ['KIC_ID', 'TCE_num'])

# generate LC plot
secondary_example1 = KOIObject(1026032, 1).total_initialize()

fig_secondary_LC = secondary_example1.plot_LC()
