# PACKAGE LOADING 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

os.chdir("..\\..\\src\\")
sys.path.append(os.getcwd())
from KOIclass import KOIObject


print(sys.path)
print(os.getcwd())

image_folder = "..\\reports\\figures\\"

# LIGHT CURVE PLOTTING
#-------------------------------------------------------------
# detrended light curve to figures

#secondary eclipse (class 2 example 1)
secondary_example1 = KOIObject(1026032, 1).total_initialize()
secondary_example1.plot_LC(mode = "save")
plt.savefig(image_folder + "lc_detrend_secondaryFP_ex1.png")
plt.clf()

# secondary eclipse detrended light curve (class 2 example 2)
secondary_example2 = KOIObject(3127817, 1).total_initialize()
secondary_example2.plot_LC(mode = "save")
plt.savefig(image_folder + "lc_detrend_secondaryFP_ex2.png")
plt.clf()

#  plot phase folded averaged LC with weak secondary 
secondary_example1.plot_phasefolded(mode = 'save')
plt.savefig(image_folder + "phasefolded_secondaryFP_ex1.png")
plt.clf()

#  plot phase folded averaged LC with weak secondary 
secondary_example1.plot_secondary(mode = 'save')
plt.savefig(image_folder + "secondarypeakonly_secondaryFP_ex1.png")
plt.clf()


secondary_example2.plot_phasefolded(mode = "save")
plt.savefig(image_folder + "phasefolded_secondaryFP_ex2.png")
plt.clf()

secondary_example2.plot_oddandeven_transit(mode = "save")
plt.savefig(image_folder + "evenoddstagger_secondaryFP_ex2.png")
plt.clf()

#non-transiting phenomena
ntp_example1 = KOIObject(3324644, 1).total_initialize()
ntp_example1.plot_phasefolded(mode = "save")
plt.savefig(image_folder + "phasefolded_ntp_ex1.png")
plt.clf()

ntp_example2 = KOIObject(4142768, 1).total_initialize()
ntp_example2.plot_phasefolded(mode = "save")
plt.savefig(image_folder + "phasefolded_ntp_ex2.png")
plt.clf()

ntp_example3 = KOIObject(4729553, 1).total_initialize()
ntp_example3.plot_phasefolded(mode = "save")
plt.savefig(image_folder + "phasefolded_ntp_ex3.png")
plt.clf()

ntp_example4 = KOIObject(3344427, 1).total_initialize()
ntp_example4.plot_phasefolded(mode = "save")
plt.savefig(image_folder + "phasefolded_ntp_ex4.png")
plt.clf()

# confirmed planet example
cp_example = KOIObject(2987027,1).total_initialize()
cp_example.plot_phasefolded(mode = "save")
plt.savefig(image_folder + "phasefolded_CP_ex1.png")
plt.clf()

cp_example.plot_transit_closeup(mode = "save")
plt.savefig(image_folder + "closeup_CP_ex1.png")
plt.clf()

#----------plot LC-BINNED xy-normalized curves----------------------------
feature_df_path = "..\\data\\interim\\feat_df_tot.csv"
df_exo = pd.read_csv(feature_df_path, index_col = ['KIC_ID', 'TCE_num'])
print(df_exo.head())

#-----------Secondary FP normalized ------
ax = df_exo.loc[(12643589,1),'LCBIN_0':'LCBIN_140'].T.reset_index().plot()
df_exo.loc[(1026032,1),'LCBIN_0':'LCBIN_140'].T.reset_index().plot(ax = ax)
df_exo.loc[(12644774,1),'LCBIN_0':'LCBIN_140'].T.reset_index().plot(ax = ax)
df_exo.loc[(2438490,1),'LCBIN_0':'LCBIN_140'].T.reset_index().plot(ax = ax)
df_exo.loc[(2437783,1),'LCBIN_0':'LCBIN_140'].T.reset_index().plot(ax = ax)
df_exo.loc[(2446623, 1),'LCBIN_0':'LCBIN_140'].T.reset_index().plot(ax = ax)
df_exo.loc[(2437488,1),'LCBIN_0':'LCBIN_140'].T.reset_index().plot(ax = ax)
df_exo.loc[(2438070, 1), 'LCBIN_0':'LCBIN_140'].T.reset_index().plot(ax = ax)
plt.xlabel('Bin Number')
plt.ylabel('XY-rescaled Rel. Flux')
plt.title("Secondary FP Transit Close-ups")
plt.savefig(image_folder + "rescalexy_secondaryFP.png")
plt.clf()

#-----------CP normalized ------

CP_df = df_exo.loc[(df_exo.target_label == 1) & (df_exo.Depth > 2000)].head(9)
CP_df.loc[:, 'LCBIN_0':'LCBIN_140'].T.reset_index().plot()
plt.xlabel('Bin Number')
plt.ylabel('XY-rescaled Rel. Flux')
plt.title("Confirmed Planet Transit Close-ups")
plt.savefig(image_folder + "rescalexy_CP.png")
plt.clf()
#-----------CP vs secondary FP xy normalized ------
ax2 = df_exo.loc[(1026032,1),'LCBIN_0':'LCBIN_140'].T.reset_index().plot()
df_exo.loc[(2987027,1),'LCBIN_0':'LCBIN_140'].T.reset_index().plot(ax = ax2)
plt.xlabel('Bin Number')
plt.ylabel('XY-rescaled Rel. Flux')
plt.title("Confirmed Planet Transit vs Secondary FP ")
plt.savefig(image_folder + "secondaryFPvsCP.png")
plt.legend()
plt.clf()