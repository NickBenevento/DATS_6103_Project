# %%
import numpy as np
import pandas as pd
import dm6103 as dm
import statsmodels.api as sm  # Importing statsmodels
import  as


df = pd.read_csv("heart_2020_balanced.csv")
dm.dfChk(df)