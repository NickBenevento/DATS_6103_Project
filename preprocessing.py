#%%
# import packages
import pandas as pd
import dm6103 as dm
import numpy as np
import matplotlib.pyplot as plt

#%%
# read in datafile and check variables 
df = pd.read_csv("heart_2020_cleaned.csv")
dm.dfChk(df)

#%%
# change age variable to continuous 

df["AgeCategory"].value_counts()

#65-69          34151
#60-64          33686
#70-74          31065
#55-59          29757
#50-54          25382
#80 or older    24153
#45-49          21791
#75-79          21482
#18-24          21064
#40-44          21006
#35-39          20550
#30-34          18753
#25-29          16955

# define function

def cleanDfAge(row):
  thisage = row["AgeCategory"]

  thisage = thisage.strip()
  if thisage == "18-24": 
      thisage = min(round(18 + 6*np.random.random()), 24)
  if thisage == "25-29": 
      thisage = min(round(25 + 4*np.random.random()), 29)
  if thisage == "30-34": 
      thisage = min(round(30 + 4*np.random.random()), 34)
  if thisage == "35-39": 
      thisage = min(round(35 + 4*np.random.random()), 39)
  if thisage == "40-44": 
      thisage = min(round(40 + 4*np.random.random()), 44)
  if thisage == "45-49": 
      thisage = min(round(45 + 4*np.random.random()), 49)
  if thisage == "50-54": 
      thisage = min(round(50 + 4*np.random.random()), 54)
  if thisage == "55-59": 
      thisage = min(round(55 + 4*np.random.random()), 59)
  if thisage == "60-64": 
      thisage = min(round(60 + 4*np.random.random()), 64)
  if thisage == "65-69": 
      thisage = min(round(65 + 4*np.random.random()), 69)
  if thisage == "70-74": 
      thisage = min(round(70 + 4*np.random.random()), 74)
  if thisage == "75-79": 
      thisage = min(round(75 + 4*np.random.random()), 79)
  if thisage == "80 or older": 
      thisage = min(round(80 + 3*np.random.chisquare(2)), 100)
  
  return thisage 
# end function cleanDfAge
print("\nReady to continue.")

df["AgeCategory"] = df.apply(cleanDfAge,axis=1)
print(df.dtypes)
print("\nReady to continue.")

# %%
# export clean dataset 
df.to_csv('heart_2020_new.csv')  
print("\nReady to continue.")

# %%
