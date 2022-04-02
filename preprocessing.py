#%%
# import packages
import pandas as pd
import dm6103 as dm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# read in datafile and check variables 
df = pd.read_csv("heart_2020_cleaned.csv")
#dm.dfChk(df)

#%%
# change age variable to continuous 

#df["AgeCategory"].value_counts()

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
      thisage = np.random.uniform(low = 18, high = 24)
  if thisage == "25-29": 
      thisage = np.random.uniform(low = 25, high = 29)
  if thisage == "30-34":
      thisage = np.random.uniform(low = 30, high = 34)
  if thisage == "35-39": 
      thisage = np.random.uniform(low = 35, high = 39)
  if thisage == "40-44": 
      thisage = np.random.uniform(low = 40, high = 44)
  if thisage == "45-49": 
      thisage = np.random.uniform(low = 45, high = 49)
  if thisage == "50-54": 
      thisage = np.random.uniform(low = 50, high = 54)
  if thisage == "55-59": 
      thisage = np.random.uniform(low = 55, high = 59)
  if thisage == "60-64": 
      thisage = np.random.uniform(low = 60, high = 64)
  if thisage == "65-69": 
      thisage = np.random.uniform(low = 65, high = 69)
  if thisage == "70-74": 
      thisage = np.random.uniform(low = 70, high = 74)
  if thisage == "75-79": 
      thisage = np.random.uniform(low = 75, high = 79)
  if thisage == "80 or older": 
      thisage = min(80 + 3*np.random.chisquare(2), 99)
  
  return thisage 
# end function cleanDfAge
print("\nReady to continue.")

df["AgeCategory"] = df.apply(cleanDfAge,axis=1)
print(df.dtypes)
print("\nReady to continue.")

# %%
# checking age variable distribution
sns.set(style="whitegrid")
sns.distplot(x=df['AgeCategory'], kde = False)
plt.show()

# %%
# export clean dataset 
df.to_csv('heart_2020_new.csv')  
print("\nReady to continue.")

# %%
