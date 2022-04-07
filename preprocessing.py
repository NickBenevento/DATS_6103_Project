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
  if thisage == "18-24": return np.random.uniform(18, 25)
  if thisage == "25-29": return np.random.uniform(25, 30)
  if thisage == "30-34": return np.random.uniform(30, 35)
  if thisage == "35-39": return np.random.uniform(35, 40)
  if thisage == "40-44": return np.random.uniform(40, 45)
  if thisage == "45-49": return np.random.uniform(45, 50)
  if thisage == "50-54": return np.random.uniform(50, 55)
  if thisage == "55-59": return np.random.uniform(55, 60)
  if thisage == "60-64": return np.random.uniform(60, 65)
  if thisage == "65-69": return np.random.uniform(65, 70)
  if thisage == "70-74": return np.random.uniform(70, 75)
  if thisage == "75-79": return np.random.uniform(75, 80)
  if thisage == "80 or older": return min(79 + 3*np.random.chisquare(2), 99)
  return np.nan

# end function cleanDfAge
print("\nReady to continue.")

df["AgeCont"] = df.apply(cleanDfAge, axis=1)
print(df.dtypes)
print("\nReady to continue.")

# %%
# check to see if dataset is as expected 
df.head()

# %%
# checking age variable distribution
sns.set(style="whitegrid")
sns.distplot(x=df['AgeCont'], kde = False)
plt.show()

# %%
# export clean dataset 
df.to_csv('heart_2020_new.csv')  
print("\nReady to continue.")

# %%
