#%%
# import packages
import pandas as pd
import dm6103 as dm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# read in datafile and check variables 
df = pd.read_csv("heart_2020_new.csv")
dm.dfChk(df)


# %%
sns.set(style="whitegrid")

# Boxplots

sns.boxplot(x='HeartDisease',y='BMI', data=df)
plt.show()

sns.boxplot(x='HeartDisease',y='MentalHealth', data=df)
plt.show()

sns.boxplot(x='HeartDisease',y='PhysicalHealth', data=df)
plt.show()

sns.boxplot(x='HeartDisease',y='AgeCategory', data=df)
plt.show()

sns.boxplot(x='HeartDisease',y='SleepTime', data=df)
plt.show()

#%%
# Barcharts 

## HeartDisease and Sex

# create crosstab
ct=pd.crosstab(columns=df['Sex'],index=df['HeartDisease'])

# now stack and reset
stacked = ct.stack().reset_index().rename(columns={0:'value'})

# plot grouped bar chart
p = sns.barplot(x=stacked.HeartDisease, y=stacked.value, hue=stacked.Sex)
sns.move_legend(p, bbox_to_anchor=(1, 1.02), loc='upper left')
p


## HeartDisease and Race

# create crosstab
ct=pd.crosstab(columns=df['Race'],index=df['HeartDisease'])

# now stack and reset
stacked = ct.stack().reset_index().rename(columns={0:'value'})

# plot grouped bar chart
p = sns.barplot(x=stacked.Race, y=stacked.value, hue=stacked.HeartDisease)
sns.move_legend(p, bbox_to_anchor=(1, 1.02), loc='upper left')
p


# %%
