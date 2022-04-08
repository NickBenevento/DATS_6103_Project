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


#%%
sns.set(style="whitegrid")

# stat test prep
import scipy.stats as stats
df_HD = df[ df['HeartDisease'] == 'Yes']
df_NO = df[ df['HeartDisease'] == 'No']

#%%
# Histograms

## BMI

list = ('BMI', 'MentalHealth', 'PhysicalHealth', 'AgeCategory', 'SleepTime')

for var in list:
    ### all
    sns.distplot(x=df[var], kde = False).set(title = var + " -All")
    plt.show()

    ### HD = Y
    sns.distplot(x=df.loc[ df['HeartDisease'] == "Yes", var], kde = False).set(title = var + " -HD")
    plt.show()

    ### HD = N
    sns.distplot(x=df.loc[ df['HeartDisease'] == "No", var], kde = False).set(title = var + " -No HD")
    plt.show()

#%%
# Boxplots

for var in list:
    sns.boxplot(x='HeartDisease',y=var, data=df).set(title = var)
    plt.show()

#%%
# Barcharts 

## HeartDisease and Race

# create crosstab
ct=pd.crosstab(columns=df['Race'],index=df['HeartDisease'])
#print(ct)

# define vars to make proportions
dfwhite = df[ df['Race'] == "White" ]
white = dfwhite.count()[0]

dfaian = df[ df['Race'] == "American Indian/Alaskan Native" ]
aian = dfaian.count()[0]

dfasian = df[ df['Race'] == "Asian" ]
asian = dfasian.count()[0]

dfblack = df[ df['Race'] == "Black" ]
black = dfblack.count()[0]

dfhis = df[ df['Race'] == "Hispanic" ]
his = dfhis.count()[0]

dfoth = df[ df['Race'] == "Other" ]
oth = dfoth.count()[0]

# now stack and reset

stacked = ct.stack().reset_index().rename(columns={0:'value'})

# create proportion column
stacked['valueProp'] = 0
stacked.loc[ stacked['Race'] == "White", 'valueProp'] = stacked['value']/white
stacked.loc[ stacked['Race'] == "American Indian/Alaskan Native", 'valueProp'] = stacked['value']/aian
stacked.loc[ stacked['Race'] == "Asian", 'valueProp'] = stacked['value']/asian
stacked.loc[ stacked['Race'] == "Black", 'valueProp'] = stacked['value']/black
stacked.loc[ stacked['Race'] == "Hispanic", 'valueProp'] = stacked['value']/his
stacked.loc[ stacked['Race'] == "Other", 'valueProp'] = stacked['value']/oth

# plot grouped bar chart
p = sns.barplot(x=stacked.Race, y=stacked.valueProp, hue=stacked.HeartDisease)
sns.move_legend(p, bbox_to_anchor=(1, 1.02), loc='upper left')
p.set(ylabel = "Share of Total")
plt.show()

# other bar charts

list1 = ('Sex', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer')
list2 = ('GenHealth', 'Diabetic')

# binary
for var in list1:
    # create crosstab
    ct=pd.crosstab(columns=df[var],index=df['HeartDisease'])
    # now stack and reset
    stacked = ct.stack().reset_index().rename(columns={0:'value'})
    # plot grouped bar chart
    p = sns.barplot(x=stacked['HeartDisease'], y=stacked['value'], hue=stacked[var])
    sns.move_legend(p, bbox_to_anchor=(1, 1.02), loc='upper left')
    p.set(ylabel = "Count")
    plt.show()

# multi-level
for var in list2:
    # create crosstab
    ct=pd.crosstab(columns=df['HeartDisease'],index=df[var])
    # now stack and reset
    stacked = ct.stack().reset_index().rename(columns={0:'value'})
    # plot grouped bar chart
    p = sns.barplot(x=stacked[var], y=stacked['value'], hue=stacked['HeartDisease'])
    sns.move_legend(p, bbox_to_anchor=(1, 1.02), loc='upper left')
    p.set(ylabel = "Count")
    plt.show()


# %%
