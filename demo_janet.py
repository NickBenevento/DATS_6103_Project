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
dm.dfChk(df)
# %%
df.describe()
# %%
for col in df:
    print(col, df[col].unique())
# %%
#Check labeled values distrbuition
sns.countplot(df['HeartDisease'])
plt.show()
# unbalance data
# %%
# the three key factors we are going to focus; smoking, high blood pressure, and high cholesterol
#Smoking and heart disease
plt.figure(figsize=(10,4))
sns.countplot(data=df, x='Smoking', hue='HeartDisease')

# %%
# balance the data
#Find length of positive cases
len(df.loc[df['HeartDisease'] == 'Yes'])

#%%
#Under sampling the data so we can get a balanced dataset
df = df.sample(frac=1)

positive_df = df.loc[df['HeartDisease'] == 'Yes']
negative_df = df.loc[df['HeartDisease'] == 'No'][0:27373]

normal_distrbuted_df = pd.concat([positive_df, negative_df])

new_df = normal_distrbuted_df.sample(frac = 1, random_state = 42)

new_df.head()

print('Distribution of the Classes in the subsample dataset')
print(new_df['HeartDisease'].value_counts()/len(new_df))


#%%
sns.countplot('HeartDisease', data=new_df)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()

#%%
#Reset indices
new_df.reset_index(inplace = True,drop = True)
new_df

#%%
#BMI and Heart Disease
plt.figure(figsize=(10,4))

# Histogram
sns.histplot(data=new_df, x='BMI', hue='HeartDisease', binwidth=1, kde=True)

# Aesthetics
plt.title('BMI Distrbution')
plt.xlabel('BMI indicator')
#We notice that the most healthy people who have BMI within 20 till about 
# 28 and the more BMI this leads to increase HeartDisease, 
# and that's make sense as BMI is indictator for obesity whic is one of direct cause of HeartDisease¶
#%%
# #Smoking and heart disease
plt.figure(figsize=(10,4))
sns.countplot(data=new_df, x='Smoking', hue='HeartDisease')
#We notice that smoking increase your chance in getting Heart Diseases by about 50%
# claculation
#%%
#Alchol and Heart Disease
plt.figure(figsize=(10,4))
sns.countplot(data=new_df, x='AlcoholDrinking', hue='HeartDisease')
#drinking alchol don't increase your chance by getting heart disease on the other side the people who don't drink and get heart disease 
# is higher than the people who don't drink and get heart disease
#%%
#Emoitional Stroke and heartdisease
plt.figure(figsize=(10,4))
sns.countplot(data=new_df, x='Stroke', hue='HeartDisease')

#%%
#Difficulty in walking and heart disease
plt.figure(figsize=(10,4))
sns.countplot(data=new_df, x='DiffWalking', hue='HeartDisease')

#It's completely related as the people who have heart disease will be more 
# likely to have difficulty in walking or climibing stairs with a precentage = 300%¶


#%%
#Gender and Heart Disease
plt.figure(figsize=(10,4))
sns.countplot(data=new_df, x='Sex', hue='HeartDisease')
#Males have more chance of getting heartdisease

#%%
#Age category and heart disease
plt.figure(figsize=(16,4))
sns.countplot(data=new_df, x='AgeCategory', hue='HeartDisease', order = new_df['AgeCategory'].sort_values().unique())
#Increasing age will lead to higher chances to get heart diseases

#%%
#Human race and heart disease
plt.figure(figsize=(16,6))
sns.countplot(data=new_df, x='Race', hue='HeartDisease')
#there is a remarkable drop in the Asian people with heart disease

#%%
#Diabete and heart disease
plt.figure(figsize=(16,4))
sns.countplot(data=new_df, x='Diabetic', hue='HeartDisease')

#Diabetic people have chance to be diagnoised by heart disease 
# with increasing chance = 300%


#%%
#Physical activity and heart disease
plt.figure(figsize=(10,4))
sns.countplot(data=new_df, x='PhysicalActivity', hue='HeartDisease')

#Logically, practicing sport will reduce the chance of getting heart disease, and not doing sport double your chance to get heart diseases

#%%
#GenHealth and heart disease
plt.figure(figsize=(16,4))
sns.countplot(data=new_df, x='GenHealth', hue='HeartDisease')
#the more better GenHealth the less chance to get heart diseases

#%%
#SleepTime and heart disease
plt.figure(figsize=(16,6))
sns.histplot(data=new_df, x='SleepTime', hue='HeartDisease', binwidth=1, binrange= (0,15))
#We notice that the best sleeping routine would be to get 7.5 hourse per night to reduce the chance of getting heart disease

#%%
#Asthma and heart disease
plt.figure(figsize=(10,4))
sns.countplot(data=new_df, x='Asthma', hue='HeartDisease')
#Having Asthma will increase your chance to get heart disease

#%%
#Kidney diesase and heart disease
plt.figure(figsize=(10,4))
sns.countplot(data=new_df, x='KidneyDisease', hue='HeartDisease')

#%%
#SkinCancer and heart disease
plt.figure(figsize=(10,4))
sns.countplot(data=new_df, x='SkinCancer', hue='HeartDisease')

#%%
