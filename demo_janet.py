#%%
# import packages
import pandas as pd
import dm6103 as dm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# read in datafile and check variables 
df = pd.read_csv("heart_2020_cleaned_default.csv")
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
# add new variable for continuous age (like in 'heart_2020_new.csv)
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

# apply function and return new column of data
new_df["AgeCont"] = new_df.apply(cleanDfAge, axis=1)
print(new_df.dtypes)
print("\nReady to continue.")

# check that this works
new_df.head()

# %%
# export dataset for us to use later
df.to_csv('heart_2020_balanced.csv')  
print("\nReady to continue.")


# stat test prep
import scipy.stats as stats
df_HD = new_df[ new_df['HeartDisease'] == 'Yes']
df_NO = new_df[ new_df['HeartDisease'] == 'No']

#%%
# BMI and Heart Disease
plt.figure(figsize=(10,4))

# Histogram
sns.histplot(data=new_df, x='BMI', hue='HeartDisease', binwidth=1, kde=True)

# Aesthetics
plt.title('BMI Distrbution')
plt.xlabel('BMI indicator')
#We notice that the most healthy people who have BMI within 20 till about 
# 28 and the more BMI this leads to increase HeartDisease, 
# and that's make sense as BMI is indictator for obesity whic is one of direct cause of HeartDisease¶

# t test
print("t test p val: \n", stats.ttest_ind(df_HD['BMI'], df_NO['BMI'])[1])
# stat sig at a = 0.01, confirm that there is a difference in means

#%%
# #Smoking and heart disease
plt.figure(figsize=(10,4))
sns.countplot(data=new_df, x='Smoking', hue='HeartDisease')
#We notice that smoking increase your chance in getting Heart Diseases by about 50%

# chi sq
ct1 = pd.crosstab(new_df['Smoking'], new_df['HeartDisease'])
print("chi sq p val: \n", stats.chi2_contingency(ct1)[1])
# stat sig at a = 0.01, confirm that there is a difference in frequencies

#%%
#Alchol and Heart Disease
plt.figure(figsize=(10,4))
sns.countplot(data=new_df, x='AlcoholDrinking', hue='HeartDisease')
#drinking alchol don't increase your chance by getting heart disease on the other side the people who don't drink and get heart disease 
# is higher than the people who don't drink and get heart disease

# chi sq
ct2 = pd.crosstab(new_df['AlcoholDrinking'], new_df['HeartDisease'])
print("chi sq p val: \n", stats.chi2_contingency(ct2)[1])
# stat sig at a = 0.01, confirm that there is a difference in frequencies
#%%
#Emotional Stroke and heartdisease
plt.figure(figsize=(10,4))
sns.countplot(data=new_df, x='Stroke', hue='HeartDisease')

# chi sq
ct3 = pd.crosstab(new_df['Stroke'], new_df['HeartDisease'])
print("chi sq p val: \n", stats.chi2_contingency(ct3)[1])
# stat sig at a = 0.01, confirm that there is a difference in frequencies

#%%
#Difficulty in walking and heart disease
plt.figure(figsize=(10,4))
sns.countplot(data=new_df, x='DiffWalking', hue='HeartDisease')

#It's completely related as the people who have heart disease will be more 
# likely to have difficulty in walking or climibing stairs with a precentage = 300%¶

# chi sq
ct4 = pd.crosstab(new_df['DiffWalking'], new_df['HeartDisease'])
print("chi sq p val: \n", stats.chi2_contingency(ct4)[1])
# stat sig at a = 0.01, confirm that there is a difference in frequencies

#%%
#Gender and Heart Disease
plt.figure(figsize=(10,4))
sns.countplot(data=new_df, x='Sex', hue='HeartDisease')
#Males have more chance of getting heartdisease

# chi sq
ct5 = pd.crosstab(new_df['Sex'], new_df['HeartDisease'])
print("chi sq p val: \n", stats.chi2_contingency(ct5)[1])
# stat sig at a = 0.01, confirm that there is a difference in frequencies

#%%
#Age category and heart disease
plt.figure(figsize=(16,4))
sns.countplot(data=new_df, x='AgeCategory', hue='HeartDisease', order = new_df['AgeCategory'].sort_values().unique())
#Increasing age will lead to higher chances to get heart diseases

# chi sq
ct6 = pd.crosstab(new_df['AgeCategory'], new_df['HeartDisease'])
print("chi sq p val: \n", stats.chi2_contingency(ct6)[1])
# stat sig at a = 0.01, confirm that there is a difference in frequencies

#%%
# AgeCont (continuous age variable) and heart disease 
sns.boxplot(x='HeartDisease',y='AgeCont', data=new_df)
# also looks like two different mean ages, with higher age associated with heart disease

# t test
print("t test p val: \n", stats.ttest_ind(df_HD['AgeCont'], df_NO['AgeCont'])[1])
# stat sig at a = 0.01, confirm that there is a difference in means

#%%
#Human race and heart disease
plt.figure(figsize=(16,6))
sns.countplot(data=new_df, x='Race', hue='HeartDisease')
#there is a remarkable drop in the Asian people with heart disease

# chi sq
ct7 = pd.crosstab(new_df['Race'], new_df['HeartDisease'])
print("chi sq p val: \n", stats.chi2_contingency(ct7)[1])
# stat sig at a = 0.01, confirm that there is a difference in frequencies

#%%
#Diabete and heart disease
plt.figure(figsize=(16,4))
sns.countplot(data=new_df, x='Diabetic', hue='HeartDisease')

#Diabetic people have chance to be diagnoised by heart disease 
# with increasing chance = 300%

# chi sq
ct8 = pd.crosstab(new_df['Diabetic'], new_df['HeartDisease'])
print("chi sq p val: \n", stats.chi2_contingency(ct8)[1])
# stat sig at a = 0.01, confirm that there is a difference in frequencies



#%%
#Physical activity and heart disease
plt.figure(figsize=(10,4))
sns.countplot(data=new_df, x='PhysicalActivity', hue='HeartDisease')

#Logically, practicing sport will reduce the chance of getting heart disease, and not doing sport double your chance to get heart diseases

# chi sq
ct9 = pd.crosstab(new_df['PhysicalActivity'], new_df['HeartDisease'])
print("chi sq p val: \n", stats.chi2_contingency(ct9)[1])
# stat sig at a = 0.01, confirm that there is a difference in frequencies

#%%
#GenHealth and heart disease
plt.figure(figsize=(16,4))
sns.countplot(data=new_df, x='GenHealth', hue='HeartDisease')
#the more better GenHealth the less chance to get heart diseases

# chi sq
ct10 = pd.crosstab(new_df['GenHealth'], new_df['HeartDisease'])
print("chi sq p val: \n", stats.chi2_contingency(ct10)[1])
# stat sig at a = 0.01, confirm that there is a difference in frequencies


#%%
#SleepTime and heart disease
plt.figure(figsize=(16,6))
sns.histplot(data=new_df, x='SleepTime', hue='HeartDisease', binwidth=1, binrange= (0,15))
#We notice that the best sleeping routine would be to get 7.5 hours per night to reduce the chance of getting heart disease

# t test
print("t test p val: \n", stats.ttest_ind(df_HD['SleepTime'], df_NO['SleepTime'])[1])
# stat sig at a = 0.01, confirm that there is a difference in means


#%%
#Asthma and heart disease
plt.figure(figsize=(10,4))
sns.countplot(data=new_df, x='Asthma', hue='HeartDisease')
#Having Asthma will increase your chance to get heart disease

# chi sq
ct11 = pd.crosstab(new_df['Asthma'], new_df['HeartDisease'])
print("chi sq p val: \n", stats.chi2_contingency(ct11)[1])
# stat sig at a = 0.01, confirm that there is a difference in frequencies

#%%
#Kidney diesase and heart disease
plt.figure(figsize=(10,4))
sns.countplot(data=new_df, x='KidneyDisease', hue='HeartDisease')

# chi sq
ct12 = pd.crosstab(new_df['KidneyDisease'], new_df['HeartDisease'])
print("chi sq p val: \n", stats.chi2_contingency(ct12)[1])
# stat sig at a = 0.01, confirm that there is a difference in frequencies
#%%
#SkinCancer and heart disease
plt.figure(figsize=(10,4))
sns.countplot(data=new_df, x='SkinCancer', hue='HeartDisease')

# chi sq
ct13 = pd.crosstab(new_df['SkinCancer'], new_df['HeartDisease'])
print("chi sq p val: \n", stats.chi2_contingency(ct13)[1])
# stat sig at a = 0.01, confirm that there is a difference in frequencies
# %%
