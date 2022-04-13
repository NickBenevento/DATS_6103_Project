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
#%%
sns.countplot('HeartDisease', data=new_df)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()
#%%
#Reset indices
new_df.reset_index(inplace = True,drop = True)
new_df

#%%
# pie chart
# Create the function to visualize to pie chart
def Draw_pie_chart(name_feature):

    fig,axes = plt.subplots(1,2,figsize=(15,8))
    labels = new_df[name_feature].unique()
    textprops = {"fontsize":15}

    axes[0].pie(new_df[new_df.HeartDisease=="No"][name_feature].value_counts(), labels=labels,autopct='%1.1f%%',textprops =textprops)
    axes[0].set_title('No Heart Disease',fontsize=15)
    axes[1].pie(new_df[new_df.HeartDisease=="Yes"][name_feature].value_counts(), labels=labels,autopct='%1.1f%%',textprops =textprops)
    axes[1].set_title('Yes Heart Disease',fontsize=15)

    plt.legend(title = name_feature, fontsize=15, title_fontsize=15)
    plt.show()
    
#%%
# pie chart
Draw_pie_chart("Smoking")

#%%
features = new_df.columns
# Take out all features is binary data (Yes, No) and (Male, Female)
binary_feature = []

for feature in new_df.columns:
    if np.isin(new_df[feature].unique(),["Yes","No"]).all() or np.isin(new_df[feature].unique(),["Male","Female"]).all():
        binary_feature.append(feature)
        
# Take out all features is continuous data
continuos_feature = ["BMI"]

# Take out all features is discrete data
discrete_feature = features[~features.isin(binary_feature+continuos_feature)]


#%%
for feature in binary_feature[2:]:
    Draw_pie_chart(feature)

#%%
sns.countplot('HeartDisease', data=new_df)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()


#%%
# countinues variable
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
# disceret variable
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
#GenHealth and heart disease
plt.figure(figsize=(16,4))
sns.countplot(data=new_df, x='GenHealth', hue='HeartDisease')
#the more better GenHealth the less chance to get heart diseases

#%%
#SleepTime and heart disease
plt.figure(figsize=(16,6))
sns.histplot(data=new_df, x='SleepTime', hue='HeartDisease', binwidth=1, binrange= (0,15))
#We notice that the best sleeping routine would be to get 7.5 hourse per night to reduce the chance of getting heart disease
