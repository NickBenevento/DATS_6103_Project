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
Draw_pie_chart("PhysicalActivity")


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
# how to change the order?
#the more better GenHealth the less chance to get heart diseases

#%%
#SleepTime and heart disease
plt.figure(figsize=(16,6))
sns.histplot(data=new_df, x='SleepTime', hue='HeartDisease', binwidth=1, binrange= (0,15))
#We notice that the best sleeping routine would be to get 7.5 hourse per night to reduce the chance of getting heart disease


#%%
one = new_df[new_df['Diabetic'] == 'No']
one.info()
# %%
two = new_df[new_df['Diabetic'] == 'Yes']

#%%
one.groupby('HeartDisease').size().plot(kind='pie', autopct='%1.0f%%', textprops={'fontsize': 20},colors=['tomato', 'gold'])
plt.title('% of HeartDisease in No Diabetic', fontsize = 12)
#%%
two.groupby('HeartDisease').size().plot(kind='pie', autopct='%1.0f%%', textprops={'fontsize': 20},colors=['tomato', 'gold'])
plt.title('% of HeartDisease in Yes Diabetic', fontsize = 12)
#%%
#fig,  (ax1,ax2) = plt.subplots(1,2,figsize=(10,10))

#axes[1].plot(one.groupby('HeartDisease'),kind='pie', figsize=(5,5), fontsize=10,  labels = ['No', 'Yes'], autopct='%1.0f%%')
#axes[1].set_title('% of HeartDisease in No Diabetic', fontsize = 12)

#axes[2].plot(two.groupby('HeartDisease'), kind='pie', figsize=(5,5), labels = ['No', 'Yes'], fontsize=10, autopct='%1.0f%%')
#axes[2].set_title('% of HeartDisease in Yes Diabetic', fontsize = 12)

#plt.show()


#%%
new_df.groupby('pclass').size().plot(kind='pie', autopct='%1.0f%%', textprops={'fontsize': 20},colors=['tomato', 'gold', 'skyblue'])

#%%
dia = pd.crosstab(new_df.Diabetic, new_df.HeartDisease)
import matplotlib.pyplot as plt
dia.plot.bar(stacked=True)
plt.legend(title='HeartDisease')
plt.show()

# %%
df3 = new_df[new_df['PhysicalActivity'] == 'No']
df3.groupby('HeartDisease').size().plot(kind='pie', autopct='%1.0f%%', textprops={'fontsize': 20},colors=['tomato', 'gold'])




##################################################################################
#%%
# read in datafile and check variables 
df = pd.read_csv("heart_2020_cleaned.csv")
df.head()
#%%
# Data understang
for feature in df.columns:
    print(feature)
    print(df[feature].unique(),"\n")

# %%
# Data understang
no = df['HeartDisease'].value_counts()[0]
yes = df['HeartDisease'].value_counts()[1] # the reason have [1][0] label is associate with '{}%'.format
print('The number of People have heart disease is {}%'.format(((yes/len(df))*100).round(2)))
#unbalance data

# %%
#data vistualizatio
features = df.columns
features

# %%
# binary feature
binary_feature=[]
for feature in df.columns:
    if np.isin(df[feature].unique(),["Yes","No"]).all() or np.isin(df[feature].unique(),["Male","Female"]).all():
        binary_feature.append(feature)
        
#binary_feature
# %%
continuos_feature = ['BMI']
#can not use df['BMI], casue is different dtype
# %%
discrete_feature = features[~features.isin(binary_feature+continuos_feature)]
#np.isin function
discrete_feature
# %%
# discrete_feature vistualization
position_index = [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]
fig,axes = plt.subplots(3,2,figsize=(20,15))
for position, feature in zip(position_index, discrete_feature):
    if len(df[feature].unique()) > 15:
        sns.histplot(ax=axes[position],bins = 15, data=df[feature].sort_values())
    else:
        if feature in ["AgeCategory", "Race"]:
            i, r = pd.factorize(df[feature])
            a = np.argsort(np.bincount(i)[i], kind='mergesort')[::-1]
            sns.histplot(ax=axes[position],y=df.iloc[a][feature])
        elif feature == "GenHealth":
            sns.histplot(ax=axes[position],data=pd.Categorical(df.GenHealth, categories=["Poor","Fair","Good","Very good","Excellent"], ordered=True))
            axes[position].set(xlabel=feature)
        else:
            sns.histplot(ax=axes[position],data=df[feature].sort_values())
    axes[position].set_title(feature)
        
fig.tight_layout()
plt.show()

#%%
# continous data vistualization
combine_features = features[~features.isin(continuos_feature)]
#%%
nrows, ncols = 9, 2

fig = plt.figure(figsize=(15,60))    
for position, name_feature in zip(range(1,18),combine_features):
    axes = fig.add_subplot(nrows, ncols, position)
    sns.kdeplot(ax=axes,data=df, x=continuos_feature[0], hue=name_feature,fill=True,bw_adjust=.8)
    
fig.tight_layout()
plt.show()

#%%
#Data selection 
df.corr()

# %%
# one hot encoder features
cat_features = []
num_features = []
for column, i in zip(df.columns, df.dtypes):
    if i == object:
        cat_features.append(column)
    else:
        num_features.append(column)

#%%
# show encoded feature
df_cat = df[cat_features].copy()
df_cat.head()
#%%
# transfer cat to num
# ordinalencoder
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
df_cat_encoded = ordinal_encoder.fit_transform(df_cat)
df_cat_encoded = pd.DataFrame(df_cat_encoded, columns = cat_features)

df_cat_encoded.info()
#%%
# check the unique value after transformation
for feature in df_cat_encoded.columns:
    print(feature)
    print(df_cat_encoded[feature].unique(),"\n")
    
#%%
#new numeric datafram
df_num = pd.merge(df_cat_encoded, df[num_features],left_index=True, right_index=True)
df_num.head()
#%%
# corr
sns.heatmap(df_num.corr())
plt.show()
#%%
# corr ranking
corr_m = df_num.corr()
corr_m['HeartDisease'].drop('HeartDisease').sort_values(ascending=False)
#vistual?
#%%
#covariance
cov_m = df_num.cov()
sns.heatmap(cov_m)
plt.show()

#%%
# cov ranking
cov_m["HeartDisease"].drop("HeartDisease").sort_values(ascending=False)
#vistual
#%%
#feture inprotance in encode feature
unscaling_cor = corr_m["HeartDisease"].drop("HeartDisease").sort_values(ascending=False)[:6]
unscaling_cov = cov_m["HeartDisease"].drop("HeartDisease").sort_values(ascending=False)[:6]

plot_bar_chart(unscaling_cor,unscaling_cov,maintitle = "Correlation vs Covariance", title1="Correlation",title2="Covariance")
#%%
# standarlized dataset
#%%
# feature importance in un-encode standarlized dataset 
# feature importance in encode standarlized dataset
# compare un-encode standarlized dataset with not encode standarlized dataset