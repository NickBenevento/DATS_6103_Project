# %%
import numpy as np
import pandas as pd
import dm6103 as dm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

plt.style.use('seaborn')

# import and check out data
df = pd.read_csv("heart_2020_balanced.csv")
dm.dfChk(df)

# create binary variable for Heart Disease 
df['HeartDiseaseBin'] = 0
df.loc[ df['HeartDisease'] == "Yes", 'HeartDiseaseBin' ] = 1

# %%
# prep data for modelling

dfX = df[[ 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer', 'AgeCont', 'BMI']]

dfY = df[['HeartDiseaseBin']]

# change X variables to either numeric or factor

for var in ('Smoking', 'AlcoholDrinking', 'Stroke',  'DiffWalking', 'Sex', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth',  'Asthma', 'KidneyDisease', 'SkinCancer'):
    dfX[var] = dfX[var].astype('category').cat.codes

# cat code mapping:
# Sex: F = 0, M = 1
# Race: White = 5, Hisp = 3, Black - 2, Other = 4, Asian = 1, American Indian/Alaskan Native = 0
# Diabetic: No = 0, Yes = 2, No borderline = 1, Yes pregnancy = 3
# GenHealth: 2 = good, 4 = very good, fair = 1, Excellent = 0, Poor = 3


# split into test and train 
X_train, X_test, y_train, y_test = train_test_split(dfX, dfY, test_size = 0.25, random_state=333)

# create empty dataframe for predictions vs. actuals
model_predictions = pd.DataFrame()
model_predictions['Actuals'] = y_test['HeartDiseaseBin']

# %%
# model

HDlogit = LogisticRegression()  # instantiate
HDlogit.fit(X_train, y_train)
print('Logit model accuracy (with the test set):', HDlogit.score(X_test, y_test))
print('Logit model accuracy (with the train set):', HDlogit.score(X_train, y_train))


# %%
# Store predictions
model_predictions['Model1'] = HDlogit.predict_proba(X_test)[:,1]
model_predictions.head()

# %%
# Scoring 
from sklearn.metrics import classification_report
y_true, y_pred = y_test, HDlogit.predict(X_test)
print(classification_report(y_true, y_pred))

#%%
# Receiver Operator Characteristics (ROC)
# Area Under the Curve (AUC)
from sklearn.metrics import roc_auc_score, roc_curve

# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = HDlogit.predict_proba(X_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()

#0.8 is a good model 

#%%
# use some dummy data to see how predictions change 
data = [[0, 0, 0, 0, 0, 0, 1, 5, 0, 1, 0, 8, 0, 0, 0, 40, 20]] 
testData = pd.DataFrame(data, columns = ['Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer', 'AgeCont', 'BMI'])
dataTable = testData.copy()
dataTable['Prediction'] = HDlogit.predict_proba(testData)[:,1]
print(dataTable)

# %%
