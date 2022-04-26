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

# encoding for binaries 
for var in ('Smoking', 'AlcoholDrinking', 'Stroke',  'DiffWalking', 'Sex', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer'):
    dfX[var] = dfX[var].astype('category').cat.codes

# encoding for categorical

from pandas.api.types import CategoricalDtype

# GH
GH_dtype = CategoricalDtype(
    categories=['Excellent', 'Very good', 'Good', 'Fair', 'Poor'], ordered=True)

dfX['GenHealth'] = dfX['GenHealth'].astype(GH_dtype).cat.codes

# Diabetic 
DB_dtype = CategoricalDtype(
    categories=['No', 'No, borderline diabetes', 'Yes (during pregnancy)', 'Yes'], ordered=True)

dfX['Diabetic'] = dfX['Diabetic'].astype(DB_dtype).cat.codes

# Race
dfX = pd.get_dummies(dfX, columns=['Race'])



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

# see coefficients 
weight = HDlogit.coef_  
factors = HDlogit.feature_names_in_
weight = weight.reshape(22,1)
factors = factors.reshape(22,1)
regression_results = pd.DataFrame(factors, columns = ['vars'])
regression_results['coefs'] = weight

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
data = [['No', 'No', 'No', 0, 0, 'No', 'Female', 'White', 'No', 'Yes', 'Excellent', 8, 'No', 'No', 'No', 40, 20], ['No', 'No', 'No', 0, 0, 'No', 'Male', 'White', 'No', 'Yes', 'Excellent', 8, 'No', 'No', 'No', 40, 20], ['Yes', 'No', 'No', 0, 0, 'No', 'Female', 'White', 'No', 'Yes', 'Good', 8, 'No', 'No', 'No', 45, 23], ['Yes', 'Yes', 'No', 0, 0, 'No', 'Female', 'White', 'No', 'Yes', 'Poor', 8, 'No', 'No', 'No', 45, 23], ['Yes', 'No', 'Yes', 0, 0, 'No', 'Male', 'White', 'No', 'Yes', 'Fair', 5, 'No', 'No', 'No', 45, 23], ['No', 'No', 'No', 0, 0, 'No', 'Female', 'White', 'Yes', 'Yes', 'Excellent', 8, 'No', 'No', 'No', 40, 20], ['No', 'No', 'No', 0, 0, 'Yes', 'Male', 'White', 'Yes', 'No', 'Good', 8, 'No', 'No', 'No', 40, 20], ['No', 'No', 'No', 0, 0, 'Yes', 'Male', 'Asian', 'Yes', 'No', 'Good', 8, 'No', 'No', 'No', 40, 20], ['No', 'No', 'No', 0, 0, 'Yes', 'Male', 'Asian', 'Yes', 'No', 'Good', 8, 'No', 'Yes', 'Yes', 40, 20], ['No', 'No', 'No', 0, 0, 'Yes', 'Male', 'Hispanic', 'Yes', 'No', 'Good', 8, 'No', 'Yes', 'Yes', 40, 20], ['No', 'No', 'No', 0, 0, 'Yes', 'Male', 'Black', 'Yes', 'No', 'Good', 8, 'No', 'Yes', 'Yes', 40, 20], ['No', 'No', 'No', 0, 0, 'Yes', 'Male', 'Other', 'Yes', 'No', 'Good', 8, 'No', 'Yes', 'Yes', 40, 20], ['No', 'No', 'No', 0, 0, 'Yes', 'Male', 'American Indian/Alaskan Native ', 'Yes', 'No', 'Good', 8, 'No', 'Yes', 'Yes', 40, 20] ] 
testData = pd.DataFrame(data, columns = ['Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer', 'AgeCont', 'BMI'])

# change X variables to either numeric or factor

# encoding for binaries 
for var in ('Smoking', 'AlcoholDrinking', 'Stroke',  'DiffWalking', 'Sex', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer'):
    testData[var] = testData[var].astype('category').cat.codes

# encoding for categorical

# GH
GH_dtype = CategoricalDtype(
    categories=['Excellent', 'Very good', 'Good', 'Fair', 'Poor'], ordered=True)

testData['GenHealth'] = testData['GenHealth'].astype(GH_dtype).cat.codes

# Diabetic 
DB_dtype = CategoricalDtype(
    categories=['No', 'No, boderline diabetes', 'Yes (during pregnancy)', 'Yes'], ordered=True)

testData['Diabetic'] = testData['Diabetic'].astype(DB_dtype).cat.codes

# Race
testData = pd.get_dummies(testData, columns=['Race'])

# predictions

dataTable = testData.copy()
dataTable['Prediction'] = HDlogit.predict_proba(testData)[:,1]
print(dataTable)
dataTable.to_csv('Test_Pred_2.csv')
# %%
