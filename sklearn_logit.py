# %%
import numpy as np
import pandas as pd
import dm6103 as dm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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
