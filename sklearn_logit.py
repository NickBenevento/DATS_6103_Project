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

dfX = df[[ 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer', 'AgeCont']]

dfY = df[['HeartDiseaseBin']]

# split into test and train 
X_train, X_test, y_train, y_test = train_test_split(DfX, DfY, test_size = 0.25, random_state=333)

# create empty dataframe for predictions vs. actuals
model_predictions = pd.DataFrame()


