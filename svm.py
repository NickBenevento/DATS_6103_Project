# %%
import pandas as pd
import dm6103 as dm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


df = pd.read_csv('heart_2020_balanced.csv')
dm.dfChk(df)

# %%
# create binary variable for Heart Disease 
df['HeartDiseaseBin'] = 0
df.loc[ df['HeartDisease'] == "Yes", 'HeartDiseaseBin' ] = 1

# %%
x_df = df[[ 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer', 'AgeCont', 'BMI']]

y_df = df[['HeartDiseaseBin']]

for var in ('Smoking', 'AlcoholDrinking', 'Stroke',  'DiffWalking', 'Sex', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth',  'Asthma', 'KidneyDisease', 'SkinCancer'):
    x_df[var] = x_df[var].astype('category').cat.codes

X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size = 0.25, random_state=333)
# %%
# use a linear kernel b/c we are predicting a binary variable (2 dimensions)
svm = SVC(kernel='linear')
svm.fit(X_train, y_train.values.ravel())

y_pred = svm.predict(X_test)

# %%
print('Accuracy: ', accuracy_score(y_test, y_pred))
# 74.74% accuracy; decent
# %%
