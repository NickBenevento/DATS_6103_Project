# %%
import pandas as pd
import dm6103 as dm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


df = pd.read_csv('heart_2020_balanced.csv')
dm.dfChk(df)

# %%
# create binary variable for Heart Disease 
df['HeartDiseaseBin'] = 0
df.loc[ df['HeartDisease'] == "Yes", 'HeartDiseaseBin' ] = 1

# %%
features = [ 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer', 'AgeCont', 'BMI']
x_df = df[[ 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer', 'AgeCont', 'BMI']]

y_df = df[['HeartDiseaseBin']]

for var in ('Smoking', 'AlcoholDrinking', 'Stroke',  'DiffWalking', 'Sex', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth',  'Asthma', 'KidneyDisease', 'SkinCancer'):
    x_df[var] = x_df[var].astype('category').cat.codes

X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size = 0.25, random_state=333)
# %%
# use a linear kernel b/c we are predicting a binary variable (2 dimensions for hyperplane)
svm = SVC(kernel='linear')
svm.fit(X_train, y_train.values.ravel())

y_pred = svm.predict(X_test)

# %%
# all observations, no split
all_observations = pd.DataFrame()
all_observations['predicted'] = svm.predict(x_df)
all_observations.head()

# %%
# Compute class predictions
all_observations['heart_disease'] = np.where(all_observations['predicted'] == 1, 1, 0)
#
# Make a cross table
confusion: pd.DataFrame = pd.crosstab(y_df.HeartDiseaseBin, all_observations['heart_disease'],
                                      rownames=['Actual'], colnames=['Predicted'],
                                      margins=True)
# print(confusion)

true_neg = confusion[0][0]
false_pos = confusion[1][0]
false_neg = confusion[0][1]
true_pos = confusion[1][1]

total = confusion['All']['All']
accuracy = (confusion.iloc[1][1] + confusion.iloc[0][0]) / total
precision = true_pos / (true_pos + false_pos)
recall = true_pos / (true_pos + false_neg)
f1 = 2 * (precision*recall) / (precision + recall)
print('f1: ', round(f1, 3))
print('accuracy: ', round(accuracy, 3))
print('precision: ', round(precision, 3))
print('recall: ', round(recall, 3))
print()
# %%
from sklearn import metrics
# Train/Test split

y_true, y_pred = y_test, svm.predict(X_test)

recall = precision_score(y_test, y_pred)
precision = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
F1 = f1_score(y_test, y_pred)

print('Recall: ', round(recall, 4))
print('Precision: ', round(precision, 4))
print('Accuracy: ', round(accuracy, 4))
print('F1: ', round(F1, 4))
print(metrics.r2_score(y_true, y_pred))
print(metrics.mean_squared_error(y_true, y_pred))
# F1: 75.34%, accuracy: 74.74%: decent

# %%
# Use some test data:
data = [[0, 0, 0, 0, 0, 0, 1, 5, 0, 1, 0, 8, 0, 0, 0, 40, 20],
        [1, 1, 1, 0, 0, 1, 1, 5, 0, 1, 0, 8, 0, 0, 0, 40, 20]] 
test_data = pd.DataFrame(data, columns = ['Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Sex', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer', 'AgeCont', 'BMI'])
data_table = test_data.copy()
data_table['Prediction'] = svm.predict(test_data)
# Predicts the first person as 0, second as 1 (which seems to make sense, 
# as person 2 is much more unhealthy and has underlying health conditions)
print(data_table)
