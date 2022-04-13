# %%
import numpy as np
import pandas as pd
import dm6103 as dm
import statsmodels.api as sm  # Importing statsmodels
from statsmodels.formula.api import glm


df = pd.read_csv("heart_2020_new.csv")
dm.dfChk(df)

# %%

# create binary variable for Heart Disease 
df['HeartDiseaseBin'] = 0
df.loc[ df['HeartDisease'] == "Yes", 'HeartDiseaseBin' ] = 1

model_predictions = pd.DataFrame()

print(df.head())
formula = 'HeartDiseaseBin ~ BMI + C(Smoking) + C(AlcoholDrinking) + C(Stroke) + \
            PhysicalHealth + MentalHealth + C(DiffWalking) + C(Sex) + C(Race) + \
            C(Diabetic) + C(PhysicalActivity) + C(GenHealth) + \
            SleepTime + C(Asthma) + C(KidneyDisease) + C(SkinCancer) + AgeCont'
heart_disease = glm(formula=formula, data=df, family=sm.families.Binomial())

heart_disease_fit = heart_disease.fit()
print(heart_disease_fit.summary())
model_predictions['heart_disease'] = heart_disease_fit.predict(df)

# add row of binary actual outcomes for comparison
model_predictions['actual'] = df['HeartDiseaseBin']

# check it out 
model_predictions.head()


# %%
print(-2*heart_disease_fit.llf)
# Compare to the null deviance
print(heart_disease_fit.null_deviance)

# %%
# Try three different cut-off values at 0.3, 0.5, and 0.7. What are the a) Total accuracy of the model b) The precision of the model (average for 0 and 1), and c) the recall rate of the model (average for 0 and 1)
cut_offs = [0.3, 0.5, 0.7]

for cut_off in cut_offs:
    # Compute class predictions
    model_predictions['heart_disease_' + str(cut_off)] = np.where(model_predictions['heart_disease'] > cut_off, 1, 0)
    #
    # Make a cross table
    confusion: pd.DataFrame = pd.crosstab(df.HeartDisease, model_predictions['heart_disease_' + str(cut_off)],
    rownames=['Actual'], colnames=['Predicted'],
    margins = True)

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
    print(f'Cutoff = {cut_off}:')
    print('f1: ', round(f1, 3))
    print('precision: ', round(precision, 3))
    print('recall: ', round(recall, 3))
    print()

# %%
 # now try all of the above with the balanced dataset 

df2 = pd.read_csv("heart_2020_balanced.csv")
dm.dfChk(df)

# %%

# create binary variable for Heart Disease 
df['HeartDiseaseBin'] = 0
df.loc[ df['HeartDisease'] == "Yes", 'HeartDiseaseBin' ] = 1

model_predictions = pd.DataFrame()

print(df.head())
formula = 'HeartDiseaseBin ~ BMI + C(Smoking) + C(AlcoholDrinking) + C(Stroke) + \
            PhysicalHealth + MentalHealth + C(DiffWalking) + C(Sex) + C(Race) + \
            C(Diabetic) + C(PhysicalActivity) + C(GenHealth) + \
            SleepTime + C(Asthma) + C(KidneyDisease) + C(SkinCancer) + AgeCont'
heart_disease = glm(formula=formula, data=df, family=sm.families.Binomial())

heart_disease_fit = heart_disease.fit()
print(heart_disease_fit.summary())
model_predictions['heart_disease'] = heart_disease_fit.predict(df)

# add row of binary actual outcomes for comparison
model_predictions['actual'] = df['HeartDiseaseBin']

# check it out 
model_predictions.head()


# %%
print(-2*heart_disease_fit.llf)
# Compare to the null deviance
print(heart_disease_fit.null_deviance)

# %%
# Try three different cut-off values at 0.3, 0.5, and 0.7. What are the a) Total accuracy of the model b) The precision of the model (average for 0 and 1), and c) the recall rate of the model (average for 0 and 1)
cut_offs = [0.3, 0.5, 0.7]

for cut_off in cut_offs:
    # Compute class predictions
    model_predictions['heart_disease_' + str(cut_off)] = np.where(model_predictions['heart_disease'] > cut_off, 1, 0)
    #
    # Make a cross table
    confusion: pd.DataFrame = pd.crosstab(df.HeartDisease, model_predictions['heart_disease_' + str(cut_off)],
    rownames=['Actual'], colnames=['Predicted'],
    margins = True)

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
    print(f'Cutoff = {cut_off}:')
    print('f1: ', round(f1, 3))
    print('precision: ', round(precision, 3))
    print('recall: ', round(recall, 3))
    print()
