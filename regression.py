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
    confusion: pd.DataFrame = pd.crosstab(df.HeartDiseaseBin, model_predictions['heart_disease_' + str(cut_off)],
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
dm.dfChk(df2)

# %%
# create binary variable for Heart Disease 
df2['HeartDiseaseBin'] = 0
df2.loc[ df2['HeartDisease'] == "Yes", 'HeartDiseaseBin' ] = 1

model_predictions_bal = pd.DataFrame()

print(df2.head())
formula = 'HeartDiseaseBin ~ BMI + C(Smoking) + C(AlcoholDrinking) + C(Stroke) + \
            PhysicalHealth + MentalHealth + C(DiffWalking) + C(Sex) + C(Race) + \
            C(Diabetic) + C(PhysicalActivity) + C(GenHealth) + \
            SleepTime + C(Asthma) + C(KidneyDisease) + C(SkinCancer) + AgeCont'
heart_disease_bal = glm(formula=formula, data=df2, family=sm.families.Binomial())

heart_disease_bal_fit = heart_disease_bal.fit()
print(heart_disease_bal_fit.summary())
model_predictions_bal['heart_disease'] = heart_disease_bal_fit.predict(df2)

# add row of binary actual outcomes for comparison
model_predictions_bal['actual'] = df2['HeartDiseaseBin']

# check it out 
model_predictions_bal.head()


# %%
print(-2*heart_disease_bal_fit.llf)
# Compare to the null deviance
print(heart_disease_bal_fit.null_deviance)

# %%
# Try three different cut-off values at 0.3, 0.5, and 0.7. What are the a) Total accuracy of the model b) The precision of the model (average for 0 and 1), and c) the recall rate of the model (average for 0 and 1)
cut_offs = [0.3, 0.5, 0.7]

for cut_off in cut_offs:
    # Compute class predictions
    model_predictions_bal['heart_disease_' + str(cut_off)] = np.where(model_predictions_bal['heart_disease'] > cut_off, 1, 0)
    #
    # Make a cross table
    confusion_bal: pd.DataFrame = pd.crosstab(df2.HeartDiseaseBin, model_predictions_bal['heart_disease_' + str(cut_off)],
    rownames=['Actual'], colnames=['Predicted'],
    margins = True)

    # print(confusion)

    true_neg_bal = confusion_bal[0][0]
    false_pos_bal = confusion_bal[1][0]
    false_neg_bal = confusion_bal[0][1]
    true_pos_bal = confusion_bal[1][1]

    total_bal = confusion_bal['All']['All']
    accuracy_bal = (confusion_bal.iloc[1][1] + confusion_bal.iloc[0][0]) / total_bal
    precision_bal = true_pos_bal / (true_pos_bal + false_pos_bal)
    recall_bal = true_pos_bal / (true_pos_bal + false_neg_bal)
    f1_bal = 2 * (precision_bal*recall_bal) / (precision_bal + recall_bal)
    print(f'Cutoff = {cut_off}:')
    print('f1: ', round(f1_bal, 3))
    print('precision: ', round(precision_bal, 3))
    print('recall: ', round(recall_bal, 3))
    print()
