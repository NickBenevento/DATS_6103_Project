# Heart Disease Prediction (DATS 6103 Project)

![download](https://github.com/NickBenevento/DATS_6103_Project/blob/main/hd.jpg)
### Basic Information

* **Person or organization developing model**: Ange Olson, Nick Benevento, Yaxin Zhuang
* **Project date**: April, 2022
* **Project topic**: Determine if the key risk factors listed by the CDC were good predictors for heart disease and also explore if there are other, less prevalent factors that were also useful predictors. 

### Data Information

* Data dictionary: 
  - HeartDisease: Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI).
  - BMI: Body Mass Index (BMI).
  - Smoking: Have you smoked at least 100 cigarettes in your entire life?
  - AlcoholDrinking: Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week
  - Stroke: (Ever told) (you had) a stroke?
  - PhysicalHealth: Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? (0-30 days).
  - MentalHealth: Thinking about your mental health, for how many days during the past 30 days was your mental health not good? (0-30 days).
  - DiffWalking: Do you have serious difficulty walking or climbing stairs?
  - Sex: Are you male or female?
  - AgeCategory: Fourteen-level age category. (then calculated the mean)
  - Race: Imputed race/ethnicity value.
  - Diabetic: (Ever told) (you had) diabetes?
  - PhysicalActivity: Adults who reported doing physical activity or exercise during the past 30 days other than their regular job.
  - GenHealth: Would you say that in general your health is...
  - SleepTime: On average, how many hours of sleep do you get in a 24-hour period?
  - Asthma: (Ever told) (you had) asthma?
  - KidneyDisease: Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?
  - SkinCancer: (Ever told) (you had) skin cancer?

* **Source of test data**: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-diseasen
* **Number of rows in test data**: 57,000


### EDA

* **Feature Importance**:
  * Correlation and Covariance
![download](https://github.com/NickBenevento/DATS_6103_Project/blob/main/correlation_features.JPG)

![download](https://github.com/NickBenevento/DATS_6103_Project/blob/main/covariance_features.JPG)

    Result in the correlation being the same between not scaled and scaled data, but there are quite big differences in covariance. The PhysicalHealth decreased from 0.37 to 0.078 which is a huge decrease, Diabetic and Diffwalking are rising from 5,6 to 3,4. 
    From the above correlation and covariance information, we can conclude features that most relate to heart disease are:AgeCategory, Diffwalking, PhysicalHeadlth, and Diabetic. Also, looking back to the binary variable analysis part, we found that Stroke, PhysicalActivity, Diffwalking, Skincancer, KidneyDisease, Diabetic, AgeCategory affect more people with heart disease than people who don’t have it. However, while Stroke,Diffwalking, KidneyDisease,Diabetic, AgeCategory in the top 6 correlations with heart disease, the rest two have every low correlation and PhysicalActivity has a negative correlation and covariance with heart disease after normalization. 
	   In conclusion, we have 6 important features: AgeCategory, Diffwalking, PhysicalHeadlth, Stroke, KidneyDisease and Diabetic.  
    
* **Age analysis**
![download](https://github.com/NickBenevento/DATS_6103_Project/blob/main/age_analysis.JPG)

    The result of covariance and correlation analysis in AgeCategory seems to have the opposite results from the AgeCategory histogram. Suppose the 70~74 and 80 or older have the most affection for heart disease in the age category histogram, but these two have very low correlation and covariance. Also, people who are 18~24, 30~34, and 45~49 suppose to have a small effect on heart disease, especially people who are 30~34, but it has the highest correlation and covariance. Therefore, 30~34, 18~24, and 45~49 are important features in Agecategory.




### Model analysis:
* **Logit Model with All Observations, No Test/Train Split**:

| Cutoff | F1 Score | Precision | Recall |
| ------ | -------- | --------- | ------ |
| 0.3 | 0.346 | 0.415 | 0.296 |
| 0.5 | 0.18 | 0.532 | 0.108 |
| 0.7 | 0.051 | 0.629 | 0.026 |

* **Logit Model with Balanced Dataset, No Test/Train Split**:

| Cutoff | F1 Score | Precision | Recall |
| ------ | -------- | --------- | ------ |
| 0.3 | 0.778 | 0.674 | 0.922 |
| 0.5 | 0.768 | 0.759 | 0.777 |
| 0.7 | 0.647 | 0.829 | 0.53 |

* **Logit Model with Balanced Dataset, Test/Train Split**:

| Heart Disease(Y/N) | F1 Score | Precision | Recall |
| ------ | -------- | --------- | ------ |
| N | 0.75 | 0.77 | 0.74 |
| Y | 0.76 | 0.75 | 0.77 |
| Weighted Average | 0.76 | 0.76 | 0.76 |

*AUC : 0.831*

* **Support Vector Machine (SVM) Model with Balanced Dataset, Test/Train Split**:

|    | F1 Score | Accuracy | Precision | Recall |
| ------ | -------- | --------- | --------- | ------ |
| No split | 0.759 | 0.753 | 0.74 | 0.78 |
| Train/Test Split | 0.753 | 0.745 | 0.775 | 0.733 |


### Conlcusion:
   Looking back at our SMART questions, we aimed to determine if the key risk factors listed by the CDC were good predictors for heart disease and also explore if there are other, less prevalent factors that were also useful predictors. Given that our dataset only included information on smoking and not blood pressure levels or high cholesterol, we knew that we would not be able to answer this entirely. What we can look at however, are factors related to high blood pressure and high cholesterol that we do have data on in terms of their predictive power for risk of heart disease.
	  While smoking did not turn out to be as important as we first hypothesized, it was still statistically significant in our models. Factors such as if a respondent has had a stroke, if they are male, and if they had previous health problems like kidney disease or difficulty walking turned out to be the most important features in our models. Overall, this makes sense; less healthy individuals or those that have pre-existing health and heart conditions are more likely to be at risk for heart disease.
	  While both the logistic regression and the SVM models performed similarly, we concluded that the logistic regression model was likely more useful in real world applications. It gives a predicted probability of a respondent having heart disease, which allows for more flexibility when determining if a patient should get a check up with their physician.

