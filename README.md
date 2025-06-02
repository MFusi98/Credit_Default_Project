# Credit_Default_Project

This project aims to predict whether a client will default on credit payments using machine learning models, and to evaluate model performance not only with standard metrics but also with cost-based decision-making.

## About the Data 

The dataset, available on [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data), consists of 150000 observations and 12 features related to clients’ financial profiles, including income, age, debt usage, and historical payment behavior.

Before training the models, we carried out a comprehensive preprocessing phase. We addressed missing values, particularly in the MonthlyIncome and NumberOfDependents variables. For income, we adopted a strategy 
based on age groups: we created an AgeGroup variable that divides the population into four bands (18–30, 31–45, 46–65, and 65+), and then used the median income within each group to fill the missing values. 
Anomalous entries, such as clients with an age of zero, were corrected by replacing them with the mean age (52).

We also handled extreme and unrealistic values in the DebtRatio feature, which in some cases exceeded 1. Since a value above 1 would imply a person is using more than 100% of their income for debt, a likely data 
error, we capped this variable at 1 to keep the values consistent and interpretable.

Finally, we engineered new variables to improve model performance, including a log transformation of MonthlyIncome (creating MonthlyIncome_log) and the capped version of DebtRatio (DebtRatio_capped). These 
adjustments helped stabilize variance, reduce the impact of outliers, and prepare the data for reliable model training.

## About the Models 

To predict which clients are likely to default, we implemented three different machine learning models. We started with logistic regression as a benchmark due to its simplicity and interpretability, and then 
introduced two tree-based models, Random Forest and XGBoost, to capture potential non-linear relationships and improve predictive performance.

In addition to training the models, we explored the effect of different classification thresholds. While 0.50 is the standard threshold, we also evaluated performance at 0.40, which in our case provided more 
balanced results in terms of false positives and false negatives.

Model performance was assessed using a variety of metrics, including precision, recall, ROC AUC, and the confusion matrix for both thresholds. These evaluations helped us understand not only the accuracy of each
model, but also how they behaved in identifying true defaulters versus non-defaulters.

Finally, we introduced a cost-based evaluation framework, assigning a hypothetical financial cost to misclassifications. In particular, we assumed that failing to identify a defaulter (false negative) is 
significantly more costly than misclassifying a good client as a defaulter (false positive). This allowed us to select the model and threshold that minimized total expected cost, aligning the analysis with real-
world business considerations.
