# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import ExtraTreesClassifier as ETC
from sklearn.ensemble import RandomForestClassifier as RDF 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score
# -

# ## Pre processing
# Based on EDA (see notebook:) data processing will be done below

# +
# import data
data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')

# Function to obtain information of null data (missing data)
def null_info(df):
    num_null_val = df.isnull().sum()
    p_null_val = 100*num_null_val/len(df)
    null_info = pd.concat([num_null_val, p_null_val], axis=1)
    null_info = null_info.rename(columns = {0: 'Counts of null', 1:'%'})
    return null_info

data_train['Sex'] = data_train['Sex'].map({"male":0, "female":1})
data_test['Sex'] = data_test['Sex'].map({"male":0, "female":1})

data_train["Age"] = data_train["Age"].fillna(data_train["Age"].median())
data_test["Age"] = data_test["Age"].fillna(data_test["Age"].median())

data_train["Embarked"] = data_train["Embarked"].fillna("S")
data_test["Embarked"] = data_test["Embarked"].fillna("S")
# Repalce strings in "Embarked" to 0 (S), 1 (C), 2 (Q)
data_train["Embarked"] = data_train["Embarked"].map({"S":0, "C":1, "Q":2})
data_test["Embarked"] = data_test["Embarked"].map({"S":0, "C":1, "Q":2})

data_test["Fare"] = data_test["Fare"].fillna(data_test["Fare"].median())


# -

PassengerId_test = np.array(data_test['PassengerId']).astype(int)

# ## Models

colmuns_train = ['Pclass','Sex','Age','Parch']
df_test = data_test[colmuns_train]

# ### Data for cross-validation

# data_train.columns
X = data_train[colmun_train]
y = data_train[['Survived']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# ### Linear Regression

X.columns

# #### Cross-validation

lm = LinearRegression()

lm.fit(X_train, y_train)

print(lm.intercept_)

lm.coef_

cdf = pd.DataFrame(lm.coef_,index=['Coeff'], columns=X.columns)

cdf

prediction = lm.predict(X_test)
res_predict = np.where(prediction>0.5, 1, 0)
accuracy_score(y_test, res_predict)

plt.scatter(y_test, prediction)

sns.distplot((y_test-prediction))

prediction_linR = lm.predict(df_test)
myPrediction_linR = pd.DataFrame(prediction_linR, PassengerId_test, columns=['Survived'])
myPrediction_linR['Survived'] = np.where(prediction_linR > 0.5, 1, 0)
myPrediction_linR.to_csv("results/Titianic_linR_model.csv", index_label = ["PassengerId"])b

myPrediction_linR

# ### Decision tree

clf_DTC = DTC(max_depth=4)

clf_DTC.fit(X_train,y_train)

y_pred_test = clf_DTC.predict(X_test)
accuracy_score(y_test, y_pred_test)

# ### Logistic Regression

logi_regg = LogisticRegression()
clf_LogiReg = logi_regg.fit(X_train,y_train)

clf_LogiReg.coef_

clf_LogiReg.intercept_

y_pred_LogiReg = clf_LogiReg.predict(X_test)
accuracy_score(y_test, y_pred_LogiReg)




