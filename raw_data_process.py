# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# # Kaggle Titanic Data - Raw Data Analysis

# ## Import required library and load data

import numpy as np
import pandas as pd
import pandas_profiling as pdp
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Input data: train.csv, test.csv  
# (Data column)  
# Sibsp: number of siblings/spouses aboard the Titanic  
# Parch: number of parents/children aboard the Titanic  
# embarked: Port of embarkation  
#     C = Cherbourg  
#     Q = Queenstown  
#     S = Southampton  
# Pclass: ticket class  
#     1 = upper  
#     2 = middle  
#     3 = low  

data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')

data_train.dtypes

data_test.dtypes

# ## Exploring raw data 

data_train.head()

data_test.head()

data_train.describe()

data_test.describe()

pdp.ProfileReport(data_train)


# Function to obtain information of null data (missing data)
def null_info(df):
    num_null_val = df.isnull().sum()
    p_null_val = 100*num_null_val/len(df)
    null_info = pd.concat([num_null_val, p_null_val], axis=1)
    null_info = null_info.rename(columns = {0: 'Counts of null', 1:'%'})
    return null_info


null_info(data_train)

null_info(data_test)

# ##  Data Cleaning
# Since ~80% of cells of cabin is NaN, cabin is discarded in training.

# ### Repalce sex to 0 (male) or 1 (female)  

# + code_folding=[]
# data_train = data_train.replace("male", 0).replace("female", 1)
# data_test = data_test.replace("male", 0).replace("female", 1)

data_train['Sex'] = data_train['Sex'].map({"male":0, "female":1})
data_test['Sex'] = data_test['Sex'].map({"male":0, "female":1})
data_train
# -

# ### Fill null data in Age with median value of age

data_train["Age"] = data_train["Age"].fillna(data_train["Age"].median())
data_test["Age"] = data_test["Age"].fillna(data_test["Age"].median())

# ### Fill null data in "Embarked"
# S is the most freqent embarked port and hence the missing cells are filled with S.

data_train["Embarked"] = data_train["Embarked"].fillna("S")
data_test["Embarked"] = data_test["Embarked"].fillna("S")
# Repalce strings in "Embarked" to 0 (S), 1 (C), 2 (Q)
data_train["Embarked"] = data_train["Embarked"].map({"S":0, "C":1, "Q":2})
data_test["Embarked"] = data_test["Embarked"].map({"S":0, "C":1, "Q":2})

data_train

null_info(data_train)

data_train.dtypes

data_test["Fare"] = data_test["Fare"].fillna(data_test["Fare"].median())
null_info(data_test)

pdp.ProfileReport(data_train)

# survival rate
data_train['Survived'].mean()

# Pclass
data_train['Survived'].groupby(data_train['Pclass']).mean()

sns.countplot(data_train['Pclass'], hue=data_train['Survived'])

# + code_folding=[]
columns = ['Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']

fig, axes = plt.subplots(len(columns), 1, figsize=(8, 20))
plt.subplots_adjust(hspace=0.3)

for column, ax in zip(columns, axes):
    sns.countplot(x='Survived', hue=column, data=data_train, ax=ax)
    ax.legend(loc='upper right')
    ax.set_title(column)
# -

data_train['bin_age'] = pd.cut(data_train['Age'],10)
pd.crosstab(data_train['bin_age'], data_train['Survived']).plot.bar(stacked=True)

data_train['bin_age'] = pd.cut(data_train['Age'],10)
pd.crosstab(data_train['bin_age'], data_train['Survived'], normalize='index').plot.bar(stacked=True)

sns.boxplot(x='Pclass', y='Age', data=data_train)

sns.pairplot(data_train)
