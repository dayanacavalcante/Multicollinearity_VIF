# Imports
from logging import warning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn import linear_model as lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# Data
housing = pd.read_csv("Data/Housing.csv")
print(housing.head())

# Data Exploration

print(housing.info())
print(housing.shape)
print(housing.describe())
print(housing.columns)

# Fast way to separate numeric columns
print(housing.describe().columns)

# For Numeric Data:
## Histograms to understand distributions
## Corrplot
## Pivot Table comparing price across numeric variables

# For Categorical Data:
## Bar charts to understand balance of classes
## Pivot Tables to understand relationship with price

# Look at numeric and categorical variables separately
var_num = housing[['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
var_cat = housing[['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']]

# Distributions for Numeric variables

for i in var_num.columns:
    plt.hist(var_num[i])
    plt.title(i)
    plt.show()

print(var_num.corr())
sns.heatmap(var_num.corr())

# Compare price across area, bedrooms, bathrooms, stories, parking
print(pd.pivot_table(housing, index='price', values=['area', 'bedrooms', 'bathrooms', 'stories', 'parking']))

# Categorical variables
for i in var_cat.columns:
    plt.figure(figsize=(8,6))
    sns.barplot(var_cat[i].value_counts().index, var_cat[i].value_counts(), palette="Blues_d").set_title(i)
    plt.show()

# Binary variables
varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

def binary_map(i):
    return i.map({"yes":1, "no":0})

housing[varlist] = housing[varlist].apply(binary_map)

print(housing.head())

# Dummy variable
status = pd.get_dummies(housing['furnishingstatus'])
print(status.head())

# Drop the original column and concat the new columns
housing = pd.concat([housing,status], axis=1)
housing.drop(['furnishingstatus'],axis=1,inplace=True)
print(housing.head())

# Compare price across categorical variables
print(pd.pivot_table(housing, index='price', values=['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnished', 'semi-furnished', 'unfurnished']))

# Apply scaler to all numeric variables
scaler = MinMaxScaler()
num_vars = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
housing[num_vars] = scaler.fit_transform(housing[num_vars])
print(housing.head())

# Split the variables between predictors and target
X = housing.drop('price', axis=1)
y = housing['price']

# Create the train and the test set
np.random.seed(0) # For the result to be the same
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,random_state=100)

print(X_train.shape)

# Train Model
# First version
# Fit the linear model

X_train_lm = sm.add_constant(X_train)
lr_1 = sm.OLS(y_train, X_train_lm).fit()
print(lr_1.summary())

# VIF
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by="VIF", ascending=False)
print(vif)

# Correlation
fig, ax = plt.subplots(figsize=(15,12))
sns.heatmap(X_train.corr(), cmap="YlGnBu", annot=True, linewidths=.5,ax=ax)

# Second version
X_train_sec = X_train.drop(['semi-furnished', 'furnished', 'bedrooms'], axis=1)

# Fit the linear model
X_train_lm = sm.add_constant(X_train_sec)
lr_2 = sm.OLS(y_train, X_train_lm).fit()
print(lr_2.summary())

# VIF

vif = pd.DataFrame()
vif['Features'] = X_train_sec.columns
vif['VIF'] = [variance_inflation_factor(X_train_sec.values, i) for i in range(X_train_sec.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
print(vif)

# Correlation
fig, ax = plt.subplots(figsize=(15,12))         
sns.heatmap(X_train_sec.corr(), cmap="YlGnBu", annot = True, linewidths=.5, ax=ax)

# Predict
# Test set
X_test = X_test.drop(['semi-furnished', 'furnished', 'bedrooms'], axis=1)

# Model
model = lm.LinearRegression()
model.fit(X_test, y_test)
y_pred = model.predict(X_test)
print('Predict: {}'.format(y_pred))

#R2
r2 = r2_score(y_test,y_pred)
print('R2: {}'.format(r2))