'''
Logistic Regression
'''
import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation
from random import randint
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

X_COLUMNS = ['Pclass', 'Title', 'Sex', 'Age', 'AgeIsNa', 'Fare', 'FareIsZero', 'Embarked']
X_CATEGORICAL_COLUMNS = ['Pclass','Title', 'Sex', 'AgeIsNa','FareIsZero', 'Embarked']
# ****************************************************************
# Read the train csv
# ****************************************************************

# Data Cleansing
# Read the training data from csv.
train_df = pd.read_csv('train.csv', header = 0)
test_df = pd.read_csv('test.csv', header = 0)
test_passengerid = test_df['PassengerId']
df = pd.concat([train_df, test_df])
df.reset_index(inplace=True)
df.drop('index', axis=1, inplace=True)

# Map male and female into 0,1 with Series name 'Gender'.
df['Sex'] = df['Sex'].map( {'female': 1, 'male': 2} ).astype(int)

# Fill the missing Embarked with the most common value.
df.loc[df['Embarked'].isnull(), 'Embarked'] = df['Embarked'].dropna().mode().values

# Map embarked into 0,1,2.
port_dict = list(enumerate(np.unique(df['Embarked'])))
df['Embarked'] = df['Embarked'].map( {name:i for i,name in port_dict} ).astype(int)

# # Fill the missing Age with the median age.
# age_median = train_df['Age'].dropna().median()
# train_df.loc[train_df['Age'].isnull(), 'Age'] = age_median
# Add a new column called AgeIsNa
# Fill the missing data with a random int
# Change the float to int
age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
df['AgeIsNa'] = df['Age'].apply(lambda x: 1 if np.isnan(x) else 0)
df['Age'] = df['Age'].apply(lambda x: randint(age_min, age_max) if np.isnan(x) else x)
df['Age'] = df['Age'].apply(lambda x: int(x)+1)

# Fill the nan value with a random
# Change 0 into a random
# Log the value to make it linear
# Add a new column called FareIsZero
df['FareIsZero'] = df['Fare'].apply(lambda x: 1 if (np.isnan(x) | (x == 0)) else 0)
fare_min, fare_max = int(df['Fare'].min()), int(df['Fare'].max())
df['Fare'] = df['Fare'].apply(lambda x: randint(fare_min, fare_max) if np.isnan(x) else x)
df['Fare'] = df['Fare'].apply(lambda x: randint(fare_min, fare_max) if x == 0 else x)
df['Fare'] = df['Fare'].apply(lambda x: np.log(x))

# Add a column called title
# Change the titles into enumerate numbers
df['Title'] = df['Name'].apply(lambda x: x.split(', ')[1].split('. ')[0])
df['Title'] = df[['Title']].apply(lambda x: pd.factorize(x, na_sentinel = -1)[0])

# Add a column called Relatives
df['RelativeNum'] = df['SibSp'] + df['Parch'] + 1

# Remove irrelevant columns.
df.drop([x for x in df.columns.values if x not in X_COLUMNS + ['Survived']], axis=1, inplace=True)
df = df.reindex_axis(X_COLUMNS + ['Survived'], axis = 1)

# ****************************************************************
# Training
# ****************************************************************

# train_X = df[X_COLUMNS][0:891].values
X = df[X_COLUMNS].values
train_y = df['Survived'][0:891].values
# test_X = df[X_COLUMNS][891:].values

categorical_indices = [i for i in range(len(X_COLUMNS)) if X_COLUMNS[i] in X_CATEGORICAL_COLUMNS]
# model = Pipeline([
#     ('onehot', OneHotEncoder(categorical_features=categorical_indices, sparse=False, n_values=39)),
#     ('scaler', MinMaxScaler(feature_range=(0, 1))),
#     ('classifier', LogisticRegression(penalty='l1')),
# ])

# Preprocessing for training data
# print 'enc and scale training data'
enc = OneHotEncoder(categorical_features=categorical_indices, sparse=False)
enc.fit(X)
X_enc = enc.transform(X)

scaler = MinMaxScaler()
scaler.fit(X_enc)
X_scaler=scaler.transform(X_enc)

train_X = X_scaler[0:891]
test_X = X_scaler[891:]

# print 'cross validating'
clf = LogisticRegression()
# score = cross_validation.cross_val_score(model, X=train_X_scaler, y=train_y, cv=10)

tuned_parameters = {
    'C': [0.01, 0.1, 1, 10, 100],
    'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    'penalty':['L1', 'L2']
}

grid = GridSearchCV(clf, tuned_parameters, cv=5, verbose=3)
grid.fit(train_X, train_y)

print 'BEST SCORE: {}'.format(grid.best_score_)
print 'BEST PARAMETERS: ' + str(grid.best_params_)


model = grid.best_estimator_
model = model.fit(train_X, train_y)

# print 'Training...'
# model = RandomForestClassifier(n_estimators=100)
# scores = cross_validation.cross_val_score(model, train_X, train_y, cv=5)
# print scores.mean()
# model.fit(train_X, train_y)

# print 'score is: {}'.format(score)
# print 'score.mean is: {}'.format(score.mean())

print 'Predicting...'
test_y = model.predict(test_X).astype(int)

out_d = {'PassengerId':test_passengerid.values, 'Survived':test_y}
out_df = pd.DataFrame(out_d)
out_df.to_csv('submission.csv', index=False)

print 'Done.'