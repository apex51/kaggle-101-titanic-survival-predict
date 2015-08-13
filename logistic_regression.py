'''
logistic regression
random forest
extra trees
'''


import numpy as np
import pandas as pd
from sklearn import preprocessing
from patsy import dmatrix
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Read train and test data.
# Merge train dataframe and test dataframe.
df_train = pd.read_csv('train.csv',header=0)
df_test = pd.read_csv('test.csv', header=0)
df = pd.concat([df_train,df_test])
df.reset_index(inplace=True)
df.drop('index', axis=1, inplace=True)

# Map 0 fare into nan.
# Fill nan with $ due to Pclass median
df['Fare'] = df['Fare'].apply(lambda x: np.nan if x==0 else x)
df.loc[(df.Fare.isnull())&(df.Pclass==1),'Fare'] = np.median(df.loc[df.Pclass==1,'Fare'].dropna())
df.loc[(df.Fare.isnull())&(df.Pclass==2),'Fare'] = np.median(df.loc[df.Pclass==2,'Fare'].dropna())
df.loc[(df.Fare.isnull())&(df.Pclass==3),'Fare'] = np.median(df.loc[df.Pclass==3,'Fare'].dropna())

# Extract title from name.
# Map titles into title categories(age based).
df['Title'] = df['Name'].apply(lambda x: x.split(', ')[1].split('. ')[0])
df['Title'][df.Title.isin(['Mr','Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col', 'Sir'])] = 'Mr'
df['Title'][df.Title.isin(['the Countess', 'Mme','Mrs','Dona'])] = 'Mrs'
df['Title'][df.Title.isin(['Mlle', 'Ms','Miss','Lady'])] = 'Miss'
df.loc[(df.Title=='Dr')&(df.Sex=='male'),'Title'] = 'Mr'
df.loc[(df.Title=='Dr')&(df.Sex=='female'),'Title'] = 'Mrs'

# Transform Parch and Sibsp into FamilySize.
df['FamilySize'] = df['Parch'] + df['SibSp'] + 1

# Fill the empty Ages in AgeFill
df['AgeFill'] = df['Age']
age_mean = np.zeros(4)
age_mean[0] = np.average(df.loc[df.Title=='Master','Age'].dropna())
age_mean[1] = np.average(df.loc[df.Title=='Mr','Age'].dropna())
age_mean[2] = np.average(df.loc[df.Title=='Mrs','Age'].dropna())
age_mean[3] = np.average(df.loc[df.Title=='Miss','Age'].dropna())
df.loc[(df.Title=='Master')&(df.Age.isnull()),'AgeFill'] = age_mean[0]
df.loc[(df.Title=='Mr')&(df.Age.isnull()),'AgeFill'] = age_mean[1]
df.loc[(df.Title=='Mrs')&(df.Age.isnull()),'AgeFill'] = age_mean[2]
df.loc[(df.Title=='Miss')&(df.Age.isnull()),'AgeFill'] = age_mean[3]

# Categorize the age
df['AgeCat'] = df['AgeFill']
df.loc[(df.AgeFill<=10),'AgeCat'] = 'Child'
df.loc[(df.AgeFill>10)&(df.AgeFill<=30),'AgeCat'] = 'Adult'
df.loc[(df.AgeFill>30)&(df.AgeFill<=60),'AgeCat'] = 'Senior'
df.loc[(df.AgeFill>60),'AgeCat'] = 'Aged'

# Fill the nan Embarked with 'S'
df.loc[df['Embarked'].isnull(),'Embarked'] = 'S'

# The nan Cabin may be a signal of unsurvival. We use a CabinFlag to flag.
df['CabinFlag'] = df['Cabin']
df.loc[df['Cabin'].isnull(),'CabinFlag'] = 0.5
df.loc[df['Cabin'].notnull(),'CabinFlag'] = 1.5

# The ticket is bought in family, so we devide it into fare per person.
df['FarePerPerson'] = df['Fare']/df['FamilySize']

# Combination: product of two feature.
df['AgeClass'] = df['AgeFill'] * df['Pclass']
df['FareClass'] = df['FarePerPerson'] * df['Pclass']

# Devide Stage into high(1) and low(0), using the quartile.
df['Stage'] = df['FarePerPerson']
df.loc[df['FarePerPerson']<=25.9, 'Stage'] = 0
df.loc[df['FarePerPerson']>25.9, 'Stage'] = 1

# Encode labels.
le = preprocessing.LabelEncoder()

le.fit(df['Embarked'])
v_embarked = le.transform(df['Embarked'])
df['Embarked'] = v_embarked.astype(float)

le.fit(df['Sex'])
v_sex = le.transform(df['Sex'])
df['Sex'] = v_sex.astype(float)

le.fit(df['Title'])
v_title = le.transform(df['Title'])
df['Title'] = v_title.astype(float)

le.fit(df['AgeCat'])
v_agecat = le.transform(df['AgeCat'])
df['AgeCat'] = v_agecat.astype(float)

formula_ml = 'Embarked+Pclass+Sex+Title+FamilySize+AgeFill+AgeCat+CabinFlag \
                +FarePerPerson+AgeClass+FareClass+Stage'

# formula_ml = 'Pclass+Sex+Title+FamilySize+AgeCat+FarePerPerson+Fare'

train_X = (dmatrix(formula_ml, data=df[0:891], return_type='dataframe')).values
test_X = (dmatrix(formula_ml, data=df[891::], return_type='dataframe')).values
train_y = (df[0:891][['Survived']].astype(int)).values.flatten()
test_idx = (df_test[['PassengerId']]).values.flatten()


# ************************************************************************************************************
# Random Forest Classifier
#
# clf=RandomForestClassifier(n_estimators=2000, criterion='entropy', max_depth=5, min_samples_split=1,
#   min_samples_leaf=1, max_features='auto',    bootstrap=False, oob_score=False, verbose=0, min_density=None, compute_importances=None)
# pipeline=Pipeline([ ('clf',clf) ])
# param_grid = {'clf__n_estimators':[500, 1000, 1500, 2000],
#     'clf__criterion':['entropy', 'gini'],
#     'clf__bootstrap':[True, False],
#     'clf__max_depth':[3, 4, 5, 6, 7, 8],
#     }

# grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=3,scoring='accuracy',\
# cv=5).fit(train_X, train_y)
# print("Best score: %0.3f" % grid_search.best_score_)
# print(grid_search.best_estimator_)

# pre_feat = grid_search.best_estimator_.predict(test_X)

# out_df = {'PassengerId':test_idx, 'Survived':pre_feat}
# pd.DataFrame(out_df).to_csv('submission.csv', index=False)


# ************************************************************************************************************
# Extremely Randomized Trees Classifier
#
# clf=ExtraTreesClassifier(n_estimators=2000, criterion='entropy', max_depth=10, min_samples_split=1,
#   min_samples_leaf=1, max_features='auto',    bootstrap=False, oob_score=False, verbose=0, min_density=None, compute_importances=None)
# param_grid = dict( )
# pipeline=Pipeline([ ('clf',clf) ])
# grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=3,scoring='accuracy',\
# cv=StratifiedShuffleSplit(train_y, n_iter=10, test_size=0.2, train_size=None, indices=None, \
#  n_iterations=None)).fit(train_X, train_y)
# print("Best score: %0.3f" % grid_search.best_score_)
# print(grid_search.best_estimator_)

# pre_feat = grid_search.best_estimator_.predict(test_X)

# out_df = {'PassengerId':test_idx, 'Survived':pre_feat}
# pd.DataFrame(out_df).to_csv('submission.csv', index=False)


# ************************************************************************************************************
# Logit Regressor 804713804714
#
model = Pipeline(steps=[
    ('scaler', preprocessing.MinMaxScaler(feature_range=(0, 1))),
    # ('scaler', preprocessing.StandardScaler()),
    ('classifier', LogisticRegression(penalty='l1')),
])
tuned_parameters = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
}
model = GridSearchCV(model, tuned_parameters, cv=10, verbose=3)
model = model.fit(train_X, train_y)

print 'BEST SCORE: {}'.format(model.best_score_)
print 'BEST PARAMETERS: ' + str(model.best_params_)

test_y = model.predict(test_X).astype(int)
out_d = {'PassengerId':test_idx, 'Survived':test_y}
pd.DataFrame(out_d).to_csv('submission.csv', index=False)