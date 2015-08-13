# Kaggle-101: Titanic Survival Prediction

*finished on 20th, May 2015*

This is the code for Kaggle's 101 game [Titanic Survival Prediction](https://www.kaggle.com/c/titanic). I used logistic regression model with some simple feature engineering. The final score got 0.81340 which ranked 169/3084 at my final submission.

## Feature Engineering

#### Missing Age

Missing age seldom appears in a survivor's data, so adding a binary feature "AgeIsNa" is necessary. The missing ages are then filled with random values between `age_min` and `age_max`. 

I also tried to fill missing ages using a random forest but in vain.

#### Name Title

Name title may reveal a person's information. This may help decide whether he/she has the chance to get into lifeboat. Using a categorical variable "Title" to record this.

#### Ticket Price Distribution

The price's distribution is very skewed. Using `Log(fare)` can alleviate the spike near low ticket fare, making it fit well to a linear model.

#### "Parch" + "SibSp"

When "Parch" (parent and child number) + "SibSp" (sibling and spouse number) equals 1,2 or 3, a person tend more likely to survive. So a variable "FamilySize" helps.

#### Categorize the Age

Make the continuous variables into discrete ones can help logistic models predict well. This information is recorded in variable `AgeCat`.

#### Dummy Variables

The magnitude in a categorical variable like "Title" is meaningless, so dumminize it.

## Model Comparison

Logistic Regression: 0.81340
`{'penalty': 'L1', 'classifier__C': 10, 'classifier__tol': 0.001}`

Support Vector Classifier: 0.79349
`{'kernel': 'linear', 'C': 1, 'tol': 0.0001}`

Random Forest Classifier: 0.78469
`{'n_estimators': 200, 'n_jobs': '-1'}`