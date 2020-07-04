import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder


trainData = pd.read_csv("train.csv")
dfTrainData = pd.DataFrame(trainData)
dfTrainData.drop(["PassengerId", "Name", "Ticket", "Cabin"],1, inplace=True)
y = dfTrainData["Survived"].values
dfTrainData.drop(["Survived"],1, inplace=True)
dfTrainData["Age"].fillna(dfTrainData["Age"].median(), inplace=True)
dfTrainData["Sex"] = LabelEncoder().fit_transform(dfTrainData["Sex"])
dfTrainData = pd.get_dummies(dfTrainData, columns=["Embarked"])
ml = LogisticRegression()
X = dfTrainData.values
X = preprocessing.scale(X)
ml.fit(X, y)
testingData = pd.read_csv("test.csv")
dfTestData = pd.DataFrame(testingData)
dfTestData.drop(["PassengerId", "Name", "Ticket", "Cabin"],1, inplace=True)
dfTestData["Age"].fillna(dfTestData["Age"].median(), inplace=True)
dfTestData["Sex"] = LabelEncoder().fit_transform(dfTestData["Sex"])
dfTestData = pd.get_dummies(dfTestData, columns=["Embarked"])
dfTestData["Fare"].fillna(dfTestData["Fare"].mean(), inplace=True)
X_test =  preprocessing.scale(dfTestData.values)
y_test = ml.predict(X_test)

testingData = pd.read_csv("test.csv")
dfTestData = pd.DataFrame(testingData)

dfTestData["Survived"] = y_test
dfTestData.to_csv("pred.csv", index=False)


