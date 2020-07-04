import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets, linear_model, metrics 
from sklearn.model_selection import train_test_split 



trainData = pd.read_csv("train.csv")
trainData = pd.DataFrame(trainData).drop(["Id"], 1)
# trainData = pd.DataFrame(trainData).drop(["Alley"], 1)
# trainData = pd.DataFrame(trainData).drop(["MasVnrType"], 1)
# trainData = pd.DataFrame(trainData).drop(["MasVnrType"], 1)
y = np.array(trainData.SalePrice)
trainData = pd.DataFrame(trainData).drop(["SalePrice"], 1)
dfTrainData = pd.DataFrame(trainData)
# print(trainData["Alley"].dtypes)
dfTrainData["Alley"].fillna('empty', inplace=True)
dfTrainData["MasVnrType"].fillna('empty', inplace=True)
dfTrainData["BsmtQual"].fillna('empty', inplace=True)
dfTrainData["BsmtCond"].fillna('empty', inplace=True)
dfTrainData["BsmtExposure"].fillna('empty', inplace=True)
dfTrainData["BsmtFinType1"].fillna('empty', inplace=True)
dfTrainData["BsmtFinType2"].fillna('empty', inplace=True)
dfTrainData["Electrical"].fillna('empty', inplace=True)
dfTrainData["FireplaceQu"].fillna('empty', inplace=True)
dfTrainData["GarageType"].fillna('empty', inplace=True)
dfTrainData["GarageFinish"].fillna('empty', inplace=True)
dfTrainData["GarageQual"].fillna('empty', inplace=True)
dfTrainData["GarageCond"].fillna('empty', inplace=True)
dfTrainData["PoolQC"].fillna('empty', inplace=True)
dfTrainData["Fence"].fillna('empty', inplace=True)
dfTrainData["MiscFeature"].fillna('empty', inplace=True)


# print(trainData["MasVnrType"])
for (colName, colVal) in dfTrainData.iteritems() :
    
    if (dfTrainData[colName].dtypes == "int64" or dfTrainData[colName].dtypes =="float64") :
        dfTrainData.fillna(0, inplace=True)
    else:
        dfTrainData[colName].fillna('empty', inplace=True)

for (colName, colVal) in dfTrainData.iteritems() :
    if (dfTrainData[colName].dtypes == "object") :
        print(colName)
        e = LabelEncoder()
        dfTrainData[colName] = e.fit_transform(dfTrainData[colName])
    
X = dfTrainData.to_numpy()
ml = linear_model.LinearRegression() 
ml.fit(X, y) 

testingData = pd.read_csv("test.csv")
testingData = pd.DataFrame(testingData).drop(["Id"], 1)
dfTestData = pd.DataFrame(testingData)
dfTestData["MSZoning"].fillna('empty', inplace=True)
dfTestData["Alley"].fillna('empty', inplace=True)
dfTestData["Utilities"].fillna('empty', inplace=True)
dfTestData["Exterior1st"].fillna('empty', inplace=True)
dfTestData["Exterior2nd"].fillna('empty', inplace=True)
dfTestData["MasVnrType"].fillna('empty', inplace=True)
dfTestData["BsmtQual"].fillna('empty', inplace=True)
dfTestData["BsmtCond"].fillna('empty', inplace=True)
dfTestData["BsmtExposure"].fillna('empty', inplace=True)
dfTestData["BsmtFinType1"].fillna('empty', inplace=True)
dfTestData["BsmtFinType2"].fillna('empty', inplace=True)
dfTestData["Electrical"].fillna('empty', inplace=True)
dfTestData["KitchenQual"].fillna('empty', inplace=True)
dfTestData["Functional"].fillna('empty', inplace=True)
dfTestData["FireplaceQu"].fillna('empty', inplace=True)
dfTestData["GarageType"].fillna('empty', inplace=True)
dfTestData["GarageFinish"].fillna('empty', inplace=True)
dfTestData["GarageQual"].fillna('empty', inplace=True)
dfTestData["GarageCond"].fillna('empty', inplace=True)
dfTestData["PoolQC"].fillna('empty', inplace=True)
dfTestData["Fence"].fillna('empty', inplace=True)
dfTestData["MiscFeature"].fillna('empty', inplace=True)
dfTestData["SaleType"].fillna('empty', inplace=True)
for (colName, colVal) in dfTestData.iteritems() :
    
    if (dfTestData[colName].dtypes == "int64" or dfTestData[colName].dtypes =="float64") :
        dfTestData.fillna(0, inplace=True)
    else:
        dfTestData[colName].fillna('empty', inplace=True)

for (colName, colVal) in dfTestData.iteritems() :
    if (dfTestData[colName].dtypes == "object") :
        print("Test" , colName)
        e = LabelEncoder()
        dfTestData[colName] = e.fit_transform(dfTestData[colName])
X_test = dfTrainData.to_numpy()
y_test = ml.predict(X_test)

testingData = pd.read_csv("test.csv")
dfTestData = pd.DataFrame(testingData)

dfTestData["SalePrice"] = y_test[1:]
dfTestData.to_csv("pred.csv", index=False)