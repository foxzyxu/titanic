import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#读取数据
train_data = pd.read_csv('train.csv')
#print(train_data.info())
#对缺失值赋均值或者众数
train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values
train_data_survived0 = train_data[train_data.Survived.isin([0])]
train_data_survived1 = train_data[train_data.Survived.isin([1])]
train_data_survived0.Age[train_data.Age.isnull()] = train_data_survived0.Age.dropna().mean()
train_data_survived1.Age[train_data.Age.isnull()] = train_data_survived1.Age.dropna().mean()
train_data = pd.concat([train_data_survived0, train_data_survived1])
#向量化属性
train_data.Cabin[train_data.Cabin.notnull()] = 1
train_data.Cabin[train_data.Cabin.isnull()] = 0
embark_dummies  = pd.get_dummies(train_data['Embarked'])
train_data = train_data.join(embark_dummies)
train_data.drop(['Embarked'], axis=1,inplace=True)
train_data.loc[train_data["Sex"] == "male","Sex"] = 0
train_data.loc[train_data["Sex"] == "female","Sex"] = 1
train_data.drop(['Name'], axis=1,inplace=True)
train_data.drop(['Ticket'], axis=1,inplace=True)
train_data.drop(['PassengerId'], axis=1,inplace=True)
#print(train_data.info())

#随机森林
train_data_featureX = train_data.iloc[:,1:]
train_data_labelY = train_data['Survived'].values
X_train, X_test, y_train, y_test = train_test_split(train_data_featureX, train_data_labelY, test_size=0.2)
random_forest = RandomForestClassifier(n_estimators = 100, criterion = 'gini', max_features = 3, min_samples_leaf = 3, min_samples_split = 3)
random_forest = random_forest.fit(X_train, y_train)
#print("train score:", random_forest.score(X_train, y_train))
#print("test score:", random_forest.score(X_test, y_test))


#预测
test_data_featureX = pd.read_csv('test.csv')
#print(test_data_featureX.info())
test_data_gender = pd.read_csv('gender_submission.csv')
#对缺失值赋均值或者众数
test_data_featureX.Embarked[test_data_featureX.Embarked.isnull()] = test_data_featureX.Embarked.dropna().mode().values
test_data_featureX.Age[test_data_featureX.Age.isnull()] = test_data_featureX.Age.dropna().mean()
test_data_featureX.Fare[test_data_featureX.Fare.isnull()] = test_data_featureX.Fare.dropna().mean()
#向量化属性
test_data_featureX.Cabin[test_data_featureX.Cabin.notnull()] = 1
test_data_featureX.Cabin[test_data_featureX.Cabin.isnull()] = 0
embark_dummies  = pd.get_dummies(test_data_featureX['Embarked'])
test_data_featureX = test_data_featureX.join(embark_dummies)
test_data_featureX.loc[test_data_featureX["Sex"] == "male","Sex"] = 0
test_data_featureX.loc[test_data_featureX["Sex"] == "female","Sex"] = 1
test_data_featureX.drop(['Name','Ticket','PassengerId','Embarked'], axis=1,inplace=True)
test_data_labelY = test_data_gender['Survived'].values
#print(test_data_featureX.info())
#print(test_data_labelY.info())

#用测试集预测
print("gender score:", random_forest.score(test_data_featureX, test_data_labelY))

#输出预测结果
predictedY = random_forest.predict(test_data_featureX)
my_submission = np.column_stack((test_data_gender['PassengerId'], predictedY))
my_submission = pd.DataFrame(my_submission,columns = ['PassengerId','Survived'])
my_submission.to_csv('my_submission.csv', index = False)

