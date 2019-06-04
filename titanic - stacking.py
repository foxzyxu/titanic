import pandas as pd
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from xgboost import XGBClassifier

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
#向量化,标准化
train_data.loc[ train_data['Fare'] <= 7.91, 'Fare']  = 0
train_data.loc[(train_data['Fare'] > 7.91) & (train_data['Fare'] <= 14.454), 'Fare'] = 1
train_data.loc[(train_data['Fare'] > 14.454) & (train_data['Fare'] <= 31), 'Fare'] = 2
train_data.loc[ train_data['Fare'] > 31, 'Fare'] = 3
train_data['Fare'] = train_data['Fare'].astype(int)
train_data.loc[ train_data['Age'] <= 16, 'Age'] = 0
train_data.loc[(train_data['Age'] > 16) & (train_data['Age'] <= 30), 'Age'] = 1
train_data.loc[(train_data['Age'] > 30) & (train_data['Age'] <= 48), 'Age'] = 2
train_data.loc[(train_data['Age'] > 48) & (train_data['Age'] <= 64), 'Age'] = 3
train_data.loc[ train_data['Age'] > 64, 'Age'] = 4
train_data.Cabin[train_data.Cabin.notnull()] = 1
train_data.Cabin[train_data.Cabin.isnull()] = 0
train_data.loc[train_data["Sex"] == "male","Sex"] = 0
train_data.loc[train_data["Sex"] == "female","Sex"] = 1
train_data['Sex'] = train_data['Sex'].astype(str)
train_data.drop(['Name'], axis=1,inplace=True)
train_data.drop(['Ticket'], axis=1,inplace=True)
train_data.drop(['PassengerId'], axis=1,inplace=True)
train_data = pd.get_dummies(train_data)
Y_train = train_data['Survived'].values
train_data.drop(['Survived'], axis=1,inplace=True)
X_train = train_data.values
#print(train_data.info())

#测试集处理
X_test = pd.read_csv('test.csv')
#print(X_test.info())
X_test.Embarked[X_test.Embarked.isnull()] = X_test.Embarked.dropna().mode().values
X_test.Age[X_test.Age.isnull()] = X_test.Age.dropna().mean()
X_test.Fare[X_test.Fare.isnull()] = X_test.Fare.dropna().mean()
X_test.loc[X_test['Fare'] <= 7.91, 'Fare']  = 0
X_test.loc[(X_test['Fare'] > 7.91) & (X_test['Fare'] <= 14.454), 'Fare'] = 1
X_test.loc[(X_test['Fare'] > 14.454) & (X_test['Fare'] <= 31), 'Fare'] = 2
X_test.loc[X_test['Fare'] > 31, 'Fare'] = 3
X_test['Fare'] = X_test['Fare'].astype(int)
X_test.loc[X_test['Age'] <= 16, 'Age'] = 0
X_test.loc[(X_test['Age'] > 16) & (X_test['Age'] <= 30), 'Age'] = 1
X_test.loc[(X_test['Age'] > 30) & (X_test['Age'] <= 48), 'Age'] = 2
X_test.loc[(X_test['Age'] > 48) & (X_test['Age'] <= 64), 'Age'] = 3
X_test.loc[X_test['Age'] > 64, 'Age'] = 4
X_test.Cabin[X_test.Cabin.notnull()] = 1
X_test.Cabin[X_test.Cabin.isnull()] = 0
X_test.loc[X_test["Sex"] == "male","Sex"] = 0
X_test.loc[X_test["Sex"] == "female","Sex"] = 1
X_test['Sex'] = X_test['Sex'].astype(str)
X_test.drop(['Name'], axis=1,inplace=True)
X_test.drop(['Ticket'], axis=1,inplace=True)
Submissions_Id = X_test['PassengerId']#用于后续提交数据
X_test.drop(['PassengerId'], axis=1,inplace=True)
X_test = pd.get_dummies(X_test)
X_test = X_test.values

#stacking
ntrain = train_data.shape[0]
ntest = X_test.shape[0]
kf = KFold(ntrain, n_folds= 5, random_state=55)
#5折训练后得到的数据
def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((5, ntest))
    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]
        clf.fit(x_tr, y_tr)
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
#1级训练器
rf = RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=500,
                            warm_start=True, max_depth=6, min_samples_leaf=2,
                            max_features = 'sqrt', verbose=0)
et = ExtraTreesClassifier(random_state=0, n_jobs=-1, n_estimators=500, 
                          max_depth=8, min_samples_leaf=2, verbose=0)
ada = AdaBoostClassifier(random_state=0, n_estimators=500, learning_rate=0.75)
gb = GradientBoostingClassifier(random_state=0, n_estimators=500, max_depth=5, 
                                min_samples_leaf=2, verbose=0)
svc = SVC(random_state=0, kernel='linear', C=0.025)
#得到一级训练器的数据
et_oof_train, et_oof_test = get_oof(et, X_train, Y_train, X_test)
rf_oof_train, rf_oof_test = get_oof(rf,X_train, Y_train, X_test)
ada_oof_train, ada_oof_test = get_oof(ada, X_train, Y_train, X_test)
gb_oof_train, gb_oof_test = get_oof(gb,X_train, Y_train, X_test) 
svc_oof_train, svc_oof_test = get_oof(svc,X_train, Y_train, X_test)
#得到二级训练所用数据
X_2_train = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
X_2_test = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)
xgb = XGBClassifier(learning_rate = 0.02, n_estimators= 2000,max_depth= 4,
                        min_child_weight= 2,gamma=0.9,subsample=0.8,
                        colsample_bytree=0.8,objective= 'binary:logistic',
                        nthread= -1,scale_pos_weight=1).fit(X_2_train, Y_train)
Y_predicteds = xgb.predict(X_2_test)
Titanic_Submission = np.concatenate((Submissions_Id, Y_predicteds))
Titanic_Submission.to_csv('Titanic_Submission.csv', index = False)