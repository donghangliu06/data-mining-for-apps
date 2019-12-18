import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import preprocessing
import xgboost as xgb
from sklearn.model_selection import cross_val_score

import os
os.environ["PATH"] += os.pathsep + 'D:/PythonU/Graphiz2.38/bin/'

df = pd.read_csv(r"E:\yanyi\DataMining\GooglePlayStore\model_Data.csv")
print(df.info())
feature_list = ['installs', 'reviews', 'size', 'content_rating', 'free', 'price', 'iap',
                'android_version', 'len_title', 'main_category', 'iap_min', 'iap_max',
                'updateDays']
X = df[feature_list].values
# X = df[['installs', 'reviews', 'size', 'free', 'price', 'iap', 'len_title',
#         'main_category', 'updateDays']].values

y = df['rating_class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf_rf = RandomForestClassifier(criterion='entropy', n_estimators=200, max_depth=9,
                                max_features=10, random_state=0)
# clf_rf = RandomForestClassifier(n_estimators=200, max_depth=9, random_state=0)

clf_rf.fit(X_train, y_train)
y_pred = clf_rf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(clf_rf.feature_importances_)
importances = list(clf_rf.feature_importances_)

feature_importances = [(feature, round(importance, 2))
                       for feature, importance in zip(feature_list, importances)]
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
for pair in feature_importances:
    print('Variable: {:20} Importance: {}'.format(*pair))

scores = cross_val_score(clf_rf, X, y, cv=10)
print(scores)
print("Accuracy: %0.4f" % scores.mean())
# print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

# =========================================================================
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
ddata = xgb.DMatrix(X, label=y)
param = {'max_depth': 3, 'eta': 1, 'objective': 'binary:logistic'}
param['eval_metric'] = 'auc'
evallist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 12
bst = xgb.train(param, dtrain, num_round, evallist)
y_pred = bst.predict(dtest)
y_pred = np.round(y_pred).astype(int)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

clf_bst = xgb.XGBClassifier(max_depth=3, objective='binary:logistic', num_round=12, eta=1)
scores = cross_val_score(clf_bst, X, y, cv=10)
print(scores)
print("Accuracy: %0.4f" % scores.mean())


xgb.plot_tree(bst, num_trees=0)
plt.savefig('./bsttree0.jpg', dpi=2000)
plt.show()
xgb.plot_importance(bst)
plt.savefig('./bstimpt.jpg')
plt.show()


