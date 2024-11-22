import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

col_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'exang', 'oldpeak','slope','ca','thal','target']
pima = pd.read_csv("cleaned_merged_heart_dataset.csv", header=None, names=col_names)
pd.set_option('display.max_columns', None)
le = LabelEncoder()
feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
for col in pima.columns:
    pima[col] = le.fit_transform(pima[col])

X = pima[feature_cols]
Y = pima.target
print(pima.head())
X_train, X_test, Y_train, Y_test, = train_test_split(X, Y, test_size=.03, random_state=1)
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(Y_test,Y_pred))
