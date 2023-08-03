#problem statement:(Bharat Intern Data Science Internship)
#Make a system which tells whether the person will be
#save from sinking. What factors were
#most likely lead to success-socio-economic
#status, age, gender and more.


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


data = pd.read_csv("titanic_dataset.csv")


features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'
data = data[features + [target]]
data = data.dropna()  


data = pd.get_dummies(data, columns=['Sex', 'Embarked'])


X = data.drop(target, axis=1)
y = data[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


plt.figure(figsize=(15, 10))
plot_tree(clf, filled=True, feature_names=X.columns.tolist(), class_names=["Not Survived", "Survived"])
plt.show()