#Import necessary modules

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

#Loading data
irisData = load_iris()

#create feature and target arrays
X = irisData.data
Y = irisData.target

#split into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

#Model fit

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, Y_train)

# Predict on dataset which model has not seen before
print(knn.predict(X_test))

#Model accuracy





