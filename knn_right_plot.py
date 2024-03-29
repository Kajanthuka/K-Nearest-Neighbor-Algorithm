#Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

# Load data
irisData = load_iris()

#Create target arrays
X = irisData.data
Y = irisData.target


#Split into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,random_state=42)

neighbors = np.arange(1,9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))


#Loop over K values
for i,k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,Y_train)

    #Compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train,Y_train)
    test_accuracy[i] = knn.score(X_test,Y_test)

#Generate plot
plt.plot(neighbors,test_accuracy,label= 'Testing dataset Accuracy')
plt.plot(neighbors,train_accuracy, label = 'Training dataset Accuracy')

plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()