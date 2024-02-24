# Import necessary libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


#Loadding data
irisData = load_iris()

# Create feature and target arrays
X = irisData.data
Y = irisData.target

# Split into training and test set 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)


#Model
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)


#Calculate the accuracy of the model
print(knn.score(X_test,Y_test))


