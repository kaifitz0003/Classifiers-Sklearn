from sklearn.neighbors import KNeighborsClassifier 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

#Data
MyData = True
if MyData:
    X = np.array([[0],[1],[2],[6],[7],[8]]) # Step 1 (Data)
    y = np.array([0,0,0,1,1,1])
    
else:
    X,y = make_classification(n_samples=1000, n_features=2, n_redundant=0)
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.9, random_state = 40)

#Algorithm
k = 3 #HYPER PAREMATER 
model = KNeighborsClassifier(n_neighbors = k)# Step 2 (Algoithm)

#Training
model.fit(X_train, y_train) # Step 3 (Training/learning)
Training_accuracy = model.score(X_train, y_train) #finds accuracy
print('Training Accuracy:' , Training_accuracy)

#Predictiom/Testing
y_test_hat = model.predict(X_test) # Step 4 (Prediction)
print(y_test_hat)
Prediction_accuracy = model.score(X_test, y_test) #finds accuracy
print('Prediction Accuracy:' , Prediction_accuracy)
