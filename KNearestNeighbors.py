#Link: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html

from sklearn.neighbors import KNeighborsClassifier 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


#Data
MyData = False
if MyData:
    X = np.array([[0],[1],[2],[4],[3], [6],[7],[8],[11]]) # Step 1 (Data), My data has 1 collumn/feature
    y = np.array([ 0,  0,  0,  0, 0,     1,  1,  1,  1]) 
    n_samples, n_features = X.shape
    
else:
    n_samples = 1000
    n_features = 2
    
    X,y = make_classification(n_samples=n_samples, n_features= n_features, n_redundant = 0)
    
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.9, random_state = 40)


#Algorithm
k = 2 #HYPER PAREMATER 

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

color = np.where(y_train == 1,'g','r')
color2 = np.where(y_test == 1,'g','r')
if n_features == 1:
    
    
    plt.scatter(X_train,np.zeros(len(X_train)), c = color, marker = 'o')
    plt.scatter(X_test, np.zeros(len(X_test)), c = color2, marker = 'x')
    plt.ylim([-0.03,1])
 
    
color3 = np.where(y_train == 1,'g','r')
color4 = np.where(y_test_hat == 1,'g','r')

if n_features == 2 :
    plt.scatter(X_train[:,0], X_train[:,1], s=5, c=color3, marker = 'o')
    plt.scatter(X_test[:,0], X_test[:,1], s=100, c=color4,marker = 'x' )
    plt.axis('equal')
    #plt.scatter(range(len(X_train)), X_train,  c = y_train)
    plt.xlabel('Person ID')
    plt.ylabel('Height')
    plt.grid()
    
elif n_features == 3 : 
    pass
    
