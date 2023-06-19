import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import pandas as pd

# The data contains 2 parts, X and y
# X is the features like student height or student weight
# y is the labels or classes
# n_features is the number of columns in X
# n_samples is the number of datapoints in X and y

###DATA


data = 'MyData'
if data == 'Sklearn DataGenerator':
    n_samples = 100
    n_features = 1
    X,y = make_classification(n_samples=n_samples, n_features= n_features, n_redundant = 0, n_informative = 1, n_clusters_per_class = 1)
    
elif data == 'MyData':
    X = np.array([[0],[1],[2],[4],[3], [6],[8],[7],[9]]) # Step 1 (Data)
    y = np.array([ 0,  0,  0,  0, 0,     1,  1,  1,  1]) 
    n_samples,n_features = X.shape


elif data == 'Iris':
    from sklearn.datasets import load_iris
    bunch = load_iris() 
    X = bunch.data
    y = bunch.target
    n_samples, n_features = X.shape
    
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.9, random_state = 40)
    
    




    


from sklearn.linear_model import Perceptron
model = Perceptron()
    
###LEARNING/FITTING
model.fit(X_train,y_train)  
m = model.coef_[0,0]
b = model.intercept_[0]

    
x_axis = np.array([X_train.min(), X_train.max()])
y_axis = m * x_axis + b 
plt.plot(x_axis,y_axis) 

plt.scatter(X_train,np.zeros(len(y_train)), c = y_train, s = 5)
#plt.axis('equal')


'''
###PREDICTION
y_test_hat = model.predict(X_test)


###PLOTTING
color = np.where(y_train == 1,'g','r') # Training Color
color2 = np.where(y_test == 1,'g','r') # Testing Color
color3 = np.where(y_test_hat == 1, 'g', 'r') # Algorithm Ouput Color
fig = plt.figure()

if n_features == 1 and algo == 'Perceptron':
    ax = fig.add_subplot()
    ax.scatter(X_train, np.zeros(len(X_train)), s=5, c=color, marker='o')
    ax.scatter(X_test, np.zeros(len(X_test)), s=100, c=color3,  marker='x')
    #ax.set_ylim([-0.03,1]) # Uncomment to make the plots on the bottom
    
elif n_features == 2:
    ax = fig.add_subplot()
    ax.scatter(X_train[:,0], X_train[:,1], s=5, c=color, marker='o')
    ax.scatter(X_test[:,0], X_test[:,1], s=100, c=color3,marker='x' )
    ax.grid()
    
elif n_features == 3:
    x = X_train[:,0]
    y = X_train[:,1]
    z = X_train[:,2]
    
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x , y, z, s=5, c=color, marker='o')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    
    x2 = X_test[:,0]
    y2 = X_test[:,1]
    z2 = X_test[:,2]
    fig = plt.figure()
   
    ax.scatter(x2, y2, z2, s=100, c=color3, marker='x')
 
else:
    x = X_train[:,1] 
    y = X_train[:,2]
    z = X_train[:,3]
    
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x,y,z, s=5, c=y_train, marker='o')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    
    x2 = X_test[:,0]
    y2 = X_test[:,1]
    z2 = X_test[:,2]
    fig = plt.figure()
   
    ax.scatter(x2 ,y2 ,z2 , s=100, c=y_test, marker='x')
    
'''
     
