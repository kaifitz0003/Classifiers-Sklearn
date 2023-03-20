import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


###DATA


data = 'Sklearn DataGenerator'
if data == 'Sklearn DataGenerator':
    n_samples = 100
    n_features = 1
    
    X,y = make_classification(n_samples=n_samples, n_features= n_features, n_redundant = 0, n_informative = 1, n_clusters_per_class = 1)
    
elif data == 'MyData':
    X = np.array([[0],[1],[2],[4],[3], [6],[7],[8],[11]]) # Step 1 (Data)
    y = np.array([ 0,  0,  0,  0, 0,     1,  1,  1,  1]) 
    
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.9, random_state = 40)
    
    



###PICK ALGORITHM
algo = 'KNN'

if algo == 'Perceptron':
    from sklearn.linear_model import Perceptron
    model = Perceptron()
    
elif algo == 'KNN':
    from sklearn.neighbors import KNeighborsClassifier 
    model = KNeighborsClassifier()
    
    
###LEARNING/FITTING
model.fit(X_train,y_train)  


###PREDICTION
y_test_hat = model.predict(X_test)


###PLOTTING
color = np.where(y_train == 1,'g','r') # Training Color
color2 = np.where(y_test == 1,'g','r') # Testing Color
color3 = np.where(y_test_hat == 1, 'g', 'r') # Algorithm Ouput Color
fig = plt.figure()

if n_features == 1:
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
    ax.scatter(x,y,z, s=5, c=color, marker='o')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    
    x = X_test[:,0]
    y = X_test[:,1]
    z = X_test[:,2]
    fig = plt.figure()
   
    ax.scatter(x,y,z, s=100, c=color3, marker='x')
 

