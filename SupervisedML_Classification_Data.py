# The data contains 2 parts, X and y
# X is the features like student height or student weight, ALWAYS GOING TO BE A 2D ARRAY!
# y is the labels or classes, ALWAYS GOING TO BE A 1D ARRAY!
# n_features is the number of columns in X
# n_samples is the number of datapoints in X and y

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

### DATA GENERATION

def generate_MyData_1Features():
    
    X = np.array([[0],[1],[2],[4],[3], [6],[7],[8],[11]]) # Step 1 (Data)
    y = np.array([ 0,  0,  0,  0, 0,     1,  1,  1,  1])
    X_train,X_test,y_train,y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test

def generate_MyData_2Features():
    X = np.array([[1,2],[3,4],[6,6],[7,8],[9,10],[11,12],[13,14]])
    y = np.array([0,0,1,1,1,1,1])
    X_train,X_test,y_train,y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test

def generate_random_data_2Features():
    from sklearn.datasets import make_classification
    X,y = make_classification(n_samples = 40, n_features = 2,n_redundant = 0, n_informative = 2)
    X_train,X_test,y_train,y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test


def import_iris():
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
    from sklearn.datasets import load_iris
    bunch = load_iris() # Since we didnt add as_frame = True, bunch.data stays as a NumPy array instead of a Pandas DataFrame
    X = bunch.data
    y = bunch.target
    X_train,X_test,y_train,y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test

def import_wine():
    from sklearn.datasets import load_wine
    bunch = load_wine()
    X = bunch.data
    y = bunch.target
    X_train,X_test,y_train,y_test = train_test_split(X, y)
    return X_train, X_test, y_train, y_test







### PLOTTING

def Plot1D_Data_Subplots(X_train, y_train, X_test, y_test):
    # We are using the ax style because we wanted multiple colorbars.
    fig = plt.figure()
    
    ax1 = fig.add_subplot(1,2,1) 
    sc=ax1.scatter(X_train, np.zeros(len(y_train)), c=y_train, cmap='copper')
    ax1.set_title('Training Data')
    ax1.set_yticks([0,1,2])
    ax1.set_xlim([min(X_train)-1,max(X_train)+1])
    plt.colorbar(sc, ax=[ax1])# , location='top'
    
    ax2 = fig.add_subplot(1,2,2)
    ax2.scatter(X_test, np.zeros(len(y_test)), c=y_test, cmap='copper')
    ax2.set_title('Testing Data')
    ax2.set_yticks([0,1,2])
    ax2.set_xlim([min(X_test)-1,max(X_test)+1])
    plt.colorbar(sc, ax=[ax2])

    #plt.subplots_adjust(top=0.725,bottom=0.535,left=0.125,right=0.9,hspace=0.2,wspace=0.05)

def plot_first_1_features(X_train,y_train,X_test,y_pred): 

    '''    
    Plots X_train and X_test on a number line and gets its color from y_train and y_pred.
    
    We want to see what our algo got, not the anwser. That is why we called it y_pred instead of y_test.
    This function is used when X_train and X_test have only 1 feature because Matplotlib can't make 1d plots. 
    '''
    plt.scatter(X_train, np.zeros(len(y_train)), c=y_train, cmap='copper', s=30, label='Training Data')
    plt.scatter(X_test, np.zeros(len(y_pred)), c=y_pred, cmap='copper', marker='x',s=100, label='Testing Data')
    plt.title('Training and Testing Data')
    
    plt.xlim([np.min([np.min(X_train), np.min(X_test)])-1,np.max([np.max(X_train), np.max(X_test)])+1]) 
    
    plt.colorbar()
    
    plt.legend()
    
    
def plot_first_2_features(X_train, y_train, X_test, y_pred):
    ''' 
    Plots first 2 columns of the input data. 
                
    X_train is a 2D NumPy array and  must have at least 2 columns. 
    The number of rows/samples/examples/observation doesnt matter.
    '''
     
    plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='copper', s=30, label='Training Data') 
    plt.scatter(X_test[:,0], X_test[:,1], c=y_pred, cmap='copper', marker='x',s=100, label='Testing Data')
    plt.title('Training and Testing Data')
    
    
    plt.colorbar()
    
    plt.legend()    
    
    
'''
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




    
    
    
