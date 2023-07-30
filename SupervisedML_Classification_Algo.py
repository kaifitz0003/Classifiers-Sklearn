import numpy as np
import matplotlib.pyplot as plt
from SupervizedML_Plot import Plot1D_Data
from SupervizedML_ClassificationData import MyData_1Feature
import matplotlib.pyplot as plt
#Data
X_train,X_test,y_train,y_test = MyData_1Feature()

 
#Algo
algo = 'Logistic_Regression'


    
if algo == 'KNN':
    from sklearn.neighbors import KNeighborsClassifier 
    model = KNeighborsClassifier()
elif algo == 'Perceptron':
    from sklearn.linear_model import Perceptron
    model = Perceptron()
elif algo == 'Logistic_Regression':
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()


#Training
model.fit(X_train,y_train)
if algo == 'Perceptron' or 'Logistic_Regression':
    m = model. coef_[0,0]
    b = model.intercept_[0]

#Prediction

#Plotting

Plot1D_Data(X_train,y_train,X_test,y_test)
if algo=='Perceptron' or 'logistic_Regression':
    x_axis = np.array([X_train.min(), X_train.max()])
    Decision_line = m * x_axis + b # Decision function/boundary
    plt.plot(x_axis, Decision_line) 

if algo=='Logistic_Regression':
        Activation_function=1/(1+np.exp(-(Decision_line))) # Activation function
        plt.plot(x_axis, Decision_line)
        plt.plot(x_axis, Activation_function)
        plt.yticks([0,1,2]) #Logistic_Regresion activation function goes from 0 to 1

