import numpy as np
import matplotlib.pyplot as plt
from SupervisedML_Classification_Data import plot_first_1_features, generate_MyData_1Features, plot_first_2_features, generate_random_data_2Features, generate_MyData_2Features, import_iris

import matplotlib.pyplot as plt

#Data

#X_train,X_test,y_train,y_test = generate_MyData_1Features()
#X_train,X_test,y_train,y_test = generate_MyData_2Features()
X_train,X_test,y_train,y_test = generate_random_data_2Features()
#X_train,X_test,y_train,y_test = import_iris()


#Algo
algo = 'MLP'
    
if algo == 'KNN':
    from sklearn.neighbors import KNeighborsClassifier 
    model = KNeighborsClassifier(n_neighbors=1)
elif algo == 'Perceptron':
    from sklearn.linear_model import Perceptron
    model = Perceptron()
elif algo == 'Logistic_Regression':
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
elif algo == 'MLP':
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(1,), max_iter=4000, activation = 'logistic', solver = 'sgd', alpha=0.01, 
                      batch_size=1, verbose = True)


#Training
model.fit(X_train,y_train)

#Prediction
y_pred = model.predict(X_test)
print(model.score(X_test, y_test))




#Plotting
if X_train.shape[1] ==1 : # Only use plot_first_1_features if the data has 1 column
    plot_first_1_features(X_train,y_train,X_test,y_pred) # 2d+ Data for all algo's
elif X_train.shape[1] >=2 : # Only use plot_first_2_features if the data has 2 column
    plot_first_2_features(X_train,y_train,X_test,y_pred) # 2d+ Data for all algo's


'''
if algo=='Perceptron' or algo == 'logistic_Regression':
    m = model. coef_[0,0]
    b = model.intercept_[0]
    
    x_axis = np.array([X_train.min(), X_train.max()])
    z = m * x_axis + b # Decision function/boundary
    plt.text(x_axis.max()-4,z.max()-3,'Decision_line',color='red')
    plt.plot(x_axis, z,color='red') 
    Plot1D_Data(X_train,y_train,X_test,y_pred)
    
if algo=='Logistic_Regression': # Only runs when algo is 'Logistic_Regression'
        Activation_function=1/(1+np.exp(-(z))) # Activation function
        plt.plot(x_axis, Activation_function)
        plt.yticks([0,1,2]) #Logistic_Regresion activation function goes from 0 to 1
'''
