from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from SupervisedML_Classification_Data import import_iris, import_wine


#X_train,X_test,y_train,y_test = import_iris()
#model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=3000, verbose=True) # Hyperparameters (settings) work well for load_iris
X_train,X_test,y_train,y_test = import_wine()
model = MLPClassifier(hidden_layer_sizes=(80,90), max_iter=3000, verbose=True) # Hyperparameters (settings) work well for load_iris
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(model.score(X_test, y_test))

