from sklearn.neighbors import KNeighborsClassifier 
import numpy as np


X_trg = np.array([[0],[1],[2],[6],[7],[8]]) # Step 1 (Data)
y_trg = np.array([0,0,0,1,1,1])
X_test = np.array([[2.5],[5]])

model = KNeighborsClassifier()# Step 2 (Algorithm)
model.fit(X_trg,y_trg) # Step 3 (Training/learning)

y_test = model.predict(X_test) # Step 4 (Prediction)
print(y_test)
