# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/home/btpbatch3/Desktop/BTP3/April/Japan data/japan_data_250mfnl.csv')
X = dataset.iloc[:, 2:8].values
y = dataset.iloc[:, 8].values


# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
# Dummy Encoding coz our algorithm would consider differnt values assigned to the catagorical classes as quantitative values
# The problem is solves with the help of Dummy Encoding
onehotencoder1 = OneHotEncoder(categorical_features = [4, 5])
X = onehotencoder1.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('one', StandardScaler(), [17, 18, 19, 20])], remainder='passthrough')
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)
for i in range (len(y_prob)):
	print(y_test[i],',',y_prob[i][1])

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# print(cm)
# print((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))

# [[2744  852]
#  [ 426 2771]]
# 0.8118651553069336

