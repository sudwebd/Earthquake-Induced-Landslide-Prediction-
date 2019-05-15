# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/home/btpbatch3/Desktop/BTP3/April/Japan data/japan_data_250mfnl.csv')
X = dataset.iloc[:, 2:8].values
y = dataset.iloc[:, 8].values
lngs = dataset.iloc[0]
lats = dataset.iloc[1]
# var = 0
# for i in range(0, len(y)):
# 	if y[i]==0:
# 		var = var + 1
# print(var, len(y)-var)


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
# Dummy Encoding coz our algorithm would consider differnt values assigned to the catagorical classes as quantitative values
# The problem is solves with the help of Dummy Encoding
onehotencoder1 = OneHotEncoder(categorical_features = [4, 5])
X = onehotencoder1.fit_transform(X).toarray()
# for i in range(0,10):
# 	print(X[i])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
lngs_train, lngs_test, lats_train, lats_test = train_test_split(lngs, lats, test_size=0.3, random_state=0)

print(X_train[0])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('one', StandardScaler(), [17, 18, 19, 20])], remainder='passthrough')
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)
# for i in range(0,10):
# 	print(X_train[i])


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# y2 = classifier.predict(X_train)
print(y_pred[0])

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))
# cm2 = confusion_matrix(y_train, y2)
# print(cm2)
# print((cm2[0][0]+cm2[1][1])/(cm2[0][0]+cm2[0][1]+cm2[1][0]+cm2[1][1]))


# # plotting
# from mpl_toolkits.basemap import Basemap

# fig = plt.figure()
# ax = plt.subplot(1,1,1)

# earth = Basemap(ax=ax, llcrnrlat=28, urcrnrlat=47.5, llcrnrlon=125, urcrnrlon=149)
# earth.shadedrelief()
# mags=[10 for row in lngs_test]
# ax.scatter(lngs_test, lats_test, mags_test, c='red',alpha=0.5, zorder=10)
# plt.show()
