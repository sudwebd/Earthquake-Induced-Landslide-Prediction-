# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend("TkAgg")
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/home/btpbatch2/Desktop/BTP3/china_250m.csv')
X = dataset.iloc[:, 2:8].values
y = dataset.iloc[:, 8].values

lngs = pd.read_csv(r'/home/btpbatch2/Desktop/BTP3/china_250m.csv',usecols=[0]).values.tolist()
lats = pd.read_csv(r'/home/btpbatch2/Desktop/BTP3/china_250m.csv',usecols=[1]).values.tolist()


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
onehotencoder1 = OneHotEncoder(categorical_features = [4, 5])
X = onehotencoder1.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
lngs_train, lngs_test, lats_train, lats_test = train_test_split(lngs, lats, test_size=0.3, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('one', StandardScaler(), [19, 20, 21, 22])], remainder='passthrough')
X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)
# for i in range (len(y_prob)):
# 	print(y_test[i],',',y_prob[i][1])
# y2 = classifier.predict(X_train)

lats_draw=[]
lngs_draw=[]
mag_draw = []
for i in range(0, len(y_pred)):
	if y_test[i] == y_pred[i]:
		lats_draw.append(lats_test[i][0])
		lngs_draw.append(lngs_test[i][0])
		mag_draw.append(y_prob[i][1])

lats=np.array(lats_draw)
lons=np.array(lngs_draw)
mags = np.array(mag_draw)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# print(cm)
# print((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]))

# import sklearn.metrics as metrics
# # calculate the fpr and tpr for all thresholds of the classification
# probs = classifier.predict_proba(X_test)
# preds = probs[:,1]
# fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
# roc_auc = metrics.auc(fpr, tpr)

# # method I: plt
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()

# cm2 = confusion_matrix(y_train, y2)
# print(cm2)
# print((cm2[0][0]+cm2[1][1])/(cm2[0][0]+cm2[0][1]+cm2[1][0]+cm2[1][1]))

# [[67211  4765]
#  [ 5144 40972]]
# 0.9160908444263794

# colors = ['white','green','yellow','orange','red']
col_draw = []
for i in range(0, len(mag_draw)):
	if(mag_draw[i]<=0.1):
		col_draw.append('#e1f6f4')
	elif(mag_draw[i]>0.1 and mag_draw[i]<=0.2):
		col_draw.append('#33ffd1')
	elif(mag_draw[i]>0.2 and mag_draw[i]<=0.3):
		col_draw.append('#33ff99')
	elif(mag_draw[i]>0.3 and mag_draw[i]<=0.4):
		col_draw.append('#33ff5e')
	elif(mag_draw[i]>0.4 and mag_draw[i]<=0.5):
		col_draw.append('#b8ff33')
	elif(mag_draw[i]>0.5 and mag_draw[i]<=0.6):
		col_draw.append('#f3ff33')
	elif(mag_draw[i]>0.6 and mag_draw[i]<=0.7):
		col_draw.append('#ffd733')
	elif(mag_draw[i]>0.7 and mag_draw[i]<=0.8):
		col_draw.append('#ffac33')
	elif(mag_draw[i]>0.8 and mag_draw[i]<=0.9):
		col_draw.append('#fb2e1d')
	elif(mag_draw[i]>0.9):
		col_draw.append('#931005')

#

# r=[0 for i in range(B[1]) for j in range(B[0])]
# for i in range(len(B[0])):
# 	for j in range(len(B[1])):
# 		avg = 0.125*H[i][j][0] + 0.375*H[i][j][1] + 0.625*H[i][j][2] + 0.875*H[i][j][3]
# 		r[i][j]=avg

# plotting
from mpl_toolkits.basemap import Basemap

fig = plt.figure()
ax = plt.subplot(1,1,1)
print(lngs_draw[0],lats_draw[0],mag_draw[0])
# 12.531685, 71.815680 
# 55.083610, 147.696487
# earth = Basemap(ax=ax, projection='merc', llcrnrlat=12.531685, urcrnrlat=55.083610, llcrnrlon=71.815680, urcrnrlon=147.696487)
earth = Basemap(projection='merc', lon_0=100, llcrnrlat=12.531685, urcrnrlat=55.083610, llcrnrlon=71.815680, urcrnrlon=147.696487, lat_ts=0.0)
# earth = Basemap(projection='merc', lon_0=100, llcrnrlat=27.0007, urcrnrlat=34.59660, llcrnrlon=101.883988, urcrnrlon=105.9884, lat_ts=0.0)
earth.fillcontinents(color='#ffe6cc', lake_color='#dae8fc')
earth.drawcountries()
earth.drawstates()
earth.drawcoastlines()
earth.plot(103.367, 31.021,'white', marker='*', markersize=10, zorder=12)
earth.plot(102.956, 30.284,'white', marker='*', markersize=10, zorder=12)
earth.plot(104.2, 34.5,'white', marker='*', markersize=10, zorder=6)
earth.plot(103.427, 27.245,'black', marker='*', markersize=10, zorder=6)

nx, ny = 500, 500

# compute appropriate bins to histogram the data into
lon_bins = np.linspace(lons.min(), lons.max(), nx+1)
lat_bins = np.linspace(lats.min(), lats.max(), ny+1)
# mag_bins = np.linspace(mags.min(), mags.max(), 10+1)

# print(lon_bins)

density=[[0 for i in range(nx+1)] for j in range(ny+1)]
freq = [[0 for i in range(nx+10)] for j in range(ny+10)]
lons_min = lons.min()
lats_min = lats.min()
lons_max = lons.max()
lats_max = lats.max()
print(lats_min, lats_max, lons_min, lons_max)
lons_diff = (lons_max-lons_min)/nx
lats_diff = (lats_max-lats_min)/ny

for i in range(len(lngs_draw)):
	off_lons = lngs_draw[i]-lons_min
	off_lats = lats_draw[i]-lats_min
	xlons = int(off_lons/lons_diff)
	xlats = int(off_lats/lats_diff)
	freq[xlats][xlons]+=1
	density[xlats][xlons]+=mag_draw[i]

for i in range(nx+1):
	for j in range(ny+1):
		if freq[i][j]>0:
			density[i][j]=density[i][j]/freq[i][j]

# Histogram the lats and lons to produce an array of frequencies in each box.
# Because histogram2d does not follow the cartesian convention 
# (as documented in the numpy.histogram2d docs)
# we need to provide lats and lons rather than lons and lats
# density, _, _ = np.histogram2d(lats, lons, [lat_bins, lon_bins])
# density, _ = np.histogramdd(lats, lons, [lat_bins, lon_bins, mag_bins])


# Turn the lon/lat bins into 2 dimensional arrays ready 
# for conversion into projected coordinates
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins, lat_bins)

# convert the xs and ys to map coordinates
xs, ys = earth(lon_bins_2d, lat_bins_2d)
# from astropy.convolution import convolve
# from astropy.convolution.kernels import Gaussian2DKernel
# convolve(density, Gaussian2DKernel(stddev=2))
plt.pcolormesh(xs, ys, density,zorder=5, alpha = 0.8)
plt.colorbar(orientation='horizontal')

# overlay the scatter points to see that the density 
# is working as expected
# plt.scatter(*earth(lons, lats), zorder = 12, color = 'white', alpha = 0.1)

plt.show()